#!/usr/bin/env python3
"""Build verified corner/wall comparison against the accepted pass4 scene."""
import argparse
import json
from pathlib import Path

from make_reference_comparison import HTML, ROOT
from make_pass4_comparison import contact_sheet, load_into

BASELINE_SHA = '2a43c179550d1d417e4f3d50207b811e81a31d092a47d7f59d07743a681f016c'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='pass5-final')
    parser.add_argument('--review', type=Path,
                        default=ROOT / 'raw/ambulance-cleanup/pass5/review')
    parser.add_argument('--notes', type=Path)
    args = parser.parse_args()
    previous = ROOT / 'raw/ambulance-cleanup/pass4/review'
    original = ROOT / 'raw/ambulance-cleanup/pass3/review'
    cameras = json.loads((args.review.parent / 'review-cameras.json').read_text())
    if len(cameras) != 19 or len({x['id'] for x in cameras}) != 19:
        raise ValueError('Expected the 19 fixed full-cabin review cameras')
    priority = ['ceiling-rear', 'cabinet-counter', 'front-wall', 'pads-close']
    cameras.sort(key=lambda x: priority.index(x['id']) if x['id'] in priority else len(priority))
    models = {key: {} for key in ['baseline', 'original', 'insta-aligned', args.candidate]}
    load_into(models['baseline'], 'pass4-final', previous, args.review)
    for label in ['original', 'insta-aligned']:
        load_into(models[label], label, original, args.review)
        load_into(models[label], label, previous, args.review)
    load_into(models[args.candidate], args.candidate, args.review, args.review, True, {})
    for camera in cameras:
        key = camera['id']
        for label in models:
            if key not in models[label]:
                raise ValueError(f'Missing required render: {label}/{key}')
        a, b = models['baseline'][key], models[args.candidate][key]
        for field in ['position', 'target', 'fov']:
            if a['camera'][field] != b['camera'][field]:
                raise ValueError(f'Candidate camera mismatch: {key}/{field}')
        if a['camera'].get('up', [0, 1, 0]) != b['camera'].get('up', [0, 1, 0]):
            raise ValueError(f'Candidate up direction mismatch: {key}')
        for field in ['fov_axis', 'near', 'resolution']:
            if a[field] != b[field]:
                raise ValueError(f'Candidate settings mismatch: {key}/{field}')
    hashes = {record['input_sha256'] for record in models[args.candidate].values()}
    if len(hashes) != 1:
        raise ValueError('Candidate views do not share a frozen source')
    baseline_hashes = {record['input_sha256'] for record in models['baseline'].values()}
    if baseline_hashes != {BASELINE_SHA}:
        raise ValueError('Baseline differs from the accepted pass4 scene')
    notes = json.loads(args.notes.read_text()) if args.notes else {}
    names = {'original': 'Original iPhone', 'baseline': 'Previously accepted scene',
             'insta-aligned': 'Aligned Insta360 reference',
             args.candidate: 'Corner and wall repair'}
    data = {'candidate': args.candidate, 'names': names, 'cameras': cameras,
            'models': models, 'notes': notes,
            'candidate_sha256': next(iter(hashes)), 'baseline_sha256': BASELINE_SHA}
    page = HTML.replace('Ambulance · reference-guided repair', 'Ambulance · corner and wall repair')
    page = page.replace('Ambulance: reference-guided repair', 'Ambulance: corner and wall repair')
    page = page.replace(
        'Compare the original iPhone capture with the revised result from the same viewpoint.',
        'Compare the previously accepted scene with the corner and wall repair from the same viewpoint.')
    page = page.replace('href="reassessment.html">Earlier comparison and rejected refinement',
                        'href="../../pass4/review/index.html">Previous accepted floor and wall repair')
    page = page.replace("['original','baseline','refined','insta-aligned']",
                        "['baseline','original','insta-aligned']")
    page = page.replace("[D.candidate,'original','baseline','refined','insta-aligned']",
                        "[D.candidate,'baseline','original','insta-aligned']")
    page = page.replace('Original comparison view', 'Previously accepted comparison view')
    page = page.replace('Historical failed attempts remain available so improvements can be judged against the original, not only against previous experiments.',
                        'The previously accepted scene is the default baseline. Original and native reference captures remain selectable for context.')
    page = page.replace('__DATA__', json.dumps(data).replace('<', '\\u003c'))
    (args.review / 'final-comparison-manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    (args.review / 'index.html').write_text(page)
    for camera, filename in [('ceiling-rear', 'comparison-corners.jpg'),
                             ('cabinet-counter', 'comparison-walls.jpg'),
                             ('pads-close', 'comparison-pads.jpg')]:
        contact_sheet(args.review, models, args.candidate, camera, filename, 'After: revised scene')
    print(args.review / 'index.html')


if __name__ == '__main__':
    main()
