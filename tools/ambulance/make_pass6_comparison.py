#!/usr/bin/env python3
"""Build verified floor/laminate comparison against the accepted pass5 scene."""
import argparse
import html
import json
from pathlib import Path

from make_reference_comparison import HTML, ROOT
from make_pass4_comparison import contact_sheet, load_into

BASELINE_SHA = '6e7aa0f7463a3e297531394aaba7054f0b7c7554099c18b168e0dcce5695da47'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='pass6-final')
    parser.add_argument('--review', type=Path,
                        default=ROOT / 'raw/ambulance-cleanup/pass6/review')
    parser.add_argument('--notes', type=Path)
    parser.add_argument('--camera-file', type=Path)
    parser.add_argument('--expected-views', type=int, default=23)
    parser.add_argument('--provenance', help='Plain-language material provenance for this candidate')
    args = parser.parse_args()
    previous = ROOT / 'raw/ambulance-cleanup/pass5/review'
    floor_context = ROOT / 'raw/ambulance-cleanup/pass4/review'
    original = ROOT / 'raw/ambulance-cleanup/pass3/review'
    camera_file = args.camera_file or args.review.parent / 'all-review-cameras.json'
    cameras = json.loads(camera_file.read_text())
    if len(cameras) != args.expected_views or len({x['id'] for x in cameras}) != args.expected_views:
        raise ValueError(f'Expected {args.expected_views} unique fixed review cameras')
    priority = ['wall-material-close', 'wall-material-grazing', 'floor-low-rear', 'floor-low-front', 'floor-cabinet-low-front']
    cameras.sort(key=lambda x: priority.index(x['id']) if x['id'] in priority else len(priority))
    models = {key: {} for key in ['baseline', 'original', 'insta-aligned', args.candidate]}
    load_into(models['baseline'], 'pass5-final', previous, args.review)
    load_into(models['baseline'], 'baseline', args.review, args.review)
    load_into(models['baseline'], 'baseline-strip', args.review, args.review)
    for label in ['original', 'insta-aligned']:
        load_into(models[label], label, original, args.review)
        load_into(models[label], label, floor_context, args.review)
    load_into(models['original'], 'original', args.review, args.review)
    load_into(models['insta-aligned'], 'native', args.review, args.review)
    load_into(models['original'], 'original-strip', args.review, args.review)
    load_into(models['insta-aligned'], 'native-strip', args.review, args.review)
    load_into(models[args.candidate], args.candidate, args.review, args.review, True, {})
    camera_ids = {camera['id'] for camera in cameras}
    models = {label: {key: record for key, record in records.items() if key in camera_ids}
              for label, records in models.items()}
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
        raise ValueError('Baseline differs from the accepted pass5 scene')
    notes = json.loads(args.notes.read_text()) if args.notes else {}
    names = {'original': 'Original iPhone', 'baseline': 'Previously accepted scene',
             'insta-aligned': 'Aligned Insta360 reference',
             args.candidate: 'Floor and laminate repair'}
    qa_path = args.review / 'candidate-qa.json'
    qa = json.loads(qa_path.read_text()) if qa_path.exists() else {}
    rejected = (qa.get('verdict', '').startswith('Rejected') and
                qa.get('source_sha256') == next(iter(hashes)))
    if rejected:
        names[args.candidate] = 'Superseded wall experiment'
    data = {'candidate': args.candidate, 'names': names, 'cameras': cameras,
            'models': models, 'notes': notes,
            'candidate_sha256': next(iter(hashes)), 'baseline_sha256': BASELINE_SHA}
    page = HTML.replace('Ambulance · reference-guided repair', 'Ambulance · floor and laminate repair')
    page = page.replace('Ambulance: reference-guided repair', 'Ambulance: floor and laminate repair')
    page = page.replace(
        'Compare the original iPhone capture with the revised result from the same viewpoint.',
        'Compare the previously accepted scene with the floor and laminate repair from the same viewpoint.')
    page = page.replace('The Insta360 capture provides observed surface material and geometry; it is also available below as a reference.',
                        'Inspect the revised floor and the wall’s actual captured grain, with stretched splats and misplaced source layers reduced. Original and Insta360 captures remain available for context.')
    page = page.replace('href="reassessment.html">Earlier comparison and rejected refinement',
                        'href="../../pass5/review/index.html">Previous accepted corner and wall repair')
    page = page.replace("['original','baseline','refined','insta-aligned']",
                        "['baseline','original','insta-aligned']")
    page = page.replace("[D.candidate,'original','baseline','refined','insta-aligned']",
                        "[D.candidate,'baseline','original','insta-aligned']")
    page = page.replace('Original comparison view', 'Previously accepted comparison view')
    page = page.replace('Historical failed attempts remain available so improvements can be judged against the original, not only against previous experiments.',
                        'The previously accepted scene is the default baseline. Original and native reference captures remain selectable for context.')
    provenance = args.provenance or ('The wall uses actual aligned Insta360 spatial grain, captured colors and directional appearance; elongated splat shapes and misplaced iPhone layers are corrected without a generated wall pattern. '
                                    'Floor coverage repeats a dense captured tread patch along the measured floor, including the cabinet-side strip with observed shadow shading. '
                                    'Floor coverage is reconstructed rather than capture-exact. Lower wall grain remains coarse at some close and grazing angles.')
    page = page.replace('<details><summary>Comparison provenance</summary>',
                        '<details><summary>Comparison provenance</summary><p>' + html.escape(provenance) + '</p>')
    if rejected:
        page = page.replace('<main>', '<main><p style="border:2px solid #e89873;padding:14px;color:#ffd8c5"><strong>Wall experiment rejected — this combined candidate is not installed.</strong> The user rejected the generated wall panel and its blending; the floor repair remains accepted. This page preserves diagnostic history while the wall is revised. The local viewer still uses the previously accepted scene.</p>')
        page = page.replace('Final model SHA256:', 'Rejected candidate SHA256:')
    elif (args.review / 'rejected-wall-experiment.html').exists():
        page = page.replace('<div class="links">', '<div class="links"><a href="rejected-wall-experiment.html">Rejected generated-wall experiment</a>')
    page = page.replace('__DATA__', json.dumps(data).replace('<', '\\u003c'))
    (args.review / 'final-comparison-manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    (args.review / 'index.html').write_text(page)
    for camera, filename in [('wall-material-close', 'comparison-wall.jpg'),
                             ('wall-material-grazing', 'comparison-wall-grazing.jpg'),
                             ('floor-low-rear', 'comparison-floor-low.jpg'),
                             ('floor-front-down', 'comparison-floor.jpg')]:
        contact_sheet(args.review, models, args.candidate, camera, filename, 'After: revised scene')
    if 'floor-cabinet-low-front' in camera_ids:
        contact_sheet(args.review, models, args.candidate, 'floor-cabinet-low-front',
                      'comparison-floor-cabinet.jpg', 'After: revised scene')
    print(args.review / 'index.html')


if __name__ == '__main__':
    main()
