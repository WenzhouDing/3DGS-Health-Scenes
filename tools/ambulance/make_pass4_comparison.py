#!/usr/bin/env python3
"""Compare the floor/wall repair against the previously accepted scene."""
import argparse
import json
import os
from pathlib import Path

from make_reference_comparison import HTML, ROOT, collect


def load_into(destination, label, directory, output, strict=False, source_cache=None):
    records, _ = collect(label, directory, strict, source_cache)
    for key, record in records.items():
        record['image'] = os.path.relpath(directory / record['image'], output)
        destination[key] = record


def contact_sheet(output, models, candidate, camera, filename, after_caption):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (1616, 696), '#10151b')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 27)
        small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except OSError:
        font = small = ImageFont.load_default()
    for i, (label, caption) in enumerate([
            ('baseline', 'Before: accepted scene'),
            (candidate, after_caption)]):
        draw.text((i * 816 + 16, 14), caption, font=font, fill='#e9eef3')
        with Image.open(output / models[label][camera]['image']) as image:
            canvas.paste(image.resize((800, 600), Image.Resampling.LANCZOS),
                         (i * 816, 60))
    draw.text((16, 671),
              'Same camera and clipping · full Gaussian renders · resized for comparison only',
              font=small, fill='#bdc9d4')
    canvas.save(output / filename, quality=96)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='pass4-final')
    parser.add_argument('--review', type=Path,
                        default=ROOT / 'raw/ambulance-cleanup/pass4/review')
    parser.add_argument('--notes', type=Path)
    args = parser.parse_args()
    previous = ROOT / 'raw/ambulance-cleanup/pass3/review'
    cameras = json.loads((args.review.parent / 'review-cameras.json').read_text())
    cameras += json.loads((previous.parent / 'diagnostic-cameras.json').read_text())
    cameras += json.loads((ROOT / 'tools/ambulance/pass2-cameras.json').read_text())
    models = {key: {} for key in ['baseline', 'original', 'insta-aligned', args.candidate]}
    cache = {}
    load_into(models['baseline'], 'reference-repair-final', previous, args.review)
    load_into(models['baseline'], 'pass4-baseline', args.review, args.review)
    for label in ['original', 'insta-aligned']:
        load_into(models[label], label, previous, args.review)
        load_into(models[label], label, args.review, args.review)
    load_into(models[args.candidate], args.candidate, args.review, args.review, True, cache)
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
    if baseline_hashes != {'038afc4739c5db173bbf2f46d7f5c49c3d41b64f903ee16206ecacd3951af347'}:
        raise ValueError('Baseline differs from the accepted pass3 scene')
    notes = json.loads(args.notes.read_text()) if args.notes else {}
    names = {'original': 'Original iPhone', 'baseline': 'Previously accepted scene',
             'insta-aligned': 'Aligned Insta360 reference',
             args.candidate: 'Floor and wall repair'}
    data = {'candidate': args.candidate, 'names': names, 'cameras': cameras,
            'models': models, 'notes': notes,
            'candidate_sha256': next(iter(hashes)),
            'baseline_sha256': next(iter(baseline_hashes))}
    page = HTML.replace('Ambulance · reference-guided repair', 'Ambulance · floor and wall repair')
    page = page.replace('Ambulance: reference-guided repair', 'Ambulance: floor and wall repair')
    page = page.replace(
        'Compare the original iPhone capture with the revised result from the same viewpoint.',
        'Compare the previously accepted scene with the floor and wall repair from the same viewpoint.')
    page = page.replace('href="reassessment.html">Earlier comparison and rejected refinement',
                        'href="../../pass3/review/index.html">Previous accepted repair and original capture')
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
    contact_sheet(args.review, models, args.candidate, 'floor-front-down',
                  'comparison-floor.jpg', 'After: floor and bench face')
    contact_sheet(args.review, models, args.candidate, 'left-grazing',
                  'comparison-wall.jpg', 'After: wall recess and protector')
    contact_sheet(args.review, models, args.candidate, 'left-seats',
                  'comparison-left-seats.jpg', 'After: backrest material')
    print(args.review / 'index.html')


if __name__ == '__main__':
    main()
