#!/usr/bin/env python3
"""Render the systematic ambulance surface inspection atlas at a fixed near plane.

Uses full anisotropic Gaussian rasterization with no point sampling. The camera
file is frozen for all candidate comparisons. Defaults intentionally expose
close-surface fog: near .02, black background, no photographic postprocessing.
"""
import argparse
import html
import json
from pathlib import Path
import subprocess
import time

from render_review import ROOT, ROTATION, commands, file_sha256

CAMERAS = Path(__file__).with_name('pass2-cameras.json')
OUT = ROOT / 'raw/ambulance-cleanup/pass2/review'


def make_sheets(out, cameras):
    manifests = sorted(out.glob('*-manifest.json'))
    labels = [m.name[:-14] for m in manifests]
    metadata = {m.name[:-14]: json.loads(m.read_text()) for m in manifests}
    configurations = {(m.get('camera_file_sha256'), m.get('near'), m.get('resolution'))
                      for m in metadata.values()}
    if len(configurations) == 1 and metadata:
        m = next(iter(metadata.values()))
        settings_note = (f"Identical camera file, near plane {m.get('near')}, "
                         f"resolution {m.get('resolution')}. ")
    else:
        settings_note = ('This atlas includes different camera files or clipping settings. '
                         'Registered reference views may use inverse-mapped cameras and '
                         'scale-adjusted clipping; consult each manifest before comparing. ')
    if 'baseline' in labels:
        labels.remove('baseline'); labels.insert(0, 'baseline')
    for region in ['all'] + list(dict.fromkeys(c['region'] for c in cameras)):
        chosen = cameras if region == 'all' else [c for c in cameras if c['region'] == region]
        body = ['<!doctype html><meta charset="utf-8"><title>Ambulance surface review</title>',
                '<style>body{background:#121820;color:#e9eef2;font:15px system-ui;margin:20px}h1{font-size:24px}h2{font-size:18px;margin-top:28px}.row{display:flex;gap:12px;overflow-x:auto}figure{margin:0;min-width:480px;width:720px}img{width:100%;display:block}figcaption{padding:7px 0;color:#bdcad5}p{color:#b6c5d0;max-width:1000px}</style>',
                f'<h1>Surface inspection: {html.escape(region)}</h1>',
                '<p>Full Gaussian renders. ' + html.escape(settings_note) +
                'Source labels identify each candidate. Click a render for full resolution. Each manifest records the source SHA256 and exact renderer command; the final comparison page identifies the accepted baseline.</p>']
        for camera in chosen:
            body += [f'<h2>{html.escape(camera["id"])} · {html.escape(camera["label"])}</h2>', '<div class="row">']
            for label in labels:
                src = f'{label}-{camera["id"]}.webp'
                if (out / src).exists():
                    body.append(f'<figure><a href="{src}"><img src="{src}" loading="lazy"></a><figcaption>{html.escape(label)}</figcaption></figure>')
            body += ['</div>']
        (out / f'atlas-{region}.html').write_text('\n'.join(body))
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        print('HTML contact sheets saved; run --sheets-only with Pillow for JPEG contact sheets.')
        return
    for region in list(dict.fromkeys(c['region'] for c in cameras)):
        chosen = [c for c in cameras if c['region'] == region]
        for label in labels:
            present = [c for c in chosen if (out / f'{label}-{c["id"]}.webp').exists()]
            if not present:
                continue
            canvas = Image.new('RGB', (1000, 58 + ((len(present) + 1) // 2) * 415), (18, 24, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((15, 12), f'{label} / {region} / full anisotropic Gaussian inspection', fill='white')
            m = metadata[label]
            draw.text((15, 31), f"Contact sheet only: open original WebP. Resolution {m.get('resolution')}; native near {m.get('near')}; camera file {str(m.get('camera_file_sha256', ''))[:12]}.", fill=(180, 195, 210))
            for i, camera in enumerate(present):
                x, y = (i % 2) * 500, 58 + (i // 2) * 415
                with Image.open(out / f'{label}-{camera["id"]}.webp') as img:
                    canvas.paste(img.resize((500, 375), Image.Resampling.LANCZOS), (x, y + 32))
                draw.text((x + 8, y + 8), camera['id'] + ' / ' + camera['label'], fill='white')
            canvas.save(out / f'{label}-{region}-contact.jpg', quality=93)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path)
    p.add_argument('--label')
    p.add_argument('--out', type=Path, default=OUT)
    p.add_argument('--camera-file', type=Path, default=CAMERAS)
    p.add_argument('--cameras', default='all')
    p.add_argument('--near', type=float, default=.02)
    p.add_argument('--resolution', default='1000x750')
    p.add_argument('--frame', choices=['raw', 'sog'], default='raw')
    p.add_argument('--converter')
    p.add_argument('--sheets-only', action='store_true')
    args = p.parse_args()
    cameras = json.loads(args.camera_file.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if args.sheets_only:
        make_sheets(args.out, cameras); return
    if not args.input or not args.label:
        p.error('--input and --label are required to render')
    args.input = args.input.resolve()
    args.fov_axis = 'vertical'
    requested = set(args.cameras.split(','))
    known = {c['id'] for c in cameras}
    if args.cameras != 'all' and requested - known:
        p.error('unknown camera(s): ' + ', '.join(sorted(requested - known)))
    selected = cameras if args.cameras == 'all' else [c for c in cameras if c['id'] in requested]
    existing = args.out / f'{args.label}-manifest.json'
    manifest = {'input': str(args.input), 'input_sha256': file_sha256(args.input),
                'input_frame': args.frame, 'rotation_degrees': ROTATION if args.frame == 'raw' else None,
                'camera_file': str(args.camera_file.resolve()), 'camera_file_sha256': file_sha256(args.camera_file),
                'near': args.near, 'resolution': args.resolution, 'renderer': '@playcanvas/splat-transform 3.4.2 full anisotropic GPU rasterizer',
                'fov_axis': 'vertical', 'renders': []}
    if existing.exists():
        old = json.loads(existing.read_text())
        if all(old.get(k) == manifest[k] for k in ['input_sha256', 'camera_file_sha256', 'near', 'resolution']):
            manifest['renders'] = [r for r in old['renders'] if r['camera']['id'] not in {c['id'] for c in selected}]
    for camera, output, command in commands(args, selected):
        start = time.monotonic()
        print(f'Render {args.label}: {camera["id"]}', flush=True)
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout); print(result.stderr); result.check_returncode()
        manifest['renders'].append({'camera': camera, 'output': str(output), 'sha256': file_sha256(output),
                                    'command': command, 'elapsed_seconds': round(time.monotonic() - start, 3)})
        existing.write_text(json.dumps(manifest, indent=2) + '\n')
        print(f'  saved {output.name}', flush=True)
    if file_sha256(args.input) != manifest['input_sha256']:
        manifest['invalid_reason'] = 'Input changed during the render batch; rerender this label.'
        existing.write_text(json.dumps(manifest, indent=2) + '\n')
        raise RuntimeError(manifest['invalid_reason'])
    manifest['batch_verified'] = True
    existing.write_text(json.dumps(manifest, indent=2) + '\n')
    make_sheets(args.out, cameras)


if __name__ == '__main__':
    main()
