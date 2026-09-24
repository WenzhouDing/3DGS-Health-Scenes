#!/usr/bin/env python3
"""Render uncompressed two-PLY repairs without repeatedly quantizing SH.

The final SOG must still be rendered separately before installation. This
helper records both input hashes, exact cameras and the unmodified image hash.
"""
import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path

from render_review import ROOT, ROTATION, comma, file_sha256, resolve_converter


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, action='append', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--camera-file', type=Path, default=Path(__file__).with_name('pass2-cameras.json'))
    ap.add_argument('--cameras', default='all')
    ap.add_argument('--resolution', default='1000x750')
    ap.add_argument('--near', type=float, default=.02)
    ap.add_argument('--converter')
    args = ap.parse_args()
    if not args.label or '/' in args.label or '\\' in args.label:
        ap.error('Label must be a single filename component')
    cameras = json.loads(args.camera_file.read_text())
    requested = set(args.cameras.split(','))
    if args.cameras != 'all' and requested - {c['id'] for c in cameras}:
        ap.error('Unknown camera')
    cameras = cameras if args.cameras == 'all' else [c for c in cameras if c['id'] in requested]
    inputs = [p.resolve() for p in args.input]
    sources = [{'path': str(p), 'sha256': file_sha256(p)} for p in inputs]
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / f'{args.label}-manifest.json'
    metadata = {'sources': sources, 'input_frame': 'raw', 'rotation_degrees': ROTATION,
                'camera_file': str(args.camera_file.resolve()), 'camera_file_sha256': file_sha256(args.camera_file),
                'near': args.near, 'resolution': args.resolution, 'fov_axis': 'vertical',
                'renderer': '@playcanvas/splat-transform 3.4.2 full anisotropic GPU rasterizer', 'renders': []}
    if dest.exists():
        old = json.loads(dest.read_text())
        if all(old.get(k) == metadata[k] for k in ['sources', 'camera_file_sha256', 'near', 'resolution']):
            metadata['renders'] = [r for r in old['renders'] if r['camera']['id'] not in {c['id'] for c in cameras}]
    for camera in cameras:
        output = args.out / f'{args.label}-{camera["id"]}.webp'
        command = [shutil.which('node'), str(resolve_converter(args.converter)), *map(str, inputs), str(output),
                   '-r', comma(ROTATION), '--camera-pos', comma(camera['position']),
                   '--camera-target', comma(camera['target']), '--camera-up', comma(camera.get('up', [0, 1, 0])),
                   '--camera-fov', str(camera['fov']), '--resolution', args.resolution,
                   '--camera-near', str(args.near), '--background', '0,0,0,1', '-w']
        print(f'Render {args.label}: {camera["id"]}', flush=True)
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            print(result.stdout); print(result.stderr); result.check_returncode()
        metadata['renders'].append({'camera': camera, 'output': str(output.resolve()),
                                    'sha256': file_sha256(output), 'command': command,
                                    'elapsed_seconds': round(time.monotonic()-start, 3)})
        dest.write_text(json.dumps(metadata, indent=2)+'\n')
        print(f'Saved {output.name}', flush=True)
    if sources != [{'path': str(p), 'sha256': file_sha256(p)} for p in inputs]:
        raise RuntimeError('Source changed during render; reject batch')
    metadata['batch_verified'] = True
    dest.write_text(json.dumps(metadata, indent=2)+'\n')


if __name__ == '__main__':
    main()
