#!/usr/bin/env python3
"""Reproducible full-Gaussian ambulance review images via splat-transform.

Uses PlayCanvas's GPU rasterizer, including perspective anisotropic covariance,
opacity and all SH bands present. The iPhone exp11 source has degree-zero SH.
No splats are sampled, decimated, denoised, or recolored by this review tool.

The current viewer does not recenter. Its fixed cameras are engine world
coordinates; source PLY to world = R(-78.243,.463,-4.499) @ diag(-1,-1,1).
The converter supplies the PLY format flip, so apply only the user rotation.
An already exported SOG needs no additional transform.

Examples:
  python3 tools/ambulance/render_review.py --input raw/ambulance_exp11_boot_sharp.ply --label original
  python3 tools/ambulance/render_review.py --input raw/ambulance-cleanup/ambulance_cleaned.ply --label cleaned
  python3 tools/ambulance/render_review.py --input viewers/ambulance/index.sog --frame sog --label published

On macOS the process needs local GPU access, which may require sandbox approval.
Use --dry-run to emit argument arrays for direct approved converter invocations.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / 'raw/ambulance-cleanup/review'
ROTATION = [-78.243, .463, -4.499]
CAMERAS = [
    {'id': 'default', 'label': 'Published interior camera position', 'position': [-1.1, .25, .45], 'target': [.3, -.05, .1], 'fov': 70},
    {'id': 'bench', 'label': 'Bench upholstery and wall pads', 'position': [-.4, .25, -.12], 'target': [-.25, -.35, .65], 'fov': 75},
    {'id': 'cabinets', 'label': 'Cabinet fronts and ceiling edges', 'position': [-.75, .24, .3], 'target': [.35, .12, -.75], 'fov': 68},
    {'id': 'rear', 'label': 'Reverse cabin view', 'position': [.9, .25, -.1], 'target': [-1.2, -.1, .15], 'fov': 75},
]


def resolve_converter(explicit):
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = list((Path.home() / '.npm/_npx').glob('*/node_modules/@playcanvas/splat-transform/bin/cli.mjs'))
    for path in candidates:
        package = json.loads((path.parent.parent / 'package.json').read_text())
        if package['version'] == '3.4.2':
            return path
    raise RuntimeError('Install @playcanvas/splat-transform@3.4.2 or pass --converter /path/to/bin/cli.mjs')


def comma(values):
    return ','.join(str(v) for v in values)


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def commands(args, cameras):
    converter = resolve_converter(args.converter)
    node = shutil.which('node')
    if not node:
        raise RuntimeError('Node.js is required')
    for camera in cameras:
        fov = camera['fov']
        if args.fov_axis == 'horizontal':
            width, height = map(int, args.resolution.lower().split('x'))
            fov = math.degrees(2 * math.atan(math.tan(math.radians(fov) / 2) * height / width))
        output = args.out / f'{args.label}-{camera["id"]}.webp'
        command = [node, str(converter), str(args.input)]
        if args.frame == 'raw':
            command += ['-r', comma(ROTATION)]
        command += [str(output), '--camera-pos', comma(camera['position']),
                    '--camera-target', comma(camera['target']), '--camera-up', comma(camera.get('up', [0, 1, 0])),
                    '--camera-fov', str(fov), '--resolution', args.resolution,
                    '--camera-near', str(args.near), '--background', '0,0,0,1', '-w']
        yield camera, output, command


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--frame', choices=['raw', 'sog'], default='raw', help='Raw training PLY or already exported SOG')
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--cameras', default='default,bench,cabinets,rear')
    parser.add_argument('--camera-file', type=Path, help='JSON array of world-space cameras, overriding built-ins')
    parser.add_argument('--resolution', default='1280x900')
    parser.add_argument('--fov-axis', choices=['vertical', 'horizontal'], default='vertical',
                        help='Camera FOV interpretation; browser uses horizontal on landscape screens, default review uses vertical')
    parser.add_argument('--near', type=float, default=.02)
    parser.add_argument('--converter')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.input = args.input.resolve()
    if not args.input.is_file():
        parser.error(f'Missing input: {args.input}')
    if '/' in args.label or '\\' in args.label:
        parser.error('--label must be a filename label')
    cameras = json.loads(args.camera_file.read_text()) if args.camera_file else CAMERAS
    selected = args.cameras.split(',')
    cameras = [c for c in cameras if c['id'] in selected]
    missing = set(selected) - {c['id'] for c in cameras}
    if missing:
        parser.error(f'Unknown cameras: {sorted(missing)}')
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        'input': str(args.input), 'input_frame': args.frame,
        'input_sha256': file_sha256(args.input),
        'rotation_degrees': ROTATION if args.frame == 'raw' else None,
        'renderer': '@playcanvas/splat-transform 3.4.2 GPU perspective anisotropic Gaussian rasterizer',
        'camera_space': 'Fixed PlayCanvas world coordinates; no centroid recentering',
        'resolution': args.resolution, 'near': args.near, 'input_fov_axis': args.fov_axis,
        'background': [0, 0, 0, 1], 'renders': [],
    }
    for camera, output, command in commands(args, cameras):
        print(json.dumps({'camera': camera['id'], 'command': command}), flush=True)
        record = {'camera': camera, 'output': str(output), 'command': command}
        if not args.dry_run:
            started = time.monotonic()
            subprocess.run(command, check=True)
            record['elapsed_seconds'] = round(time.monotonic() - started, 3)
            record['sha256'] = hashlib.sha256(output.read_bytes()).hexdigest()
        manifest['renders'].append(record)
    suffix = 'commands' if args.dry_run else 'manifest'
    (args.out / f'{args.label}-{suffix}.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
