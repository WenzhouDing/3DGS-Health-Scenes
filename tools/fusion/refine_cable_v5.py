#!/usr/bin/env python3
"""Read-only native source feature audit for the screen-right gray cable/socket.

This is the mannequin's anatomical LEFT arm. The right arm has yellow tubes.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, read_scan, transform_gaussians
from render_gaussians import atlas, render

OUT = ROOT / 'raw/fusion-work/refinement-v5/cable'
EVIDENCE = ROOT / 'tools/fusion/right-arm-axial-evidence.json'
BASE = ROOT / 'raw/fusion-work/refinement-v5/baseline'
VIEWS = [('Front', [0, -1, 0]), ('Back', [0, 1, 0]),
         ('Outer', [-1, 0, 0]), ('Inner', [1, 0, 0]),
         ('Back outer', [-.6, 1, .1]), ('Back inner', [.6, 1, .1])]
# Source-native centers, measured from the visible cable's neutral-colored
# Gaussian support and checked in posterior, lateral, and oblique renders.
# Distances are uncalibrated scan units. This selects existing rows only.
BACK_CABLE_PATH = np.array([
    [-.325226665, -.056645535, -.177797079],
    [-.335654855, -.066558499, -.156659901],
    [-.352352649, -.063008990, -.137839079],
    [-.377889365, -.047244374, -.113113210],
    [-.396680921, -.035138723, -.096713796],
    [-.419605762, -.018297317, -.077197969],
    [-.436803728, +.000826395, -.059817389],
    [-.447356045, +.022476419, -.044226285],
    [-.453683197, +.051098667, -.024798727],
    [-.449847728, +.081102382, -.009476733],
    [-.438820004, +.101033665, +.006951213],
    [-.432499498, +.110606745, +.016455023],
    [-.427000000, +.118000000, +.022000000],
])
CABLE_RADIUS = .008


def load_native():
    fits = json.loads((BASE / 'feature-transforms.json').read_text())
    evidence = {'sourceHashes': fits['sourceHashes'],
                'parentSourceToFrontRaw': fits['parts']['left_upper_arm']['sourceToFrontRaw'],
                'childSourceToFrontRaw': fits['parts']['left_forearm']['sourceToFrontRaw']}
    result = {}
    for source in ['front', 'back']:
        data, meta = read_scan(ROOT / f'raw/mannequin_{source}_119999.ply')
        assert meta['sha256'] == evidence['sourceHashes'][source]
        lm = json.loads((BASE / f'{source}-landmarks.json').read_text())['landmarks']
        pivot = np.array(lm['left_elbow'])
        axis = np.array(lm['left_wrist']) - pivot
        axis /= np.linalg.norm(axis)
        delta = data[:, :3] - pivot
        station = np.einsum('ni,i->n', delta, axis)
        radius = np.linalg.norm(delta - station[:, None] * axis, axis=1)
        keep = ((station > -.19) & (station < .30) & (radius < .15)
                & (data[:, 10] > -4.6) & (np.exp(data[:, 7:10].max(1)) < .04))
        result[source] = data[keep]
        np.savez_compressed(OUT / f'{source}-native.npz', data=data[keep],
                            indices=np.flatnonzero(keep).astype(np.uint32),
                            station=station[keep], radius=radius[keep])
        print(source, 'native ROI', keep.sum(), flush=True)
    return result, evidence


def mapped(data, transform):
    return transform_gaussians(data, np.array(transform['rotation']),
                               np.array(transform['translation']), transform['scale'])


def distance_to_polyline(points, path):
    nearest = np.full(len(points), np.inf)
    for a, b in zip(path[:-1], path[1:]):
        axis = b - a
        along = np.clip(np.einsum('ni,i->n', points - a, axis) / np.dot(axis, axis), 0, 1)
        nearest = np.minimum(nearest, np.linalg.norm(points - (a + along[:, None] * axis), axis=1))
    return nearest


def export_membership(native, evidence, skip_render=False):
    record = np.load(OUT / 'back-native.npz')
    data = native['back']
    chosen = distance_to_polyline(data[:, :3], BACK_CABLE_PATH) < CABLE_RADIUS
    indices = np.unique(record['indices'][chosen]).astype(np.uint32)
    mask_path = OUT / 'back-left-gray-cable.npy'
    np.save(mask_path, indices)
    hashes = {f'raw/mannequin_{source}_119999.ply': value for source, value in evidence['sourceHashes'].items()}
    meta = {
        'version': 1, 'purpose': 'source part ownership: keep the continuous gray cable attached to its upper-arm socket',
        'sourceCapture': 'back', 'sourceFile': 'raw/mannequin_back_119999.ply',
        'sourceSha256': evidence['sourceHashes']['back'], 'sourceHashes': hashes,
        'baseline': str(BASE.relative_to(ROOT)), 'maskFile': str(mask_path.relative_to(ROOT)),
        'uniqueSourceIndices': len(indices), 'maskMeaning': 'sorted original source PLY vertex rows belonging to the visible gray cable',
        'applyStage': 'after joint boundaries, assign existing retained cable rows to left_upper_arm',
        'part': 'left_upper_arm', 'allowedParts': ['left_upper_arm', 'left_forearm'],
        'anatomicalSide': 'left; appears on screen-right in the user screenshot',
        'algorithm': {'frame': 'back original native XYZ', 'method': 'union of narrow capsules around reviewed native cable centerline',
                      'path': BACK_CABLE_PATH.tolist(), 'radius': CABLE_RADIUS,
                      'maxSourceScale': .04, 'minSourceOpacityLogit': -4.6,
                      'colorSelection': 'none; neutral color was used only to measure path, not discard cable texture'},
        'observations': [
            'The back original capture contains one smooth unbroken cable crossing the rigid arm collar.',
            'The socket lies on the upper arm; splitting this flexible cable between upper and lower arm rigid transforms creates an artificial kink.',
            'The front capture does not provide a reliable matching posterior cable or socket. No paired feature or cable-derived twist is claimed.',
            'Posterior, lateral, and oblique renders of cable-only and cable-removed data show selection follows the cable without removing a skin strip.',
            'The connector at the free tip is included. The surrounding rubber grommet remains naturally upper-arm-owned.',
        ],
        'socketExitNative': BACK_CABLE_PATH[0].tolist(),
        'recommendation': 'Apply the upper-arm transform to every selected retained cable row, regardless of which side of the collar it crosses. Treat future flexible cable animation separately from skin axial articulation.',
        'proof': str((OUT / 'back-cable-final-review.png').relative_to(ROOT)),
        'reproduce': '.venv-fusion/bin/python -B tools/fusion/refine_cable_v5.py --extract-only',
    }
    mask_path.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print('CABLE MEMBERSHIP', len(indices), str(mask_path), flush=True)
    if not skip_render:
        atlas({'Original back source': data, 'Cable membership only': data[chosen], 'Cable omitted for inspection': data[~chosen]},
              OUT / 'back-cable-final-review.png', [-.4, .02, -.08], .35, size=850,
              views=[('Posterior', [0, -1, 0]), ('Side', [1, 0, 0]), ('Posterior oblique', [.65, -1, .15])],
              title='Anatomical left gray cable: original-row membership, native source frame')
    return indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-render', action='store_true')
    parser.add_argument('--extract-only', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    native, evidence = load_native()
    export_membership(native, evidence, args.skip_render)
    if args.extract_only:
        return
    aligned_back = mapped(native['back'], evidence['parentSourceToFrontRaw'])
    if not args.skip_render:
        atlas({'Front original': native['front'], 'Back original under parent fit': aligned_back},
              OUT / 'left-original-source-collar.png', [.415, .03, .31], .49,
              size=900, views=VIEWS, title='Anatomical left collar / gray cable / black socket; no fusion or pruning')


if __name__ == '__main__':
    main()
