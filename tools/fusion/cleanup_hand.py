#!/usr/bin/env python3
"""Reproduce the reviewed left-hand cleanup, retaining immutable source indices."""
import argparse
import json
import warnings
import numpy as np
from scipy.spatial import cKDTree
from pipeline import (ROOT, BODY_PARTS, read_scan, clean_scan, transform_gaussians,
                      coverage_weights, attenuate)
from render_gaussians import load_fused, atlas

OUT = ROOT / 'raw/fusion-work/cleanup-v3/hand'
BASE = ROOT / 'raw/fusion-work/cleanup-v3/baseline'
MASKS = ROOT / 'tools/fusion/cleanup-masks'
BLEND = {'depthBias': .008, 'feather': .004,
         'preserveUniqueFront': False, 'preserveUniqueBack': False}
PEEL_TOLERANCE = .015
SHELL_MARGIN = .003


def observed_surface_keep(data, tolerance=PEEL_TOLERANCE):
    """Keep near the observed native low-Y shell in small XZ neighborhoods."""
    core = data[(data[:, 10] > 0) & (np.exp(data[:, 7:10].max(1)) < .012)]
    distance, neighbors = cKDTree(core[:, [0, 2]]).query(data[:, [0, 2]], k=32, workers=-1)
    depth = np.where(distance < .007, core[neighbors, 1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        observed = np.nanquantile(depth, .2, axis=1)
    return ~np.isfinite(observed) | (data[:, 1] <= observed + tolerance)


def inside_opposite_shell_keep(data, opposite, margin=SHELL_MARGIN, front=True):
    """Reject opposite-side leakage only where the other capture has coverage."""
    distance, neighbors = cKDTree(opposite[:, [0, 2]]).query(data[:, [0, 2]], k=24, workers=-1)
    depth = np.where(distance < .005, opposite[neighbors, 1], np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        boundary = np.nanquantile(depth, .8 if front else .2, axis=1)
    return ~np.isfinite(boundary) | ((data[:, 1] < boundary-margin) if front else
                                    (data[:, 1] > boundary+margin))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-render', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    MASKS.mkdir(parents=True, exist_ok=True)
    config = json.loads((BASE / 'fusion-config.json').read_text())
    data, labels, _, report = load_fused(BASE)
    ids = {p['id']: i for i, p in enumerate(report['parts'])}
    cervical = json.loads((ROOT / config['cervicalSegmentation']).read_text())
    part = BODY_PARTS[ids['left_hand']]
    lm = json.loads((ROOT / config['frontLandmarks']).read_text())['landmarks']
    expected_hashes = {m['file']: m['sha256'] for m in report['sources']}
    native, source_ids, metadata = {}, {}, {}
    for source in ['front', 'back']:
        raw, meta = read_scan(ROOT / config[source])
        if meta['sha256'] != expected_hashes[meta['file']]:
            raise ValueError('Hand baseline belongs to different source scans')
        landmarks = json.loads((ROOT / config[source + 'Landmarks']).read_text())['landmarks']
        raw, segmentation, _, indices = clean_scan(
            raw, landmarks, config['filter'], BODY_PARTS,
            config['exclusions'].get(source, []), config['segmentationOverrides'].get(source, []),
            config['filter'].get('partOverrides', {}).get(source, {}),
            cervical['cervicalSegmentation'][source])
        selected = segmentation == ids['left_hand']
        native[source], source_ids[source], metadata[source] = raw[selected], indices[selected], meta
    tr = report['parts'][ids['left_hand']]['sourceToFrontRaw']
    aligned = {'front': native['front'], 'back': transform_gaussians(
        native['back'], np.array(tr['rotation']), np.array(tr['translation']), tr['scale'])}
    keep = {source: observed_surface_keep(d) for source, d in native.items()}
    peeled = {source: aligned[source][selected] for source, selected in keep.items()}
    # Evaluate both shells against the same peeled inputs, avoiding sequential drift.
    for source, opposite in [('front', 'back'), ('back', 'front')]:
        keep[source][keep[source]] = inside_opposite_shell_keep(
            peeled[source], peeled[opposite], front=source == 'front')
        rejected = np.sort(source_ids[source][~keep[source]]).astype(np.uint32)
        path = MASKS / ('hand-' + source + '.npy')
        np.save(path, rejected)
        evidence = {
            'version': 1, 'purpose': 'remove opposite-side hand leakage while preserving the observed palm and back',
            'sourceCapture': source, 'sourceFile': metadata[source]['file'],
            'sourceSha256': metadata[source]['sha256'], 'sourceHashes': expected_hashes,
            'baseline': str(BASE.relative_to(ROOT)), 'maskFile': str(path.relative_to(ROOT)),
            'uniqueSourceIndices': len(rejected), 'maskMeaning': 'sorted original source PLY vertex rows',
            'applyStage': 'after coverage confidence weighting and opacity attenuation',
            'part': 'left_hand', 'fusionOverride': BLEND,
            'observedSurface': {'nativeAxis': 'low Y', 'projectedAxes': ['X', 'Z'],
                                'neighbors': 32, 'radius': .007, 'quantile': .2,
                                'coreMinOpacityLogit': 0, 'coreMaxScale': .012,
                                'depthTolerance': PEEL_TOLERANCE},
            'oppositeShell': {'frame': 'aligned front raw', 'neighbors': 24, 'radius': .005,
                              'frontQuantile': .8, 'backQuantile': .2, 'margin': SHELL_MARGIN,
                              'unsupportedColumns': 'preserve'},
            'rejectedTrials': 'Global .005/.008 scale caps opened holes; aggressive feather/bias exposed finger interiors.',
            'proof': str((OUT / 'accepted.png').relative_to(ROOT)),
            'reproduce': '.venv-fusion/bin/python -B tools/fusion/cleanup_hand.py',
        }
        path.with_suffix('.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print(source, 'selected original vertices', len(rejected), flush=True)
    settings = {**config['fusion'], 'partOverrides': {'left_hand': BLEND}}
    weights = coverage_weights(aligned['front'], aligned['back'], lm, part, settings)
    chosen = []
    for source, weight in zip(['front', 'back'], weights):
        contribution, retained = attenuate(aligned[source], weight, settings['minWeight'])
        chosen.append(contribution[keep[source][retained]])
    before = data[np.isin(labels, [ids['left_hand'], ids['left_forearm']])]
    after = np.concatenate([data[labels == ids['left_forearm']], *chosen])
    np.save(OUT / 'accepted.npy', after)
    if not args.skip_render:
        atlas({'Before': before, 'Cleaned': after}, OUT / 'accepted.png', [.76, .09, .64], .30, 640,
              title='Left palm and back / local source-shell cleanup / identical cameras')


if __name__ == '__main__':
    main()
