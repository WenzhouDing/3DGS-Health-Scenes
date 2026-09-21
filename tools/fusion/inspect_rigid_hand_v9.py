#!/usr/bin/env python3
"""Diagnostic-only source authenticity gate for one rigid right forearm/hand.

The default input is the V8 whole-rigid-hand trial, never the deformed export.
No source coordinates or active configuration/output are changed. Native
observed-alpha confidence is inherited from the full-source V6 calculation and
matched by exact original vertex ID. Reproduce with this script's --render flag.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, sha256_file, transform_gaussians
from render_gaussians import atlas, load_fused

OUT = ROOT / 'raw/fusion-work/refinement-v9/visual-audit'
DEFAULT = ROOT / 'raw/fusion-work/refinement-v8/rigid-hand-trial'
NATIVE = ROOT / 'raw/fusion-work/refinement-v8/pad-inspection'
VISIBILITY = ROOT / 'raw/fusion-work/refinement-v6/right-hand'
ORIGIN = np.array([-.697, .03, .597])
AXIS = np.array([-.626, 0., .7798]); AXIS /= np.linalg.norm(AXIS)
VIEWS = [('Below', [0, 0, 1]), ('Outer side', [-1, 0, 0]),
         ('Distal', [-.7, 0, .7]), ('Palm', [0, -1, 0]),
         ('Back oblique', [-.5, 1, .2])]


def mapped(data, transform):
    return transform_gaussians(data, np.array(transform['rotation']),
                               np.array(transform['translation']), transform['scale'])


def run(directory, render_images, size):
    OUT.mkdir(parents=True, exist_ok=True)
    data, labels, sources, report = load_fused(directory)
    ids = np.load(directory / 'source-vertex-indices.npy')
    part_ids = {p['id']: i for i, p in enumerate(report['parts'])}
    hand, forearm = part_ids['right_hand'], part_ids['right_forearm']
    transforms = [report['parts'][i]['sourceToFrontRaw'] for i in [hand, forearm]]
    if not all(np.allclose(transforms[0][key], transforms[1][key], atol=1e-12, rtol=0)
               for key in ['scale', 'rotation', 'translation']):
        raise ValueError('Audit input must use exactly one rigid hand/forearm transform')
    if report['parameters'].get('rightWristRegistration'):
        raise ValueError('Audit excludes any wrist/knuckle deformation')
    selected = np.isin(labels, [hand, forearm]) & (data[:, 0] < -.59)
    data, labels, sources, ids = [a[selected] for a in [data, labels, sources, ids]]
    flux = np.full(len(data), np.inf); ratio = np.ones(len(data))
    other = np.zeros(len(data)); native_rows = {}; native_strong = {}
    metadata = {'sourceHashes': {s['file']: s['sha256'] for s in report['sources']},
                'input': str(directory.relative_to(ROOT)),
                'method': 'no deformation; original-source observed-alpha confidence',
                'nativeConfidence': 'full quality-clean native hand, observed camera -Y, opposite camera +Y',
                'scope': 'only hand rows beyond reference longitudinal station .065; complete pad protected',
                'thresholds': {}, 'referenceTransform': transforms[1]}
    for si, source in enumerate(['front', 'back']):
        meta = report['sources'][si]
        if sha256_file(ROOT / meta['file']) != meta['sha256']:
            raise ValueError('Original source hash mismatch')
        native = np.load(NATIVE / f'{source}-full-native.npz')
        visibility = np.load(VISIBILITY / f'{source}-full-visibility.npz')
        hand_mask = native['labels'] == hand
        if not np.array_equal(native['indices'][hand_mask], visibility['indices']):
            raise ValueError('Native confidence cache has different source-index order')
        order = np.argsort(visibility['indices'])
        rows = (sources == si) & (labels == hand)
        lookup = np.searchsorted(visibility['indices'][order], ids[rows])
        if not np.array_equal(visibility['indices'][order][lookup], ids[rows]):
            raise ValueError('Candidate rows absent from native confidence cache')
        for values, key in [(flux, 'seen'), (other, 'opposite_seen'), (ratio, 'observed_ratio')]:
            values[rows] = visibility[key][order][lookup]
        original = native['data'][hand_mask]
        supported = ((visibility['seen'] > visibility['opposite_seen']) &
                     (visibility['seen'] > .05) & (visibility['observed_ratio'] > .1))
        raw_supported = original[supported]
        native_strong[source] = raw_supported if si == 0 else mapped(raw_supported, transforms[1])
        original = native['data']
        native_rows[source] = original if si == 0 else mapped(original, transforms[1])

    longitudinal = np.einsum('ni,i->n', data[:, :3] - ORIGIN, AXIS)
    scope = (labels == hand) & (longitudinal > .065)
    pad_metadata = json.loads((ROOT / 'tools/fusion/part-masks/back-right-wrist-pad.json').read_text())
    pad_ids = np.load(ROOT / pad_metadata['maskFile'])
    scope &= ~((sources == 1) & np.isin(ids, pad_ids))
    rows = {'One rigid pose / retained': data}
    for threshold in [.03, .10, .25]:
        keep = ~scope | ((ratio > threshold) & (flux > .05) & (flux >= other))
        rows[f'Native confidence > {threshold:g}'] = data[keep]
        metadata['thresholds'][str(threshold)] = {
            source: {'removed': int(np.sum(~keep & (sources == si))),
                     'retainedHand': int(np.sum(keep & (sources == si) & (labels == hand)))}
            for si, source in enumerate(['front', 'back'])}
        np.savez_compressed(OUT / f'confidence-{threshold:g}.npz', data=data[keep],
                            labels=labels[keep], sources=sources[keep], indices=ids[keep])
    (OUT / 'confidence-audit.json').write_text(json.dumps(metadata, indent=2) + '\n')
    np.savez_compressed(OUT / 'rigid-input.npz', data=data, labels=labels, sources=sources, indices=ids)
    if render_images:
        atlas(rows, OUT / 'rigid-confidence-trials.png', [-.764, .05, .665], .31,
              size=size, views=VIEWS, title='One rigid right wrist / native-support pruning diagnostics only')
        atlas({'Front native observed / strong': native_strong['front'],
               'Back native observed / strong': native_strong['back'],
               'Strong observed both / original opacity': np.concatenate(list(native_strong.values()))},
              OUT / 'observed-source-authenticity.png', [-.764, .05, .665], .31,
              size=size, views=VIEWS, title='Original supported hand surfaces in one forearm frame / no bending')
    print(json.dumps(metadata, indent=2))


def compare_axial():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = {}
    for degrees in [0, 10, 15, 20]:
        trial = np.load(ROOT / f'raw/fusion-work/refinement-v9/registration/axial-extra-{degrees}.npz')
        rows[f'Rigid shared pose / extra {degrees} degrees'] = trial['data']
    atlas(rows, OUT / 'axial-close-review.png', [-.764, .05, .665], .31,
          size=900, views=VIEWS, title='Independent rigid assembly gate / observed surfaces retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--size', type=int, default=700)
    parser.add_argument('--compare-axial', action='store_true')
    args = parser.parse_args()
    if args.compare_axial:
        compare_axial()
    else:
        run(args.input.resolve(), args.render, args.size)
