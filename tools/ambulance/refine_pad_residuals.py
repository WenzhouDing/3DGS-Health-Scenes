#!/usr/bin/env python3
"""Remove attributed original gray/white overlays from already replaced pads.

The source geometry/color and accepted reference splats remain exact. Dense
interior and top-edge rays locate surviving iPhone fragments in the mixed scene;
only neutral, behind-pad contributors are eligible. This is a trial requiring
frontal, grazing and neighbouring-wall review, not a global dehazing filter.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cleanup import ROOT, read_ply, sha256_file, write_ply
from transfer_reference_wall_details import attenuate, positions, trace


GROUP = {
    'cameras': ['pads-close', 'right-wall', 'right-grazing', 'bench-grazing'],
    'candidate_rgb_min': .43,
    'candidate_chroma_max': .25,
    'surfaces': [
        {'id': 'upper-pad-top-fringe', 'axis': 2, 'uv_axes': [0, 1],
         'uv_bounds': [[-.77, .657], [.48, .699]],
         'plane': [-.08291, .06183, .78042], 'behind_sign': 1,
         'deep_bounds': [.78, 5.5], 'grid': [81, 9]},
        {'id': 'upper-pad-interior', 'axis': 2, 'uv_axes': [0, 1],
         'uv_bounds': [[-.77, .462], [.48, .656]],
         'plane': [-.08291, .06183, .78042], 'behind_sign': 1,
         'deep_bounds': [.78, 5.5], 'grid': [39, 11]},
        {'id': 'lower-pad-interior', 'axis': 2, 'uv_axes': [0, 1],
         'uv_bounds': [[-.77, -.015], [.48, .245]],
         'plane': [-.08171, -.06387, .79002], 'behind_sign': 1,
         'deep_bounds': [.79, 5.5], 'grid': [45, 15]},
    ],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, default=ROOT/'raw/ambulance-cleanup/pass4/combined-v1')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--rounds', type=int, default=3)
    ap.add_argument('--seed-selection', type=Path)
    ap.add_argument('--manual-indices', default='')
    args = ap.parse_args()
    if args.baseline.resolve() == args.out.resolve():
        raise ValueError('Keep accepted baseline immutable')
    prior = json.loads((args.baseline/'report.json').read_text())
    source = Path(prior['iphone']); reference = Path(prior['reference'])
    phone, _, _ = read_ply(args.baseline/'iphone.ply')
    original, _, _ = read_ply(source)
    ref, _, _ = read_ply(args.baseline/'reference-patches.ply')
    extra = np.empty(len(ref), dtype=phone.dtype)
    for name in phone.dtype.names:
        extra[name] = ref[name]
    pp, rp = positions(phone), positions(ref)
    current = phone.copy()
    picked_all = np.zeros(len(phone), bool)
    if args.seed_selection:
        seed = np.load(args.seed_selection)
        if np.any(seed['multipliers'] != 0):
            raise ValueError('Only full-suppression seeds are supported')
        picked_all[seed['indices']] = True
    manual = [int(i) for i in args.manual_indices.split(',') if i]
    if any(i < 0 or i >= len(phone) for i in manual):
        raise ValueError('Manual source ID out of range')
    picked_all[manual] = True
    selected = np.flatnonzero(picked_all)
    current[selected] = attenuate(original[selected], np.zeros(len(selected)))
    rounds = []
    args.out.mkdir(parents=True, exist_ok=True)
    for iteration in range(args.rounds):
        scene = np.concatenate([current, extra])
        selected, observations = trace(scene, np.concatenate([pp, rp]), GROUP, len(phone))
        added = selected[~picked_all[selected]]
        picked_all[selected] = True
        current[added] = attenuate(original[added], np.zeros(len(added)))
        rounds.append({'iteration': iteration, 'additional_original_rows': len(added),
                       'trace': observations})
        print('Additional original overlay rows:', len(added), flush=True)
        if not len(added):
            break
    ids = np.flatnonzero(picked_all)
    np.savez_compressed(args.out/'iphone-opacity-selection.npz', indices=ids, multipliers=np.zeros(len(ids)))
    np.savez_compressed(args.out/'reference-selection.npz', indices=np.empty(0, np.int64), multipliers=np.empty(0))
    assembled = args.out/'assembled'; assembled.mkdir(exist_ok=True)
    write_ply(assembled/'iphone.ply', current)
    shutil.copy2(args.baseline/'reference-patches.ply', assembled/'reference-patches.ply')
    checks = {
        'all_original_fields_except_opacity_exact': all(np.array_equal(current[n], original[n]) for n in original.dtype.names if n != 'opacity'),
        'unselected_accepted_rows_exact': np.array_equal(current[~picked_all], phone[~picked_all]),
        'accepted_reference_file_exact': sha256_file(assembled/'reference-patches.ply') == sha256_file(args.baseline/'reference-patches.ply'),
        'all_rows_finite': all(bool(np.isfinite(current[n]).all()) for n in current.dtype.names),
    }
    if not all(checks.values()):
        raise RuntimeError(checks)
    shutil.copy2(__file__, args.out/'generator.py')
    report = {'status': 'Unreviewed mixed-scene pad overlay candidate',
              'iphone': str(source), 'iphone_sha256': sha256_file(source),
              'reference': str(reference), 'reference_sha256': sha256_file(reference),
              'baseline': str(args.baseline), 'baseline_selections_sha256': sha256_file(args.baseline/'selections.npz'),
              'iphone_changed_rows': len(ids), 'reference_added_rows': 0,
              'seed_selection': str(args.seed_selection) if args.seed_selection else None,
              'seed_selection_sha256': sha256_file(args.seed_selection) if args.seed_selection else None,
              'manual_original_indices': manual,
              'recipe': GROUP, 'rounds': rounds, 'checks': checks,
              'generator_sha256': sha256_file(args.out/'generator.py'),
              'remote_publish': False}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'rows': len(ids), 'checks': checks}), flush=True)


if __name__ == '__main__':
    main()
