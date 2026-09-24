#!/usr/bin/env python3
"""Prune explicitly attributed, exterior ceiling artifacts; preserve all other rows."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--selection', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--compact', action='store_true')
    args = ap.parse_args()
    if args.out.exists():
        ap.error('Use a new output directory')
    choice = json.loads(args.selection.read_text())
    ids = np.unique(np.asarray(choice['indices'], dtype=np.int64))
    phone, _, _ = read_ply(args.baseline / 'iphone.ply')
    assert len(ids) and ids.min() >= 0 and ids.max() < len(phone)
    p = columns(phone[ids], ['x', 'y', 'z']).astype(float) @ W.T
    assert np.all(p[:, 0] > 2.2) and np.all(p[:, 1] > 1.4)
    keep = np.ones(len(phone), bool)
    keep[ids] = False
    output = phone[keep].copy() if args.compact else phone.copy()
    if not args.compact:
        output['opacity'][ids] = np.log(1e-8 / (1 - 1e-8))
    unchanged = output if args.compact else output[keep]
    assert unchanged.tobytes() == phone[keep].tobytes()
    args.out.mkdir(parents=True)
    write_ply(args.out / 'iphone.ply', output)
    shutil.copy2(args.baseline / 'reference-patches.ply', args.out / 'reference-patches.ply')
    reread, _, _ = read_ply(args.out / 'iphone.ply')
    assert reread.dtype == output.dtype and reread.tobytes() == output.tobytes()
    np.savez_compressed(args.out / 'changes.npz', iphone_removed_indices=ids,
        iphone_retained_baseline_indices=np.flatnonzero(keep) if args.compact else np.arange(len(phone)))
    shutil.copy2(__file__, args.out / 'generator.py')
    shutil.copy2(args.selection, args.out / 'selection.json')
    hashes = {stream: sha256_file(args.baseline / f) for stream, f in
              [('iphone', 'iphone.ply'), ('reference', 'reference-patches.ply')]}
    reference, _, _ = read_ply(args.out / 'reference-patches.ply')
    report = {'status': 'Unreviewed attributed exterior-haze pruning',
        'baseline': str(args.baseline.resolve()), 'baseline_hashes': hashes,
        'baseline_report_sha256': sha256_file(args.baseline / 'report.json'),
        'method': 'Only exact ray-attributed iPhone artifacts beyond the measured cabin/roof; no new or retextured surfaces',
        'selection': choice, 'selected_world_positions': p.tolist(), 'selected_count': len(ids),
        'physically_removed': args.compact, 'combined_count': len(output) + len(reference),
        'checks': {'all_remaining_iphone_records_byte_exact': True,
            'entire_reference_file_byte_exact': sha256_file(args.out / 'reference-patches.ply') == hashes['reference'],
            'all_selected_far_outside_cabin': True, 'serialized_records_exact': True,
            'no_added_records': True, 'no_geometry_color_or_material_edits': True},
        'output_hashes': {stream: sha256_file(args.out / f) for stream, f in
              [('iphone', 'iphone.ply'), ('reference', 'reference-patches.ply')]},
        'generator_sha256': sha256_file(args.out / 'generator.py'),
        'selection_sha256': sha256_file(args.out / 'selection.json'),
        'changes_sha256': sha256_file(args.out / 'changes.npz'),
        'limitations': ['Some exterior reconstruction support remains because deleting it exposes gaps.',
            'This bounded cleanup does not rebuild the separate collider or certify scene collision accuracy.'],
        'remote_publish': False}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'selected': len(ids), 'physically_removed': args.compact, 'checks': report['checks']}))


if __name__ == '__main__':
    main()
