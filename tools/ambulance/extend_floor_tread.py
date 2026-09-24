#!/usr/bin/env python3
"""Extend a reviewed dense captured tread patch over the repaired floor.

The selected physical pattern is translated along the fitted floor plane.
This is explicit material reconstruction, not a recovery of unobserved detail.
The original scene and candidate remain immutable; additions retain source IDs.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def alpha(v):
    return 1/(1+np.exp(-np.clip(v['opacity'].astype(float), -40, 40)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--source-rectangle', type=float, nargs=4, required=True, metavar=('X0', 'Z0', 'X1', 'Z1'))
    args = ap.parse_args()
    if args.out.exists():
        ap.error('Use a fresh output directory')
    report = json.loads((args.parent / 'report.json').read_text())
    phone, _, _ = read_ply(args.parent / 'iphone.ply')
    baseline, _, _ = read_ply(Path(report['baseline']) / 'iphone.ply')
    changes = dict(np.load(args.parent / 'changes.npz'))
    recovered = changes['recovered_original_phone_indices']
    world = np.einsum('ij,nj->ni', W, columns(phone, ['x', 'y', 'z']).astype(float))
    uv = world[:, [0, 2]]
    rectangle = np.asarray(args.source_rectangle).reshape(2, 2)
    size = rectangle[1]-rectangle[0]
    assert np.all(size > .01)
    selected = ((uv[recovered] >= rectangle[0]) & (uv[recovered] < rectangle[1])).all(1)
    template_ids = recovered[selected]
    assert len(template_ids) >= 100
    template = phone[template_ids].copy()
    source_positions = world[template_ids]
    cf = np.asarray(report['floor_fit']['plane_y_from_xz1'])
    region = report['region']
    lo, hi = np.asarray(region['xz_bounds'])
    bench = np.asarray(region['bench_face_z_from_xy1'])

    def distance(p):
        edge = np.minimum(p[:, [0, 2]]-lo, hi-p[:, [0, 2]]).min(1)
        front = np.einsum('ij,j->i', p[:, :2], bench[:2])+bench[2]-p[:, 2]-region['bench_face_minimum_front_clearance']
        return np.minimum(edge, front)

    additions, provenance, shifts = [], [], []
    start = np.floor((lo-rectangle[1])/size).astype(int)
    stop = np.ceil((hi-rectangle[0])/size).astype(int)
    for ix in range(start[0], stop[0]+1):
        for iz in range(start[1], stop[1]+1):
            du = size*np.array([ix, iz])
            translation = np.array([du[0], cf[0]*du[0]+cf[1]*du[1], du[1]])
            positions = source_positions+translation
            edge = distance(positions)
            keep = edge > .0005
            if not keep.any():
                continue
            v = template[keep].copy()
            raw = np.einsum('ij,nj->ni', W.T, positions[keep])
            for i, field in enumerate(['x', 'y', 'z']):
                v[field] = raw[:, i]
            for field in ['scale_1', 'scale_2']:
                v[field] = np.log(np.minimum(np.exp(v[field].astype(float)), edge[keep]/3.2))
            a = np.clip(alpha(v)*smooth(edge[keep]/.025), 1e-8, 1-1e-8)
            v['opacity'] = np.log(a/(1-a))
            additions.append(v)
            provenance.append(template_ids[keep])
            shifts.append(np.tile(translation, (len(v), 1)))
    extra = np.concatenate(additions)
    out = phone.copy()
    # Replace uneven recovered density with the sampled continuous pattern;
    # preserve a feathered transition to the captured material at boundaries.
    weight = smooth(distance(world[recovered])/.025)
    a = np.clip(alpha(out[recovered])*(1-weight), 1e-8, 1-1e-8)
    out['opacity'][recovered] = np.log(a/(1-a))
    changed = np.zeros(len(baseline), bool)
    for field in baseline.dtype.names:
        changed |= out[field][:len(baseline)] != baseline[field]
    changes['phone_changed_indices'] = np.flatnonzero(changed)
    changes['phone_delete_indices'] = np.flatnonzero(alpha(out[:len(baseline)]) <= 1.01e-8)
    args.out.mkdir(parents=True)
    write_ply(args.out / 'iphone.ply', np.concatenate([out, extra]))
    shutil.copy2(args.parent / 'reference-patches.ply', args.out / 'reference-patches.ply')
    write_ply(args.out / 'tread-additions.ply', extra)
    np.savez_compressed(args.out / 'changes.npz', **changes)
    np.savez_compressed(args.out / 'tread-sampling.npz', template_original_phone_ids=template_ids,
        added_original_phone_ids=np.concatenate(provenance), added_world_translations=np.concatenate(shifts))
    shutil.copyfile(__file__, args.out / 'generator.py')
    report['status'] = 'Unreviewed floor tread continuity reconstruction'
    report['parent_patch'] = {'directory': str(args.parent.resolve()), 'report_sha256': sha256_file(args.parent / 'report.json')}
    report['tread_reconstruction'] = {'template_rectangle_xz': rectangle.tolist(), 'template_rows': len(template_ids),
        'added_rows': len(extra), 'method': 'Translations of captured compact floor grains along the measured floor; exact sampled source colors retained',
        'sampling_sha256': sha256_file(args.out / 'tread-sampling.npz')}
    report['parent_recovery'] = report.pop('recovery')
    report['parent_stage_counts'] = report.pop('counts')
    report['counts'] = {'template_rows': len(template_ids), 'new_tread_additions': len(extra),
        'parent_floor_base_additions': len(phone)-len(baseline),
        'baseline_phone_changed': int(changed.sum()),
        'baseline_phone_suppressed_for_compaction': len(changes['phone_delete_indices']),
        'reference_deleted_on_compaction': len(changes['reference_delete_indices'])}
    report['checks'] = {'unchanged_phone_prefix_exact': out[:len(baseline)][~changed].tobytes() == baseline[~changed].tobytes(),
        'all_values_finite': all(bool(np.isfinite(v[f]).all()) for v in [out, extra] for f in v.dtype.names),
        'reference_parent_exact': sha256_file(args.out / 'reference-patches.ply') == sha256_file(args.parent / 'reference-patches.ply')}
    report['generator_sha256'] = sha256_file(args.out / 'generator.py')
    report['changes_sha256'] = sha256_file(args.out / 'changes.npz')
    report['frozen_ply_hashes'] = {f: sha256_file(args.out / f) for f in ['iphone.ply', 'reference-patches.ply']}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['tread_reconstruction'], indent=2))


if __name__ == '__main__':
    main()
