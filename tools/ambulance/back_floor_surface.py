#!/usr/bin/env python3
"""Add thin sampled floor support behind retained tread after haze deletion.

This reconstructs missing continuous material. It does not restore deleted
volumetric support or invent a tread pattern. Fine captured floor splats remain
in front; the backing uses robust local color from measured floor anchors.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W


def smooth(x):
    x = np.clip(x, 0, 1)
    return x*x*(3-2*x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--patch', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--spacing', type=float, default=.006)
    ap.add_argument('--color-gain', type=float, default=1.)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('Use a fresh output directory')
    parent = json.loads((args.patch / 'report.json').read_text())
    baseline = Path(parent['baseline'])
    phone, _, _ = read_ply(args.patch / 'iphone.ply')
    reference, _, _ = read_ply(baseline / 'reference-patches.ply')
    selection = np.load(args.patch / 'changes.npz')
    anchor_ids = selection['floor_anchor_reference_indices']
    anchors = reference[anchor_ids]
    p = np.einsum('ij,nj->ni', W, columns(anchors, ['x', 'y', 'z']).astype(float))
    rgb = .5 + .28209479177387814*columns(anchors, ['f_dc_0', 'f_dc_1', 'f_dc_2']).astype(float)
    valid = (rgb.min(1) > .12) & (rgb.max(1) < .65) & (np.ptp(rgb, axis=1) < .18)
    p, rgb, anchor_ids = p[valid], rgb[valid], anchor_ids[valid]
    assert len(p) > 100
    cf = np.asarray(parent['floor_fit']['plane_y_from_xz1'])
    lo, hi = np.array([[-1.50, -.045], [1.48, .60]])
    xx = np.arange(lo[0]+args.spacing/2, hi[0], args.spacing)
    zz = np.arange(lo[1]+args.spacing/2, hi[1], args.spacing)
    xz = np.stack(np.meshgrid(xx, zz), axis=-1).reshape(-1, 2)
    points = np.c_[xz[:, 0], np.einsum('ij,j->i', xz, cf[:2]) + cf[2], xz[:, 1]]
    # Actual bench-face boundary: a generic rectangular aisle mask includes
    # its vertical face and caused the rejected first trial's coverage leak.
    bench = np.array([-.08432874, .02431246, .38086325])
    edge = np.minimum(xz-lo, hi-xz).min(1)
    bench_clearance = bench[0]*points[:, 0]+bench[1]*points[:, 1]+bench[2]-points[:, 2]-.045
    edge = np.minimum(edge, bench_clearance)
    inside = edge > .001
    points, xz, edge = points[inside], xz[inside], edge[inside]
    distances, neighbors = cKDTree(p[:, [0, 2]]).query(xz, k=24)
    local = np.median(rgb[neighbors], axis=1)
    median = np.median(rgb, axis=0)
    colors = np.clip(local, median-.045, median+.045)*args.color_gain
    colors = np.clip(colors, .03, .8)
    normal = np.array([-cf[0], 1., -cf[1]])
    normal /= np.linalg.norm(normal)
    tangent = np.array([1., cf[0], 0.])
    tangent /= np.linalg.norm(tangent)
    axes = np.column_stack([normal, tangent, np.cross(normal, tangent)])
    quaternion = Rotation.from_matrix(W.T @ axes).as_quat()[[3, 0, 1, 2]]
    # Keep the entire thin skin below the fitted plane. Bound lateral support
    # at both bed and bench margins instead of allowing large tails there.
    depth = -.009
    points += depth*normal
    raw = np.einsum('ij,nj->ni', W.T, points)
    sigma = np.minimum(args.spacing*.85, edge/3.2)
    additions = np.zeros(len(points), dtype=phone.dtype)
    for i, field in enumerate(['x', 'y', 'z']):
        additions[field] = raw[:, i]
    for i, field in enumerate(['rot_0', 'rot_1', 'rot_2', 'rot_3']):
        additions[field] = quaternion[i]
    additions['scale_0'] = np.log(.00015)
    additions['scale_1'] = np.log(sigma)
    additions['scale_2'] = np.log(sigma)
    for i, field in enumerate(['f_dc_0', 'f_dc_1', 'f_dc_2']):
        additions[field] = (colors[:, i]-.5)/.28209479177387814
    opacity = np.clip(.96*smooth(edge/.025), 1e-8, 1-1e-8)
    additions['opacity'] = np.log(opacity/(1-opacity))
    args.out.mkdir(parents=True)
    write_ply(args.out / 'iphone.ply', np.concatenate([phone, additions]))
    shutil.copy2(args.patch / 'reference-patches.ply', args.out / 'reference-patches.ply')
    shutil.copy2(args.patch / 'changes.npz', args.out / 'changes.npz')
    write_ply(args.out / 'floor-additions.ply', additions)
    np.savez_compressed(args.out / 'floor-support-sampling.npz',
        anchor_baseline_reference_indices=anchor_ids, neighbor_anchor_indices=neighbors,
        nearest_anchor_distances=distances[:, 0], world_positions=points)
    shutil.copyfile(__file__, args.out / 'generator.py')
    report = dict(parent)
    report['status'] = 'Unreviewed thin floor material support after contributor removal'
    report['parent_patch'] = {'directory': str(args.patch.resolve()),
        'report_sha256': sha256_file(args.patch / 'report.json'),
        'iphone_sha256': sha256_file(args.patch / 'iphone.ply'),
        'reference_sha256': sha256_file(args.patch / 'reference-patches.ply')}
    report['floor_support'] = {'added_rows': len(additions), 'spacing': args.spacing,
        'normal_sigma': .00015, 'normal_offset': depth, 'top_three_sigma': depth+3*.00015,
        'color_gain': args.color_gain, 'observed_median_rgb': median.tolist(),
        'sample_count': len(anchor_ids), 'local_color_neighborhood': 24,
        'maximum_nearest_anchor_distance': float(distances[:, 0].max()),
        'bounds': [points.min(0).tolist(), points.max(0).tolist()],
        'appearance': 'Smooth observed floor base behind retained captured tread, not a synthesized tread pattern'}
    report['checks'] = {**parent['checks'], 'support_values_finite': all(bool(np.isfinite(additions[f]).all()) for f in additions.dtype.names),
        'support_below_fitted_plane': depth+3*.00015 < 0}
    if 'phone_byte_exact' in report['checks']:
        report['checks']['phone_prefix_byte_exact'] = report['checks'].pop('phone_byte_exact')
    report['counts'] = {**parent['counts'], 'phone_added_floor_support': len(additions)}
    if 'frozen_ply_hashes' in report:
        report['parent_frozen_ply_hashes'] = report.pop('frozen_ply_hashes')
    report['frozen_ply_hashes'] = {'iphone': sha256_file(args.out / 'iphone.ply'),
                                 'reference': sha256_file(args.out / 'reference-patches.ply')}
    report['generator_sha256'] = sha256_file(args.out / 'generator.py')
    report['changes_sha256'] = sha256_file(args.out / 'changes.npz')
    report['sampling_sha256'] = sha256_file(args.out / 'floor-support-sampling.npz')
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report['floor_support'], indent=2))


if __name__ == '__main__':
    main()
