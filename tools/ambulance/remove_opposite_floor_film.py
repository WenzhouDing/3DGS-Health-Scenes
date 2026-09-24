#!/usr/bin/env python3
"""Remove an attributed, guarded source film beside the cabinet-side floor.

This source-only trial adds no points/material and changes no covariance or
position. Exact original IDs receive a deletion sentinel; the surface composer
must physically compact them. All other records remain byte-exact.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a new output directory; frozen trials cannot be overwritten.')
    floor = np.array([-.0071857297284330405, -.017685424335699158, -.7750895496740308])
    cabinet = np.array([-.08237276, .02981509, -.92014449])
    phone, _, _ = read_ply(args.baseline / 'iphone.ply')
    reference, _, _ = read_ply(args.baseline / 'reference-patches.ply')
    world = np.einsum('ij,nj->ni', W, columns(phone, ['x', 'y', 'z']).astype(float))
    height = world[:, 1] - np.einsum('ij,j->i', world[:, [0, 2]], floor[:2]) - floor[2]
    cabinet_z = np.einsum('ij,j->i', world[:, :2], cabinet[:2]) + cabinet[2]
    # The front seat/loose-item cluster is around x=.77, outside this strip.
    # The cabinet toe face has measured slope; a constant z box alone clips it.
    region = ((world[:, 0] > -1.25) & (world[:, 0] < .50)
              & (world[:, 2] > -.84) & (world[:, 2] < -.56)
              & (world[:, 2] > cabinet_z + .04)
              & (height > .02) & (height < .30))
    candidates = np.flatnonzero(region)
    block = phone[candidates]
    scales = np.exp(columns(block, ['scale_0', 'scale_1', 'scale_2']).astype(float))
    rotations = Rotation.from_quat(columns(block, ['rot_1', 'rot_2', 'rot_3', 'rot_0']).astype(float)).as_matrix()
    rotations = np.einsum('ij,njk->nik', W, rotations)
    normal = np.array([-floor[0], 1., -floor[1]])
    normal /= np.linalg.norm(normal)
    sigma_n = np.sqrt(np.sum((np.einsum('i,nij->nj', normal, rotations) * scales) ** 2, axis=1))
    rgb = .5 + .28209479 * columns(block, ['f_dc_0', 'f_dc_1', 'f_dc_2'])
    opacity = 1 / (1 + np.exp(-np.clip(block['opacity'].astype(float), -40, 40)))
    # Geometry/ray attribution defines the candidate region. This radiance guard
    # additionally excludes dark, colored and thin equipment/support records.
    diffuse = ((sigma_n > .0075) & (scales.max(1) > .012)
               & (rgb.mean(1) > .23) & (rgb.mean(1) < .70)
               & (np.ptp(rgb, axis=1) < .15) & (opacity > 1.1e-8))
    selected = candidates[diffuse]
    out = phone.copy()
    out['opacity'][selected] = np.log(1e-8 / (1-1e-8))
    unchanged = np.ones(len(phone), bool)
    unchanged[selected] = False
    assert out[unchanged].tobytes() == phone[unchanged].tobytes()
    assert all(np.array_equal(out[field], phone[field]) for field in phone.dtype.names if field != 'opacity')
    args.out.mkdir(parents=True)
    write_ply(args.out / 'iphone.ply', out)
    shutil.copy2(args.baseline / 'reference-patches.ply', args.out / 'reference-patches.ply')
    np.savez_compressed(args.out / 'changes.npz',
                        phone_changed_indices=selected, phone_delete_indices=selected,
                        reference_changed_indices=np.empty(0, np.int64),
                        reference_delete_indices=np.empty(0, np.int64),
                        original_phone_indices=selected,
                        removed_world_centers=world[selected],
                        removed_normal_sigma=sigma_n[diffuse],
                        removed_floor_height=height[selected])
    shutil.copy2(__file__, args.out / 'generator.py')
    report = {
        'status': 'Unreviewed bounded source-only cabinet-side floor film removal',
        'baseline': str(args.baseline.resolve()),
        'baseline_phone_sha256': sha256_file(args.baseline / 'iphone.ply'),
        'baseline_reference_sha256': sha256_file(args.baseline / 'reference-patches.ply'),
        'scope': 'Original iPhone coarse off-floor radiance only; no new points, material, texture, position or covariance edits.',
        'region': {'x': [-1.25, .50], 'z': [-.84, -.56], 'height_above_fitted_floor': [.02, .30],
                   'cabinet_z_from_xy1': cabinet.tolist(), 'cabinet_front_clearance': .04,
                   'floor_y_from_xz1': floor.tolist()},
        'selection': {'normal_sigma_min': .0075, 'maximum_axis_sigma_min': .012,
                      'mean_rgb_bounds': [.23, .70], 'rgb_chroma_max': .15,
                      'input_alpha_min': 1.1e-8,
                      'rationale': 'Confirmed coarse contributors dominate three cabinet-side strip rays; thin elevated seat/bed supports and cabinet toe face are guarded.'},
        'counts': {'phone_deletion_ids': len(selected), 'reference_changed': 0, 'added': 0,
                   'unchanged_phone_records': int(unchanged.sum())},
        'removed_geometry': {
            'center_bounds': [world[selected].min(0).tolist(), world[selected].max(0).tolist()],
            'height_quantiles': np.quantile(height[selected], [0, .5, .95, 1]).tolist(),
            'normal_sigma_quantiles': np.quantile(sigma_n[diffuse], [0, .5, .95, 1]).tolist()},
        'checks': {'all_unselected_phone_records_byte_exact': True, 'phone_nonopacity_fields_exact': True,
                   'reference_byte_exact': sha256_file(args.out / 'reference-patches.ply') == sha256_file(args.baseline / 'reference-patches.ply'),
                   'no_additions': len(out) == len(phone),
                   'all_selected_centers_outside_main_aisle_repair': bool(np.all(world[selected, 2] < -.56)),
                   'all_selected_centers_behind_seat_tangle_x': bool(np.all(world[selected, 0] < .50))},
        'deletion': {'sentinel_alpha': 1e-8, 'physical_removal_in_trial': False,
                     'final_compositor_must_compact_selected_rows': True},
        'evidence': 'raw/ambulance-cleanup/pass6/floor-audit/opposite-bed-strip-attribution.json',
        'limitations': ['Only traced coarse material is removed; fine captured texture and possible sparse gaps are retained.',
                       'Seat/loose-item reconstruction around x=.77 is intentionally outside this edit.',
                       'This does not certify the whole scene or the separate legacy collider.'],
        'remote_publish': False,
        'generator_sha256': sha256_file(args.out / 'generator.py'),
        'changes_sha256': sha256_file(args.out / 'changes.npz'),
        'frozen_ply_hashes': {name: sha256_file(args.out / name) for name in ['iphone.ply', 'reference-patches.ply']},
    }
    if not all(report['checks'].values()):
        raise RuntimeError(report['checks'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
