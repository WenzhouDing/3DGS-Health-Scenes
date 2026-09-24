#!/usr/bin/env python3
"""Trial of registered same-strip floor support; no cloned/tiled material.

Selected captured Insta splats retain DC/SH and their tangential location.
Only normal position and covariance are confined to measured floor geometry.
Original iPhone floating film is suppressed for physical compaction. Fine
elevated structure, seat cluster and cabinet plinth remain outside the masks.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W

FLOOR = np.array([-.0071857297284330405, -.017685424335699158, -.7750895496740308])
CABINET = np.array([-.08237276, .02981509, -.92014449])
NORMAL = np.array([-FLOOR[0], 1., -FLOOR[1]])
NORMAL_LENGTH = np.linalg.norm(NORMAL)
NORMAL /= NORMAL_LENGTH


def point_geometry(v):
    p = np.einsum('ij,nj->ni', W, columns(v, ['x', 'y', 'z']).astype(float))
    d = (p[:, 1] - np.einsum('ij,j->i', p[:, [0, 2]], FLOOR[:2]) - FLOOR[2]) / NORMAL_LENGTH
    return p, d


def clearance(p):
    edge = np.minimum(p[:, [0, 2]] - [-1.25, -.84], [.50, -.56] - p[:, [0, 2]]).min(1)
    cabinet = p[:, 2] - np.einsum('ij,j->i', p[:, :2], CABINET[:2]) - CABINET[2] - .04
    return np.minimum(edge, cabinet / np.linalg.norm([-CABINET[0], -CABINET[1], 1.]))


def covariances(v):
    s = np.exp(columns(v, ['scale_0', 'scale_1', 'scale_2']).astype(float))
    q = np.einsum('ij,njk->nik', W, Rotation.from_quat(columns(v, ['rot_1', 'rot_2', 'rot_3', 'rot_0']).astype(float)).as_matrix())
    a = q * s[:, None, :]
    c = np.einsum('nik,njk->nij', a, a)
    sn = np.sqrt(np.einsum('i,nij,j->n', NORMAL, c, NORMAL))
    return c, sn, s


def alpha(v):
    return 1 / (1 + np.exp(-np.clip(v['opacity'].astype(float), -40, 40)))


def smooth(t):
    t = np.clip(t, 0, 1)
    return t*t*(3-2*t)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output must be new.')
    phone, _, _ = read_ply(args.baseline / 'iphone.ply')
    reference, _, _ = read_ply(args.baseline / 'reference-patches.ply')
    donor, _, _ = read_ply(args.reference)
    p, d = point_geometry(phone)
    region_ids = np.flatnonzero((clearance(p) > 0) & (d > -.04) & (d < .30))
    _, sn, scales = covariances(phone[region_ids])
    rgb = .5 + .28209479 * columns(phone[region_ids], ['f_dc_0', 'f_dc_1', 'f_dc_2'])
    # The expanded rule catches the traced thin but long elevated sheet at
    # original ID210423 and near-plane broad covariance at ID2203002.
    wrong = (((d[region_ids] > .025) & (sn > .002) & (scales.max(1) > .012))
             | ((np.abs(d[region_ids]) <= .025) & (sn > .012) & (scales.max(1) > .025)))
    material = (rgb.mean(1) > .23) & (rgb.mean(1) < .70) & (np.ptp(rgb, axis=1) < .15)
    phone_ids = region_ids[wrong & material & (alpha(phone[region_ids]) > 1.1e-8)]
    phone_out = phone.copy()
    phone_out['opacity'][phone_ids] = np.log(1e-8/(1-1e-8))

    rp, rd = point_geometry(donor)
    rough_ids = np.flatnonzero((clearance(rp) > .001) & (rd > -.10) & (rd < .18))
    rc, rsn, rs = covariances(donor[rough_ids])
    rgb = .5 + .28209479 * columns(donor[rough_ids], ['f_dc_0', 'f_dc_1', 'f_dc_2'])
    floor_support = ((np.abs(rd[rough_ids]) < .025) | (rsn > .0075))
    floor_material = (rgb.mean(1) > .07) & (rgb.mean(1) < .60) & (np.ptp(rgb, axis=1) < .18)
    local = np.flatnonzero(floor_support & floor_material & (alpha(donor[rough_ids]) > .01))
    ids = rough_ids[local]
    patch = donor[ids].copy()
    cov = rc[local]
    height = rd[ids]
    target = np.where(np.abs(height) > .012, -.008, np.clip(height, -.010, .002))
    points = rp[ids] + (target-height)[:, None] * NORMAL
    projector = np.eye(3) - np.outer(NORMAL, NORMAL)
    tangent = np.einsum('ij,njk,lk->nil', projector, cov, projector)
    values, vectors = np.linalg.eigh(tangent)
    tangent_sigma = np.sqrt(np.maximum(values[:, 1:], 1e-12))
    # Cap only gross stretched floor needles. No pattern is repeated or moved
    # laterally; this is a bounded reconstruction of captured covariance.
    needle = (tangent_sigma[:, 1] > .04) & (tangent_sigma[:, 1] > 6*tangent_sigma[:, 0])
    tangent_sigma[needle, 1] = .025
    edge = clearance(points)
    tangent_sigma = np.minimum(tangent_sigma, np.maximum(edge/3.2, 1e-6)[:, None])
    n_sigma = np.minimum(rsn[local], .0015)
    normalized = np.einsum('nik,nk,njk->nij', vectors[:, :, 1:], tangent_sigma**2, vectors[:, :, 1:])
    normalized += n_sigma[:, None, None]**2 * np.outer(NORMAL, NORMAL)
    ev, rot = np.linalg.eigh(normalized)
    rot[np.linalg.det(rot) < 0, :, 0] *= -1
    raw_rot = np.einsum('ij,njk->nik', W.T, rot)
    quat = Rotation.from_matrix(raw_rot).as_quat()
    raw_p = np.einsum('ij,nj->ni', W.T, points)
    for i, field in enumerate(['x', 'y', 'z']):
        patch[field] = raw_p[:, i]
    for i, field in enumerate(['scale_0', 'scale_1', 'scale_2']):
        patch[field] = .5*np.log(np.maximum(ev[:, i], 1e-12))
    for i, field in enumerate(['rot_1', 'rot_2', 'rot_3', 'rot_0']):
        patch[field] = quat[:, i]
    a = np.clip(alpha(patch)*smooth(edge/.020), 1e-8, 1-1e-8)
    patch['opacity'] = np.log(a/(1-a))
    reference_ids = np.load(args.baseline / 'selections.npz')['reference_indices']
    lookup = np.full(len(donor), -1, np.int64)
    lookup[reference_ids] = np.arange(len(reference_ids))
    existing = lookup[ids] >= 0
    reference_out = reference.copy()
    reference_out[lookup[ids[existing]]] = patch[existing]
    extra = patch[~existing]
    all_reference = np.concatenate([reference_out, extra])
    geometric_fields = {'x', 'y', 'z', 'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'opacity'}
    original_material_exact = all(np.array_equal(patch[f], donor[ids][f]) for f in donor.dtype.names if f not in geometric_fields)
    changed_reference = np.zeros(len(reference), bool)
    for field in reference.dtype.names:
        changed_reference |= reference_out[field] != reference[field]
    args.out.mkdir(parents=True)
    write_ply(args.out / 'iphone.ply', phone_out)
    write_ply(args.out / 'reference-patches.ply', all_reference)
    write_ply(args.out / 'selected-reference.ply', patch)
    np.savez_compressed(args.out / 'changes.npz', phone_changed_indices=phone_ids,
                        phone_delete_indices=phone_ids,
                        reference_changed_indices=np.flatnonzero(changed_reference),
                        reference_delete_indices=np.empty(0, np.int64),
                        selected_reference_original_ids=ids,
                        added_reference_original_ids=ids[~existing],
                        modified_reference_original_ids=ids[existing],
                        reference_normal_shift=target-height,
                        reference_tangent_needle_ids=ids[needle])
    shutil.copy2(__file__, args.out / 'generator.py')
    report = {'status': 'Unreviewed same-strip registered reference floor trial',
              'baseline': str(args.baseline.resolve()),
              'baseline_phone_sha256': sha256_file(args.baseline / 'iphone.ply'),
              'baseline_reference_sha256': sha256_file(args.baseline / 'reference-patches.ply'),
              'reference_capture': str(args.reference.resolve()), 'reference_capture_sha256': sha256_file(args.reference),
              'region': {'x': [-1.25, .50], 'z': [-.84, -.56], 'cabinet_clearance': .04, 'cabinet_z_from_xy1': CABINET.tolist()},
              'floor_y_from_xz1': FLOOR.tolist(),
              'counts': {'source_phone_deletions': len(phone_ids), 'same_strip_reference_rows': len(ids),
                         'existing_reference_rows_changed': int(changed_reference.sum()), 'new_reference_rows': len(extra),
                         'coarse_tangent_needles_shortened': int(needle.sum())},
              'geometry': {'maximum_normal_sigma': float(n_sigma.max()),
                           'maximum_target_upper_three_sigma': float((target+3*n_sigma).max()),
                           'normal_shift_range': [float((target-height).min()), float((target-height).max())]},
              'checks': {'source_phone_nonopacity_exact': all(np.array_equal(phone_out[f], phone[f]) for f in phone.dtype.names if f != 'opacity'),
                         'all_captured_DC_SH_exact': original_material_exact,
                         'no_tangential_translation': bool(np.allclose(np.einsum('ij,nj->ni', projector, points-rp[ids]), 0, atol=1e-12)),
                         'no_capture_record_duplicated': len(np.unique(ids)) == len(ids),
                         'all_reference_additions_new_to_baseline': bool(np.all(lookup[ids[~existing]] == -1)),
                         'no_tiled_or_procedural_material': True},
              'limitations': ['Registered donor covariance is manually confined to the measured floor; this is not recovered ground truth.',
                              'Fine elevated structures and front seat are guarded; visual review is required for support boundaries.',
                              'Opacity deletion markers require physical compaction; no collision certification is implied.'],
              'generator_sha256': sha256_file(args.out / 'generator.py'), 'changes_sha256': sha256_file(args.out / 'changes.npz'),
              'remote_publish': False}
    if not all(report['checks'].values()):
        raise RuntimeError(report['checks'])
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
