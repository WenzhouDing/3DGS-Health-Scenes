#!/usr/bin/env python3
"""Extend the accepted floor reconstruction into the guarded opposite strip.

Explicit reconstructed material: copies the same captured grain template used
by accepted floor V7, with captured shadow shading for this darker strip. No
wall material is changed. Bed, cabinet toe and front-seat regions are guarded.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W
from repair_opposite_floor_reference import FLOOR, NORMAL, CABINET, point_geometry, clearance, covariances, alpha, smooth


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--source-trial', type=Path, required=True)
    parser.add_argument('--accepted-floor', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Output must be a fresh directory.')
    phone, _, _ = read_ply(args.baseline / 'iphone.ply')
    # Reuse only the reviewed geometric selection, never the rejected V2 donors.
    removed = np.load(args.source_trial / 'changes.npz')['phone_delete_indices']
    prefix = phone.copy()
    prefix['opacity'][removed] = np.log(1e-8/(1-1e-8))
    floor_report = json.loads((args.accepted_floor / 'report.json').read_text())
    source_dir = Path(floor_report['parent_patch']['directory'])
    source, _, _ = read_ply(source_dir / 'iphone.ply')
    source_ids = np.load(args.accepted_floor / 'tread-sampling.npz')['template_original_phone_ids']
    template = source[source_ids].copy()
    source_points, _ = point_geometry(template)
    rectangle = np.asarray(floor_report['tread_reconstruction']['template_rectangle_xz'])
    period = rectangle[1] - rectangle[0]
    reference, _, _ = read_ply(args.reference)
    rp, rd = point_geometry(reference)
    broad = np.flatnonzero((clearance(rp) > 0) & (rd > -.10) & (rd < .18))
    _, sn, ss = covariances(reference[broad])
    rgb = .5 + .28209479 * columns(reference[broad], ['f_dc_0', 'f_dc_1', 'f_dc_2'])
    material = ((ss.max(1) > .035) & (sn > .015) & (alpha(reference[broad]) > .1)
                & (rgb.mean(1) > .08) & (rgb.mean(1) < .4) & (np.ptp(rgb, axis=1) < .15))
    palette_ids = broad[material]
    assert len(palette_ids) >= 50
    strip_color = np.median(rgb[material], axis=0)
    main_color = np.asarray(floor_report['floor_support']['observed_median_rgb'])
    shading_gain = strip_color/main_color
    assert np.all((shading_gain > .5) & (shading_gain < .9))

    lo = np.array([-1.25, -.84])
    hi = np.array([.50, -.56])
    spacing = .006
    xz = np.stack(np.meshgrid(np.arange(lo[0]+spacing/2, hi[0], spacing),
                              np.arange(lo[1]+spacing/2, hi[1], spacing)), axis=-1).reshape(-1, 2)
    points = np.c_[xz[:, 0], np.einsum('ij,j->i', xz, FLOOR[:2])+FLOOR[2], xz[:, 1]]
    points -= .009*NORMAL
    edge = clearance(points)
    points, edge = points[edge > .001], edge[edge > .001]
    tangent = np.array([1., FLOOR[0], 0.]); tangent /= np.linalg.norm(tangent)
    axes = np.column_stack([NORMAL, tangent, np.cross(NORMAL, tangent)])
    quat = Rotation.from_matrix(W.T @ axes).as_quat()[[3, 0, 1, 2]]
    base = np.zeros(len(points), dtype=phone.dtype)
    raw = np.einsum('ij,nj->ni', W.T, points)
    for i, f in enumerate(['x', 'y', 'z']): base[f] = raw[:, i]
    for i, f in enumerate(['rot_0', 'rot_1', 'rot_2', 'rot_3']): base[f] = quat[i]
    base['scale_0'] = np.log(.00015)
    base['scale_1'] = base['scale_2'] = np.log(np.minimum(spacing*.85, edge/3.3))
    for i, f in enumerate(['f_dc_0', 'f_dc_1', 'f_dc_2']): base[f] = (strip_color[i]-.5)/.28209479177387814
    a = np.clip(.96*smooth(edge/.025), 1e-8, 1-1e-8)
    base['opacity'] = np.log(a/(1-a))

    grains, provenance, translations = [], [], []
    start = np.floor((lo-rectangle[1])/period).astype(int)
    stop = np.ceil((hi-rectangle[0])/period).astype(int)
    for ix in range(start[0], stop[0]+1):
        for iz in range(start[1], stop[1]+1):
            shift_uv = period*np.array([ix, iz])
            shift = np.array([shift_uv[0], FLOOR[0]*shift_uv[0]+FLOOR[1]*shift_uv[1], shift_uv[1]])
            q = source_points+shift
            edges = clearance(q)
            keep = edges > .0005
            if not keep.any(): continue
            grain = template[keep].copy()
            raw = np.einsum('ij,nj->ni', W.T, q[keep])
            for i, f in enumerate(['x', 'y', 'z']): grain[f] = raw[:, i]
            for f in ['scale_1', 'scale_2']:
                grain[f] = np.log(np.minimum(np.exp(grain[f].astype(float)), edges[keep]/3.3))
            for i, f in enumerate(['f_dc_0', 'f_dc_1', 'f_dc_2']):
                color = (.5+.28209479177387814*grain[f].astype(float))*shading_gain[i]
                grain[f] = (color-.5)/.28209479177387814
            a = np.clip(alpha(grain)*smooth(edges[keep]/.025), 1e-8, 1-1e-8)
            grain['opacity'] = np.log(a/(1-a))
            grains.append(grain)
            provenance.append(source_ids[keep])
            translations.append(np.tile(shift, (int(keep.sum()), 1)))
    tread = np.concatenate(grains)
    additions = np.concatenate([base, tread])
    args.out.mkdir(parents=True)
    write_ply(args.out / 'iphone.ply', np.concatenate([prefix, additions]))
    shutil.copy2(args.baseline / 'reference-patches.ply', args.out / 'reference-patches.ply')
    write_ply(args.out / 'floor-additions.ply', additions)
    np.savez_compressed(args.out / 'changes.npz', phone_changed_indices=removed, phone_delete_indices=removed,
                        reference_changed_indices=np.empty(0, np.int64), reference_delete_indices=np.empty(0, np.int64))
    np.savez_compressed(args.out / 'sampling.npz', template_original_phone_ids=source_ids,
                        added_original_phone_ids=np.concatenate(provenance),
                        added_world_translations=np.concatenate(translations),
                        shadow_palette_reference_ids=palette_ids)
    shutil.copy2(__file__, args.out / 'generator.py')
    report = {'status': 'Unreviewed opposite-strip reconstruction using accepted floor material method',
              'baseline': str(args.baseline.resolve()),
              'baseline_phone_sha256': sha256_file(args.baseline / 'iphone.ply'),
              'baseline_reference_sha256': sha256_file(args.baseline / 'reference-patches.ply'),
              'source_trial': str(args.source_trial.resolve()), 'source_trial_report_sha256': sha256_file(args.source_trial/'report.json'),
              'accepted_floor': str(args.accepted_floor.resolve()), 'accepted_floor_report_sha256': sha256_file(args.accepted_floor/'report.json'),
              'template_source': str(source_dir.resolve()),
              'region': {'x': lo[[0]].tolist()+hi[[0]].tolist(), 'z': lo[[1]].tolist()+hi[[1]].tolist(),
                         'cabinet_z_from_xy1': CABINET.tolist(), 'cabinet_clearance': .04},
              'floor_y_from_xz1': FLOOR.tolist(),
              'counts': {'phone_deletions': len(removed), 'thin_base_added': len(base), 'captured_tread_added': len(tread),
                         'template_source_records': len(source_ids), 'shadow_palette_records': len(palette_ids)},
              'shading': {'observed_strip_median_rgb': strip_color.tolist(), 'observed_main_floor_median_rgb': main_color.tolist(),
                          'tread_rgb_gain': shading_gain.tolist()},
              'geometry': {'normal_sigma': .00015, 'base_depth': -.009, 'tread_depth': -.002,
                           'lateral_three_sigma_margin_factor': 3.3},
              'checks': {'reference_byte_exact_to_baseline': sha256_file(args.out/'reference-patches.ply') == sha256_file(args.baseline/'reference-patches.ply'),
                         'all_phone_nonopacity_prefix_fields_exact': all(np.array_equal(prefix[f],phone[f]) for f in phone.dtype.names if f!='opacity'),
                         'all_additions_finite': all(bool(np.isfinite(additions[f]).all()) for f in additions.dtype.names)},
              'limitations': ['Explicit reconstructed floor material repeats a measured captured tread patch; not exact unobserved detail.',
                              'Shadow shading is derived from captured local broad floor radiance.',
                              'The guarded front-seat/loose-item cluster is not altered.',
                              'Physical deletion must be confirmed in the final compositor, and this does not certify the whole scene collider.'],
              'generator_sha256': sha256_file(args.out/'generator.py'), 'changes_sha256': sha256_file(args.out/'changes.npz'),
              'sampling_sha256': sha256_file(args.out/'sampling.npz'), 'remote_publish': False}
    if not all(report['checks'].values()): raise RuntimeError(report['checks'])
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
