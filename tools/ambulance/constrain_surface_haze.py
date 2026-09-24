#!/usr/bin/env python3
"""Reduce Gaussian thickness at measured ceiling/corner surfaces.

No texture transfer, new splats, recoloring, position projection or opacity
change. A normal-direction contraction preserves the full tangent covariance.
Every trial starts from immutable baseline files and requires visual review.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W

FIELDS = {'scale_0', 'scale_1', 'scale_2', 'rot_0', 'rot_1', 'rot_2', 'rot_3'}


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def covariance(v):
    q = columns(v, ['rot_1', 'rot_2', 'rot_3', 'rot_0']).astype(float)
    axes = np.einsum('ij,njk->nik', W, Rotation.from_quat(q).as_matrix())
    scaled = axes * np.exp(columns(v, ['scale_0', 'scale_1', 'scale_2']).astype(float))[:, None, :]
    return np.einsum('nik,njk->nij', scaled, scaled)


def constrain(v, recipe):
    out = v.copy()
    p = np.einsum('ij,nj->ni', W, columns(v, ['x', 'y', 'z']).astype(float))
    rgb = .5 + .28209479177387814 * columns(v, ['f_dc_0', 'f_dc_1', 'f_dc_2'])
    alpha = 1 / (1 + np.exp(-np.clip(v['opacity'].astype(float), -40, 40)))
    edited = np.zeros(len(v), bool)
    evidence = []
    provenance = {}
    for region in recipe['regions']:
        axis, uv_axes = region['axis'], region['uv_axes']
        cf = np.asarray(region['plane'], float)
        uv = p[:, uv_axes]
        lo, hi = np.asarray(region['uv_bounds'], float)
        edge = np.minimum(uv - lo, hi - uv).min(1)
        normal = np.zeros(3)
        normal[axis] = 1
        normal[uv_axes] = -cf[:2]
        length = np.linalg.norm(normal)
        normal /= length
        distance = (p[:, axis] - np.einsum('ni,i->n', uv, cf[:2]) - cf[2]) / length
        dlo, dhi = region['depth_band']
        weight = smooth(edge / region.get('feather', .025))
        weight *= smooth(np.minimum(distance - dlo, dhi - distance) / region.get('depth_feather', .006))
        weight[(alpha < region.get('alpha_min', .01)) | (np.ptp(rgb, axis=1) > region.get('chroma_max', .22))] = 0
        if 'luma_min' in region:
            weight[rgb.mean(1) < region['luma_min']] = 0
        if 'luma_max' in region:
            weight[rgb.mean(1) > region['luma_max']] = 0
        # Explicit immutable lower-scene guard: floors and rejected wall panel.
        weight[p[:, 1] < recipe.get('minimum_world_y', .70)] = 0
        for box in region.get('protected_boxes', []):
            blo, bhi = np.asarray(box['bounds'], float)
            weight[((p >= blo) & (p <= bhi)).all(1)] = 0
        for disc in region.get('protected_discs', []):
            center = np.asarray(disc['uv_center'])
            weight[np.linalg.norm(uv - center, axis=1) < disc['radius']] = 0
        ids = np.flatnonzero(weight > 0)
        before = covariance(v[ids])
        sig = np.sqrt(np.maximum(np.einsum('i,nij,j->n', normal, before, normal), 0))
        selected = sig > region.get('normal_sigma_min', region['normal_sigma'] * 1.5)
        radius = 3 * np.sqrt(np.maximum(np.diagonal(before, axis1=1, axis2=2), 0))
        if region.get('require_support_inside_uv', False):
            selected &= ((uv[ids] - radius[:, uv_axes] >= lo) & (uv[ids] + radius[:, uv_axes] <= hi)).all(1)
        if 'maximum_axis_sigma' in region:
            selected &= np.sqrt(np.linalg.eigvalsh(before)[:, -1]) <= region['maximum_axis_sigma']
        # Protect entire three-sigma support, not merely the Gaussian center.
        for box in region.get('protected_boxes', []):
            blo, bhi = np.asarray(box['bounds'], float)
            selected &= ~((p[ids] + radius >= blo) & (p[ids] - radius <= bhi)).all(1)
        for disc in region.get('protected_discs', []):
            c_uv = before[:, uv_axes, :][:, :, uv_axes]
            uv_radius = 3 * np.sqrt(np.maximum(np.linalg.eigvalsh(c_uv)[:, -1], 0))
            selected &= np.linalg.norm(uv[ids] - np.asarray(disc['uv_center']), axis=1) > disc['radius'] + uv_radius
        ids, before, sig = ids[selected], before[selected], sig[selected]
        if np.any(edited[ids]):
            raise ValueError(f"Overlapping edits in {region['id']}; use disjoint physical regions")
        if not len(ids):
            evidence.append({'id': region['id'], 'changed': 0})
            continue
        k = 1 - weight[ids] * (1 - np.minimum(1, region['normal_sigma'] / sig))
        nn = np.outer(normal, normal)
        projection = np.eye(3) - nn
        contract = np.eye(3)[None] - (1 - k[:, None, None]) * nn
        after = contract @ before @ contract.transpose(0, 2, 1)
        eigen, axes = np.linalg.eigh(after)
        if not np.all(eigen > 0):
            raise ValueError('Contraction produced a nonpositive covariance')
        axes[np.linalg.det(axes) < 0, :, 0] *= -1
        quat = Rotation.from_matrix(np.einsum('ij,njk->nik', W.T, axes)).as_quat()
        for i, field in enumerate(['scale_0', 'scale_1', 'scale_2']):
            out[field][ids] = .5 * np.log(eigen[:, i])
        for i, field in enumerate(['rot_1', 'rot_2', 'rot_3', 'rot_0']):
            out[field][ids] = quat[:, i]
        actual = covariance(out[ids])
        actual_sig = np.sqrt(np.einsum('i,nij,j->n', normal, actual, normal))
        tangent_error = np.max(np.abs(projection @ (actual - before) @ projection), axis=(1, 2))
        tolerance = 2e-6 * np.maximum(np.max(np.abs(before), axis=(1, 2)), 1e-6)
        assert np.all(tangent_error < tolerance)
        assert np.all(actual_sig <= sig + 1e-7)
        assert np.allclose(actual_sig, sig * k, rtol=2e-5, atol=1e-8)
        edited[ids] = True
        provenance[region['id'] + '_indices'] = ids
        provenance[region['id'] + '_normal_sigma_before'] = sig
        provenance[region['id'] + '_normal_sigma_after'] = actual_sig
        provenance[region['id'] + '_weight'] = weight[ids]
        evidence.append({'id': region['id'], 'changed': len(ids),
            'normal': normal.tolist(), 'normal_distance_range': [float(distance[ids].min()), float(distance[ids].max())],
            'normal_sigma_before_q0_q50_q95_q100': np.quantile(sig, [0, .5, .95, 1]).tolist(),
            'normal_sigma_after_q0_q50_q95_q100': np.quantile(actual_sig, [0, .5, .95, 1]).tolist(),
            'maximum_tangent_covariance_error': float(tangent_error.max())})
    checks = {
        'no_additions_or_removals': len(out) == len(v),
        'centers_colors_SH_opacity_exact': all(np.array_equal(v[f], out[f]) for f in v.dtype.names if f not in FIELDS),
        'unselected_records_exact': v[~edited].tobytes() == out[~edited].tobytes(),
        'lower_scene_records_exact': v[p[:, 1] < recipe.get('minimum_world_y', .70)].tobytes() == out[p[:, 1] < recipe.get('minimum_world_y', .70)].tobytes(),
        'all_fields_finite': all(bool(np.isfinite(out[f]).all()) for f in out.dtype.names),
    }
    assert all(checks.values()), checks
    return out, {'changed': int(edited.sum()), 'regions': evidence, 'checks': checks}, provenance


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--recipe', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        ap.error('Use a new trial directory')
    recipe = json.loads(args.recipe.read_text())
    args.out.mkdir(parents=True)
    report = {'status': 'Unreviewed surface-thickness trial', 'baseline': str(args.baseline.resolve()),
        'baseline_report_sha256': sha256_file(args.baseline / 'report.json'), 'recipe': recipe,
        'baseline_hashes': {}, 'streams': {}, 'remote_publish': False}
    provenance = {}
    for stream, filename in [('iphone', 'iphone.ply'), ('reference', 'reference-patches.ply')]:
        v, _, _ = read_ply(args.baseline / filename)
        out, result, changes = constrain(v, recipe)
        write_ply(args.out / filename, out)
        reread, _, _ = read_ply(args.out / filename)
        assert reread.dtype == out.dtype and reread.tobytes() == out.tobytes()
        result['output_sha256'] = sha256_file(args.out / filename)
        report['baseline_hashes'][stream] = sha256_file(args.baseline / filename)
        report['streams'][stream] = result
        provenance.update({stream + '_' + key: value for key, value in changes.items()})
    np.savez_compressed(args.out / 'changes.npz', **provenance)
    shutil.copy2(__file__, args.out / 'generator.py')
    shutil.copy2(args.recipe, args.out / 'recipe.json')
    report['generator_sha256'] = sha256_file(args.out / 'generator.py')
    report['changes_sha256'] = sha256_file(args.out / 'changes.npz')
    report['recipe_sha256'] = sha256_file(args.out / 'recipe.json')
    report['limitations'] = ['Numerical thickness reduction alone does not establish visual improvement.',
        'Surface centers and tangential support are preserved; baked texture/shading and other haze may remain.']
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
