#!/usr/bin/env python3
"""Independently verify a native ambulance cleanup and its source provenance.

Checks all Gaussian records, not a sample. Retained records must preserve every
source field except decreasing opacity; manual repair must append records. This
is a structural check, not a substitute for reviewing fixed-camera renders.
Requires NumPy only. Writes <output>.verification.json and exits nonzero on a
failed invariant. Neither source nor generated PLY is modified.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_mannequin import read_ply, sha256_file

CHUNK = 250000


def world_rotation():
    """Documented fixed viewer transform, independently built without SciPy."""
    x, y, z = np.deg2rad([-78.243, .463, -4.499])
    rx = np.array([[1, 0, 0], [0, np.cos(x), -np.sin(x)], [0, np.sin(x), np.cos(x)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rz = np.array([[np.cos(z), -np.sin(z), 0], [np.sin(z), np.cos(z), 0], [0, 0, 1]])
    return rz @ ry @ rx @ np.diag([-1., -1., 1.])


def verify_patch_records(cleaned, retained_count, patches):
    results = []
    cursor = retained_count
    transform = world_rotation()
    for patch in patches:
        count = patch['added_count']
        rows = cleaned[cursor:cursor + count]
        cursor += count
        if not count:
            continue
        xyz = np.einsum('ij,nj->ni', transform, np.column_stack([rows[n] for n in ['x', 'y', 'z']]).astype(np.float64))
        q = np.column_stack([rows[n] for n in ['rot_0', 'rot_1', 'rot_2', 'rot_3']]).astype(np.float64)
        q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-20)
        w, x, y, z = q.T
        raw_normal = np.column_stack([2 * (x*z + w*y), 2 * (y*z - w*x), 1 - 2 * (x*x + y*y)])
        actual_normal = np.einsum('ij,nj->ni', transform, raw_normal)
        expected_normal = np.asarray(patch['normal_world'])
        normal_agreement = np.abs(np.sum(actual_normal * expected_normal, axis=1))
        scale = np.exp(np.column_stack([rows[f'scale_{i}'] for i in range(3)]).astype(np.float64))
        expected_scale = [patch['sigma_tangent'], patch['sigma_tangent'], patch['sigma_normal']]
        rgb = .5 + .28209479177387814 * np.column_stack([rows[f'f_dc_{i}'] for i in range(3)])
        lo, hi = np.asarray(patch['uv_bounds'])
        uv = xyz[:, patch['uv_axes']]
        dlo, dhi = patch['depth_bounds']
        c = {
            'countMatches': len(rows) == count,
            'withinManualSurfaceBounds': bool(np.all(uv >= lo - 1e-6) and np.all(uv <= hi + 1e-6)
                                             and np.all(xyz[:, patch['axis']] >= dlo - 1e-6)
                                             and np.all(xyz[:, patch['axis']] <= dhi + 1e-6)),
            'worldBoundsMatchReport': bool(np.allclose([xyz.min(0), xyz.max(0)], patch['world_bounds'], atol=1e-6)),
            'thinAxisMatchesFittedSurfaceNormal': bool(np.all(normal_agreement > 1 - 1e-5)),
            'scalesMatchReport': bool(np.allclose(scale, expected_scale, rtol=1e-5, atol=1e-8)),
            'colorWithinReportedRange': bool(np.all(rgb >= np.asarray(patch['color_min']) - 1e-6)
                                            and np.all(rgb <= np.asarray(patch['color_max']) + 1e-6)),
            'higherSHZero': all(not np.any(rows[n]) for n in rows.dtype.names if n.startswith('f_rest_')),
        }
        results.append({'id': patch['id'], 'passed': all(c.values()), 'checks': c,
                        'minimumNormalAgreement': float(normal_agreement.min())})
    return results


def verify(source, output, report_path=None, indices_path=None):
    report_path = report_path or output.with_suffix('.json')
    indices_path = indices_path or output.with_suffix('.source-indices.npy')
    report = json.loads(report_path.read_text())
    original, source_count, source_names = read_ply(source)
    cleaned, output_count, output_names = read_ply(output)
    indices = np.load(indices_path, mmap_mode='r', allow_pickle=False)
    checks = {}
    source_hash = sha256_file(source)
    output_hash = sha256_file(output)
    checks['sourcePathDistinct'] = source.resolve() != output.resolve()
    checks['sourceHashMatchesGenerationReport'] = source_hash == report['sourceSha256']
    checks['outputHashMatchesGenerationReport'] = output_hash == report['outputSha256']
    checks['sourceCountMatchesReport'] = source_count == report['inputGaussians']
    checks['outputCountMatchesReport'] = output_count == report['outputGaussians']
    checks['fieldNamesAndOrderPreserved'] = source_names == output_names
    checks['indicesOneDimensionalInteger'] = indices.ndim == 1 and np.issubdtype(indices.dtype, np.integer)
    if not checks['indicesOneDimensionalInteger']:
        raise ValueError('Source indices must be a one-dimensional integer array')
    retained_count = len(indices)
    checks['indexCountMatchesReport'] = retained_count == report['retainedBeforeRepair']
    checks['indicesInBounds'] = bool(np.all(indices >= 0) and np.all(indices < source_count))
    checks['indicesStrictlyIncreasing'] = bool(np.all(np.diff(indices) > 0))
    checks['outputContainsRetainedPrefix'] = output_count >= retained_count
    checks['removalCountMatchesReport'] = source_count - retained_count == report['totalRemoved']
    added_count = output_count - retained_count
    repairs = report.get('manualRepairs', {})
    checks['addedCountMatchesReport'] = added_count == repairs.get('added_count', 0)
    if not all(checks[n] for n in ('fieldNamesAndOrderPreserved', 'indicesInBounds', 'outputContainsRetainedPrefix')):
        raise ValueError('Cannot compare records with invalid schema, indices or retained prefix')

    changed_fields = {name: 0 for name in source_names}
    finite_failures = {name: 0 for name in output_names}
    opacity_increases = zero_quaternions = invalid_scales = 0
    source_quaternion_range = [float('inf'), 0.]
    added_quaternion_max_error = 0.
    for start in range(0, output_count, CHUNK):
        chunk = cleaned[start:start + CHUNK]
        for name in output_names:
            finite_failures[name] += int(np.count_nonzero(~np.isfinite(chunk[name])))
        q = np.column_stack([chunk[n] for n in ['rot_0', 'rot_1', 'rot_2', 'rot_3']]).astype(np.float64)
        norms = np.linalg.norm(q, axis=1)
        zero_quaternions += int(np.count_nonzero(norms < 1e-10))
        retained_here = min(len(chunk), max(0, retained_count - start))
        if retained_here:
            source_quaternion_range[0] = min(source_quaternion_range[0], float(norms[:retained_here].min()))
            source_quaternion_range[1] = max(source_quaternion_range[1], float(norms[:retained_here].max()))
        if retained_here < len(chunk):
            added_quaternion_max_error = max(added_quaternion_max_error, float(np.abs(norms[retained_here:] - 1).max()))
        scales = np.column_stack([chunk[n] for n in ['scale_0', 'scale_1', 'scale_2']]).astype(np.float64)
        with np.errstate(over='ignore', under='ignore', invalid='ignore'):
            scales = np.exp(scales)
        invalid_scales += int(np.count_nonzero((scales <= 0) | ~np.isfinite(scales)))
    for start in range(0, retained_count, CHUNK):
        end = min(start + CHUNK, retained_count)
        old = original[indices[start:end]]
        new = cleaned[start:end]
        for name in source_names:
            changed_fields[name] += int(np.count_nonzero(old[name] != new[name]))
        opacity_increases += int(np.count_nonzero(new['opacity'] > old['opacity']))
    checks['allOutputPropertiesFinite'] = not any(finite_failures.values())
    checks['allQuaternionsNonzero'] = zero_quaternions == 0
    checks['addedQuaternionsUnitLength'] = added_quaternion_max_error <= 1e-5
    checks['allScalesPositiveAndFinite'] = invalid_scales == 0
    checks['retainedGeometryColorAndSHExactlyPreserved'] = not any(v for k, v in changed_fields.items() if k != 'opacity')
    checks['retainedOpacityNeverIncreased'] = opacity_increases == 0
    expected_opacity_changes = report.get('finalOpacityModifiedRetained', report['opacityModifiedRetained'])
    checks['opacityChangeCountMatchesReport'] = changed_fields['opacity'] == expected_opacity_changes
    patch_checks = []
    if 'patches' in repairs:
        checks['patchAdditionCountsSumCorrectly'] = sum(p['added_count'] for p in repairs['patches']) == added_count
        checks['allAddedPatchesHaveObservedSupport'] = all(p['support_count'] > 0 for p in repairs['patches'] if p['added_count'] > 0)
        if checks['patchAdditionCountsSumCorrectly']:
            patch_checks = verify_patch_records(cleaned, retained_count, repairs['patches'])
            checks['allAddedPatchRecordsMatchSurfaceRecipe'] = all(p['passed'] for p in patch_checks)
    result = {
        'passed': all(checks.values()), 'checks': checks,
        'source': str(source.resolve()), 'sourceSha256': source_hash,
        'output': str(output.resolve()), 'outputSha256': output_hash,
        'sourceGaussians': source_count, 'retainedSourceGaussians': retained_count,
        'appendedRepairGaussians': added_count, 'outputGaussians': output_count,
        'retainedChangedPropertyCounts': changed_fields,
        'nonfinitePropertyCounts': finite_failures,
        'retainedQuaternionNormRange': source_quaternion_range,
        'addedQuaternionMaximumUnitError': added_quaternion_max_error,
        'manualPatchVerification': patch_checks,
        'note': 'Structural and provenance verification only; visual quality requires fixed-camera render review.',
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'raw/ambulance_exp11_boot_sharp.ply')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--indices', type=Path)
    args = parser.parse_args()
    result = verify(args.source, args.output, args.report, args.indices)
    destination = args.output.with_suffix('.verification.json')
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    sys.exit(0 if result['passed'] else 1)


if __name__ == '__main__':
    main()
