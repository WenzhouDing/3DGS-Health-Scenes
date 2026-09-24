#!/usr/bin/env python3
"""Independently audit every field of a composed capture-backed repair.

Numeric validity establishes provenance/preservation only. Final perspective
review remains mandatory before installing the result in the local viewer.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cleanup import read_ply, sha256_file

EXPECTED_PHONE = '196cd3bc65e3301cea3d382f0fe10b1d44e9a0f6dd4c383836c385df7419f3c3'
EXPECTED_REFERENCE = '4531829be9fc4a827cb367d76a78b9cee818782ff2a612a5dec66f4f9e80a988'


def alpha(logits):
    return 1/(1+np.exp(-np.clip(logits.astype(np.float64), -40, 40)))


def selection(path, count):
    with np.load(path) as data:
        ids, values = data['indices'], data['multipliers']
    if (ids.ndim != 1 or values.shape != ids.shape or not np.issubdtype(ids.dtype, np.integer)
        or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= count)
        or not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1)):
        raise ValueError(f'Invalid component selection: {path}')
    return ids, values


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    args = ap.parse_args()
    report = json.loads((args.directory/'report.json').read_text())
    phone_path, ref_path = Path(report['iphone']), Path(report['reference'])
    phone, _, _ = read_ply(phone_path)
    ref, _, _ = read_ply(ref_path)
    out, _, _ = read_ply(args.directory/'iphone.ply')
    patch, _, _ = read_ply(args.directory/'reference-patches.ply')
    ph, rh = sha256_file(phone_path), sha256_file(ref_path)
    checks = {
        'original_phone_hash_unchanged': ph == EXPECTED_PHONE,
        'aligned_reference_hash_unchanged': rh == EXPECTED_REFERENCE,
        'reported_source_hashes_correct': ph == report['iphone_sha256'] and rh == report['reference_sha256'],
    }
    pm, rm = np.ones(len(phone)), np.zeros(len(ref))
    components = []
    reference_total = 0
    for component in report['components']:
        d = Path(component['directory'])
        pi, pw = selection(d/'iphone-opacity-selection.npz', len(phone))
        ri, rw = selection(d/'reference-selection.npz', len(ref))
        pm[pi] = np.minimum(pm[pi], pw)
        rm[ri] = np.maximum(rm[ri], rw)
        reference_total += len(ri)
        components.append({'component': str(d),
            'report_hash_matches': sha256_file(d/'report.json') == component['report_sha256'],
            'iphone_hash_matches': sha256_file(d/'iphone-opacity-selection.npz') == component['iphone_selection_sha256'],
            'reference_hash_matches': sha256_file(d/'reference-selection.npz') == component['reference_selection_sha256'],
            'iphone_rows': len(pi), 'reference_rows': len(ri)})
    pi, ri = np.flatnonzero(pm < 1), np.flatnonzero(rm > 0)
    s = np.load(args.directory/'selections.npz')
    checks['phone_selection_is_component_min'] = np.array_equal(pi, s['iphone_indices']) and np.array_equal(pm[pi], s['iphone_multipliers'])
    checks['reference_selection_is_component_max'] = np.array_equal(ri, s['reference_indices']) and np.array_equal(rm[ri], s['reference_multipliers'])
    checks['phone_count_and_dtype_exact'] = out.shape == phone.shape and out.dtype == phone.dtype
    checks['reference_count_and_dtype_exact'] = len(patch) == len(ri) and patch.dtype == ref.dtype
    checks['phone_all_nonopacity_fields_exact'] = all(np.array_equal(out[f], phone[f]) for f in phone.dtype.names if f != 'opacity')
    checks['unselected_phone_records_byte_exact'] = out[pm == 1].tobytes() == phone[pm == 1].tobytes()
    checks['reference_all_nonopacity_fields_exact'] = all(np.array_equal(patch[f], ref[f][ri]) for f in ref.dtype.names if f != 'opacity')
    checks['reference_retains_all_45_directional_coefficients'] = all(f'f_rest_{i}' in patch.dtype.names for i in range(45))
    checks['all_properties_finite'] = all(bool(np.isfinite(v[f]).all()) for v in [out, patch] for f in v.dtype.names)
    pa, ra = np.clip(alpha(phone['opacity'][pi])*pm[pi], 1e-8, 1-1e-8), np.clip(alpha(ref['opacity'][ri])*rm[ri], 1e-8, 1-1e-8)
    checks['phone_opacity_formula_float32_exact'] = np.array_equal(out['opacity'][pi], np.log(pa/(1-pa)).astype(out['opacity'].dtype))
    checks['reference_opacity_formula_float32_exact'] = np.array_equal(patch['opacity'], np.log(ra/(1-ra)).astype(patch['opacity'].dtype))
    checks['phone_alpha_not_increased_beyond_float32_rounding'] = bool(np.all(alpha(out['opacity'][pi]) <= alpha(phone['opacity'][pi])+1e-7))
    checks['reference_alpha_not_increased_beyond_float32_rounding'] = bool(np.all(alpha(patch['opacity']) <= alpha(ref['opacity'][ri])+1e-7))
    hashes = {'iphone': sha256_file(args.directory/'iphone.ply'), 'reference': sha256_file(args.directory/'reference-patches.ply')}
    checks['phone_output_hash_matches'] = hashes['iphone'] == report['iphone_output_sha256']
    checks['reference_output_hash_matches'] = hashes['reference'] == report['reference_output_sha256']
    frozen = all(c[k] for c in components for k in ['report_hash_matches', 'iphone_hash_matches', 'reference_hash_matches'])
    checks['all_component_reports_and_selections_frozen'] = frozen
    bs = np.load(args.baseline/'selections.npz')
    checks['accepted_phone_opacity_not_restored'] = bool(np.all(pm[bs['iphone_indices']] <= bs['iphone_multipliers']))
    checks['accepted_reference_support_retained'] = bool(np.all(rm[bs['reference_indices']] >= bs['reference_multipliers']))
    result = {'status': 'PASS numerical and frozen provenance hashes' if all(checks.values()) else 'FAIL',
        'scope': 'Independent all-record composition/provenance check; not visual approval.',
        'checks': checks, 'source_hashes': {'iphone': ph, 'reference': rh}, 'output_hashes': hashes,
        'counts': {'components': len(components), 'phone_source': len(phone), 'phone_selected': len(pi),
                   'phone_actually_modified': int(np.count_nonzero(out['opacity'] != phone['opacity'])),
                   'phone_unselected_exact': int((pm == 1).sum()), 'reference_unique': len(ri),
                   'reference_selection_sum_before_dedup': reference_total, 'combined': len(out)+len(patch)},
        'components': components, 'metadata_checks': {'component_reports_and_selections_frozen': frozen},
        'baseline': str(args.baseline), 'baseline_selections_sha256': sha256_file(args.baseline/'selections.npz')}
    (args.directory/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'checks': checks, 'counts': result['counts']}, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
