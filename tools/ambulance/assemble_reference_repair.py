#!/usr/bin/env python3
"""Compose reviewed capture-backed patches without duplicating reference rows.

The original iPhone model is retained in its native frame. Only selected
opacity values change. Reference geometry, covariance and degree-three SH
come from the validated aligned Insta360 capture, without repainting.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cleanup import ROOT, read_ply, sha256_file, write_ply


def alpha(logits):
    return 1 / (1 + np.exp(-np.clip(logits.astype(np.float64), -40, 40)))


def selection(path, count):
    data = np.load(path)
    ids, weights = data['indices'], data['multipliers']
    if ids.ndim != 1 or weights.shape != ids.shape:
        raise ValueError(f'Invalid selection shape: {path}')
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError(f'Invalid or duplicate row indices: {path}')
    if np.any(ids < 0) or np.any(ids >= count):
        raise ValueError(f'Out of range row indices: {path}')
    if not np.isfinite(weights).all() or np.any(weights < 0) or np.any(weights > 1):
        raise ValueError(f'Invalid opacity multipliers: {path}')
    return ids, weights


def apply_opacity(records, multipliers):
    out = records.copy()
    a = np.clip(alpha(records['opacity']) * multipliers, 1e-8, 1-1e-8)
    out['opacity'] = np.log(a / (1-a))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iphone', type=Path, default=ROOT/'raw/ambulance_exp11_boot_sharp.ply')
    ap.add_argument('--reference', type=Path, default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply')
    ap.add_argument('--patch', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    sources = {args.iphone.resolve(), args.reference.resolve()}
    outputs = { (args.out/name).resolve() for name in
                ['iphone.ply', 'reference-patches.ply', 'selections.npz', 'report.json'] }
    if sources & outputs:
        raise ValueError('Output paths must not overwrite either capture input')
    phone, _, _ = read_ply(args.iphone)
    ref, _, _ = read_ply(args.reference)
    ph, rh = sha256_file(args.iphone), sha256_file(args.reference)
    if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):
        raise ValueError('Reference must retain degree-three SH')
    pw, rw = np.ones(len(phone)), np.zeros(len(ref))
    components = []
    for folder in args.patch:
        report = json.loads((folder/'report.json').read_text())
        if report['iphone_sha256'] != ph or report['reference_sha256'] != rh:
            raise ValueError(f'Patch source hashes do not match: {folder}')
        pi, pm = selection(folder/'iphone-opacity-selection.npz', len(phone))
        ri, rm = selection(folder/'reference-selection.npz', len(ref))
        # A patch overlap replaces once, rather than double-attenuating or
        # drawing a second copy of the same reference Gaussian.
        pw[pi] = np.minimum(pw[pi], pm)
        rw[ri] = np.maximum(rw[ri], rm)
        components.append({'directory': str(folder.resolve()),
                           'report_sha256': sha256_file(folder/'report.json'),
                           'iphone_selection_sha256': sha256_file(folder/'iphone-opacity-selection.npz'),
                           'reference_selection_sha256': sha256_file(folder/'reference-selection.npz'),
                           'iphone_rows': len(pi), 'reference_rows': len(ri)})
    pi, ri = np.flatnonzero(pw < 1), np.flatnonzero(rw > 0)
    out = phone.copy()
    out[pi] = apply_opacity(phone[pi], pw[pi])
    patch = apply_opacity(ref[ri], rw[ri])
    checks = {}
    checks['phone_geometry_covariance_color_exact'] = all(
        np.array_equal(out[f], phone[f]) for f in phone.dtype.names if f != 'opacity')
    unchanged = np.ones(len(phone), bool); unchanged[pi] = False
    checks['unselected_phone_rows_exact'] = np.array_equal(out[unchanged], phone[unchanged])
    checks['reference_geometry_covariance_sh_exact'] = all(
        np.array_equal(patch[f], ref[f][ri]) for f in ref.dtype.names if f != 'opacity')
    checks['phone_opacity_never_increased'] = bool(np.all(alpha(out['opacity'][pi]) <= alpha(phone['opacity'][pi])+1e-7))
    checks['reference_opacity_never_increased'] = bool(np.all(alpha(patch['opacity']) <= alpha(ref['opacity'][ri])+1e-7))
    checks['all_output_properties_finite'] = all(
        np.isfinite(v[f]).all().item() for v in [out, patch] for f in v.dtype.names)
    if not all(checks.values()):
        raise RuntimeError(f'Numeric verification failed: {checks}')
    args.out.mkdir(parents=True, exist_ok=True)
    write_ply(args.out/'iphone.ply', out)
    write_ply(args.out/'reference-patches.ply', patch)
    np.savez_compressed(args.out/'selections.npz', iphone_indices=pi,
                        iphone_multipliers=pw[pi], reference_indices=ri,
                        reference_multipliers=rw[ri])
    report = {'status': 'numerically verified; visual review required',
              'method': 'Manual bounded transfer of actual aligned Insta360 Gaussians into original iPhone scan',
              'iphone': str(args.iphone.resolve()), 'iphone_sha256': ph,
              'reference': str(args.reference.resolve()), 'reference_sha256': rh,
              'components': components, 'iphone_source_count': len(phone),
              'iphone_selected_count': len(pi),
              'iphone_opacity_changed_count': int(np.count_nonzero(out['opacity'] != phone['opacity'])),
              'reference_added_count': len(ri),
              'combined_count': len(out)+len(patch), 'checks': checks,
              'iphone_output_sha256': sha256_file(args.out/'iphone.ply'),
              'reference_output_sha256': sha256_file(args.out/'reference-patches.ply'),
              'limitations': ['Capture differences prohibit unrestricted scene fusion.',
                              'Unpatched iPhone objects may retain haze.',
                              'Numeric validity does not establish visual improvement.'],
              'remote_publish': False}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
