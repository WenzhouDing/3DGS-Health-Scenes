#!/usr/bin/env python3
"""Trial-only transfer of observed Insta360 roof Gaussians into the iPhone scan.

Both PLY inputs must be in the native iPhone coordinate frame. This retains
Insta360 geometry, covariance and rotated SH shading; it invents no roof sheet.
Writes two complementary inputs so the iPhone need not be padded to SH degree 3
on disk. The final export must transform both inputs together to viewer space.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cleanup import ROOT, columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW


def roof_weight(vertices, fascias=False):
    p = np.einsum('ij,nj->ni', WORLD_FROM_RAW,
                  columns(vertices, ['x', 'y', 'z']).astype(np.float64))
    # The shear follows the fixed cabin side edges in the original scan.
    u, v = p[:, 0], p[:, 2] + .085 * p[:, 0]
    uv = np.column_stack([u, v])
    lo, hi = np.array([-1.43, -.69]), np.array([1.43, .57])
    edge = np.minimum(uv - lo, hi - uv).min(axis=1)
    w = np.clip(edge / .035, 0, 1)
    w = w * w * (3 - 2 * w)
    w *= (p[:, 1] >= .87) & (p[:, 1] < 1.1)
    if fascias:
        # Capture-backed inclined lamp panels. Bounds stop above cabinets and
        # wall pads; each transition is feathered without flattening the scan.
        for lo, hi in [([-1.45, .735, -1.2], [1.43, .98, -.66]),
                       ([-.95, .795, .56], [.68, .98, .94])]:
            edge = np.minimum(p - lo, np.asarray(hi) - p).min(axis=1)
            fw = np.clip(edge / .025, 0, 1)
            w = np.maximum(w, fw * fw * (3 - 2 * fw))
    rgb = .5 + .28209479177387814 * columns(vertices, ['f_dc_0','f_dc_1','f_dc_2'])
    # Hanging colored restraints belong to their original capture; no second set.
    w *= np.ptp(rgb, axis=1) < .27
    return w, p


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--iphone', type=Path, default=ROOT/'raw/ambulance_exp11_boot_sharp.ply')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=ROOT/'raw/ambulance-cleanup/pass3/trials/roof')
    parser.add_argument('--fascias', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    phone, _, _ = read_ply(args.iphone)
    ref, _, _ = read_ply(args.reference)
    if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):
        raise ValueError('Reference must retain all degree-3 SH coefficients')
    pw, pp = roof_weight(phone, args.fascias)
    rw, rp = roof_weight(ref, args.fascias)
    out = phone.copy()
    ids = np.flatnonzero(pw > 0)
    alpha = 1 / (1 + np.exp(-np.clip(phone['opacity'][ids], -40, 40)))
    alpha = np.clip(alpha * (1 - pw[ids]), 1e-8, 1-1e-8)
    out['opacity'][ids] = np.log(alpha / (1 - alpha))
    selected = np.flatnonzero(rw > 0)
    patch = ref[selected].copy()
    alpha = 1 / (1 + np.exp(-np.clip(patch['opacity'], -40, 40)))
    alpha = np.clip(alpha * rw[selected], 1e-8, 1-1e-8)
    patch['opacity'] = np.log(alpha / (1 - alpha))
    write_ply(args.out/'iphone.ply', out)
    write_ply(args.out/'reference-roof.ply', patch)
    np.savez_compressed(args.out/'selection.npz', iphone_indices=ids,
                        reference_indices=selected, iphone_weights=pw[ids],
                        reference_weights=rw[selected])
    np.savez_compressed(args.out/'iphone-opacity-selection.npz',
                        indices=ids, multipliers=1-pw[ids])
    np.savez_compressed(args.out/'reference-selection.npz',
                        indices=selected, multipliers=rw[selected])
    report = {'status':'unreviewed trial', 'iphone':str(args.iphone),
              'iphone_sha256':sha256_file(args.iphone),
              'aligned_reference':str(args.reference),
              'reference_sha256':sha256_file(args.reference),
              'iphone_opacity_changed_count':len(ids),
              'reference_gaussians_added_count':len(patch),
              'reference_bounds_world':[rp[selected].min(0).tolist(),rp[selected].max(0).tolist()],
              'phone_geometry_and_colors_unchanged':True,
              'reference_geometry_covariance_and_sh_unchanged':True,
              'distant_iphone_splats_unchanged':True,
              'includes_inclined_lamp_fascias':args.fascias,
              'world_from_raw_euler':[-78.243,.463,-4.499],
              'note':'Actual roof material and fixture support from the other capture. Must reject if matched views reveal seams or double fixtures.'}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
