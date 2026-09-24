#!/usr/bin/env python3
"""Conservative, reproducible cleanup of the full-resolution iPhone ambulance.

Works on native PLY records (log scales, opacity logits, wxyz quaternions).
High-opacity surface samples constrain removal of detached neutral haze. Small
surface details and saturated straps are not subjected to a global opacity cut.
The input is never overwritten. Manual upholstery repair is a separate stage.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_mannequin import read_ply, sha256_file

DEFAULTS = {
    'anchorOpacity': 0.8, 'anchorVoxel': 0.004,
    'anchorMaxScale': 0.025, 'anchorMaxThickness': 0.0015,
    'candidateOpacity': 0.7,
    'surfaceTolerance': 0.005, 'shellFadeDistance': 0.016,
    'maximumAnchorDistance': 0.07, 'maximumPlaneSpread': 0.005,
    'shellStrength': 0.94, 'minimumOutputOpacity': 0.015,
    'minimumNeutralLuma': 0.42,
}


def columns(vertices, names):
    return np.column_stack([vertices[n] for n in names])


def write_ply(path, records):
    """Preserve every named property, including any higher SH coefficients."""
    header = ['ply', 'format binary_little_endian 1.0',
              'comment iPhone ambulance appearance cleanup; source preserved',
              f'element vertex {len(records)}']
    data = np.empty(len(records), dtype=[(n, '<f4') for n in records.dtype.names])
    for name in records.dtype.names:
        header.append(f'property float {name}')
        data[name] = records[name]
    header.append('end_header')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(('\n'.join(header) + '\n').encode('ascii'))
        data.tofile(stream)
    temporary.replace(path)


def surface_cleanup(vertices, config):
    p = columns(vertices, ['x', 'y', 'z']).astype(np.float64)
    s = np.exp(np.clip(columns(vertices, ['scale_0', 'scale_1', 'scale_2']), -40, 20))
    alpha = 1 / (1 + np.exp(-np.clip(vertices['opacity'], -40, 40)))
    rgb = np.clip(.5 + .28209479177387814 * columns(vertices, ['f_dc_0', 'f_dc_1', 'f_dc_2']), 0, 1)
    q = columns(vertices, ['rot_1', 'rot_2', 'rot_3', 'rot_0']).astype(np.float64)
    finite = np.ones(len(vertices), bool)
    for name in vertices.dtype.names:
        finite &= np.isfinite(vertices[name])
    finite &= np.linalg.norm(q, axis=1) > 1e-10
    largest = s.max(axis=1)
    smallest = s.min(axis=1)
    # Large Gaussians are essential on the textureless ceiling: size alone
    # cannot distinguish a floater from a valid low-texture surface.
    keep = finite.copy()
    anchors = np.flatnonzero(keep & (alpha >= config['anchorOpacity']) &
                             (largest < config['anchorMaxScale']) &
                             (smallest < config['anchorMaxThickness']))
    # Highest-confidence sample in each small voxel; deterministic ordering.
    anchors = anchors[np.argsort(-alpha[anchors], kind='stable')]
    voxels = np.floor(p[anchors] / config['anchorVoxel']).astype(np.int64)
    _, selected = np.unique(voxels, axis=0, return_index=True)
    anchors = anchors[selected]
    ap = p[anchors]
    tree = cKDTree(ap)
    rotation = Rotation.from_quat(q[anchors]).as_matrix()
    normal = rotation[np.arange(len(anchors)), :, np.argmin(s[anchors], axis=1)]
    candidates = np.flatnonzero(keep & ((alpha < config['candidateOpacity']) | (largest > .04)))
    factor = np.ones(len(vertices), np.float32)
    shell_count = isolated_count = 0
    print(f'Surface anchors: {len(anchors):,}; candidates: {len(candidates):,}', flush=True)
    for start in range(0, len(candidates), 100000):
        ids = candidates[start:start + 100000]
        distance, neighbor = tree.query(p[ids], k=8, workers=4)
        near = ap[neighbor]
        ns = normal[neighbor]
        n0 = ns[:, :1]
        agreement = np.abs(np.sum(ns * n0, axis=2))
        plane_distance = np.abs(np.sum((p[ids, None, :] - near) * ns, axis=2))
        # A consistent neighborhood must agree both in normal and offset.
        spread = np.median(np.abs(np.sum((near - near[:, :1]) * n0, axis=2)), axis=1)
        coherent = (np.median(agreement, axis=1) > .9) & (spread < config['maximumPlaneSpread'])
        supported = (distance[:, -1] < config['maximumAnchorDistance']) & coherent
        off_surface = np.median(plane_distance, axis=1)
        neutral = ((rgb[ids].max(1) - rgb[ids].min(1)) < .28) & (rgb[ids].mean(1) > config['minimumNeutralLuma'])
        low_confidence = alpha[ids] < config['candidateOpacity']
        shell = supported & neutral & low_confidence
        ramp = np.clip((off_surface - config['surfaceTolerance']) / config['shellFadeDistance'], 0, 1)
        local_factor = 1 - config['shellStrength'] * ramp
        factor[ids[shell]] *= local_factor[shell]
        shell_count += int(np.sum(shell & (ramp > 0)))
        # Isolated faint blobs away from any reliable surface, never sparse
        # dark upholstery; these are independently supported by size + color.
        bright = neutral & (rgb[ids].mean(1) > .42)
        isolated = (distance[:, 0] > .09) & (largest[ids] > .018) & bright & (alpha[ids] < .45)
        keep[ids[isolated]] = False
        isolated_count += int(isolated.sum())
    changed_alpha = alpha * factor
    keep &= changed_alpha >= config['minimumOutputOpacity']
    output = np.array(vertices[keep], copy=True)
    changed = factor[keep] < 1
    a = np.clip(changed_alpha[keep][changed], 1e-7, 1-1e-7)
    output['opacity'][changed] = np.log(a / (1 - a))
    report = {
        'inputGaussians': len(vertices), 'surfaceAnchors': len(anchors),
        'invalidRemoved': int((~finite).sum()),
        'detachedShellAttenuated': shell_count, 'isolatedBlobsRemoved': isolated_count,
        'totalRemoved': int((~keep).sum()), 'retainedBeforeRepair': len(output),
        'opacityModifiedRetained': int(changed.sum()),
        'retainedPositionsAndCovariancesUnchanged': True,
    }
    return output, report, np.flatnonzero(keep)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT/'raw/ambulance_exp11_boot_sharp.ply')
    parser.add_argument('--out', type=Path, default=ROOT/'raw/ambulance-cleanup/cleaned.ply')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('cleanup-config.json'))
    parser.add_argument('--skip-repair', action='store_true')
    args = parser.parse_args()
    args.input = args.input.resolve()
    args.out = args.out.resolve()
    if args.input.resolve() == args.out.resolve():
        parser.error('Choose an output distinct from the source scan')
    config = DEFAULTS.copy()
    if args.config:
        config.update(json.loads(args.config.read_text()))
    source_hash = sha256_file(args.input)
    vertices, _, _ = read_ply(args.input)
    output, report, indices = surface_cleanup(vertices, config)
    if not args.skip_repair:
        from repair_surfaces import repair
        output, report['manualRepairs'] = repair(output)
    report['finalOpacityModifiedRetained'] = int(np.count_nonzero(
        output['opacity'][:len(indices)] != vertices['opacity'][indices]))
    write_ply(args.out, output)
    np.save(args.out.with_suffix('.source-indices.npy'), indices)
    def label(path):
        try:
            return str(path.relative_to(ROOT))
        except ValueError:
            return str(path)
    report.update({'source': label(args.input), 'sourceSha256': source_hash,
                   'output': label(args.out), 'outputGaussians': len(output),
                   'outputSha256': sha256_file(args.out), 'parameters': config,
                   'nativePlyCoordinates': True,
                   'viewerConversionEulerDegrees': [-78.243, .463, -4.499]})
    args.out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
