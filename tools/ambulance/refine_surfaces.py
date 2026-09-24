#!/usr/bin/env python3
"""Apply reviewed, bounded surface corrections to the initial ambulance cleanup.

Keeps row identity, an immutable baseline, and a per-property change map. The
architectural and object recipes are independent stages so either can be
rendered or disabled in isolation. The original training PLY is never edited.
"""
import argparse
import importlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT, columns, read_ply, sha256_file, write_ply

BASELINE = ROOT/'raw/ambulance-cleanup/pass2/baseline/cleaned.ply'
OUTPUT = ROOT/'raw/ambulance-cleanup/refined.ply'
STAGES = {'architecture': 'architectural_cleanup', 'objects': 'object_cleanup'}


def difference(before, after):
    if len(before) > len(after) or before.dtype.names != after.dtype.names:
        raise ValueError('Surface refinement must preserve the input prefix and schema; additions must be appended')
    if len(before.dtype.names) > 64:
        raise ValueError('Change map supports at most 64 properties')
    bits = np.zeros(len(before), dtype=np.uint64)
    counts = {}
    for i, name in enumerate(before.dtype.names):
        modified = before[name] != after[name][:len(before)]
        counts[name] = int(modified.sum())
        bits[modified] |= np.uint64(1 << i)
    ids = np.flatnonzero(bits)
    report = {'modifiedRows': len(ids), 'modifiedProperties': counts,
              'addedRows': len(after)-len(before)}
    if len(ids):
        delta = columns(after[ids], ['x','y','z']) - columns(before[ids], ['x','y','z'])
        move = np.linalg.norm(delta.astype(np.float64), axis=1)
        qa = columns(before[ids], ['rot_1','rot_2','rot_3','rot_0']).astype(np.float64)
        qb = columns(after[ids], ['rot_1','rot_2','rot_3','rot_0']).astype(np.float64)
        rotations = (Rotation.from_quat(qa).inv()*Rotation.from_quat(qb)).magnitude()
        report.update({'maximumCenterDisplacement': float(move.max()),
                       'movedCenterCount': int(np.count_nonzero(move)),
                       'displacementQuantiles': np.quantile(move,[.5,.9,.99]).tolist(),
                       'maximumQuaternionRotationDegrees': float(np.rad2deg(rotations.max()))})
    return report, ids, bits[ids]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASELINE)
    parser.add_argument('--out', type=Path, default=OUTPUT)
    parser.add_argument('--stages', default='architecture,objects')
    args = parser.parse_args()
    args.input = args.input.resolve(); args.out = args.out.resolve()
    if args.input == args.out:
        parser.error('Output must differ from the immutable input')
    selected = args.stages.split(',')
    if not selected or any(name not in STAGES for name in selected) or len(set(selected)) != len(selected):
        parser.error('Choose distinct stages: architecture,objects')
    source_hash = sha256_file(args.input)
    baseline, _, _ = read_ply(args.input)
    current = baseline
    reports = []
    stage_maps = {}
    for stage in selected:
        print('Applying', stage, flush=True)
        implementation = importlib.import_module(STAGES[stage])
        candidate, details = implementation.apply(current)
        metrics, ids, bits = difference(current, candidate)
        details = dict(details)
        claimed_ids = details.pop('affected_indices', details.pop('changed_indices', None))
        if claimed_ids is not None and not np.array_equal(np.asarray(claimed_ids), ids):
            raise ValueError(f'{stage}: affected-row report differs from actual changes')
        stage_maps[stage+'_indices'] = ids.astype(np.uint32)
        stage_maps[stage+'_property_bits'] = bits
        stage_maps[stage+'_added_indices'] = np.arange(len(current),len(candidate),dtype=np.uint32)
        reports.append({'stage':stage, 'changes':metrics, 'details':details})
        print(json.dumps({'stage':stage, **metrics}), flush=True)
        current = candidate
    metrics, ids, bits = difference(baseline, current)
    write_ply(args.out,current)
    np.savez_compressed(args.out.with_suffix('.changes.npz'),
                        indices=ids.astype(np.uint32), property_bits=bits,
                        added_indices=np.arange(len(baseline),len(current),dtype=np.uint32),
                        property_names=np.asarray(current.dtype.names), **stage_maps)
    def label(path):
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
    training = ROOT/'raw/ambulance_exp11_boot_sharp.ply'
    report = {'version':2,'baseline':label(args.input),'baselineSha256':source_hash,
              'output':label(args.out),'outputSha256':sha256_file(args.out),
              'baselineGaussianCount':len(baseline),'gaussianCount':len(current),'originalRowIdentityPreserved':True,
              'changes':metrics,'stages':reports,
              'viewerConversionEulerDegrees':[-78.243,.463,-4.499],
              'sourceTrainingPly':label(training),
              'sourceTrainingSha256':sha256_file(training)}
    args.out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':report['output'],'sha256':report['outputSha256'],**metrics}),flush=True)


if __name__ == '__main__':
    main()
