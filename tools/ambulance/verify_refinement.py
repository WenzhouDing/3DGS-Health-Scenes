#!/usr/bin/env python3
"""Verify all records of the manual surface refinement against its saved baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from cleanup import ROOT, columns, read_ply, sha256_file
from refine_surfaces import difference
from repair_surfaces import WORLD_FROM_RAW


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'raw/ambulance-cleanup/pass2/baseline/cleaned.ply')
    p.add_argument('--output',type=Path,default=ROOT/'raw/ambulance-cleanup/refined.ply')
    p.add_argument('--max-center-move',type=float,default=.07)
    a=p.parse_args()
    before, _, names=read_ply(a.input);after, _, new_names=read_ply(a.output)
    report=json.loads(a.output.with_suffix('.json').read_text())
    changes=np.load(a.output.with_suffix('.changes.npz'),allow_pickle=False)
    metrics,ids,bits=difference(before,after)
    finite={name:bool(np.isfinite(after[name]).all()) for name in names}
    checks={
        'baselineHashMatches':sha256_file(a.input)==report['baselineSha256'],
        'outputHashMatches':sha256_file(a.output)==report['outputSha256'],
        'originalTrainingSourceHashMatches':sha256_file(ROOT/report['sourceTrainingPly'])==report['sourceTrainingSha256'],
        'originalRowsPreservedAdditionsAppended':len(before)==report['baselineGaussianCount'] and len(after)==report['gaussianCount'] and len(after)>=len(before),
        'fieldNamesAndOrderPreserved':names==new_names,
        'allPropertiesFinite':all(finite.values()),
        'changeMapIdsExactlyMatch':np.array_equal(ids,changes['indices']),
        'changeMapPropertiesExactlyMatch':np.array_equal(bits,changes['property_bits']),
        'changeMapSchemaMatches':names==changes['property_names'].tolist(),
        'changeCountsMatch':metrics['modifiedProperties']==report['changes']['modifiedProperties'],
        'additionCountsMatch':metrics['addedRows']==report['changes']['addedRows'],
        'addedIndexMapMatches':np.array_equal(changes['added_indices'],np.arange(len(before),len(after))),
        'centerMovesWithinReviewedLimit':metrics.get('maximumCenterDisplacement',0)<=a.max_center_move,
    }
    zero_quaternions=0;invalid_scales=0
    for start in range(0,len(after),250000):
        rows=after[start:start+250000]
        q=columns(rows,['rot_0','rot_1','rot_2','rot_3']).astype(np.float64)
        zero_quaternions+=int((np.linalg.norm(q,axis=1)<1e-10).sum())
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            s=np.exp(columns(rows,['scale_0','scale_1','scale_2']).astype(np.float64))
        invalid_scales+=int(((s<=0)|~np.isfinite(s)).sum())
    checks['allQuaternionsNonzero']=zero_quaternions==0
    checks['allScalesPositiveFinite']=invalid_scales==0
    added=after[len(before):]
    if len(added):
        q=columns(added,['rot_0','rot_1','rot_2','rot_3']).astype(np.float64)
        scales=np.exp(columns(added,['scale_0','scale_1','scale_2']).astype(np.float64))
        positions=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(added,['x','y','z']).astype(np.float64))
        checks['addedQuaternionsUnitLength']=bool(np.allclose(np.linalg.norm(q,axis=1),1,atol=2e-6))
        checks['addedSupportInsideCabin']=bool(((positions>=[-1.9,-.9,-1.55])&(positions<=[1.9,1.1,1.1])).all())
        checks['addedSupportLocalScale']=bool((scales.max(axis=1)<.06).all())
        checks['addedSupportThin']=bool((scales.min(axis=1)<=.003).all())
    stages=report['stages']
    checks['stageAdditionCountsSumCorrectly']=sum(s['changes']['addedRows'] for s in stages)==len(added)
    for stage in stages:
        name=stage['stage']
        stage_ids=changes[name+'_indices'];stage_bits=changes[name+'_property_bits']
        checks[name+'StageMapValid']=bool(len(stage_ids)==stage['changes']['modifiedRows'] and
            len(stage_ids)==len(stage_bits) and (np.diff(stage_ids.astype(np.int64))>0).all() and
            (stage_bits>0).all() and len(changes[name+'_added_indices'])==stage['changes']['addedRows'])
    result={'passed':all(checks.values()),'checks':checks,'changes':metrics,
            'maximumAllowedCenterMove':a.max_center_move,
            'note':'All-record numeric/provenance check. Fixed-camera visual inspection remains required.'}
    a.output.with_suffix('.verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    if not result['passed']:raise SystemExit(1)


if __name__=='__main__':main()
