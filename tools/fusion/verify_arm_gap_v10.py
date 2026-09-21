"""Check a rigid arm correction or source-only hand repair against its baseline."""
import argparse
import json
from pathlib import Path
import numpy as np
from pipeline import ROOT, FIELDS, read_ply, transform_gaussians
from verify_rigid_wrist_v9 import load
from inspect_pad_v8 import verify_pad, OUT as PAD_AUDIT
from appearance_colors import load_color_overrides, apply_color_values
from cleanup_masks import load_cleanup_masks


def verify_hand_only_changes(before, after, restoration_ids, color_ids, pad_ids):
    """Reject undeclared row/alpha/color changes or any hand-only pose drift.

    Inputs use original source vertex IDs, independently of export row order.
    The source-parameter loop in main also verifies every retained arm Gaussian
    against its original rigidly transformed mean and covariance.
    """
    previous, current = before[4]['parts'], after[4]['parts']
    assert len(previous) == len(current) == 16, 'Expected all 16 mannequin parts'
    assert [p['id'] for p in previous] == [p['id'] for p in current], 'Part identities changed'
    for a, b in zip(previous, current):
        assert a['sourceToFrontRaw'] == b['sourceToFrontRaw'], 'Hand-only repair changed rigid pose: ' + a['id']
    hand = next(i for i, p in enumerate(current) if p['id'] == 'right_hand')
    records = []
    for capture, source in enumerate(['front', 'back']):
        old_rows = np.flatnonzero((before[1] == hand) & (before[2] == capture))
        new_rows = np.flatnonzero((after[1] == hand) & (after[2] == capture))
        old_ids, new_ids = before[3][old_rows], after[3][new_rows]
        assert len(np.unique(old_ids)) == len(old_ids), 'Duplicate baseline hand source rows'
        exported_ids = after[3][after[2] == capture]
        assert len(np.unique(exported_ids)) == len(exported_ids), 'A source row was duplicated in the export'
        common, old_at, new_at = np.intersect1d(old_ids, new_ids, return_indices=True)
        removed = np.setdiff1d(old_ids, new_ids)
        added = np.setdiff1d(new_ids, old_ids)
        assert not len(removed), 'Source restoration removed previously retained hand rows'
        assert np.isin(added, restoration_ids[source]).all(), 'New hand rows are outside declared restorations'
        assert not np.isin(added, before[3][before[2] == capture]).any(), 'New hand rows belonged to another baseline part'
        a, b = old_rows[old_at], new_rows[new_at]
        for field in FIELDS[:10]:
            np.testing.assert_array_equal(before[0][field][a], after[0][field][b],
                                          err_msg='Restoration changed retained hand means or covariance')
        alpha_changed = before[0][FIELDS[10]][a] != after[0][FIELDS[10]][b]
        assert np.isin(common[alpha_changed], restoration_ids[source]).all(), 'Hand opacity changed outside declared restorations'
        assert np.all(after[0][FIELDS[10]][b][alpha_changed] >= before[0][FIELDS[10]][a][alpha_changed]), 'Restoration reduced original retained opacity'
        old_colors = np.column_stack([before[0][f][a] for f in FIELDS[11:14]])
        new_colors = np.column_stack([after[0][f][b] for f in FIELDS[11:14]])
        color_changed = np.any(old_colors != new_colors, axis=1)
        assert np.isin(common[color_changed], color_ids[source]).all(), 'Hand colors changed outside declared color repairs'
        for name, selection in [('restoration', restoration_ids[source]), ('color repair', color_ids[source])]:
            selected = (after[2] == capture) & np.isin(after[3], selection)
            assert np.all(after[1][selected] == hand), name + ' selection crosses the hand boundary'
        records.append({'source': source, 'baselineRows': len(old_ids), 'outputRows': len(new_ids),
                        'retainedCommonRowsWithExactMeansAndCovariance': len(common),
                        'newRowsInsideDeclaredRestoration': len(added), 'removedRows': len(removed),
                        'retainedOpacityChangesInsideDeclaredRestoration': int(alpha_changed.sum()),
                        'retainedColorChangesInsideDeclaredRepair': int(color_changed.sum()),
                        'newRowsWithDeclaredColorRepair': int(np.isin(added, color_ids[source]).sum()),
                        'declaredRestorationRows': len(restoration_ids[source]),
                        'declaredColorRepairRows': len(color_ids[source]),
                        'retainedDeclaredColorRepairRows': int(np.isin(new_ids, color_ids[source]).sum())})
    old_pad = np.flatnonzero((before[2] == 1) & np.isin(before[3], pad_ids))
    new_pad = np.flatnonzero((after[2] == 1) & np.isin(after[3], pad_ids))
    old_pad = old_pad[np.argsort(before[3][old_pad])]
    new_pad = new_pad[np.argsort(after[3][new_pad])]
    assert len(old_pad), 'No measured wrist pad rows in baseline'
    for field in FIELDS:
        np.testing.assert_array_equal(before[0][field][old_pad], after[0][field][new_pad],
                                      err_msg='Hand restoration changed the measured wrist pad')
    for index in [1, 2, 3]:
        np.testing.assert_array_equal(before[index][old_pad], after[index][new_pad])
    return {'allRigidTransformsBitExactToBaseline': 16, 'handSourceChanges': records,
            'measuredBackPadRowsBitExactToBaseline': len(old_pad),
            'padFieldsLabelsSourceIdsAndRetentionUnchanged': True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'raw/fusion-work/refinement-v10/baseline')
    parser.add_argument('--evidence',type=Path,default=ROOT/'tools/fusion/right-arm-gap-evidence.json')
    parser.add_argument('--review',default='raw/mannequin-fused/hand-gap-review/')
    parser.add_argument('--changed-parts',default='right_upper_arm,right_forearm,right_hand')
    parser.add_argument('--allow-hand-color-repair',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Run checks without replacing export verification files')
    args=parser.parse_args()
    baseline=args.baseline.resolve()
    output=ROOT/'raw/mannequin-fused'
    before,after=load(baseline),load(output)
    report=after[4];by_name={p['id']:p for p in report['parts']}
    allowed_changes=set(args.changed_parts.split(','))
    assert allowed_changes<=set(by_name),'Unknown allowed changed region'
    color_repairs,_=load_color_overrides(report['parameters'].get('colorOverrides',[]),dict(zip(['front','back'],report['sources'])),ROOT)
    arm_ids=[10,11,12]
    assert report['parameters']['rigidParts']['right_hand']=='right_forearm'
    assert not report['parameters'].get('rightWristRegistration')
    assert by_name['right_hand']['sourceToFrontRaw']==by_name['right_forearm']['sourceToFrontRaw']
    assert by_name['right_hand']['rigidWithPart']=='right_forearm'
    assert report['parameters']['cleanupMasks']==before[4]['parameters']['cleanupMasks']
    previous_colors=before[4]['parameters']['colorOverrides']
    if args.allow_hand_color_repair:
        assert allowed_changes=={'right_hand'}
        assert report['parameters']['colorOverrides'][:len(previous_colors)]==previous_colors
    else:assert report['parameters']['colorOverrides']==previous_colors
    hand_only_proof=None
    if allowed_changes=={'right_hand'}:
        assert report['sources']==before[4]['sources'], 'Original scan metadata changed'
        assert report['parameters'].get('colorRepairStrength',1.)==before[4]['parameters'].get('colorRepairStrength',1.), 'Existing color repair strength changed'
        previous_rules=before[4]['parameters'].get('surfaceRestorations',[])
        rules=report['parameters'].get('surfaceRestorations',[])
        assert rules[:len(previous_rules)]==previous_rules, 'Previous surface restoration recipe changed'
        added_rules=rules[len(previous_rules):]
        assert all(rule['part']=='right_hand' for rule in added_rules), 'New restoration targets another part'
        sources=dict(zip(['front','back'],report['sources']))
        restorations,_=load_cleanup_masks([r['selection'] for r in added_rules],sources,ROOT)
        added_colors,_=load_color_overrides(report['parameters']['colorOverrides'][len(previous_colors):],sources,ROOT)
        hand_only_proof=verify_hand_only_changes(before,after,restorations,
                                               {source:item['indices'] for source,item in added_colors.items()},
                                               np.load(PAD_AUDIT/'back-right-rigid-pad.npy'))
    unchanged=[]
    for label,entry in enumerate(before[4]['parts']):
        if entry['id'] in allowed_changes:continue
        assert entry['sourceToFrontRaw']==report['parts'][label]['sourceToFrontRaw']
        a,b=before[1]==label,after[1]==label
        for i in [2,3]:np.testing.assert_array_equal(before[i][a],after[i][b])
        for field in FIELDS:np.testing.assert_array_equal(before[0][field][a],after[0][field][b])
        unchanged.append({'part':entry['id'],'count':int(a.sum()),'allFieldsSourceIdsAndRigidPoseBitExact':True})
    checks=[]
    boundary_pairs=np.load(PAD_AUDIT/'pad-cross-part-pairs.npz')
    boundary_check=None
    for capture,meta in enumerate(report['sources']):
        raw,_,_=read_ply(ROOT/meta['file'])
        for label in arm_ids:
            rows=np.flatnonzero((after[1]==label)&(after[2]==capture))
            original=np.column_stack([raw[k][after[3][rows]] for k in FIELDS])
            tr=report['parts'][label]['sourceToFrontRaw']
            expected=transform_gaussians(original,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']) if capture else original
            expected=transform_gaussians(expected,np.diag([1.,-1.,-1.]),np.zeros(3))
            expected[:,11:],recolored=apply_color_values(expected[:,11:],after[3][rows],color_repairs[['front','back'][capture]],report['parameters'].get('colorRepairStrength',1.))
            actual=np.column_stack([after[0][k][rows] for k in FIELDS])
            np.testing.assert_array_equal(actual[:,:10],expected[:,:10])
            np.testing.assert_array_equal(actual[:,11:],expected[:,11:])
            color_changed=int(np.any(actual[:,11:]!=original[:,11:],axis=1).sum())
            checks.append({'part':report['parts'][label]['id'],'capture':['front','back'][capture],
                           'everyRetainedRowChecked':len(rows),'meansExactOriginalRigidTransform':True,
                           'covarianceExactOriginalRigidTransform':True,
                           'colorsOriginal':color_changed==0,'colorsExactDeclaredSourceRecipe':True,
                           'colorRepairRows':recolored,'actualColorChangedRows':color_changed,
                           'localMeanDeformation':False})
        if capture:
            rows=np.flatnonzero(np.isin(after[1],[11,12])&(after[2]==capture))
            lookup={int(after[3][row]):int(row) for row in rows}
            pairs=[(int(a),int(b)) for a,b in zip(boundary_pairs['forearmSourceIndices'],boundary_pairs['handSourceIndices'])
                   if int(a) in lookup and int(b) in lookup]
            assert pairs,'No original cross-label pad pairs remain'
            a,b=np.array(pairs).T
            native_a=np.column_stack([raw[field][a] for field in FIELDS[:3]])
            native_b=np.column_stack([raw[field][b] for field in FIELDS[:3]])
            actual_a=np.column_stack([after[0][field][[lookup[int(i)] for i in a]] for field in FIELDS[:3]])
            actual_b=np.column_stack([after[0][field][[lookup[int(i)] for i in b]] for field in FIELDS[:3]])
            expected=np.linalg.norm(native_a-native_b,axis=1)*by_name['right_forearm']['sourceToFrontRaw']['scale']
            error=abs(np.linalg.norm(actual_a-actual_b,axis=1)-expected)
            assert error.max()<2e-7
            boundary_check={'retainedCrossBoundaryPairs':len(pairs),'maximumDistanceError':float(error.max()),
                            'meaning':'Original pad distances across the hand/forearm labels are preserved at the shared scale.'}
    verify_pad(output)
    evidence=json.loads(args.evidence.read_text())
    constraint=by_name['right_forearm']['mechanicalConstraint']
    assert constraint['independentSwingDegrees']==0
    assert constraint['independentTranslation']==[0,0,0]
    proof={'passed':True,'revision':report['parameters']['revision'],
           'baseline':str(baseline.relative_to(ROOT)),'gaussianCount':report['fusedCount'],
           'unchangedParts':unchanged,'allRightArmGaussianChecks':checks,
           'rigidAssembly':['right_forearm','right_hand'],'wristJointOrDeformation':False,
           'padBoundaryDistanceCheck':boundary_check,
           'fitEvidence':evidence,'rigidPad':json.loads((PAD_AUDIT/'verification-mannequin-fused.json').read_text()),
           'fullOutputVerification':json.loads((output/'verification.json').read_text()),
           'review':args.review}
    if hand_only_proof is not None:proof['handOnlySourceRepairChecks']=hand_only_proof
    if not args.check_only:
        for name in ['right-arm-gap-verification.json','right-wrist-verification.json','hand-refinement-verification.json']:
            (output/name).write_text(json.dumps(proof,indent=2)+'\n')
    print(json.dumps({'passed':True,'unchangedParts':len(unchanged),'allRightArmGaussiansChecked':sum(c['everyRetainedRowChecked'] for c in checks),'gaussianCount':report['fusedCount']},indent=2))


if __name__=='__main__':main()
