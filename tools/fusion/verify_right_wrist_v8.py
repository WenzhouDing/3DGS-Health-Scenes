"""Prove the wrist-pad update preserves the other 15 exported regions."""
import json
import numpy as np
from pipeline import ROOT, FIELDS, read_ply
from inspect_pad_v8 import verify_pad, OUT as PAD_AUDIT


def main():
    baseline=ROOT/'raw/fusion-work/refinement-v8/baseline'
    output=ROOT/'raw/mannequin-fused'
    records=[]
    for folder in [baseline,output]:
        vertices,count,_=read_ply(folder/'mannequin_fused.ply')
        records.append((vertices,np.load(folder/'part-labels.npy'),np.load(folder/'capture-labels.npy'),
                        np.load(folder/'source-vertex-indices.npy'),json.loads((folder/'report.json').read_text())))
    before,after=records;unchanged=[]
    for part,entry in enumerate(before[4]['parts']):
        if entry['id']=='right_hand':continue
        assert entry['sourceToFrontRaw']==after[4]['parts'][part]['sourceToFrontRaw']
        a=before[1]==part;b=after[1]==part
        np.testing.assert_array_equal(before[2][a],after[2][b])
        np.testing.assert_array_equal(before[3][a],after[3][b])
        for field in FIELDS:np.testing.assert_array_equal(before[0][field][a],after[0][field][b])
        unchanged.append({'part':entry['id'],'count':int(a.sum()),'allFieldsSourceIdsAndRigidPoseBitExact':True})
    hand=12;counts={}
    for source in [0,1]:
        a=(before[1]==hand)&(before[2]==source);b=(after[1]==hand)&(after[2]==source)
        common,ia,ib=np.intersect1d(before[3][a],after[3][b],return_indices=True)
        if source==0:
            np.testing.assert_array_equal(before[3][a],after[3][b])
            for field in FIELDS[:10]+FIELDS[11:]:
                np.testing.assert_array_equal(before[0][field][a],after[0][field][b])
        counts[['front','back'][source]]={'before':int(a.sum()),'after':int(b.sum()),
            'addedOriginalRows':int(b.sum()-len(common)),'removedOriginalRows':int(a.sum()-len(common))}
    wrist=after[4]['parts'][hand]['sourceToFrontRaw'];forearm=after[4]['parts'][11]['sourceToFrontRaw']
    for key in ['rotation','translation','scale']:
        np.testing.assert_allclose(wrist[key],forearm[key],atol=1e-10,rtol=0)
    verify_pad(output)
    pad=json.loads((PAD_AUDIT/'verification-mannequin-fused.json').read_text())
    result={'passed':True,'revision':after[4]['parameters']['revision'],'baseline':str(baseline.relative_to(ROOT)),
            'gaussianCount':after[4]['fusedCount'],'unchangedParts':unchanged,'rightHandCounts':counts,
            'rightFrontHandGeometryAndColorsBitExact':True,'rightWristBaseSharesForearmRigidTransform':True,
            'rigidPad':pad,'fullOutputVerification':json.loads((output/'verification.json').read_text()),
            'transition':after[4]['parts'][hand]['wristRegistration'],
            'review':'raw/mannequin-fused/wrist-pad-review/',
            'residual':'Small thumb-index web seam remains; optional stronger thumb warp was rejected'}
    for name in ['right-wrist-verification.json','hand-refinement-verification.json']:
        (output/name).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'unchangedParts':len(unchanged),'rightHandCounts':counts,
                      'gaussianCount':after[4]['fusedCount']},indent=2))


if __name__=='__main__':main()
