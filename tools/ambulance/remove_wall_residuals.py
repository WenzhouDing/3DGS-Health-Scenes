#!/usr/bin/env python3
"""Remove two explicitly attributed giant splats from the repaired matte wall.

V3 actual-hybrid trace at mattress-top locates the vertical gray float and the
lower triangle. This changes only two original-source opacity values, leaving
all reconstructed material and reference rows byte-identical to V3.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from cleanup import ROOT,read_ply,write_ply,sha256_file
IDS=np.array([311509,3813605],dtype=np.int64)
EVIDENCE=[{'original_index':311509,'camera':'mattress-top','pixel':[43,316],'visible_weight':.3742,'world':[-.9146,-.2824,-.9225],'sigma_max':.08526,'defect':'Gray triangle crossing the matte wall.'},{'original_index':3813605,'camera':'mattress-top','pixel':[147,114],'visible_weight':.0951,'world':[-.4595,.2547,-1.2296],'sigma_max':.16116,'defect':'Black/gray vertical floating smear; large off-panel splat center causes false foreground sorting.'}]

def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,default=ROOT/'raw/ambulance-cleanup/pass6/wall-grain-v3');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);v,_,_=read_ply(a.base/'iphone.ply');r=json.loads((a.base/'report.json').read_text());baseline=Path(r['baseline']);b,_,_=read_ply(baseline/'iphone.ply');out=v.copy();out['opacity'][IDS]=np.log(1e-8/(1-1e-8));write_ply(a.out/'iphone.ply',out)
 for n in ['reference-patches.ply','additions.ply']:shutil.copyfile(a.base/n,a.out/n)
 z=np.load(a.base/'changes.npz');changes={k:z[k] for k in z.files};ix=np.flatnonzero(out['opacity'][:len(b)]!=b['opacity']);changes['iphone_indices']=ix;changes['iphone_opacity_indices']=ix;changes['iphone_opacity_before']=b['opacity'][ix];changes['iphone_opacity_after']=out['opacity'][ix];changes['attributed_residual_original_indices']=IDS;np.savez_compressed(a.out/'changes.npz',**changes);shutil.copyfile(__file__,a.out/'generator.py');shutil.copyfile(a.base/'generator.py',a.out/'material-generator.py');r.update({'status':'Unreviewed V4 two-source residual correction; material identical to reviewed V3','base_candidate':str(a.base),'base_candidate_report_sha256':sha256_file(a.base/'report.json'),'attributed_residuals':EVIDENCE,'attribution_file':'raw/ambulance-cleanup/pass6/wall-diagnostics/v3-residual-pixel-contributors.json','generator_sha256':sha256_file(a.out/'generator.py'),'material_generator_sha256':sha256_file(a.out/'material-generator.py')});r['changes']['iphone']['changed_rows']=len(ix);r['checks']['only_two_source_opacities_differ_from_V3']=all(np.array_equal(out[n],v[n]) for n in v.dtype.names if n!='opacity') and np.array_equal(np.flatnonzero(out['opacity']!=v['opacity']),IDS);r['checks']['additions_byte_identical_to_V3']=sha256_file(a.out/'additions.ply')==sha256_file(a.base/'additions.ply');r['checks']['reference_byte_identical_to_V3']=sha256_file(a.out/'reference-patches.ply')==sha256_file(a.base/'reference-patches.ply');r['review_required']=['wall-material-close','wall-material-grazing','left-wall','mattress-top','cabinet-counter']
 if not all(r['checks'].values()):raise RuntimeError(r['checks'])
 (a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'changed_phone_rows':len(ix),'new_residual_source_ids':IDS.tolist(),'checks':r['checks']},indent=2))


if __name__=='__main__':main()
