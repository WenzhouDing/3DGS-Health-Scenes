#!/usr/bin/env python3
"""Restore captured iPhone wall texture and remove only coarse off-surface support.

No colors, texture samples, copied material, centers or fine-grain shapes are
created or moved. Undo the old lower-cabinet reference-material component while
preserving every other accepted component. Use local fine iPhone points to
locate the observed skin; suppress coarse detached layers and constrain only
the normal covariance of coarse near-surface layers. Actual grain/lighting stays.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
CF=np.array([-.08237276,.02981509,-.92014449]);UV=np.array([[-1.405,-.52],[-.345,.135]]);LABEL=np.array([[-1.50,-.52],[-1.315,.015]])
R={'uv_bounds':UV.tolist(),'bottom_edge_y_from_x1':[-.058,-.477],'label_protection_uv':LABEL.tolist(),'feather':.03,'source_coarse_maxsigma_min':.006,'local_skin_neighbors':64,'local_skin_max_neighbor_distance':.025,'local_offset_tolerance_min':.008,'local_offset_MAD_factor':4,'coarse_normal_sigma_limit':.002,'coarse_normal_edit_min_sigma':.004,'clear_global_depth_band':.16,'note':'Reference plane only finds the vicinity; local fine captured phone depth determines detached-layer selection.'}

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def pos(v):return np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))
def alpha(v):return 1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
def logit(a):
 a=np.clip(a,1e-8,1-1e-8);return np.log(a/(1-a))
def regionweight(p):
 uv=p[:,:2];e=np.minimum(uv-UV[0],UV[1]-uv).min(1);e=np.minimum(e,uv[:,1]-(-.058*uv[:,0]-.477));ld=np.linalg.norm(np.maximum(np.maximum(LABEL[0]-uv,uv-LABEL[1]),0),axis=1);return smooth(e/R['feather'])*smooth(ld/.018)
def dglobal(p):return p[:,2]-np.einsum('ij,j->i',p[:,:2],CF[:2])-CF[2]


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()==a.baseline.resolve():raise ValueError('Baseline immutable')
 prior=json.loads((a.baseline/'report.json').read_text());source,_,_=read_ply(Path(prior['iphone']));base,_,_=read_ply(a.baseline/'iphone.ply');ref,_,_=read_ply(a.baseline/'reference-patches.ply');mapping=np.load(a.baseline/'selections.npz');pm=np.ones(len(source));rm=np.zeros(999410);omitted=[]
 for component in prior['components']:
  folder=Path(component['directory'])
  if str(folder).endswith('/pass3/trials/walls-v3'):
   omitted.append(str(folder));continue
  z=np.load(folder/'iphone-opacity-selection.npz');np.minimum.at(pm,z['indices'],z['multipliers']);z=np.load(folder/'reference-selection.npz');np.maximum.at(rm,z['indices'],z['multipliers'])
 if len(omitted)!=1:raise ValueError('Expected exactly the legacy lower-cabinet wall donor component')
 bpm=np.ones(len(source));bpm[mapping['iphone_indices']]=mapping['iphone_multipliers'];brm=np.zeros(len(rm));brm[mapping['reference_indices']]=mapping['reference_multipliers'];phone=base.copy();refout=ref.copy();restored=np.flatnonzero(pm>bpm);phone['opacity'][restored]=logit(alpha(source[restored])*pm[restored]);rid=mapping['reference_indices'];drop=np.flatnonzero(rm[rid]<brm[rid]);refout['opacity'][drop]=logit(alpha(ref[drop])*(rm[rid[drop]]/brm[rid[drop]]))
 p=pos(source);w=regionweight(p);dg=dglobal(p);s=np.exp(columns(source,['scale_0','scale_1','scale_2']).astype(float));rgb=.5+.28209479177387814*columns(source,['f_dc_0','f_dc_1','f_dc_2']);aa=alpha(source);maxs=s.max(1)
 anchor=(w>0)&(abs(dg)<.025)&(maxs<.0035)&(aa>.35)&(np.ptp(rgb,axis=1)<.2);ai=np.flatnonzero(anchor);tree=cKDTree(p[ai,:2]);candidate=(w>0)&(abs(dg)<R['clear_global_depth_band'])&(maxs>R['source_coarse_maxsigma_min'])&(aa>.01);ci=np.flatnonzero(candidate);dist,ni=tree.query(p[ci,:2],k=R['local_skin_neighbors']);neighbor_d=dg[ai[ni]];local_offset=np.median(neighbor_d,axis=1);mad=1.4826*np.median(abs(neighbor_d-local_offset[:,None]),axis=1);good=(dist[:,-1]<R['local_skin_max_neighbor_distance'])&(mad<.008);dl=dg[ci]-local_offset;tol=np.maximum(R['local_offset_tolerance_min'],R['local_offset_MAD_factor']*mad);bad=good&(abs(dl)>tol);removed=ci[bad];phone['opacity'][removed]=logit(alpha(phone[removed])*(1-w[removed]))
 # Near-surface coarse splats keep their center, tangent covariance, color and
 # opacity. Only protruding normal support is reduced, with covariance feather.
 near=ci[good&~bad];Q=Rotation.from_quat(columns(source[near],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',W,Q);B=Q*s[near,None,:];C=np.einsum('nik,njk->nij',B,B);n=np.array([-CF[0],-CF[1],1.]);n/=np.linalg.norm(n);sn=np.sqrt(np.einsum('i,nij,j->n',n,C,n));edit=sn>R['coarse_normal_edit_min_sigma'];ei=near[edit];C=C[edit];sn=sn[edit];P=np.eye(3)-np.outer(n,n);thin=np.minimum(sn,R['coarse_normal_sigma_limit']);target=np.einsum('ij,njk,lk->nil',P,C,P)+thin[:,None,None]**2*np.outer(n,n);blend=C*(1-w[ei,None,None])+target*w[ei,None,None];ev,axes=np.linalg.eigh(blend);axes[np.linalg.det(axes)<0,:,0]*=-1;q=Rotation.from_matrix(np.einsum('ij,njk->nik',W.T,axes)).as_quat();logs=.5*np.log(np.maximum(ev,1e-20))
 for i,f in enumerate(['rot_1','rot_2','rot_3','rot_0']):phone[f][ei]=q[:,i]
 for i,f in enumerate(['scale_0','scale_1','scale_2']):phone[f][ei]=logs[:,i]
 a.out.mkdir(parents=True,exist_ok=True);write_ply(a.out/'iphone.ply',phone);write_ply(a.out/'reference-patches.ply',refout);changes={'restored_captured_phone_indices':restored,'coarse_detached_phone_indices':removed,'coarse_normal_covariance_phone_indices':ei,'fine_anchor_original_indices':ai};summary={}
 for label,b,v in [('iphone',base,phone),('reference',ref,refout)]:
  union=np.zeros(len(b),bool);fields={}
  for f in b.dtype.names:
   ix=np.flatnonzero(b[f]!=v[f]);union[ix]=True
   if len(ix):changes[label+'_'+f+'_indices']=ix;changes[label+'_'+f+'_before']=b[f][ix];changes[label+'_'+f+'_after']=v[f][ix];fields[f]=len(ix)
  ix=np.flatnonzero(union);changes[label+'_indices']=ix;summary[label]={'changed_rows':len(ix),'fields':fields,'unselected_rows_exact':bool(np.array_equal(v[~union],b[~union]))}
 np.savez_compressed(a.out/'changes.npz',**changes);shutil.copyfile(__file__,a.out/'generator.py');checks={'source_positions_exact':all(np.array_equal(phone[f],base[f]) for f in ['x','y','z']),'source_DC_exact':all(np.array_equal(phone[f],base[f]) for f in ['f_dc_0','f_dc_1','f_dc_2']),'reference_nonopacity_exact':all(np.array_equal(refout[f],ref[f]) for f in ref.dtype.names if f!='opacity'),'fine_shape_unchanged':all(np.array_equal(phone[f][maxs<=R['source_coarse_maxsigma_min']],base[f][maxs<=R['source_coarse_maxsigma_min']]) for f in ['scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']),'all_finite':all(np.isfinite(v[f]).all() for v in [phone,refout] for f in v.dtype.names)}
 if not all(checks.values()):raise RuntimeError(checks)
 report={'status':'Unreviewed captured-texture-only wall cleanup','baseline':str(a.baseline),'baseline_hashes':{'iphone':sha256_file(a.baseline/'iphone.ply'),'reference':sha256_file(a.baseline/'reference-patches.ply')},'original_phone':prior['iphone'],'original_phone_sha256':prior['iphone_sha256'],'omitted_legacy_material_component':omitted,'recipe':R,'counts':{'original_phone_rows_restored':len(restored),'reference_rows_legacy_weight_reduced':len(drop),'fine_original_skin_anchors':len(ai),'coarse_candidates':len(ci),'coarse_candidates_rejected_uncertain':int((~good).sum()),'coarse_offsurface_rows_attenuated':len(removed),'coarse_normal_covariances_constrained':len(ei)},'changes':summary,'added_rows':0,'source_field_policy':'Captured centers,DC and fine-grain covariance remain byte-exact; source opacity restored or coarse haze suppressed; coarse normal covariance only may change. No copied/densified texture, no synthetic backing, no lighting normalization.','checks':checks,'generator_sha256':sha256_file(a.out/'generator.py'),'review_required':['wall-material-close','wall-material-grazing','left-wall','mattress-top'],'limitations':['Natural captured tone variation remains; this does not aim for a perfectly uniform synthetic finish.','Coarse-source haze removal can reveal genuinely missing support; GPU review must reject holes or hard boundaries.'],'remote_publish':False};(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'counts':report['counts'],'changes':summary,'checks':checks},indent=2),flush=True)


if __name__=='__main__':main()
