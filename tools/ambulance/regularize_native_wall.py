#!/usr/bin/env python3
"""Diagnostic actual-capture wall covariance repair; no material replacement."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
from regularize_wall_material import RECIPE as ORIGINAL_RECIPE,smooth
RECIPE={**ORIGINAL_RECIPE,'uv_bounds':[[-1.53,-.52],[-.345,.135]],'bottom_edge_y_from_x1':[-.058,-.477], 'label_protection_uv':[[-1.56,-.52],[-1.315,.015]],'label_feather':.02,'exclusions':'True sloped lower metal lip, original Stryker board, glass/rail above y=.135, foreground and unrelated depth layers.'}
def weights(p):
 uv=p[:,:2];lo,hi=np.asarray(RECIPE['uv_bounds']);e=np.minimum(uv-lo,hi-uv).min(1);bc=RECIPE['bottom_edge_y_from_x1'];e=np.minimum(e,uv[:,1]-(bc[0]*uv[:,0]+bc[1]));
 if 'top_edge_y_from_x1' in RECIPE:
  tc=RECIPE['top_edge_y_from_x1'];e=np.minimum(e,tc[0]*uv[:,0]+tc[1]-uv[:,1]);
 llo,lhi=np.asarray(RECIPE['label_protection_uv']);ld=np.linalg.norm(np.maximum(np.maximum(llo-uv,uv-lhi),0),axis=1);cf=np.asarray(RECIPE['plane']);d=p[:,2]-np.einsum('ij,j->i',uv,cf[:2])-cf[2];a,b=RECIPE['depth_bounds'];return smooth(e/RECIPE['feather'])*smooth(ld/RECIPE['label_feather'])*smooth(np.minimum(d-a,b-d)/RECIPE['depth_feather'])
def apply(v):
 p=np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float));weight=weights(p);alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));ids=np.flatnonzero((weight>0)&(alpha>RECIPE['live_alpha_min']));sc=columns(v[ids],['scale_0','scale_1','scale_2']).astype(float);order=np.argsort(sc,axis=1);sr=np.take_along_axis(sc,order,axis=1);ratio=np.exp(sr[:,2]-sr[:,1]);Q=np.einsum('ij,njk->nik',W,Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix());normal=Q[np.arange(len(ids)),:,order[:,0]];cf=np.asarray(RECIPE['plane']);n=np.array([-cf[0],-cf[1],1]);n/=np.linalg.norm(n);align=abs(np.einsum('ni,i->n',normal,n));good=(ratio>RECIPE['max_tangent_aspect'])&(np.exp(sr[:,2])>RECIPE['long_sigma_min'])&(align>RECIPE['normal_alignment_min']);ids=ids[good];sc=sc[good];order=order[good];sr=sr[good];delta=.5*(sr[:,2]-sr[:,1]-np.log(RECIPE['max_tangent_aspect']))*weight[ids];sc[np.arange(len(ids)),order[:,2]]-=delta;sc[np.arange(len(ids)),order[:,1]]+=delta;out=v.copy()
 for i,f in enumerate(['scale_0','scale_1','scale_2']):out[f][ids]=sc[:,i]
 after=np.sort(sc,axis=1);checks={'all_nonscale_fields_exact':all(np.array_equal(v[f],out[f]) for f in v.dtype.names if not f.startswith('scale_')),'all_unselected_exact':bool(np.array_equal(v[weight==0],out[weight==0])),'tangent_log_area_conserved':bool(np.allclose(after[:,1:].sum(1),sr[:,1:].sum(1),atol=1e-6)),'all_finite':all(np.isfinite(out[f]).all() for f in out.dtype.names)}
 if not all(checks.values()):raise RuntimeError(checks)
 return out,ids,{'selected_rows':len(ids),'checks':checks,'before_aspect_q10_q50_q90':np.quantile(ratio[good],[.1,.5,.9]).tolist(),'after_aspect_q10_q50_q90':np.quantile(np.exp(after[:,2]-after[:,1]),[.1,.5,.9]).tolist()}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--physical-bounds',action='store_true');a=ap.parse_args()
 if a.physical_bounds:RECIPE.update({'uv_bounds':[[-1.53,-.52],[-.345,.175]],'bottom_edge_y_from_x1':[-.032,-.480],'top_edge_y_from_x1':[.009,.1575],'label_protection_uv':[[-1.56,-.52],[-1.343,.015]],'feather':.012,'label_feather':.008,'physical_boundary_evidence':'Wall-material-close metal lip pixels(0,747),(500,730),(824,714) and upper rail(50,224),(500,229),(850,232), intersected with measured wall plane; inset .004 scene units, original frame protected.'})
 if a.out.exists():raise ValueError('New diagnostic output only')
 v,_,_=read_ply(a.source);out,ids,result=apply(v);a.out.mkdir(parents=True);write_ply(a.out/'native.ply',out);np.savez_compressed(a.out/'changes.npz',reference_original_indices=ids,scale_before=columns(v[ids],['scale_0','scale_1','scale_2']),scale_after=columns(out[ids],['scale_0','scale_1','scale_2']));shutil.copyfile(__file__,a.out/'generator.py');r={'status':'Unreviewed full native wall covariance diagnostic','source':str(a.source),'source_sha256':sha256_file(a.source),'output_sha256':sha256_file(a.out/'native.ply'),'recipe':RECIPE,'result':result,'added_rows':0,'removed_rows':0,'generator_sha256':sha256_file(a.out/'generator.py'),'note':'Only source Gaussian tangent scales change, preserving area and all captured position/color/SH/opacity. No iPhone hybrid yet.'};(a.out/'report.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
