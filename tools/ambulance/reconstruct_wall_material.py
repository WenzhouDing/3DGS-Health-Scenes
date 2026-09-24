#!/usr/bin/env python3
"""Uniform fine laminate sampled from a clean captured border, fitted to the wall.

Manual material reconstruction authorized by the user's homogeneous fine
salt-and-pepper request. Grain density is uniform; color statistics are sampled
from captured clean material. This is not claimed as recovered photo truth.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
CF=np.array([-.08237276,.02981509,-.92014449]);UV=np.array([[-1.405,-.52],[-.36,.135]]);LABEL=np.array([[-1.50,-.52],[-1.315,.015]])
R={'uv_bounds':UV.tolist(),'plane_z_from_xy1':CF.tolist(),'bottom_edge_y_from_x1':[-.058,-.477],'bottom_note':'Conservative .01 inset above photographed white-panel/metal lower edge; prevents V2 backing rim.','feather':.024,'label_box':LABEL.tolist(),'label_feather':.018,'clear_depth_bounds':[-.065,.065],'grain_spacing':.00125,'grain_sigma_major_bounds':[.00040,.00072],'grain_sigma_normal':.00007,'grain_opacity':.86,'grain_depth_offset':.0007,'backing_spacing':.007,'backing_sigma':.0065,'backing_opacity':.985,'backing_depth_offset':-.002,'border_palette_luma_contrast_factor':.65,'seed':230923,'synthesis':'Deterministic uniform jittered sampling of observed clean-border palette; new matte material, not capture-exact detail.'}

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def pos(v):return np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))
def alpha(v):return 1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
def logit(a):
 a=np.clip(a,1e-8,1-1e-8);return np.log(a/(1-a))
def edge_distance(xy):
 edge=np.minimum(xy-UV[0],UV[1]-xy).min(1);bottom=xy[:,1]-(R['bottom_edge_y_from_x1'][0]*xy[:,0]+R['bottom_edge_y_from_x1'][1]);label=np.linalg.norm(np.maximum(np.maximum(LABEL[0]-xy,xy-LABEL[1]),0),axis=1);return np.minimum(edge,bottom),label

def weight(xy):
 e,l=edge_distance(xy);return smooth(e/R['feather'])*smooth(l/R['label_feather'])
def resid(p):return p[:,2]-np.einsum('ij,j->i',p[:,:2],CF[:2])-CF[2]
def setc(v,names,data):
 for i,n in enumerate(names):v[n]=data[:,i]
def layer(template,xy,colors,major,minor,angles,opacity,offset):
 w=weight(xy);keep=w>0;xy=xy[keep];colors=colors[keep];major=np.asarray(major)[keep];minor=np.asarray(minor)[keep];angles=np.asarray(angles)[keep];opacity=np.asarray(opacity)[keep];w=w[keep];e,l=edge_distance(xy);maxsig=np.maximum(.00003,np.minimum(e,l)/3.2);major=np.minimum(major,maxsig);minor=np.minimum(minor,maxsig)
 p=np.c_[xy,np.einsum('ij,j->i',xy,CF[:2])+CF[2]+offset];out=np.zeros(len(p),dtype=template.dtype);setc(out,['x','y','z'],np.einsum('ij,nj->ni',W.T,p));n=np.array([-CF[0],-CF[1],1.]);n/=np.linalg.norm(n);t=np.array([1,0,CF[0]]);t/=np.linalg.norm(t);s=np.cross(n,t);majoraxis=np.cos(angles)[:,None]*t+np.sin(angles)[:,None]*s;minoraxis=np.cross(np.broadcast_to(n,majoraxis.shape),majoraxis);basis=np.stack([np.broadcast_to(n,majoraxis.shape),majoraxis,minoraxis],axis=2);qb=Rotation.from_matrix(np.einsum('ij,njk->nik',W.T,basis)).as_quat()[:,[3,0,1,2]];setc(out,['rot_0','rot_1','rot_2','rot_3'],qb);setc(out,['scale_0','scale_1','scale_2'],np.log(np.c_[np.full(len(out),R['grain_sigma_normal']),major,minor]));setc(out,['f_dc_0','f_dc_1','f_dc_2'],(colors-.5)/.28209479177387814);out['opacity']=logit(opacity*w);return out

def grid(step,rng,jitter):
 x=np.arange(UV[0,0]+step/2,UV[1,0],step);y=np.arange(UV[0,1]+step/2,UV[1,1],step);xy=np.stack(np.meshgrid(x,y),axis=-1).reshape(-1,2);xy+=rng.uniform(-jitter,jitter,xy.shape)*step;return xy


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()==a.baseline.resolve():raise ValueError('Baseline immutable')
 a.out.mkdir(parents=True,exist_ok=True);phone,_,_=read_ply(a.baseline/'iphone.ply');ref,_,_=read_ply(a.baseline/'reference-patches.ply');src=ROOT/'raw/ambulance_exp11_boot_sharp.ply';orig,_,_=read_ply(src);p=pos(orig);ss=np.sort(np.exp(columns(orig,['scale_0','scale_1','scale_2']).astype(float)),axis=1);rgb=.5+.28209479177387814*columns(orig,['f_dc_0','f_dc_1','f_dc_2']).astype(float);oa=alpha(orig)
 sample=(p[:,0]>-1.30)&(p[:,0]<-.42)&(p[:,1]>.093)&(p[:,1]<.135)&(abs(resid(p))<.007)&(ss[:,2]<.0023)&(oa>.10)&(np.ptp(rgb,axis=1)<.15);sid=np.flatnonzero(sample);pal=rgb[sid];prob=oa[sid]*ss[sid,1]*ss[sid,2];prob/=prob.sum();mean=np.average(pal,axis=0,weights=prob);luma=pal.mean(1);mu=mean.mean();chroma=mean-mu
 rng=np.random.default_rng(R['seed']);xy=grid(R['grain_spacing'],rng,.38);n=len(xy);pi=rng.choice(len(sid),size=n,p=prob);brightness=np.clip(mu+R['border_palette_luma_contrast_factor']*(luma[pi]-mu),.30,.985);shade=.02*np.clip((.135-xy[:,1])/.64,0,1);colors=np.clip((brightness-shade)[:,None]+chroma[None,:],.02,.995);maj=np.clip(np.sqrt(ss[sid[pi],1]*ss[sid[pi],2]),*R['grain_sigma_major_bounds']);mi=maj/rng.uniform(1,1.6,n);grain=layer(phone,xy,colors,maj,mi,rng.uniform(-np.pi,np.pi,n),np.full(n,R['grain_opacity']),R['grain_depth_offset'])
 bx=grid(R['backing_spacing'],rng,0);nb=len(bx);bc=mean[None,:]-.02*np.clip((.135-bx[:,1])/.64,0,1)[:,None];back=layer(phone,bx,bc,np.full(nb,R['backing_sigma']),np.full(nb,R['backing_sigma']),np.zeros(nb),np.full(nb,R['backing_opacity']),R['backing_depth_offset']);extra=np.concatenate([back,grain]);changes={'palette_original_indices':sid};reports={};outarrays={}
 for name,v in [('iphone',phone),('reference',ref)]:
  pp=pos(v);w=weight(pp[:,:2]);eligible=(w>0)&(abs(resid(pp))<.065);out=v.copy();out['opacity'][eligible]=logit(alpha(v)[eligible]*(1-w[eligible]));ix=np.flatnonzero(out['opacity']!=v['opacity']);changes[name+'_indices']=ix;changes[name+'_opacity_indices']=ix;changes[name+'_opacity_before']=v['opacity'][ix];changes[name+'_opacity_after']=out['opacity'][ix];reports[name]={'changed_rows':len(ix),'changed_fields':['opacity'],'outside_selected_rows_exact':bool(np.array_equal(out[np.setdiff1d(np.arange(len(v)),ix)],v[np.setdiff1d(np.arange(len(v)),ix)])),'nonopacity_fields_exact':all(np.array_equal(out[f],v[f]) for f in v.dtype.names if f!='opacity'),'max_center_displacement':0,'baseline_rows':len(v),'baseline_sha256':sha256_file(a.baseline/('iphone.ply' if name=='iphone' else 'reference-patches.ply'))};outarrays[name]=out
 write_ply(a.out/'iphone.ply',np.concatenate([outarrays['iphone'],extra]));write_ply(a.out/'reference-patches.ply',outarrays['reference']);write_ply(a.out/'additions.ply',extra);np.savez_compressed(a.out/'changes.npz',**changes);shutil.copyfile(__file__,a.out/'generator.py')
 checks={'all_added_fields_finite':all(np.isfinite(extra[f]).all() for f in extra.dtype.names),'added_quaternions_unit':bool(np.max(abs(np.linalg.norm(columns(extra,['rot_0','rot_1','rot_2','rot_3']),axis=1)-1))<1e-5),'baseline_prefix_preserved_except_reported_opacity':all(q['nonopacity_fields_exact'] and q['outside_selected_rows_exact'] for q in reports.values())}
 report={'status':'Unreviewed uniformly resampled observed laminate material','baseline':str(a.baseline),'baseline_hashes':{'iphone':reports['iphone']['baseline_sha256'],'reference':reports['reference']['baseline_sha256']},'recipe':R,'clean_border_source':str(src),'clean_border_source_sha256':sha256_file(src),'palette_count':len(sid),'palette_rgb_mean':mean.tolist(),'palette_luma_std':float(np.sqrt(np.average((luma-mu)**2,weights=prob))),'grain_output_luma_std':float(np.std(brightness)),'changes':reports,'added_rows':len(extra),'backing_rows':len(back),'grain_rows':len(grain),'additions':{'file':'additions.ply','trailing_in':'iphone.ply','start_index':len(phone),'purpose':'Thin opaque matte support plus uniform compact grains using observed clean-border material statistics.','provenance':'Manually reconstructed material; not original captured row identity.'},'index_mapping':'All baseline prefix rows retain index. Changed originals/references opacity only; additions trailing.', 'checks':checks,'review_required':['wall-material-close','wall-material-grazing','left-wall','mattress-top'],'limitations':['Reconstructed fine grain is intentionally spatially resampled; it does not preserve exact original stochastic pattern.','User requested uniform fine salt/pepper, so prior coarse reflectance clouds and donor view-dependent ghosts are removed within the bounded matte panel.','Panel lower boundary and label exclusions require GPU controls.'],'generator_sha256':sha256_file(a.out/'generator.py'),'remote_publish':False}
 if not all(checks.values()):raise RuntimeError(checks)
 (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
