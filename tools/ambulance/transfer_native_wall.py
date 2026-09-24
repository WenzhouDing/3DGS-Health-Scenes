#!/usr/bin/env python3
"""Contributor-aware transfer of the actual native wall, with bounded shape repair.

All captured native positions/DC/SH are retained. Tangent scale repair uses the
reviewed actual-native diagnostic. Original iPhone wall contributors are removed
only within the wall silhouette; no material is synthesized or tiled.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
from regularize_native_wall import RECIPE,smooth
CF=np.asarray(RECIPE['plane']);N=np.array([-CF[0],-CF[1],1.]);UV=np.asarray(RECIPE['uv_bounds']);LABEL=np.asarray(RECIPE['label_protection_uv'])
CAMERA_PATH=ROOT/'raw/ambulance-cleanup/pass6/review-cameras.json'

def alpha(v):return 1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
def logit(a):
 a=np.clip(a,1e-8,1-1e-8);return np.log(a/(1-a))
def pos(v):return np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))
def residual(p):return p[:,2]-np.einsum('ij,j->i',p[:,:2],CF[:2])-CF[2]
def panel_weight(p):
 uv=p[:,:2];edge=np.minimum(uv-UV[0],UV[1]-uv).min(1);bc=RECIPE['bottom_edge_y_from_x1'];edge=np.minimum(edge,uv[:,1]-(bc[0]*uv[:,0]+bc[1]));
 if 'top_edge_y_from_x1' in RECIPE:
  tc=RECIPE['top_edge_y_from_x1'];edge=np.minimum(edge,tc[0]*uv[:,0]+tc[1]-uv[:,1]);
 ld=np.linalg.norm(np.maximum(np.maximum(LABEL[0]-uv,uv-LABEL[1]),0),axis=1);return smooth(edge/RECIPE['feather'])*smooth(ld/RECIPE.get('label_feather',.02))
def rays():
 x=np.linspace(UV[0,0]+.022,UV[1,0]-.022,39);y=np.linspace(UV[0,1]+.022,UV[1,1]-.022,21);uv=np.array(np.meshgrid(x,y)).reshape(2,-1).T;p=np.c_[uv,np.einsum('ij,j->i',uv,CF[:2])+CF[2]];return p[panel_weight(p)>.88]
def cameras():return [c for c in json.loads(CAMERA_PATH.read_text()) if c['id'] in ['wall-material-close','wall-material-grazing']]
def projected_inside(p,every=True):
 allowed=np.ones(len(p),bool) if every else np.zeros(len(p),bool)
 for camera in cameras():
  eye=np.asarray(camera['position']);direction=p-eye;den=np.einsum('ni,i->n',direction,N);t=(CF[2]-np.dot(eye,N))/np.where(abs(den)>1e-12,den,1e-12);q=eye+direction*t[:,None];hit=(t>0)&(panel_weight(q)>.85)
  if every:allowed&=hit
  else:allowed|=hit
 return allowed

def visible(v,eligible,tag):
 """Exact review-renderer alpha tracing at fixed physical wall sample rays."""
 p=pos(v);a0=alpha(v);s=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float));world=rays();selected=np.zeros(len(v),bool);stats=[]
 for camera in cameras():
  eye=np.asarray(camera['position']);fwd=np.asarray(camera['target'])-eye;fwd/=np.linalg.norm(fwd);right=np.cross(fwd,[0,1,0]);right/=np.linalg.norm(right);down=-np.cross(right,fwd);V=np.array([right,down,fwd]);cp=np.einsum('ij,nj->ni',V,p-eye);z=cp[:,2];safe=np.maximum(z,.02);f=375/np.tan(np.radians(camera['fov'])/2);screen=np.c_[500+f*cp[:,0]/safe,375+f*cp[:,1]/safe];q=np.einsum('ij,nj->ni',V,world-eye);pixels=np.c_[500+f*q[:,0]/q[:,2],375+f*q[:,1]/q[:,2]];ok=(q[:,2]>.02)&(pixels[:,0]>4)&(pixels[:,0]<996)&(pixels[:,1]>4)&(pixels[:,1]<746);pixels=pixels[ok];low,high=pixels.min(0),pixels.max(0);radius=4*f*s.max(1)/safe+2;box=(z>.02)&(a0>1.1e-8)&(screen[:,0]+radius>low[0])&(screen[:,0]-radius<high[0])&(screen[:,1]+radius>low[1])&(screen[:,1]-radius<high[1]);ids=np.flatnonzero(box);ids=ids[np.argsort(z[ids])];Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@W,Q);J=np.zeros((len(ids),2,3));J[:,0,0]=f/z[ids];J[:,1,1]=f/z[ids];J[:,0,2]=-f*np.clip(cp[ids,0]/z[ids],-1.3*500/f,1.3*500/f)/z[ids];J[:,1,2]=-f*np.clip(cp[ids,1]/z[ids],-1.3*375/f,1.3*375/f)/z[ids];B=np.einsum('nij,njk->nik',J,Q)*s[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inv=np.linalg.inv(C);det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2;rad=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-rad)/(1024/1080*750),0,1);picked=np.zeros(len(ids),bool);coverage=[];mass=[]
  for pixel in pixels:
   d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inv,d);a=np.minimum(.99,a0[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)));trans=np.r_[1,np.cumprod(1-a[:-1])];weight=a*trans;picked|=(weight>.003)&eligible[ids];coverage.append(float(weight.sum()));mass.append(float(weight[eligible[ids]].sum()))
  selected[ids[picked]]=True;record={'camera':camera['id'],'rays':len(pixels),'candidate_scene_rows':len(ids),'selected_contributors':int(picked.sum()),'alpha_q05_q50':np.quantile(coverage,[.05,.5]).tolist(),'eligible_visible_mass_q50_q95':np.quantile(mass,[.5,.95]).tolist()};stats.append(record);print(tag,record,flush=True)
 return selected,stats

def main():
 global RECIPE,CF,N,UV,LABEL
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--native',type=Path,default=ROOT/'raw/ambulance-cleanup/pass6/native-clean-v2/native.ply');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise ValueError('New output directory only')
 RECIPE=json.loads((a.native.parent/'report.json').read_text())['recipe'];CF=np.asarray(RECIPE['plane']);N=np.array([-CF[0],-CF[1],1.]);UV=np.asarray(RECIPE['uv_bounds']);LABEL=np.asarray(RECIPE['label_protection_uv'])
 base,_,_=read_ply(a.baseline/'iphone.ply');ref,_,_=read_ply(a.baseline/'reference-patches.ply');native,_,_=read_ply(a.native);source,_,_=read_ply(ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply');mapping=np.load(a.baseline/'selections.npz');refids=mapping['reference_indices'];lookup=np.full(len(native),-1,np.int64);lookup[refids]=np.arange(len(refids));rp=pos(native);rd=residual(rp);rw=panel_weight(rp)*smooth((rd+.06)/.012)*smooth((.035-rd)/.010);rw[(abs(rp)>8).any(1)]=0
 deep=(rd<-.025)&(rd> -5)&projected_inside(rp)&(abs(rp[:,0])<7)&(abs(rp[:,1])<5);deep_ids,donor_trace=visible(native,deep,'native');rw[deep_ids]=1;selected=np.flatnonzero(rw>1e-6);existing=lookup[selected]>=0;refout=ref.copy();changed_existing=[]
 # Existing accepted reference alpha is never decreased. Every selected source
 # is deduplicated against its baseline identity before appending.
 foriginal=alpha(native[selected])*rw[selected];patch=native[selected].copy();patch['opacity']=logit(foriginal);old=lookup[selected[existing]];patch_existing=patch[existing].copy();patch_existing['opacity']=logit(np.maximum(alpha(ref[old]),alpha(patch_existing)));refout[old]=patch_existing;extra=patch[~existing];fullref=np.concatenate([refout,extra]);fullids=np.r_[refids,selected[~existing]]
 p=pos(base);d=residual(p);pw=panel_weight(p)*smooth((d+.06)/.012)*smooth((.06-d)/.015);phone=base.copy();initial=np.flatnonzero(pw>1e-6);phone['opacity'][initial]=logit(alpha(base[initial])*(1-pw[initial]));original_eligible=(d> -5)&(d<.23)&(projected_inside(p)|((d<-.025)&projected_inside(p,every=False)))&(abs(p[:,0])<7)&(abs(p[:,1])<5)
 common=np.empty(len(fullref),dtype=phone.dtype)
 for f in phone.dtype.names:common[f]=fullref[f]
 traces=[];peeled=[]
 # Two exact colored background contributors, independently traced against native.
 exact_ids=np.array([1915340,4372762],dtype=np.int64);phone['opacity'][exact_ids]=logit(np.full(len(exact_ids),1e-8))
 for iteration in range(8):
  scene=np.concatenate([phone,common]);eligible=np.r_[original_eligible,np.zeros(len(common),bool)];found,trace=visible(scene,eligible,'hybrid-'+str(iteration));ids=np.flatnonzero(found[:len(phone)]&(alpha(phone)>1.1e-8));traces.append({'iteration':iteration,'removed_rows':len(ids),'views':trace});print('Peel',iteration,len(ids),flush=True)
  if not len(ids):break
  phone['opacity'][ids]=logit(np.full(len(ids),1e-8));peeled.extend(ids.tolist())
 a.out.mkdir(parents=True);write_ply(a.out/'iphone.ply',phone);write_ply(a.out/'reference-patches.ply',fullref);changes={'native_reference_selected_ids':selected,'native_reference_added_ids':selected[~existing],'reference_source_ids':fullids,'native_deep_support_source_ids':np.flatnonzero(deep_ids),'phone_initial_ids':initial,'phone_exact_attributed_ids':exact_ids,'phone_hybrid_peeled_ids':np.unique(peeled).astype(np.int64)};summary={}
 for label,b,v in [('iphone',base,phone),('reference',ref,fullref[:len(ref)])]:
  union=np.zeros(len(b),bool);fields={}
  for f in b.dtype.names:
   ids=np.flatnonzero(b[f]!=v[f]);union[ids]=True
   if len(ids):changes[label+'_'+f+'_indices']=ids;changes[label+'_'+f+'_before']=b[f][ids];changes[label+'_'+f+'_after']=v[f][ids];fields[f]=len(ids)
  ids=np.flatnonzero(union);changes[label+'_indices']=ids;summary[label]={'changed_rows':len(ids),'fields':fields,'unselected_exact':bool(np.array_equal(b[~union],v[~union]))}
 np.savez_compressed(a.out/'changes.npz',**changes);shutil.copyfile(__file__,a.out/'generator.py');captured_fields=[f for f in source.dtype.names if f not in ['opacity','scale_0','scale_1','scale_2']];check={'phone_nonopacity_exact':all(np.array_equal(phone[f],base[f]) for f in phone.dtype.names if f!='opacity'),'no_phone_opacity_increases':bool(np.all(phone['opacity']<=base['opacity']+2e-6)),'selected_native_capture_positions_colors_SH_exact':all(np.array_equal(native[f][selected],source[f][selected]) for f in captured_fields),'reference_source_ids_unique':len(np.unique(fullids))==len(fullids),'all_finite':all(np.isfinite(v[f]).all() for v in [phone,fullref] for f in v.dtype.names)}
 if not all(check.values()):raise RuntimeError(check)
 report={'status':'Unreviewed actual-native captured wall transfer with area-preserving covariance repair','baseline':str(a.baseline.resolve()),'baseline_hashes':{'iphone':sha256_file(a.baseline/'iphone.ply'),'reference':sha256_file(a.baseline/'reference-patches.ply')},'native_source':str(a.native.resolve()),'native_sha256':sha256_file(a.native),'recipe':RECIPE,'counts':{'selected_actual_reference_rows':len(selected),'new_actual_reference_rows':len(extra),'deep_visible_reference_rows':int(deep_ids.sum()),'initial_phone_mask_rows':len(initial),'hybrid_original_rows_peeled':len(set(peeled)),'exact_attributed_original_rows_removed':exact_ids.tolist()},'changes':summary,'donor_trace':donor_trace,'hybrid_trace':traces,'checks':check,'added_reference_rows':len(extra),'added_phone_rows':0,'generator_sha256':sha256_file(a.out/'generator.py'),'changes_sha256':sha256_file(a.out/'changes.npz'),'review_required':['wall-material-close','wall-material-grazing','left-wall','mattress-top','cabinet-counter'],'limitations':['Actual registered native support can be nonphysical in depth; selected by visible contribution in two wall views.','Source color and spatial grain are retained. Only native tangent shape and both captures opacity are edited.','No synthetic grain, new colors, translated material copies or planar backing.'],'remote_publish':False};(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
