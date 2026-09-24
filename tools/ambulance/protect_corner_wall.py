#!/usr/bin/env python3
"""Preserve measured neighbouring white-wall support during corner cleanup.

Restores only this trial's attenuation of visible original wall contributors. The
accepted baseline is immutable. No painting, geometry edits or SH edits occur.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
from transfer_reference_wall_details import positions,attenuate

WALL_GUARD={'camera':'equipment-bag','pixel_box':[[392,46],[416,138]],
       'plane_z_from_xy1':[-.08850324,.01756119,.86515643],'grid':[13,47],
       'cameras':['equipment-bag','right-grazing','ceiling-front'],
       'visible_weight_threshold':.002}


def wall_points():
 c=next(c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id']==WALL_GUARD['camera'])
 eye=np.asarray(c['position']);forward=np.asarray(c['target'])-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);V=np.array([right,-np.cross(right,forward),forward]);f=375/np.tan(np.radians(c['fov'])/2)
 lo,hi=np.asarray(WALL_GUARD['pixel_box']);nu,nv=WALL_GUARD['grid'];pixels=np.array(np.meshgrid(np.linspace(lo[0],hi[0],nu),np.linspace(lo[1],hi[1],nv))).reshape(2,-1).T
 rays=np.einsum('ni,ij->nj',np.c_[(pixels[:,0]-500)/f,(pixels[:,1]-375)/f,np.ones(len(pixels))],V)
 coef=np.asarray(WALL_GUARD['plane_z_from_xy1']);n=np.array([-coef[0],-coef[1],1.]);denominator=np.einsum('ni,i->n',rays,n)
 if np.any(np.abs(denominator)<1e-6):raise ValueError('Wall rays must intersect the measured plane')
 distance=(coef[2]-np.sum(n*eye))/denominator;points=eye+rays*distance[:,None]
 if not np.isfinite(points).all():raise ValueError('Nonfinite projected wall guard')
 return points


def visible(v,p):
 scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float));alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
 points=wall_points()
 cams=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in WALL_GUARD['cameras']]
 all_ids=np.zeros(len(v),bool);reports=[]
 for c in cams:
  eye=np.asarray(c['position']);fwd=np.asarray(c['target'])-eye;fwd/=np.linalg.norm(fwd);right=np.cross(fwd,[0,1,0]);right/=np.linalg.norm(right);V=np.array([right,-np.cross(right,fwd),fwd]);q=np.einsum('ij,nj->ni',V,p-eye);z=q[:,2];safe=np.maximum(z,.02);f=375/np.tan(np.radians(c['fov'])/2);screen=np.c_[500+f*q[:,0]/safe,375+f*q[:,1]/safe];radius=4*f*scales.max(1)/safe+2
  qp=np.einsum('ij,nj->ni',V,points-eye);pixels=np.c_[500+f*qp[:,0]/qp[:,2],375+f*qp[:,1]/qp[:,2]];ok=(qp[:,2]>.02)&(pixels[:,0]>=0)&(pixels[:,0]<1000)&(pixels[:,1]>=0)&(pixels[:,1]<750);pixels=pixels[ok]
  if not len(pixels):continue
  a,b=pixels.min(0),pixels.max(0);m=(z>.02)&(screen[:,0]+radius>a[0])&(screen[:,0]-radius<b[0])&(screen[:,1]+radius>a[1])&(screen[:,1]-radius<b[1]);ids=np.flatnonzero(m);ids=ids[np.argsort(z[ids])]
  Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@W,Q);J=np.zeros((len(ids),2,3));J[:,0,0]=f/z[ids];J[:,1,1]=f/z[ids];J[:,0,2]=-f*np.clip(q[ids,0]/z[ids],-1.3*500/f,1.3*500/f)/z[ids];J[:,1,2]=-f*np.clip(q[ids,1]/z[ids],-1.3*375/f,1.3*375/f)/z[ids]
  B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C);det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2;radius=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-radius)/(1024/1080*750),0,1);picked=np.zeros(len(ids),bool)
  for pixel in pixels:
   d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d);a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)));T=np.r_[1,np.cumprod(1-a[:-1])];picked|=(a*T)>WALL_GUARD['visible_weight_threshold']
  all_ids[ids[picked]]=True;record={'camera':c['id'],'rays':len(pixels),'selected_rows':int(picked.sum())};reports.append(record);print(record,flush=True)
 return np.flatnonzero(all_ids),reports


def scene(source,reference,pm,rm):
 ids=np.flatnonzero(pm<1);p=source.copy();p[ids]=attenuate(source[ids],pm[ids]);ri=np.flatnonzero(rm>0);r=attenuate(reference[ri],rm[ri]);extra=np.empty(len(r),dtype=source.dtype)
 for name in source.dtype.names:extra[name]=r[name]
 return p,r,ri,np.concatenate([p,extra])


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--base',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/front-vertical-v1');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.base.resolve()==args.out.resolve():raise ValueError('Keep V1 evidence immutable')
 report=json.loads((args.base/'report.json').read_text());baseline=Path(report['accepted_baseline']);source,_,_=read_ply(Path(report['iphone']));reference,_,_=read_ply(Path(report['reference']));bs=np.load(baseline/'selections.npz');bpm=np.ones(len(source));brm=np.zeros(len(reference));bpm[bs['iphone_indices']]=bs['iphone_multipliers'];brm[bs['reference_indices']]=bs['reference_multipliers'];ps=np.load(args.base/'iphone-opacity-selection.npz');rs=np.load(args.base/'reference-selection.npz');pm=np.ones(len(source));rm=np.zeros(len(reference));pm[ps['indices']]=ps['multipliers'];rm[rs['indices']]=rs['multipliers'];pp=positions(source);rp=positions(reference)
 _,_,bri,base_scene=scene(source,reference,bpm,brm);bids,btrace=visible(base_scene,np.concatenate([pp,rp[bri]]));pg=bids[bids<len(source)]
 phone_restored=int((pm[pg]<1).sum());pm[pg]=1

 p,r,ri,_=scene(source,reference,np.minimum(bpm,pm),np.maximum(brm,rm));pi=np.flatnonzero(pm<1);newri=np.flatnonzero(rm>0);args.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=pm[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=newri,multipliers=rm[newri]);np.savez_compressed(args.out/'projected-wall-guards.npz',iphone_indices=pg);dest=args.out/'assembled';dest.mkdir(exist_ok=True);write_ply(dest/'iphone.ply',p);write_ply(dest/'reference-patches.ply',r);shutil.copyfile(__file__,args.out/'generator.py')
 checks={'original_nonopacity_fields_exact':all(np.array_equal(p[n],source[n]) for n in source.dtype.names if n!='opacity'),'reference_nonopacity_fields_exact':all(np.array_equal(r[n],reference[n][ri]) for n in reference.dtype.names if n!='opacity'),'guarded_original_change_restored_to_baseline':bool(np.all(pm[pg]==1)),'candidate_reference_file_exact':sha256_file(dest/'reference-patches.ply')==sha256_file(args.base/'assembled/reference-patches.ply')}
 if not all(checks.values()):raise RuntimeError(checks)
 report.update({'status':'Unreviewed projected neighbouring-wall guard correction','base_candidate':str(args.base),'base_candidate_report_sha256':sha256_file(args.base/'report.json'),'iphone_changed_rows':len(pi),'reference_added_rows':len(newri),'wall_guard':WALL_GUARD,'wall_guard_trace':btrace,'wall_guard_restored_phone_rows':phone_restored,'checks':checks,'generator_sha256':sha256_file(args.out/'generator.py'),'review_cameras':WALL_GUARD['cameras']});(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'phone_rows':len(pi),'reference_rows':len(newri),'restored_phone':phone_restored,'checks':checks},indent=2),flush=True)



if __name__=='__main__':main()
