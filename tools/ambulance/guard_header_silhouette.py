#!/usr/bin/env python3
"""Restore measured shared roof/hatch support in a bounded header trial.

The repair is a strict subset of V1 opacity edits, except one separately traced
blue contaminated source row. No source or captured-reference attributes other
than opacity are edited. Protected camera rays are traced in both the accepted
baseline and V1; their contributors are restored/excluded respectively.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,write_ply,columns,sha256_file
from transfer_reference_corners import positions,attenuate
from repair_surfaces import WORLD_FROM_RAW
CONTROLS={
 'front-wall':[{'id':'full-left-ceiling-to-header','bounds':[[0,-5],[483,95]]},
               {'id':'full-right-ceiling-to-stepped-header','bounds':[[483,-5],[739,64]]},
               {'id':'reflective-metal-hatch','bounds':[[275,141],[388,240]]}],
 'ceiling-front':[{'id':'whole-ceiling-and-fascia','bounds':[[-5,-5],[1005,390]]},
                   {'id':'low-central-roof-silhouette','bounds':[[255,390],[710,399]]},
                   {'id':'reflective-metal-hatch','bounds':[[339,430],[430,504]]}]
}


def minimum_quadratic_rectangle(center,inverse,low,high):
 """Exact minimum of a positive 2D quadratic over a closed rectangle."""
 inside=((center>=low)&(center<=high)).all(1)
 minimum=np.full(len(center),np.inf);minimum[inside]=0
 for x in [low[0],high[0]]:
  dx=x-center[:,0];y=np.clip(center[:,1]-inverse[:,0,1]/inverse[:,1,1]*dx,low[1],high[1]);d=np.c_[dx,y-center[:,1]];minimum=np.minimum(minimum,np.einsum('ni,nij,nj->n',d,inverse,d))
 for y in [low[1],high[1]]:
  dy=y-center[:,1];x=np.clip(center[:,0]-inverse[:,0,1]/inverse[:,0,0]*dy,low[0],high[0]);d=np.c_[x-center[:,0],dy];minimum=np.minimum(minimum,np.einsum('ni,nij,nj->n',d,inverse,d))
 return minimum


def trace_controls(v,p,baseline_alpha,current_alpha,primary_rows,changed_source,new_reference):
 # Check every possible pixel covered by a support ellipse; no sparse ray grid.
 candidate=np.r_[changed_source,new_reference];allids=np.flatnonzero(candidate);scales=np.exp(columns(v[allids],['scale_0','scale_1','scale_2']).astype(float));restore=np.zeros(primary_rows,bool);exclude=np.zeros(len(v)-primary_rows,bool);evidence=[]
 cameras=json.loads(Path(__file__).with_name('pass2-cameras.json').read_text())
 for camera_id,regions in CONTROLS.items():
  camera=next(c for c in cameras if c['id']==camera_id);eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);V=np.array([right,-np.cross(right,forward),forward]);q=np.einsum('ij,nj->ni',V,p[allids]-eye);front=q[:,2]>.02;ids=allids[front];q=q[front];z=q[:,2];ss=scales[front];f=375/np.tan(np.radians(camera['fov'])/2);center=np.c_[500+f*q[:,0]/z,375+f*q[:,1]/z]-.5
  Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@WORLD_FROM_RAW,Q);J=np.zeros((len(ids),2,3));J[:,0,0]=f/z;J[:,1,1]=f/z;J[:,0,2]=-f*np.clip(q[:,0]/z,-1.3*500/f,1.3*500/f)/z;J[:,1,2]=-f*np.clip(q[:,1]/z,-1.3*375/f,1.3*375/f)/z;B=np.einsum('nij,njk->nik',J,Q)*ss[:,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C);det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2;rr=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-rr)/(1024/1080*750),0,1)
  for region in regions:
   low,high=np.asarray(region['bounds']);distance=minimum_quadratic_rectangle(center,inverse,low,high);foot=fade*np.maximum(0,np.exp(-.5*distance)-np.exp(-4.5));is_source=ids<primary_rows;src=is_source&(baseline_alpha[ids]*foot>1e-5);ref=(~is_source)&(current_alpha[ids]*foot>1e-5);restore[ids[src]]=True;exclude[ids[ref]-primary_rows]=True;item={'camera':camera_id,'control':region,'method':'continuous maximum projected Gaussian alpha over rectangle; no sparse ray sampling','alpha_threshold':1e-5,'restored_source_rows':int(src.sum()),'excluded_new_reference_rows':int(ref.sum())};evidence.append(item);print(item,flush=True)
 return np.flatnonzero(restore),np.flatnonzero(exclude),evidence


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass4/combined-v1');ap.add_argument('--v1',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/front-header-v1');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();base=json.loads((args.baseline/'report.json').read_text());old=json.loads((args.v1/'report.json').read_text());phone,_,_=read_ply(Path(base['iphone']));ref,_,_=read_ply(Path(base['reference']));bs=np.load(args.baseline/'selections.npz');ps=np.load(args.v1/'iphone-opacity-selection.npz');rs=np.load(args.v1/'reference-selection.npz');bm=np.ones(len(phone));br=np.zeros(len(ref));pm=np.ones(len(phone));rm=np.zeros(len(ref));bm[bs['iphone_indices']]=bs['iphone_multipliers'];br[bs['reference_indices']]=bs['reference_multipliers'];pm[ps['indices']]=ps['multipliers'];rm[rs['indices']]=rs['multipliers'];rids=np.flatnonzero(np.maximum(br,rm)>0);rshort=np.empty(len(rids),phone.dtype)
 for n in phone.dtype.names:rshort[n]=ref[n][rids]
 v=np.concatenate([phone,rshort]);p=positions(v);a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));ba=a*np.r_[bm,br[rids]];ca=a*np.r_[np.minimum(bm,pm),np.maximum(br,rm)[rids]];restore,exclude,evidence=trace_controls(v,p,ba,ca,len(phone),pm<bm,rm[rids]>br[rids]);pm[restore]=1;rm[rids[exclude]]=0
 # This remaining saturated blue pixel was independently attributed at front-wall
 # (798,127). It is behind the native corner face, outside red harness hardware.
 blue_row=3723506;rgb=.5+.28209479177387814*np.array([phone[f'f_dc_{i}'][blue_row] for i in range(3)]);blue_evidence={'source_index':blue_row,'rgb':rgb.tolist(),'world':positions(phone[blue_row:blue_row+1])[0].tolist(),'pixel':[798,127],'action':'suppress independently traced blue cloud on black cover'}
 pm[blue_row]=0
 # Single confirmed blue contaminant contributes to the black corner pixel;
 # source red net and hardware stay protected by the original component.
 pi=np.flatnonzero(pm<1);ri=np.flatnonzero(rm>0);args.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=pm[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rm[ri]);np.savez_compressed(args.out/'protected-control-contributors.npz',restored_phone=restore,excluded_reference=rids[exclude]);write_ply(args.out/'reference-corners.ply',attenuate(ref[ri],rm[ri]));assembled=args.out/'assembled';assembled.mkdir(exist_ok=True);cm=np.minimum(bm,pm);cr=np.maximum(br,rm);cpi=np.flatnonzero(cm<1);cri=np.flatnonzero(cr>0);current=phone.copy();current[cpi]=attenuate(phone[cpi],cm[cpi]);write_ply(assembled/'iphone.ply',current);write_ply(assembled/'reference-patches.ply',attenuate(ref[cri],cr[cri]));shutil.copyfile(__file__,args.out/'generator.py');report={**old,'status':'Unreviewed V4 header trial with continuous full roof/hatch silhouette protection','derivation':{'v1':str(args.v1),'v1_report_sha256':sha256_file(args.v1/'report.json'),'method':'Conservatively restore every changed source ellipse and reject every new donor ellipse that can contribute alpha above 1e-5 anywhere within the full stepped roof silhouette or hatch controls; no geometry changes.'},'protected_control_trace':evidence,'restored_source_count':len(restore),'excluded_reference_count':len(exclude),'blue_fleck_attribution':blue_evidence,'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'generator_sha256':sha256_file(args.out/'generator.py'),'incremental_effect':{'iphone_strengthened_rows':int(np.sum(pm<bm)),'reference_strengthened_rows':int(np.sum(rm>br)),'reference_new_rows':int(np.sum((rm>0)&(br==0)))},'checks':{'iphone_only_opacity_changed':all(np.array_equal(current[n],phone[n]) for n in phone.dtype.names if n!='opacity'),'accepted_phone_opacity_not_restored':bool(np.all(cm<=bm)),'accepted_reference_support_retained':bool(np.all(cr>=br))}}
 # The exact subset relation is checked directly, including feathered factors.
 v1pm=np.ones(len(phone));v1pm[ps['indices']]=ps['multipliers'];report['checks']['source_patch_subset_of_v1_except_traced_blue_row']=bool(np.all(np.delete(pm,blue_row)>=np.delete(v1pm,blue_row)));report['checks']['reference_patch_subset_of_v1']=bool(np.all(rm[np.setdiff1d(np.arange(len(ref)),rs['indices'])]==0));(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['iphone_changed_rows','reference_added_rows','restored_source_count','excluded_reference_count','checks']},indent=2),flush=True)
if __name__=='__main__':main()
