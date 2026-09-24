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
 'front-wall':[{'id':'shared-roof-support','bounds':[[185,8],[560,84]],'grid':[27,7]},
               {'id':'roof-header-silhouette','bounds':[[285,74],[490,94]],'grid':[31,6]},
               {'id':'reflective-metal-hatch','bounds':[[278,145],[384,237]],'grid':[13,11]}],
 'ceiling-front':[{'id':'accepted-roof-field','bounds':[[225,5],[745,360]],'grid':[29,21]},
                   {'id':'reflective-metal-hatch','bounds':[[343,435],[425,500]],'grid':[11,9]}]
}

def trace_controls(v,p,baseline_alpha,current_alpha,primary_rows,changed_source,new_reference):
 scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float)); restore=np.zeros(primary_rows,bool);exclude=np.zeros(len(v)-primary_rows,bool);evidence=[]
 cameras=json.loads(Path(__file__).with_name('pass2-cameras.json').read_text())
 for camera_id,regions in CONTROLS.items():
  camera=next(c for c in cameras if c['id']==camera_id);eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);V=np.array([right,-np.cross(right,forward),forward]);q=np.einsum('ij,nj->ni',V,p-eye);z=q[:,2];safe=np.maximum(z,.02);f=375/np.tan(np.radians(camera['fov'])/2);screen=np.c_[500+f*q[:,0]/safe,375+f*q[:,1]/safe];radius=4*f*scales.max(1)/safe+2
  for region in regions:
   lo,hi=np.asarray(region['bounds']);nu,nv=region['grid'];pixels=np.array(np.meshgrid(np.linspace(lo[0],hi[0],nu),np.linspace(lo[1],hi[1],nv))).reshape(2,-1).T
   ok=(z>.02)&(screen[:,0]+radius>lo[0])&(screen[:,0]-radius<hi[0])&(screen[:,1]+radius>lo[1])&(screen[:,1]-radius<hi[1]);ids=np.flatnonzero(ok);ids=ids[np.argsort(z[ids])]
   Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@WORLD_FROM_RAW,Q);J=np.zeros((len(ids),2,3));J[:,0,0]=f/z[ids];J[:,1,1]=f/z[ids];J[:,0,2]=-f*np.clip(q[ids,0]/z[ids],-1.3*500/f,1.3*500/f)/z[ids];J[:,1,2]=-f*np.clip(q[ids,1]/z[ids],-1.3*375/f,1.3*375/f)/z[ids];B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C);det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2;rr=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-rr)/(1024/1080*750),0,1)
   found_source=np.zeros(len(ids),bool);found_reference=np.zeros(len(ids),bool);is_source=ids<primary_rows;source_eligible=is_source.copy();source_eligible[is_source]&=changed_source[ids[is_source]];ref_eligible=~is_source;ref_eligible[~is_source]&=new_reference[ids[~is_source]-primary_rows];bc=[];cc=[]
   for pixel in pixels:
    d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d);foot=fade*np.maximum(0,np.exp(power)-np.exp(-4.5));a=np.minimum(.99,baseline_alpha[ids]*foot);visibility=a*np.r_[1,np.cumprod(1-a[:-1])];found_source|=(visibility>.001)&source_eligible;bc.append(float(visibility.sum()));a=np.minimum(.99,current_alpha[ids]*foot);visibility=a*np.r_[1,np.cumprod(1-a[:-1])];found_reference|=(visibility>.001)&ref_eligible;cc.append(float(visibility.sum()))
   restore[ids[found_source]]=True;exclude[ids[found_reference]-primary_rows]=True;item={'camera':camera_id,'control':region,'rays':len(pixels),'restored_source_rows':int(found_source.sum()),'excluded_new_reference_rows':int(found_reference.sum()),'median_baseline_opacity':float(np.median(bc)),'median_rejected_v1_opacity':float(np.median(cc))};evidence.append(item);print(item,flush=True)
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
 pi=np.flatnonzero(pm<1);ri=np.flatnonzero(rm>0);args.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=pm[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rm[ri]);np.savez_compressed(args.out/'protected-control-contributors.npz',restored_phone=restore,excluded_reference=rids[exclude]);write_ply(args.out/'reference-corners.ply',attenuate(ref[ri],rm[ri]));assembled=args.out/'assembled';assembled.mkdir(exist_ok=True);cm=np.minimum(bm,pm);cr=np.maximum(br,rm);cpi=np.flatnonzero(cm<1);cri=np.flatnonzero(cr>0);current=phone.copy();current[cpi]=attenuate(phone[cpi],cm[cpi]);write_ply(assembled/'iphone.ply',current);write_ply(assembled/'reference-patches.ply',attenuate(ref[cri],cr[cri]));shutil.copyfile(__file__,args.out/'generator.py');report={**old,'status':'Unreviewed V3 header trial with measured roof/hatch/silhouette support guards','derivation':{'v1':str(args.v1),'v1_report_sha256':sha256_file(args.v1/'report.json'),'method':'Restore baseline source contributors and exclude newly strengthened donor contributors visible on protected controls; no geometry changes.'},'protected_control_trace':evidence,'restored_source_count':len(restore),'excluded_reference_count':len(exclude),'blue_fleck_attribution':blue_evidence,'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'generator_sha256':sha256_file(args.out/'generator.py'),'incremental_effect':{'iphone_strengthened_rows':int(np.sum(pm<bm)),'reference_strengthened_rows':int(np.sum(rm>br)),'reference_new_rows':int(np.sum((rm>0)&(br==0)))},'checks':{'iphone_only_opacity_changed':all(np.array_equal(current[n],phone[n]) for n in phone.dtype.names if n!='opacity'),'accepted_phone_opacity_not_restored':bool(np.all(cm<=bm)),'accepted_reference_support_retained':bool(np.all(cr>=br))}}
 # The exact subset relation is checked directly, including feathered factors.
 v1pm=np.ones(len(phone));v1pm[ps['indices']]=ps['multipliers'];report['checks']['source_patch_subset_of_v1_except_traced_blue_row']=bool(np.all(np.delete(pm,blue_row)>=np.delete(v1pm,blue_row)));report['checks']['reference_patch_subset_of_v1']=bool(np.all(rm[np.setdiff1d(np.arange(len(ref)),rs['indices'])]==0));(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['iphone_changed_rows','reference_added_rows','restored_source_count','excluded_reference_count','checks']},indent=2),flush=True)
if __name__=='__main__':main()
