#!/usr/bin/env python3
"""Pass5 capture-backed white wall material repair, with equipment exclusions.

The reference's real laminate pattern and SH3 are preserved, never replaced by
uniform paint. Surface fits delimit material and attribution rays only. They
do not move or flatten any captured Gaussian. Each region is a separate trial.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cleanup import ROOT,read_ply,sha256_file,write_ply
from transfer_reference_wall_details import positions,smoothstep,trace,attenuate


COLUMN=[-.07972403373575151,.017460167860638722,-.9304070506527942]
BACK=[-.08845000064167072,.03128498190332061,-1.3707081719330318]
SHELF=[-.010380875134517653,-.022884346953055454,.491910368037645]


def region(name,axis,uv,bounds,plane,depth,grid,deep,sign,feather=.02):
 return {'id':name,'axis':axis,'uv_axes':uv,'uv_bounds':bounds,'plane':plane,
         'depth_bounds':depth,'feather':feather,'depth_feather':.008,
         'grid':grid,'deep_bounds':deep,'behind_sign':sign}


GROUPS={
 'right-paint':{
  'description':'Exposed right-wall laminate left of the fixed pads and below their central gap. Pad outlines, restraint columns, oxygen plate and movable bag remain outside the material regions.',
  'cameras':['right-wall','pads-close','right-grazing'],
  'regions':[
   region('paint-left-of-right-pads',2,[0,1],[[.60,-.22],[.90,.58]],[-.08850324,.01756119,.86515643],[-.045,.10],[13,29],[.70,5.5],1,.022),
   region('paint-below-pad-center',2,[0,1],[[-.397,-.47],[.112,-.075]],[-.08850324,.01756119,.86515643],[-.045,.10],[21,17],[.70,5.5],1,.022)],
  'protected_boxes':[
   {'id':'oxygen-label-and-fixture','bounds':[[.83,.54,-7],[1.10,.82,7]]},
   {'id':'upper-pad','bounds':[[-.89,.415,-7],[.59,.735,7]]},
   {'id':'lower-pad','bounds':[[-.89,-.075,-7],[.59,.29,7]]},
   {'id':'left-restraint-column','bounds':[[.112,-.6,-7],[.35,.415,7]]},
   {'id':'right-restraint-column','bounds':[[-.66,-.6,-7],[-.397,.415,7]]}]},
 'counter-column':{
  'description':'Full exposed white column to the right of the monitor/cabinet; genuine laminate grain retained. Protruding tissue holder and front-seat harness are outside the material depth band.',
  'cameras':['cabinet-counter','front-wall','left-grazing'],
  'regions':[region('counter-right-white-column',2,[0,1],[[1.295,-.61],[1.665,.82]],COLUMN,[-.10,.055],[13,35],[-5.5,-.94],-1,.025)],
  'protected_boxes':[
   {'id':'protruding-tissue-holder','bounds':[[1.35,.46,-1.03],[1.80,.95,-.6]]},
   {'id':'front-bulkhead-handles','bounds':[[1.59,-.30,-1.00],[1.85,.50,-.55]]}]},
 'counter-backsplash':{
  'description':'Exposed backsplash paint islands around the original oxygen labels, monitor, outlets and medical gear; original printed labels/screens/hardware remain excluded.',
  'cameras':['cabinet-counter','left-wall','left-grazing'],
  'regions':[
   region('left-backsplash-paint',2,[0,1],[[.318,.066],[.506,.515]],BACK,[-.08,.045],[11,21],[-5.5,-1.34],-1,.014),
   region('label-monitor-gap',2,[0,1],[[.762,.255],[.829,.50]],BACK,[-.08,.045],[5,15],[-5.5,-1.34],-1,.009),
   region('paint-below-labels',2,[0,1],[[.50,.064],[.821,.252]],BACK,[-.08,.04],[15,9],[-5.5,-1.34],-1,.014),
   region('paint-below-phone',2,[0,1],[[1.115,.066],[1.328,.246]],BACK,[-.08,.04],[11,11],[-5.5,-1.34],-1,.014)],
  'protected_boxes':[
   {'id':'oxygen-and-no-smoking-labels','bounds':[[.506,.252,-7],[.762,.54,7]]},
   {'id':'screen-and-phone','bounds':[[.829,.246,-7],[1.38,.56,7]]},
   {'id':'electrical-outlets-thermometer','bounds':[[.821,.04,-7],[1.115,.255,7]]}]},
 'counter-shelf':{
  'description':'Fixed shelf underside over the monitor. Actual blue reflection support is retained where captured; glass supplies and the screen are not transferred.',
  'cameras':['cabinet-counter','left-wall','left-grazing'],
  'regions':[region('monitor-shelf-underside',1,[0,2],[[.35,-1.49],[1.33,-1.065]],SHELF,[-.037,.085],[29,13],[.49,.603],1,.02)],
  'candidate_bounds':[[.30,.47,-1.65],[1.38,.603,-1.035]],
  'protected_boxes':[
   {'id':'cabinet-glass-and-contents-above-shelf','bounds':[[-7,.603,-7],[7,7,7]]}]}
}


def protect_mask(p,group):
 protect=np.zeros(len(p),bool)
 for box in group.get('protected_boxes',[]):
  lo,hi=np.asarray(box['bounds']);protect|=((p>lo)&(p<hi)).all(1)
 return protect


def material(p,group):
 w=np.zeros(len(p),np.float32);stats=[]
 for r in group['regions']:
  uv=p[:,r['uv_axes']];lo,hi=np.asarray(r['uv_bounds']);coef=np.asarray(r['plane']);d=p[:,r['axis']]-np.einsum('ij,j->i',uv,coef[:2])-coef[2]
  edge=np.minimum(uv-lo,hi-uv).min(1);a,b=r['depth_bounds'];candidate=smoothstep(edge/r['feather'])*smoothstep(np.minimum(d-a,b-d)/r['depth_feather']);w=np.maximum(w,candidate.astype(np.float32));stats.append({**r,'rows':int((candidate>0).sum())})
 w[protect_mask(p,group)]=0
 return w,stats


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--group',choices=GROUPS,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass4/combined-v1');args=ap.parse_args()
 if args.out.resolve()==args.baseline.resolve():raise ValueError('The accepted baseline is immutable')
 prior=json.loads((args.baseline/'report.json').read_text());src=Path(prior['iphone']);rsrc=Path(prior['reference']);phone,_,_=read_ply(src);ref,_,_=read_ply(rsrc)
 if sha256_file(src)!=prior['iphone_sha256'] or sha256_file(rsrc)!=prior['reference_sha256']:raise ValueError('Capture provenance mismatch')
 if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):raise ValueError('Full SH3 reference required')
 bs=np.load(args.baseline/'selections.npz');bpm=np.ones(len(phone));brm=np.zeros(len(ref));bpm[bs['iphone_indices']]=bs['iphone_multipliers'];brm[bs['reference_indices']]=bs['reference_multipliers']
 group=GROUPS[args.group];pp=positions(phone);rp=positions(ref);pw,ps=material(pp,group);rw,rs=material(rp,group);tracer={**group,'surfaces':group['regions']}
 rid,rtrace=trace(ref,rp,tracer);rw[rid]=1;rw[protect_mask(rp,group)]=0
 crm=np.maximum(brm,rw);cri=np.flatnonzero(crm>0);patch=attenuate(ref[cri],crm[cri]);extra=np.empty(len(patch),dtype=phone.dtype)
 for name in phone.dtype.names:extra[name]=patch[name]
 rounds=[];new_all=[]
 for i in range(4):
  cm=np.minimum(bpm,1-pw);ci=np.flatnonzero(cm<1);current=phone.copy();current[ci]=attenuate(phone[ci],cm[ci]);scene=np.concatenate([current,extra])
  picked,records=trace(scene,np.concatenate([pp,rp[cri]]),tracer,len(phone));new=picked[pw[picked]<1-1e-6];rounds.append({'iteration':i,'new_original_rows':len(new),'rays':records});print('Original hybrid peel',i,len(new),flush=True)
  if not len(new):break
  pw[new]=1;new_all.extend(new.tolist())
 pi=np.flatnonzero(pw>0);ri=np.flatnonzero(rw>0);cm=np.minimum(bpm,1-pw);ci=np.flatnonzero(cm<1);current=phone.copy();current[ci]=attenuate(phone[ci],cm[ci])
 checks={'original_all_nonopacity_fields_exact':all(np.array_equal(current[n],phone[n]) for n in phone.dtype.names if n!='opacity'),'reference_all_nonopacity_fields_exact':all(np.array_equal(patch[n],ref[n][cri]) for n in ref.dtype.names if n!='opacity'),'phone_protected_regions_not_selected':not bool(np.any(protect_mask(pp,group)[pi])),'reference_protected_regions_not_selected':not bool(np.any(protect_mask(rp,group)[ri])),'accepted_source_opacity_not_restored':bool(np.all(cm<=bpm)),'accepted_reference_weights_not_reduced':bool(np.all(crm>=brm))}
 if not all(checks.values()):raise RuntimeError(checks)
 args.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rw[ri]);np.savez_compressed(args.out/'traced-contributors.npz',reference=rid,iphone=np.asarray(new_all,dtype=np.int64));write_ply(args.out/'reference-white-wall.ply',attenuate(ref[ri],rw[ri]));context=args.out/'assembled';context.mkdir(exist_ok=True);write_ply(context/'iphone.ply',current);write_ply(context/'reference-patches.ply',patch)
 shutil.copyfile(__file__,args.out/'generator.py')
 report={'status':'Unreviewed pass5 white-wall material candidate','group':args.group,'description':group['description'],'iphone':str(src),'iphone_sha256':sha256_file(src),'reference':str(rsrc),'reference_sha256':sha256_file(rsrc),'baseline':str(args.baseline),'baseline_report_sha256':sha256_file(args.baseline/'report.json'),'baseline_selections_sha256':sha256_file(args.baseline/'selections.npz'),'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'incremental_source_rows_strengthened':int(np.sum(cm<bpm)),'incremental_reference_rows':int(np.sum((crm>0)&(brm==0))),'iphone_regions':ps,'reference_regions':rs,'protected_regions':group.get('protected_boxes',[]),'reference_attribution':rtrace,'hybrid_attribution':rounds,'checks':checks,'generator_sha256':sha256_file(args.out/'generator.py'),'review_cameras':group['cameras'],'limitations':['The native laminate is genuinely patterned; do not claim uniform clean paint.','Exact source radiance support may extend behind a material surface.','Original equipment, glass, cables, labels and accepted repairs require multi-view regression review.'],'remote_publish':False}
 (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['group','iphone_changed_rows','reference_added_rows','checks']},indent=2),flush=True)


if __name__=='__main__':main()
