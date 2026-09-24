#!/usr/bin/env python3
"""Incremental, capture-backed wall repair on the accepted ambulance baseline.

Only original iPhone opacity is suppressed. Reference geometry, covariance,
colors and all directional coefficients are copied unchanged. Actual mixed
scene visibility is traced so newly exposed old layers are diagnosed rather
than guessed from source point centers. Outputs use the assembler NPZ contract.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT,columns,read_ply,sha256_file,write_ply
from repair_surfaces import WORLD_FROM_RAW

GROUPS={
 'pad-edge':{
  'description':'Opacity-only removal of exactly attributed bright iPhone support forming a false pale ribbon along the upper right pad. Dark rounded pad and all reference material stay unchanged.',
  'cameras':['pads-close','right-wall','right-grazing'],'source_only':True,
  'context_components':['raw/ambulance-cleanup/pass4/walls-right-trim-v1'],
  'boxes':[],
  'candidate_bounds':[[-.84,.686,.77],[.54,.725,1.03]],
  'candidate_rgb_min':.60,'candidate_chroma_max':.25,
  'surfaces':[
   {'id':'false-upper-pad-ribbon','axis':2,'uv_axes':[0,1],'uv_bounds':[[-.77,.663],[.48,.699]],'plane':[-.08291,.06183,.78042],'behind_sign':1,'deep_bounds':[.77,1.03],'grid':[51,5]}]},
 'recess':{
  'description':'Dark fixed cabinet underside above left backrest and its right vertical side; glass contents and touchscreen remain outside boxes.',
  'cameras':['left-wall','left-grazing','left-seats'],
  'boxes':[
   {'id':'overhead-recess-underside','bounds':[[-.34,.635,-1.46],[.39,.83,-.87]],'feather':.02},
   {'id':'overhead-recess-right-side','bounds':[[.32,.48,-1.47],[.44,.79,-.98]],'feather':.02}],
  'surfaces':[
   {'id':'recess-underside','axis':1,'uv_axes':[0,2],'uv_bounds':[[-.29,-1.29],[.31,-.95]],'plane':[.00301445,.19654815,.93823979],'behind_sign':1,'deep_bounds':[.635,3.5],'grid':[19,11]},
   {'id':'recess-right-side','axis':0,'uv_axes':[2,1],'uv_bounds':[[-1.36,.52],[-1.01,.73]],'plane':[.0931732,-.02435573,.5083562],'behind_sign':1,'deep_bounds':[.32,3.5],'grid':[13,9]}]},
 'right-trim':{
  'description':'Fixed bolted horizontal trim above right backrest, plus exposed white wall between central strap columns. Preserves pad bodies and strap anchors.',
  'cameras':['pads-close','right-wall','right-grazing'],
  'boxes':[
   {'id':'right-bolted-trim','bounds':[[-.90,.719,.68],[.64,.811,1.04]],'feather':.014},
   {'id':'right-central-white-wall','bounds':[[-.38,.282,.79],[.08,.421,1.03]],'feather':.018}],
  'surfaces':[
   {'id':'right-bolted-trim','axis':2,'uv_axes':[0,1],'uv_bounds':[[-.86,.735],[.60,.796]],'plane':[-.08835326,.02612805,.84218627],'behind_sign':1,'deep_bounds':[.70,5.5],'grid':[29,5]},
   {'id':'right-central-white-wall','axis':2,'uv_axes':[0,1],'uv_bounds':[[-.35,.3],[.05,.402]],'plane':[-.08850324,.01756119,.86515643],'behind_sign':1,'deep_bounds':[.80,4.5],'grid':[13,7]}]},
 'protector':{
  'description':'Small fixed dark protective panel on the cabinet side below the monitor, including its four captured corner fasteners. Surrounding belongings remain original.',
  'cameras':['left-wall','left-grazing','left-seats','cabinet-counter'],
  'boxes':[
   {'id':'cabinet-side-dark-protector','bounds':[[.235,-.204,-1.21],[.345,.035,-.976]],'feather':.012}],
  'surfaces':[
   {'id':'cabinet-side-dark-protector','axis':0,'uv_axes':[2,1],'uv_bounds':[[-1.18,-.18],[-1.005,.012]],'plane':[.1111032565,-.0001854620,.4133779861],'behind_sign':1,'deep_bounds':[.245,1.4],'grid':[13,13]}]},
 'rear-paint':{
  'description':'Two lower rear-door paint interiors. The red center gasket, handles, black windows and original warning-stripe transition remain outside the regions.',
  'cameras':['rear-wall','ceiling-rear'],
  'protected_boxes':[
   {'id':'original-warning-stripe-near-door','bounds':[[-1.85,-7,-7],[-1.4,-.51,7]]},
   {'id':'original-center-gasket-near-door','bounds':[[-1.85,-7,-.08],[-1.4,7,.04]]}],
  'boxes':[
   {'id':'rear-right-paint-interior','bounds':[[-1.68,-.51,.04],[-1.46,-.27,.54]],'feather':.025},
   {'id':'rear-left-paint-interior','bounds':[[-1.71,-.51,-.69],[-1.51,-.27,-.08]],'feather':.025}],
  'surfaces':[
   {'id':'rear-right-paint-interior','axis':0,'uv_axes':[2,1],'uv_bounds':[[.075,-.48],[.505,-.30]],'plane':[.09023278,.00961142,-1.57076640],'behind_sign':-1,'deep_bounds':[-4,-1.46],'grid':[19,9]},
   {'id':'rear-left-paint-interior','axis':0,'uv_axes':[2,1],'uv_bounds':[[-.655,-.48],[-.115,-.30]],'plane':[.09023278,.00961142,-1.57076640],'behind_sign':-1,'deep_bounds':[-4,-1.51],'grid':[21,9]}]}
}


def smoothstep(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)


def positions(v):
 return np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float))


def base_weights(p,group):
 w=np.zeros(len(p),np.float32);report=[]
 for box in group['boxes']:
  lo,hi=np.asarray(box['bounds']);edge=np.minimum(p-lo,hi-p).min(1);bw=smoothstep(edge/box['feather']);w=np.maximum(w,bw.astype(np.float32))
  report.append({**box,'selected_count':int((bw>0).sum())})
 return w,report


def trace(v,p,group,primary_rows=None):
 scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float));alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
 cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in group['cameras']]
 selected_all=np.zeros(len(v),bool);records=[]
 for camera in cameras:
  eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=-np.cross(right,forward);V=np.array([right,down,forward])
  q=np.einsum('ij,nj->ni',V,p-eye);z=q[:,2];safe=np.maximum(z,.02);f=375/np.tan(np.radians(camera['fov'])/2)
  screen=np.c_[500+f*q[:,0]/safe,375+f*q[:,1]/safe];radius=4*f*scales.max(1)/safe+2
  for surface in group['surfaces']:
   ax=surface['axis'];axes=surface['uv_axes'];lo,hi=np.asarray(surface['uv_bounds']);nu,nv=surface['grid'];coef=np.asarray(surface['plane'])
   uv=np.array(np.meshgrid(np.linspace(lo[0],hi[0],nu),np.linspace(lo[1],hi[1],nv))).reshape(2,-1).T
   points=np.empty((len(uv),3));points[:,axes]=uv;points[:,ax]=np.einsum('ij,j->i',uv,coef[:2])+coef[2]
   projected=np.einsum('ij,nj->ni',V,points-eye);pixels=np.c_[500+f*projected[:,0]/projected[:,2],375+f*projected[:,1]/projected[:,2]]
   ok=(projected[:,2]>.02)&(pixels[:,0]>=0)&(pixels[:,0]<1000)&(pixels[:,1]>=0)&(pixels[:,1]<750);pixels=pixels[ok]
   if not len(pixels):continue
   low,high=pixels.min(0),pixels.max(0);overlap=(z>.02)&(screen[:,0]+radius>low[0])&(screen[:,0]-radius<high[0])&(screen[:,1]+radius>low[1])&(screen[:,1]-radius<high[1])
   ids=np.flatnonzero(overlap);ids=ids[np.argsort(z[ids])]
   Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@WORLD_FROM_RAW,Q)
   J=np.zeros((len(ids),2,3));J[:,0,0]=f/z[ids];J[:,1,1]=f/z[ids]
   J[:,0,2]=-f*np.clip(q[ids,0]/z[ids],-1.3*500/f,1.3*500/f)/z[ids];J[:,1,2]=-f*np.clip(q[ids,1]/z[ids],-1.3*375/f,1.3*375/f)/z[ids]
   B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C)
   det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2;splat_radius=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-splat_radius)/(1024/1080*750),0,1)
   residual=p[ids,ax]-np.einsum('ij,j->i',p[ids][:,axes],coef[:2])-coef[2];dlo,dhi=surface['deep_bounds']
   eligible=(surface['behind_sign']*residual>.025)&(p[ids,ax]>dlo)&(p[ids,ax]<dhi)&(np.abs(p[ids])<7).all(1)
   if 'candidate_bounds' in group:
    candidate_lo,candidate_hi=np.asarray(group['candidate_bounds']);eligible&=((p[ids]>candidate_lo)&(p[ids]<candidate_hi)).all(1)
   if 'candidate_rgb_min' in group:
    rgb=.5+.28209479177387814*columns(v[ids],['f_dc_0','f_dc_1','f_dc_2']);eligible&=(rgb.min(1)>group['candidate_rgb_min'])&(np.ptp(rgb,axis=1)<group['candidate_chroma_max'])
   for protected in group.get('protected_boxes',[]):
    protected_lo,protected_hi=np.asarray(protected['bounds']);eligible&=~((p[ids]>protected_lo)&(p[ids]<protected_hi)).all(1)
   if primary_rows is not None:eligible&=ids<primary_rows
   picked=np.zeros(len(ids),bool);coverage=[]
   for pixel in pixels:
    d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d);a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)));T=np.r_[1,np.cumprod(1-a[:-1])];visible=a*T
    picked|=(visible>.003)&eligible;coverage.append(float(visible.sum()))
   selected_all[ids[picked]]=True
   record={'camera':camera['id'],'surface':surface['id'],'rays':len(pixels),'contributors':int(picked.sum()),'median_scene_opacity':float(np.median(coverage))};records.append(record);print(record,flush=True)
 return np.flatnonzero(selected_all),records


def attenuate(v,multipliers):
 out=v.copy();a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));a=np.clip(a*multipliers,1e-8,1-1e-8);out['opacity']=np.log(a/(1-a));return out


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--group',choices=GROUPS,required=True);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/final-bench');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.resolve()==args.baseline.resolve():raise ValueError('Accepted baseline is immutable')
 base_report=json.loads((args.baseline/'report.json').read_text());src=Path(base_report['iphone']);refsrc=Path(base_report['reference']);phone,_,_=read_ply(src);ref,_,_=read_ply(refsrc)
 if sha256_file(src)!=base_report['iphone_sha256'] or sha256_file(refsrc)!=base_report['reference_sha256']:raise ValueError('Baseline source hash mismatch')
 if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):raise ValueError('Reference SH3 required')
 bs=np.load(args.baseline/'selections.npz');baseline_pm=np.ones(len(phone));baseline_rm=np.zeros(len(ref));baseline_pm[bs['iphone_indices']]=bs['iphone_multipliers'];baseline_rm[bs['reference_indices']]=bs['reference_multipliers']
 group=GROUPS[args.group]
 for context in group.get('context_components',[]):
  context_path=ROOT/context;cs=np.load(context_path/'iphone-opacity-selection.npz');cr=np.load(context_path/'reference-selection.npz')
  baseline_pm[cs['indices']]=np.minimum(baseline_pm[cs['indices']],cs['multipliers']);baseline_rm[cr['indices']]=np.maximum(baseline_rm[cr['indices']],cr['multipliers'])
 pp=positions(phone);rp=positions(ref);pw,pr=base_weights(pp,group);rw,rr=base_weights(rp,group)
 if group.get('source_only'):rid=np.empty(0,dtype=np.int64);reference_trace=[]
 else:rid,reference_trace=trace(ref,rp,group);rw[rid]=1
 composite_rw=np.maximum(baseline_rm,rw);cri=np.flatnonzero(composite_rw>0);composite_ref=attenuate(ref[cri],composite_rw[cri]);extra=np.empty(len(composite_ref),dtype=phone.dtype)
 for name in phone.dtype.names:extra[name]=composite_ref[name]
 rounds=[];additional=[]
 for iteration in range(4):
  composite_pm=np.minimum(baseline_pm,1-pw);cpi=np.flatnonzero(composite_pm<1);current=phone.copy();current[cpi]=attenuate(phone[cpi],composite_pm[cpi])
  scene=np.concatenate([current,extra]);picked,trace_report=trace(scene,np.concatenate([pp,rp[cri]]),group,len(phone));new=picked[pw[picked]<1-1e-6]
  rounds.append({'iteration':iteration,'new_original_rows':len(new),'traces':trace_report});print('Hybrid pass',iteration,'additional',len(new),flush=True)
  if not len(new):break
  pw[new]=1;additional.extend(new.tolist())
 pi=np.flatnonzero(pw>0);ri=np.flatnonzero(rw>0);args.out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rw[ri]);np.save(args.out/'additional-cloud-indices.npy',np.asarray(additional,dtype=np.int64));np.save(args.out/'reference-contributor-indices.npy',rid)
 patch=attenuate(ref[ri],rw[ri]);write_ply(args.out/'reference-wall-details.ply',patch)
 composite_pm=np.minimum(baseline_pm,1-pw);cpi=np.flatnonzero(composite_pm<1);current=phone.copy();current[cpi]=attenuate(phone[cpi],composite_pm[cpi]);assembled=args.out/'assembled';assembled.mkdir(exist_ok=True);write_ply(assembled/'iphone.ply',current);write_ply(assembled/'reference-patches.ply',composite_ref)
 checks={'iphone_only_opacity_changed':all(np.array_equal(current[n],phone[n]) for n in phone.dtype.names if n!='opacity'),'reference_geometry_covariance_DC_SH_exact':all(np.array_equal(patch[n],ref[n][ri]) for n in ref.dtype.names if n!='opacity'),'accepted_phone_opacity_not_restored':bool(np.all(composite_pm<=baseline_pm)),'accepted_reference_support_retained':bool(np.all(composite_rw>=baseline_rm))}
 if not all(checks.values()):raise RuntimeError(checks)
 report={'status':'Unreviewed incremental wall-detail candidate','group':args.group,'description':group['description'],'iphone':str(src),'iphone_sha256':sha256_file(src),'reference':str(refsrc),'reference_sha256':sha256_file(refsrc),'accepted_baseline':str(args.baseline),'baseline_report_sha256':sha256_file(args.baseline/'report.json'),'baseline_selections_sha256':sha256_file(args.baseline/'selections.npz'),'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'iphone_regions':pr,'reference_regions':rr,'trace_surfaces':group['surfaces'],'reference_trace':reference_trace,'hybrid_passes':rounds,'checks':checks,'review_cameras':group['cameras'],'reference_unchanged_fields':'All except feathered opacity','limitations':['Static-region alignment is not exact ground truth.','Deep radiance support is not physical surface geometry.','Surrounding glass, fixtures, floor and accepted repairs require visual regression review.'],'remote_publish':False}
 report['protected_regions']=group.get('protected_boxes',[])
 report['source_only']=group.get('source_only',False)
 report['context_components']=group.get('context_components',[])
 report['attribution_candidate_guards']={key:group[key] for key in ['candidate_bounds','candidate_rgb_min','candidate_chroma_max'] if key in group}
 report['incremental_effect']={'iphone_new_selected_rows':int(np.sum(baseline_pm[pi]==1)),
  'iphone_strengthened_rows':int(np.sum((1-pw[pi])<baseline_pm[pi])),
  'reference_new_rows':int(np.sum(baseline_rm[ri]==0)),
  'reference_strengthened_rows':int(np.sum(rw[ri]>baseline_rm[ri]))}
 shutil.copyfile(__file__,args.out/'generator.py');report['generator_sha256']=sha256_file(args.out/'generator.py')
 (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['group','iphone_changed_rows','reference_added_rows','checks']},indent=2),flush=True)


if __name__=='__main__':main()
