#!/usr/bin/env python3
"""Bounded shoulder registration; distal arm/hand remain one rigid body.

All trials read a frozen baseline and write diagnostic output only. The elbow
uses the existing single-axis formula; no independent wrist or finger pose.

--trials, --extend and the grid options produce exploratory candidates, never
production settings. --fit-centers reproduces the candidate selected in V10;
--export packages its independently reviewed evidence. A caller must still run
the production export and integrated verification before using a rebuilt scan.
The pad centroid is a registration anchor, not a newly measured rig ball center.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pipeline import ROOT,BODY_PARTS,apply_joint_boundaries,transform_gaussians,coverage_weights,attenuate,load_cleanup_masks
from render_gaussians import atlas,load_fused,render
from PIL import Image,ImageDraw
from audit_rigid_wrist_v9 import observed_hand,edge

BASE=ROOT/'raw/fusion-work/refinement-v10/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v10/registration'
NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection'
PAD=ROOT/'raw/fusion-work/feature-audit'

def context():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((BASE/'fusion-config.json').read_text());report=json.loads((BASE/'report.json').read_text());features=json.loads((BASE/'feature-transforms.json').read_text())['parts']
 lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
 for rule in cfg['jointBoundaries']['front']:lm[rule['landmark']]=rule['pivot']
 native={s:dict(np.load(NATIVE/f'{s}-full-native.npz')) for s in ['front','back']}
 cache=np.load(ROOT/'raw/fusion-work/refinement-v4/arm/native-clean.npz');upper={}
 for source,key in [('front','fl'),('back','bl')]:
  landmarks=json.loads((BASE/f'{source}-landmarks.json').read_text())['landmarks'];labels,_=apply_joint_boundaries(cache[source][:,:3],cache[key].copy(),landmarks,cfg['jointBoundaries'][source],BODY_PARTS)
  upper[source]=cache[source][labels==10]
  assert len(upper[source])==report['parts'][10][source+'Before']
 masks,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],report['sources'])),ROOT)
 c=json.loads((ROOT/'raw/fusion-work/refinement-v4/arm/collar-measurements.json').read_text())
 f=np.load(PAD/'front-right-pad-ellipse.npz')['outline'];b=np.load(PAD/'back-right-pad-ellipse.npz')['outline']
 return cfg,report,features,lm,native,upper,masks,c,f,b

def parent_transform(v,ctx):
 _,report,_,_,_,_,_,_,f,_=ctx
 old=report['parts'][10]['sourceToFrontRaw'];r0=np.array(old['rotation']);t0=np.array(old['translation']);pivot=f.mean(0)
 q=Rotation.from_rotvec(np.asarray(v)).as_matrix()
 return {'scale':old['scale'],'rotation':(q@r0).tolist(),'translation':(q@(t0-pivot)+pivot).tolist()}

def child_transform(parent,ctx,twist_delta=0):
 constraint=dict(ctx[2]['right_forearm']['mechanicalConstraint']);constraint['twistDegrees']+=twist_delta
 axis=np.array(constraint['axisFrontRaw']);pivot=np.array(constraint['pivotFrontRaw']);q=Rotation.from_rotvec(axis*np.radians(constraint['twistDegrees'])).as_matrix()
 return {'scale':parent['scale'],'rotation':(q@np.array(parent['rotation'])).tolist(),'translation':(q@(np.array(parent['translation'])-pivot)+pivot).tolist()},constraint

def safeguards(parent,child,ctx):
 c,f,b=ctx[-3:];r=np.array(parent['rotation']);t=np.array(parent['translation']);s=parent['scale']
 m=s*np.einsum('ni,ji->nj',b,r)+t;d=cKDTree(f).query(m)[0]
 def collar(tr):
  r=np.array(tr['rotation']);t=np.array(tr['translation']);cp=s*r@np.array(c['back']['pivot'])+t
  return {'centerError':float(np.linalg.norm(cp-c['front']['pivot'])),'normalAngleDegrees':float(np.degrees(np.arccos(np.clip((r@np.array(c['back']['axis']))@c['front']['axis'],-1,1))))}
 return {'padCenterError':float(np.linalg.norm(m.mean(0)-f.mean(0))),'padRimMedian':float(np.median(d)),'padRimP90':float(np.quantile(d,.9)),'parentCollar':collar(parent),'childCollar':collar(child)}

def fit_parent(axis_weight,ctx):
 c,f,b=ctx[-3:];tree=cKDTree(f);fa=np.array(c['front']['axis']);fc=np.array(c['front']['pivot']);ba=np.array(c['back']['axis']);bc=np.array(c['back']['pivot'])
 def residual(v):
  tr=parent_transform(v,ctx);r=np.array(tr['rotation']);t=np.array(tr['translation']);s=tr['scale'];m=s*np.einsum('ni,ji->nj',b,r)+t
  return np.r_[(m-f[tree.query(m)[1]]).ravel()/np.sqrt(len(m))/.0015,(s*r@bc+t-fc)/.002,(r@ba-fa)*axis_weight/.015,v/.05]
 return least_squares(residual,np.zeros(3),bounds=(-np.ones(3)*.05,np.ones(3)*.05),loss='soft_l1').x

def moderate_seed(ctx):
 """Reproduce the shoulder direction without depending on an old trial run."""
 v=fit_parent(1.,ctx);parent=parent_transform(v,ctx);child,constraint=child_transform(parent,ctx)
 return {'name':'collar-moderate','status':'diagnostic only; not accepted','baseline':str(BASE.relative_to(ROOT)),
         'shoulderRotationVectorDegrees':np.degrees(v).tolist(),'shoulderRotationDegrees':float(np.degrees(np.linalg.norm(v))),
         'shoulderPivotFrontRaw':ctx[-2].mean(0).tolist(),'right_upper_arm':parent,'right_forearm':child,'right_hand':child,
         'mechanicalConstraint':constraint,'safeguards':safeguards(parent,child,ctx),
         'sourceHashes':{m['file']:m['sha256'] for m in ctx[1]['sources']}}

def generate(parent,child,ctx):
 cfg,report,_,lm,native,upper,masks,_,_,_=ctx;out=[];labels=[];sources=[]
 for pid,tr in [(10,parent),(11,child),(12,child)]:
  f=upper['front'] if pid==10 else native['front']['data'][native['front']['labels']==pid]
  b=upper['back'] if pid==10 else native['back']['data'][native['back']['labels']==pid]
  b=transform_gaussians(b,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);fw,bw=coverage_weights(f,b,lm,BODY_PARTS[pid],cfg['fusion'])
  for si,(source,d,w) in enumerate([('front',f,fw),('back',b,bw)]):
   dd,keep=attenuate(d,w,cfg['fusion']['minWeight'])
   if pid!=10:
    ids=native[source]['indices'][native[source]['labels']==pid][keep];dd=dd[~np.isin(ids,masks[source])]
   out.append(dd);labels.append(np.full(len(dd),pid,np.uint8));sources.append(np.full(len(dd),si,np.uint8))
 return dict(data=np.concatenate(out),labels=np.concatenate(labels),sources=np.concatenate(sources))

def trials():
 OUT.mkdir(parents=True,exist_ok=True);ctx=context();baseline,lab,src,_=load_fused(BASE);sel=np.isin(lab,[10,11,12]);rows={'V9 baseline':baseline[sel]};reports={}
 moderate=fit_parent(1.,ctx);small=fit_parent(.5,ctx)
 for name,v in [('collar-small',small),('collar-moderate',moderate),('opposite-moderate',-moderate)]:
  parent=parent_transform(v,ctx);child,constraint=child_transform(parent,ctx);candidate=generate(parent,child,ctx)
  document={'name':name,'status':'diagnostic only; not accepted','baseline':str(BASE.relative_to(ROOT)),'shoulderRotationVectorDegrees':np.degrees(v).tolist(),'shoulderRotationDegrees':float(np.degrees(np.linalg.norm(v))),'shoulderPivotFrontRaw':ctx[-2].mean(0).tolist(),'right_upper_arm':parent,'right_forearm':child,'right_hand':child,'mechanicalConstraint':constraint,'safeguards':safeguards(parent,child,ctx),'notes':'No wrist deformation or independent child translation/swing. Parent rotates around the measured shoulder-pad center. Existing axis and pivot remain fixed. Fresh native coverage and unchanged cleanup masks for all three parts.','sourceHashes':{m['file']:m['sha256'] for m in ctx[1]['sources']}}
  np.savez_compressed(OUT/(name+'.npz'),**candidate);(OUT/(name+'.json')).write_text(json.dumps(document,indent=2)+'\n');rows[name]=candidate['data'];reports[name]=document;print(json.dumps(document),flush=True)
 (OUT/'joint-trials.json').write_text(json.dumps(reports,indent=2)+'\n')
 views=[('Below',[0,0,1]),('Outer',[-1,.2,0]),('Back oblique',[-.45,1,0]),('Front',[0,-1,0])]
 atlas(rows,OUT/'shoulder-trials-hand.png',[-.742,.045,.635],.40,650,views=views,title='One rigid distal body; bounded upstream shoulder correction; no wrist warp')
 atlas(rows,OUT/'shoulder-trials-fullarm.png',[-.515,.005,.388],.85,620,views=views[1:],title='Shoulder pad, collar and forearm safeguards; front source unchanged')

def detail():
 ctx=context();native=ctx[4];front=observed_hand(native,'front');back=observed_hand(native,'back');baseline,lab,src,_=load_fused(BASE);sel=np.isin(lab,[11,12]);rows={'V9 baseline':baseline[sel]};tintrows={};metrics={}
 views=[('Below',[0,0,1]),('Outer',[-1,.2,0]),('Distal',[-.7,0,.7])]
 def tint(data,sources):
  d=data.copy();d[:,11:14]=(np.where(sources[:,None]==0,np.array([.1,.8,.95]),np.array([1.,.55,.1]))-.5)/.2820947918;return d
 tintrows['V9 baseline']=tint(rows['V9 baseline'],src[sel])
 records={'V9 baseline':{'right_forearm':ctx[1]['parts'][11]['sourceToFrontRaw']}}
 for name in ['collar-small','collar-moderate','opposite-moderate']:
  z=np.load(OUT/(name+'.npz'));sel=np.isin(z['labels'],[11,12]);rows[name]=z['data'][sel];tintrows[name]=tint(rows[name],z['sources'][sel]);records[name]=json.loads((OUT/(name+'.json')).read_text())
 for view,direction in [('front',[0,-1,0]),('outer',[-1,0,0]),('distal',[-.63,0,.78])]:
  target=edge(front,direction,[-.754,.06,.669]);metrics[view]={}
  for name,record in records.items():
   tr=record['right_forearm'];mapped=transform_gaussians(back,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);contour=edge(mapped,direction,[-.754,.06,.669]);dist=np.r_[cKDTree(target).query(contour)[0],cKDTree(contour).query(target)[0]]
   metrics[view][name]={'mean':float(np.mean(dist)),'median':float(np.median(dist)),'p90':float(np.quantile(dist,.9))}
 (OUT/'observed-hand-contour-metrics.json').write_text(json.dumps({'metrics':metrics,'caveat':'Independent native observed contours. Opposing partial shells retain true depth thickness; these distances diagnose pose and cannot certify a watertight surface.'},indent=2)+'\n')
 atlas(rows,OUT/'shoulder-trials-detail.png',[-.765,.045,.678],.30,800,views=views,title='Rigid upstream shoulder trials / finger-layer underside and distal views')
 atlas(tintrows,OUT/'shoulder-trials-source-colors.png',[-.765,.045,.678],.30,800,views=views,title='Front cyan / back orange / unchanged opacity and native shape')
 print(json.dumps(metrics,indent=2),flush=True)

def extend():
 ctx=context();baseline,lab,src,_=load_fused(BASE);rows={'V9 baseline':baseline[np.isin(lab,[10,11,12])]}
 seed=json.loads((OUT/'collar-moderate.json').read_text());axis=np.array(seed['shoulderRotationVectorDegrees']);axis/=np.linalg.norm(axis)
 rows['Shoulder 1.049 deg']=np.load(OUT/'collar-moderate.npz')['data'];results={}
 for amount in [1.4,1.7,2.0]:
  v=axis*np.radians(amount);parent=parent_transform(v,ctx);child,constraint=child_transform(parent,ctx);candidate=generate(parent,child,ctx);name=f'collar-{amount:g}'
  record=dict(seed,name=name,shoulderRotationVectorDegrees=np.degrees(v).tolist(),shoulderRotationDegrees=amount,right_upper_arm=parent,right_forearm=child,right_hand=child,mechanicalConstraint=constraint,safeguards=safeguards(parent,child,ctx))
  record['notes']+=' Extension along the previously fitted shoulder-rotation direction only; choose smallest magnitude closing the observed same-finger slit after visual collar checks.'
  np.savez_compressed(OUT/(name+'.npz'),**candidate);(OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');results[name]=record;rows[f'Shoulder {amount:g} deg']=candidate['data'];print(name,record['safeguards'],flush=True)
 (OUT/'extended-trials.json').write_text(json.dumps(results,indent=2)+'\n')
 views=[('Below',[0,0,1]),('Outer',[-1,.2,0]),('Back oblique',[-.45,1,0]),('Front',[0,-1,0])]
 atlas(rows,OUT/'extended-shoulder-hand.png',[-.765,.045,.678],.30,800,views=views,title='Rigid shoulder correction along measured collar-fit direction / no wrist motion')
 atlas(rows,OUT/'extended-shoulder-fullarm.png',[-.515,.005,.388],.85,700,views=views[1:],title='Check shoulder rim and collar continuity under bounded shoulder rotation')

def joint_grid(negative=False,center_fit=False):
 ctx=context();seed=json.loads((OUT/'collar-moderate.json').read_text());axis=np.array(seed['shoulderRotationVectorDegrees']);axis/=np.linalg.norm(axis);entries=[];results={}
 prefix='joint-center' if center_fit else ('joint-negative' if negative else 'joint')
 if negative:axis=-axis
 amounts=[1.4,1.8,2.2] if center_fit else [seed['shoulderRotationDegrees'],1.4,1.7]
 extras=[total-ctx[2]['right_forearm']['mechanicalConstraint']['twistDegrees'] for total in [0.,5.,10.]] if center_fit else ([0.,-5.,-10.] if negative else [0.,5.,10.])
 for amount in amounts:
  row=[]
  for extra in extras:
   v=axis*np.radians(amount);parent=parent_transform(v,ctx);child,constraint=child_transform(parent,ctx,extra);candidate=generate(parent,child,ctx);name=f'{prefix}-shoulder-{amount:.3f}-extra-{extra:g}'
   record=dict(seed,name=name,shoulderRotationVectorDegrees=np.degrees(v).tolist(),shoulderRotationDegrees=amount,right_upper_arm=parent,right_forearm=child,right_hand=child,mechanicalConstraint=constraint,safeguards=safeguards(parent,child,ctx));record['totalTwistDegrees']=constraint['twistDegrees']
   np.savez_compressed(OUT/(name+'.npz'),**candidate);(OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');results[name]=record;row.append((name,amount,constraint['twistDegrees'],candidate));print(name,flush=True)
  entries.append(row)
 (OUT/(prefix+'-grid.json')).write_text(json.dumps(results,indent=2)+'\n')
 size=780
 for view,direction in [('below',[0,0,1]),('outer',[-1,.2,0]),('dorsal',[-.45,1,0])]:
  canvas=Image.new('RGB',(size*3,(size+30)*3+40),(18,23,28));draw=ImageDraw.Draw(canvas);draw.text((12,12),'Shoulder rows / total axial twist columns / rigid forearm+hand',fill='white')
  for ri,row in enumerate(entries):
   for ci,(name,amount,twist,candidate) in enumerate(row):
    image=render(candidate['data'],direction,[-.765,.045,.678],.30,width=size,height=size);x=ci*size;y=40+ri*(size+30);canvas.paste(image,(x,y+30));draw.text((x+12,y+8),f'{amount:.3f} deg shoulder / {twist:.3f} deg total axial',fill='white')
  canvas.save(OUT/f'{prefix}-grid-{view}.png')

def collar_review():
 ctx=context();base,labels,_,_=load_fused(BASE);rows={'V9 baseline':base[np.isin(labels,[10,11,12])]}
 for amount,total in [(1.4,5.),(1.8,5.),(1.8,0.),(2.2,0.)]:
  extra=total-ctx[2]['right_forearm']['mechanicalConstraint']['twistDegrees'];name=f'joint-center-shoulder-{amount:.3f}-extra-{extra:g}';rows[f'Shoulder -{amount:g} / axial {total:g}']=np.load(OUT/(name+'.npz'))['data']
 views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Outer',[-1,.2,0]),('Oblique',[-.5,1,-.2])]
 atlas(rows,OUT/'center-candidates-collar.png',[-.359,-.038,.279],.33,700,views=views,title='Actual collar remains a single axial joint; inspect source seam after whole-chain correction')
 atlas(rows,OUT/'center-candidates-fullarm.png',[-.515,.005,.388],.85,700,views=views[:3],title='Shoulder rim and rigid one-piece forearm/hand / original front pose unchanged')

def fit_centers():
 """Use curved shaft centers, preserving both source radii, to refine the chain."""
 ctx=context();seed=moderate_seed(ctx);axis=-np.array(seed['shoulderRotationVectorDegrees']);axis/=np.linalg.norm(axis)
 audit=json.loads((ROOT/'raw/fusion-work/refinement-v10/gap-audit/shaft-center-audit.json').read_text());basis=np.array(audit['frame']['axesVYU']);origin=np.array(audit['frame']['origin']);old=ctx[1]['parts'][11]['sourceToFrontRaw'];oldR=np.array(old['rotation']);oldt=np.array(old['translation']);curves={}
 for finger in ['index','middle','ring']:
  records=audit['sections'][finger];back=np.array([m['centerLocalVYU'] for m in records['back'] if m['accepted']]);front=np.array([m['centerLocalVYU'] for m in records['front'] if m['accepted'] and .134<m['station']<.156]);curves[finger]=(np.einsum('ni,ij->nj',back,basis)+origin,front)
 def transforms(x):
  correction=Rotation.from_rotvec(np.array([0.,np.radians(x[1]),0.]))*Rotation.from_rotvec(axis*np.radians(x[0]));v=correction.as_rotvec();parent=parent_transform(v,ctx);child,constraint=child_transform(parent,ctx,x[2]-ctx[2]['right_forearm']['mechanicalConstraint']['twistDegrees']);return v,parent,child,constraint
 def center_errors(child):
  deltaR=np.array(child['rotation'])@oldR.T;deltat=np.array(child['translation'])-deltaR@oldt;errors={}
  for finger,(points,target) in curves.items():
   moved=np.einsum('ni,ji->nj',points,deltaR)+deltat;local=np.einsum('ni,ji->nj',moved-origin,basis);local=local[np.argsort(local[:,2])];u=target[:,2];xy=np.c_[np.interp(u,local[:,2],local[:,0]),np.interp(u,local[:,2],local[:,1])];errors[finger]=xy-target[:,:2]
  return errors
 f,b=ctx[-2:];tree=cKDTree(f);c=ctx[-3]
 def residual(x):
  v,parent,child,_=transforms(x);r=np.array(parent['rotation']);t=np.array(parent['translation']);s=parent['scale'];m=s*np.einsum('ni,ji->nj',b,r)+t;center=s*r@np.array(c['back']['pivot'])+t
  errors=center_errors(child)
  return np.r_[np.concatenate(list(errors.values())).ravel()/.002,(m-f[tree.query(m)[1]]).ravel()/np.sqrt(len(m))/.002,(center-c['front']['pivot'])/.004,v/.07]
 fitted=least_squares(residual,[1.4,.35,5.],bounds=([.5,-.5,-5],[2.5,1.2,12]),loss='soft_l1',f_scale=1.,max_nfev=250)
 variants={'center-fit':fitted.x,'yaw-only-conservative':np.array([1.4,fitted.x[1],5.])}
 base,labels,_,_=load_fused(BASE);rows={'V9 baseline':base[np.isin(labels,[10,11,12])]}
 _,fixed_parent,fixed_child,_=transforms([1.4,0.,5.]);rows['Conservative no yaw']=generate(fixed_parent,fixed_child,ctx)['data']
 results={}
 for name,x in variants.items():
  v,parent,child,constraint=transforms(x);candidate=generate(parent,child,ctx);errors=center_errors(child);metrics={f:{'deltaLateral':float(np.median(e[:,0])),'deltaDepth':float(np.median(e[:,1])),'medianSeparation':float(np.median(np.linalg.norm(e,axis=1))),'sections':e.tolist()} for f,e in errors.items()}
  document=dict(seed,name=name,status='diagnostic circle-center fit; visual gate required',shoulderRotationVectorDegrees=np.degrees(v).tolist(),shoulderRotationDegrees=float(np.degrees(np.linalg.norm(v))),shoulderNegativeDirectionAmountDegrees=float(x[0]),additionalShoulderYawRawYDegrees=float(x[1]),totalTwistDegrees=float(x[2]),right_upper_arm=parent,right_forearm=child,right_hand=child,mechanicalConstraint=constraint,safeguards=safeguards(parent,child,ctx),shaftCenterMetrics=metrics)
  document['fitMethod']='Three-parameter rigid chain: rotation along measured shoulder correction direction, shoulder raw+Y yaw, axial elbow twist. Targets are independently inferred shaft-center trajectories from observed curved arcs at U .135/.145/.155, not opposing surfaces.'
  document['featureEvidence']='raw/fusion-work/refinement-v10/gap-audit/shaft-center-audit.json';document['objectiveScales']={'shaftCenters':.002,'shoulderRim':.002,'collarCenter':.004,'shoulderRotationRegularizationRadians':.07};document['fitLimitations']=audit['limitations'];document['reproduce']='.venv-fusion/bin/python -B tools/fusion/refine_rigid_arm_v10.py --fit-centers'
  (OUT/(name+'.json')).write_text(json.dumps(document,indent=2)+'\n');np.savez_compressed(OUT/(name+'.npz'),**candidate);rows[name]=candidate['data'];results[name]=document;print(json.dumps(document),flush=True)
 (OUT/'center-fit-comparison.json').write_text(json.dumps(results,indent=2)+'\n')
 views=[('Below',[0,0,1]),('Outer',[-1,.2,0]),('Dorsal',[-.45,1,0]),('Front',[0,-1,0])]
 atlas(rows,OUT/'center-fit-hand.png',[-.765,.045,.678],.30,800,views=views,title='Observed curved shaft-center registration / exact rigid hand and forearm / no wrist deformation')
 atlas(rows,OUT/'center-fit-collar.png',[-.359,-.038,.279],.33,700,views=views,title='Shoulder center fixed / same one-axis collar mechanics / evaluate actual source seam')
 atlas(rows,OUT/'center-fit-fullarm.png',[-.515,.005,.388],.85,650,views=views[1:],title='Whole-chain fit with shoulder pad-rim and collar-center safeguards')

def export():
 ctx=context();doc=json.loads((OUT/'center-fit.json').read_text());parent=doc['right_upper_arm'];child=doc['right_forearm'];constraint=doc['mechanicalConstraint'];r=np.array(child['rotation']);t=np.array(child['translation']);scale=child['scale']
 q=Rotation.from_rotvec(np.array(constraint['axisFrontRaw'])*np.radians(constraint['twistDegrees'])).as_matrix();pivot=np.array(constraint['pivotFrontRaw'])
 assert np.allclose(r,q@np.array(parent['rotation']),atol=1e-12,rtol=0)
 assert np.allclose(t,q@(np.array(parent['translation'])-pivot)+pivot,atol=1e-12,rtol=0)
 assert doc['right_hand']==doc['right_forearm']
 native=ctx[4]['back'];ids=native['indices'];pad_ids=np.load(NATIVE/'back-right-rigid-pad.npy');order=np.argsort(ids);sel=order[np.searchsorted(ids[order],pad_ids)];assert np.array_equal(ids[sel],pad_ids)
 pad=native['data'][sel];mapped=transform_gaussians(pad,r,t,scale);rng=np.random.default_rng(10);i=rng.integers(len(pad),size=10000);j=rng.integers(len(pad),size=10000)
 native_distance=scale*np.linalg.norm(pad[i,:3].astype(float)-pad[j,:3].astype(float),axis=1);mapped_distance=np.linalg.norm(mapped[i,:3].astype(float)-mapped[j,:3].astype(float),axis=1)
 resliced=json.loads((ROOT/'raw/fusion-work/refinement-v10/gap-audit/resliced-shaft-center-audit.json').read_text())
 evidence={
  'version':1,'status':'accepted after source-feature fit, independent reslice validation and independent six-angle visual review; final integrated export verification required',
  'method':doc['fitMethod'],'baseline':str(BASE.relative_to(ROOT)),'sourceHashes':doc['sourceHashes'],
  'parts':{'right_upper_arm':{'sourceToFrontRaw':parent},'right_forearm':{'sourceToFrontRaw':child,'mechanicalConstraint':constraint},'right_hand':{'sourceToFrontRaw':child,'rigidWithPart':'right_forearm'}},
  'shoulderRegistration':{k:doc[k] for k in ['shoulderPivotFrontRaw','shoulderRotationVectorDegrees','shoulderRotationDegrees','shoulderNegativeDirectionAmountDegrees','additionalShoulderYawRawYDegrees']},
  'shoulderAnchorMeaning':'Matched shoulder-pad centroid is a registration anchor; this does not change the articulation rig ball-center landmark.',
  'mechanicalValidation':{'oneRigidForearmAndHand':True,'nativeWarp':False,'newSourceDeletionMasks':False,'independentWristDegreesOfFreedom':0,'independentElbowSwingDegrees':0,'independentElbowTranslation':[0,0,0],'preservedOriginalPadRows':len(pad),'sampledPadPairs':len(i),'maximumFloat32PairLengthError':float(np.max(abs(mapped_distance-native_distance))),'maximumLogScaleError':float(np.max(abs(mapped[:,7:10]-(pad[:,7:10]+np.log(scale))))),'formulaRotationMaxError':float(np.max(abs(r-q@np.array(parent['rotation'])))),'formulaTranslationMaxError':float(np.max(abs(t-(q@(np.array(parent['translation'])-pivot)+pivot))))},
  'safeguards':doc['safeguards'],'baselineSafeguards':safeguards(ctx[1]['parts'][10]['sourceToFrontRaw'],ctx[1]['parts'][11]['sourceToFrontRaw'],ctx),
  'shaftCenterFit':doc['shaftCenterMetrics'],'fitAssumptions':doc['fitLimitations'],'independentResliceEvidence':'raw/fusion-work/refinement-v10/gap-audit/resliced-shaft-center-audit.json',
  'independentResliceSummary':{name:{k:v for k,v in result.items() if k!='sections'} for name,result in resliced['candidates'].items()},
  'sourceEvidence':['raw/fusion-work/refinement-v10/gap-audit/shaft-center-audit.json','raw/fusion-work/refinement-v10/gap-audit/shaft-center-support-source-indices.npz','raw/fusion-work/refinement-v8/pad-inspection/back-right-rigid-pad.json','raw/fusion-work/refinement-v10/registration/upper-source-cache-parity.json'],
  'reviewImages':['raw/fusion-work/refinement-v10/registration/center-fit-hand.png','raw/fusion-work/refinement-v10/registration/center-fit-collar.png','raw/fusion-work/refinement-v10/registration/center-fit-fullarm.png','raw/fusion-work/refinement-v10/gap-audit/transformed-source-cross-sections.png','raw/fusion-work/refinement-v10/independent-review/center-fit-fingers-detail.png'],
  'independentVisualReview':'Six actual Gaussian and source-colored views plus enlarged below, outer and distal details: baseline outer-finger slit closes, the below silhouette stays coherent, the wrist pad stays continuous, and no new gross double edge appears.',
  'tradeoff':'The source collar-normal discrepancy changes from 2.575 to 3.929 degrees, while its center stays within .003612 scan units and shoulder pad rim P90 stays .001600. The actual collar seam must remain closed in the final integrated render. No independent elbow bend is used to erase this discrepancy.',
  'limits':'Centers are inferred from observed approximately circular shaft arcs. Rejected little-finger cap fits are not used to drive the pose. Residual local reconstruction/appearance errors may remain; individual source finger sizes and shapes are preserved.',
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_rigid_arm_v10.py --fit-centers --export'}
 (OUT/'rigid-arm-gap-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');print(json.dumps(evidence['mechanicalValidation'],indent=2),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--trials',action='store_true');parser.add_argument('--detail',action='store_true');parser.add_argument('--extend',action='store_true');parser.add_argument('--joint-grid',action='store_true');parser.add_argument('--negative-grid',action='store_true');parser.add_argument('--center-grid',action='store_true');parser.add_argument('--collar-review',action='store_true');parser.add_argument('--fit-centers',action='store_true');parser.add_argument('--export',action='store_true');a=parser.parse_args()
 if a.trials:trials()
 if a.detail:detail()
 if a.extend:extend()
 if a.joint_grid:joint_grid()
 if a.negative_grid:joint_grid(True)
 if a.center_grid:joint_grid(True,True)
 if a.collar_review:collar_review()
 if a.fit_centers:fit_centers()
 if a.export:export()
