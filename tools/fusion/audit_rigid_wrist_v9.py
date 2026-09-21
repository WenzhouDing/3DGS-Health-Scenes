#!/usr/bin/env python3
"""Strict one-body right forearm/hand registration using one axial DOF only."""
import json,argparse,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mannequin-rigid-v9-mpl')
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes,binary_closing,binary_erosion,label
from pipeline import ROOT,BODY_PARTS,transform_gaussians,rotate,dot,coverage_weights,attenuate,load_cleanup_masks
from render_gaussians import render,atlas,load_fused
BASE=ROOT/'raw/fusion-work/refinement-v9/baseline';OUT=ROOT/'raw/fusion-work/refinement-v9/registration';OUT.mkdir(parents=True,exist_ok=True)
NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection'

def context():
 cfg=json.loads((BASE/'fusion-config.json').read_text());r=json.loads((BASE/'report.json').read_text());lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks'];features=json.loads((BASE/'feature-transforms.json').read_text())['parts']
 for rule in cfg.get('jointBoundaries',{}).get('front',[]):lm[rule['landmark']]=rule['pivot']
 native={s:dict(np.load(NATIVE/f'{s}-full-native.npz')) for s in ['front','back']};mask,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],r['sources'])),ROOT)
 return cfg,r,lm,features,native,mask

def shared_transform(extra,report,features):
 constraint=features['right_forearm']['mechanicalConstraint'];f=report['parts'][11]['sourceToFrontRaw'];axis=np.array(constraint['axisFrontRaw']);pivot=np.array(constraint['pivotFrontRaw']);Q=Rotation.from_rotvec(axis*np.radians(extra)).as_matrix();R=Q@np.array(f['rotation']);t=Q@(np.array(f['translation'])-pivot)+pivot
 return {'scale':f['scale'],'rotation':R.tolist(),'translation':t.tolist()},dict(constraint,twistDegrees=constraint['twistDegrees']+extra)

def generate(extra,ctx):
 cfg,r,lm,features,native,masks=ctx;tr,constraint=shared_transform(extra,r,features);output=[];labels=[];sources=[];indices=[]
 for pid in [11,12]:
  f=native['front']['data'][native['front']['labels']==pid];b=native['back']['data'][native['back']['labels']==pid];b=transform_gaussians(b,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);fw,bw=coverage_weights(f,b,lm,BODY_PARTS[pid],cfg['fusion'])
  for si,(source,d,w) in enumerate([('front',f,fw),('back',b,bw)]):
   ix=native[source]['indices'][native[source]['labels']==pid];dd,kept=attenuate(d,w,cfg['fusion']['minWeight']);keep=~np.isin(ix[kept],masks[source]);dd=dd[keep];ix=ix[kept][keep];output.append(dd);labels.append(np.full(len(dd),pid,np.uint8));sources.append(np.full(len(dd),si,np.uint8));indices.append(ix)
 return {'data':np.concatenate(output),'labels':np.concatenate(labels),'sources':np.concatenate(sources),'indices':np.concatenate(indices)},tr,constraint

def observed_hand(native,source):
 hand=native[source]['labels']==12;d=native[source]['data'][hand];ids=native[source]['indices'][hand];v=np.load(ROOT/f'raw/fusion-work/refinement-v6/right-hand/{source}-full-visibility.npz');assert np.array_equal(ids,v['indices']);keep=(v['seen']>v['opposite_seen'])&(v['observed_ratio']>.03)&(v['seen']>.05)
 return d[keep]

def edge(d,direction,center,span=.29,size=650):
 white=d.copy();white[:,11:14]=(1-.5)/.2820947918;im=np.asarray(render(white,direction,center,span,width=size,height=size));mask=im[:,:,0]>145;mask=binary_fill_holes(binary_closing(mask,iterations=2));labs,n=label(mask);counts=np.bincount(labs.ravel());counts[0]=0;mask=labs==np.argmax(counts);boundary=mask&~binary_erosion(mask);iy,ix=np.nonzero(boundary);return np.c_[ix+.5-size/2,iy+.5-size/2]*(span/size)

def sweep():
 ctx=context();cfg,r,lm,features,native,masks=ctx;front=observed_hand(native,'front');back=observed_hand(native,'back');center=np.array([-.754,.06,.669]);directions={'front':[0,-1,0],'outer':[-1,0,0],'distal':[-.63,0,.78]};target={key:edge(front,d,center) for key,d in directions.items()};results={};rows={};handrows={}
 baseline,l,s,_=load_fused(BASE);current=np.isin(l,[11,12]);rows['V8 nonrigid baseline']=baseline[current];handrows['V8 nonrigid baseline']=baseline[current]
 for extra in [-10.,0.,10.,15.,20.,25.,30.]:
  candidate,tr,constraint=generate(extra,ctx);name=f'axial-extra-{extra:g}';np.savez_compressed(OUT/(name+'.npz'),**candidate);mapped=transform_gaussians(back,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);metrics={}
  for key,direction in directions.items():
   contour=edge(mapped,direction,center);dist=np.r_[cKDTree(target[key]).query(contour)[0],cKDTree(contour).query(target[key])[0]];metrics[key]={'mean':float(np.mean(dist)),'median':float(np.median(dist)),'p90':float(np.quantile(dist,.9))}
  document={'status':'diagnostic candidate; one rigid body with unchanged native source geometry','extraTwistDegrees':extra,'totalTwistDegrees':constraint['twistDegrees'],
   'sourceHashes':{m['file']:m['sha256'] for m in r['sources']},'sharedParts':['right_forearm','right_hand'],'sourceToFrontRaw':tr,'mechanicalConstraint':constraint,
   'handObservedContourMetrics':metrics,'metricCaveat':'front projection compares finger outlines; outer/distal partial-surface contours also include real complementary thickness and are visual diagnostics, not a shell-collapse objective',
   'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_rigid_wrist_v9.py --sweep'}
  (OUT/(name+'.json')).write_text(json.dumps(document,indent=2)+'\n');results[name]=document
  label=f'Rigid total twist {constraint["twistDegrees"]:.1f} deg';rows[label]=candidate['data'];handrows[label]=candidate['data'];print(name,metrics,flush=True)
 (OUT/'axial-sweep.json').write_text(json.dumps(results,indent=2)+'\n')
 views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Outer',[-1,.2,0]),('Below',[0,0,1]),('Distal',[-.7,0,.7])]
 atlas(handrows,OUT/'rigid-axial-hand-sweep.png',[-.742,.045,.635],.40,550,views=views,title='One rigid forearm and hand; no wrist deformation; unchanged upper-arm parent and elbow axis')
 # Full distal-body review includes the axial collar and both long skin pads.
 upper=baseline[l==10];armrows={k:np.concatenate([upper,d]) for k,d in rows.items()}
 atlas(armrows,OUT/'rigid-axial-fullarm-sweep.png',[-.558,.02,.448],.77,550,views=views[:3],title='Shared rigid body / assess fixed shoulder and axial collar plus forearm and wrist pad')

def export():
 """Package the reviewed +10-degree candidate and reproducible safeguards."""
 from right_wrist_transition import preserve_pad_align_fingers
 ctx=context();cfg,r,lm,features,native,masks=ctx
 candidate,tr,constraint=generate(10.,ctx)
 np.savez_compressed(OUT/'accepted-rigid-arm.npz',**candidate)
 R=np.array(tr['rotation']);t=np.array(tr['translation']);scale=tr['scale']
 c=json.loads((ROOT/'raw/fusion-work/refinement-v4/arm/collar-measurements.json').read_text())
 front_center=np.array(c['front']['pivot']);back_center=np.array(c['back']['pivot']);front_axis=np.array(c['front']['axis']);back_axis=np.array(c['back']['axis'])
 def collar(transform):
  rr=np.array(transform['rotation']);tt=np.array(transform['translation'])
  delta=transform['scale']*rr@back_center+tt-front_center
  return {'centerError':float(np.linalg.norm(delta)),'centerErrorXYZ':delta.tolist(),'normalAngleDegrees':float(np.degrees(np.arccos(np.clip((rr@back_axis)@front_axis,-1,1))))}
 pairs=np.load(NATIVE/'pad-cross-part-pairs.npz');ids=native['back']['indices'];order=np.argsort(ids)
 def locate(values):
  index=order[np.searchsorted(ids[order],values)];assert np.array_equal(ids[index],values);return index
 a=native['back']['data'][locate(pairs['forearmSourceIndices']),:3].astype(float)
 b=native['back']['data'][locate(pairs['handSourceIndices']),:3].astype(float)
 native_lengths=np.linalg.norm(a-b,axis=1)*scale
 mapped_lengths=np.linalg.norm(scale*np.einsum('ni,ji->nj',a,R)+t-(scale*np.einsum('ni,ji->nj',b,R)+t),axis=1)
 pad_ids=np.load(NATIVE/'back-right-rigid-pad.npy');pad=native['back']['data'][locate(pad_ids)]
 mapped_pad=transform_gaussians(pad,R,t,scale)
 # Check arbitrary pairs throughout the whole body as well as the pad cut.
 rng=np.random.default_rng(913);na=rng.integers(len(ids),size=10000);nb=rng.integers(len(ids),size=10000)
 p=native['back']['data'][:,:3].astype(float);pp=scale*np.einsum('ni,ji->nj',p,R)+t
 rigid_error=np.abs(np.linalg.norm(pp[na]-pp[nb],axis=1)-scale*np.linalg.norm(p[na]-p[nb],axis=1))
 front=observed_hand(native,'front');back=observed_hand(native,'back')
 old_back,_=preserve_pad_align_fingers(back,cfg['rightWristRegistration'])
 old_transform=r['parts'][11]['sourceToFrontRaw'];old_back=transform_gaussians(old_back,np.array(old_transform['rotation']),np.array(old_transform['translation']),old_transform['scale'])
 new_back=transform_gaussians(back,R,t,scale)
 metrics={}
 for name,direction in {'front':[0,-1,0],'outer':[-1,0,0],'distal':[-.63,0,.78]}.items():
  target=edge(front,direction,[-.754,.06,.669]);metrics[name]={}
  for key,data in [('v8Deformed',old_back),('v9Rigid',new_back)]:
   contour=edge(data,direction,[-.754,.06,.669]);dist=np.r_[cKDTree(target).query(contour)[0],cKDTree(contour).query(target)[0]]
   metrics[name][key]={'mean':float(np.mean(dist)),'median':float(np.median(dist)),'p90':float(np.quantile(dist,.9))}
 doc=json.loads((OUT/'axial-extra-10.json').read_text());doc.update({
  'status':'accepted after independent multi-angle review; one rigid forearm and hand',
  'baseline':'raw/fusion-work/refinement-v9/baseline',
  'coordinateUnits':'original scan units; physical millimeter scale is not calibrated',
  'configurationChange':{'remove':'rightWristRegistration','rigidParts':{'right_hand':'right_forearm'},'totalRightForearmTwistDegrees':constraint['twistDegrees']},
  'sourceGeometryPreservation':{'originalScansModified':False,'nativeWarp':False,'nativeRowDeletionAdded':False,'uniformBackScale':scale,'sharedRigidTransform':True,'wholeBodySampledPairs':10000,'maximumPairLengthError':float(rigid_error.max()),'rotationDeterminant':float(np.linalg.det(R)),'rotationOrthonormalError':float(np.max(np.abs(R@R.T-np.eye(3))))},
  'padContinuity':{'selectedCleanOriginalRows':len(pad_ids),'forearmRows':8420,'handRows':3802,'crossPartPairs':len(a),'referenceScaledNativeDistanceMedian':float(np.median(native_lengths)),'acceptedPairDistanceMedian':float(np.median(mapped_lengths)),'maximumPairLengthError':float(np.max(np.abs(mapped_lengths-native_lengths))),'maximumLogScalePreservationError':float(np.max(np.abs(mapped_pad[:,7:10]-(pad[:,7:10]+np.log(scale))))),'selection':'raw/fusion-work/refinement-v8/pad-inspection/back-right-rigid-pad.json','limitation':'These are quality-clean observed back-source pad rows, not proof of front/back surface coincidence or full post-coverage retention.'},
  'collarSafeguard':{'nativeFitEvidence':'raw/fusion-work/refinement-v4/arm/collar-measurements.json','baseline':collar(old_transform),'accepted':collar(tr),'parentRightUpperArmUnchanged':True,'axisAndPivotUnchanged':True,'independentSwingDegrees':0,'independentTranslation':[0,0,0],'notes':'The pre-existing normal discrepancy remains unchanged under axial rotation; no artificial bend is introduced to erase it.'},
  'observedSourceContourComparison':metrics,
  'metricLimitations':'Partial front/back surfaces include real complementary thickness. Silhouette distances are diagnostics, not a minimization target that justifies flattening or deforming the source. The V8 reference includes the now-rejected nonrigid wrist field.',
  'selectionReason':'Independent review of true Gaussian views favors extra 10 degrees over 15 or 20: improved underside/outer overlap with less fuzzy tip and forearm distortion. The entire distal body rotates together around the existing measured swivel; the pad remains rigid and continuous.',
  'remainingLimitations':'Residual source seams and partial opposite-side surfaces remain. No rigid pose can guarantee pixel-identical matching when the two reconstructed sources disagree. No unsupported hand/finger warps or extra visibility deletions were introduced.',
  'reviewImages':['raw/fusion-work/refinement-v9/registration/rigid-axial-hand-sweep.png','raw/fusion-work/refinement-v9/registration/rigid-axial-fullarm-sweep.png','raw/fusion-work/refinement-v9/root-review/below.png','raw/fusion-work/refinement-v9/root-review/outer.png','raw/fusion-work/refinement-v9/root-review/pad.png'],
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_rigid_wrist_v9.py --sweep --export'})
 (OUT/'rigid-arm-evidence.json').write_text(json.dumps(doc,indent=2)+'\n')
 print(json.dumps({'evidence':str(OUT/'rigid-arm-evidence.json'),'collar':doc['collarSafeguard'],'pad':doc['padContinuity'],'contours':metrics},indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--sweep',action='store_true');p.add_argument('--export',action='store_true');a=p.parse_args()
 if a.sweep:sweep()
 if a.export:export()
