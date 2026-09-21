#!/usr/bin/env python3
"""V7 independent left fingertip 3D center registration; immutable sources."""
import os,json,argparse
os.environ.setdefault('MPLCONFIGDIR','/tmp/mannequin-tip-v7-mpl')
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT,transform_gaussians
from render_gaussians import atlas,render,load_fused
BASE=ROOT/'raw/fusion-work/refinement-v7/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v7/tip-fit';OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'raw/fusion-work/refinement-v6/left-hand'
WRIST=np.array([.704,.065,.584]);U=np.array([.68,0,.7332]);U/=np.linalg.norm(U);V=np.array([U[2],0,-U[0]]);BASIS=np.array([U,V,[0,1.,0]])
CENTER=[.76,.095,.642]
VIEWS=[('Palm',[0,-1,0]),('Back',[0,1,0]),('Under / raw +Z',[0,0,1]),('Distal along fingers',[.7,0,.7]),('Palm underside',[.65,-.65,.8]),('Back underside',[.65,.65,.8]),('Outer edge',[1,0,0]),('Thumb edge',[-1,0,0])]
def local(p):return np.einsum('ni,ji->nj',p-WRIST,BASIS)
def world(p):return np.einsum('ni,ij->nj',p,BASIS)+WRIST
def get():
 d=np.load(OLD/'stage-data.npz');report=json.loads((BASE/'report.json').read_text());old=json.loads((OLD/'left-hand-v6-evidence.json').read_text());assert old['sourceToFrontRaw']==report['parts'][6]['sourceToFrontRaw'];assert old['sourceHashes']=={m['file']:m['sha256'] for m in report['sources']}
 out={}
 for source,key in [('front','front'),('back','backAligned')]:
  a=d[key];vis=np.load(OLD/f'native-{source}-visibility.npz');good=(vis['ratio']>.015)&(vis['seen']>.05)&(vis['seen']>vis['oppositeSeen'])&(a[:,10]>-.5)&(np.exp(a[:,7:10]).max(1)<.008)
  q=a[:,3:7];m=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix();j=a[:,7:10].argmin(1);n=m[np.arange(len(m)),:,j];nl=np.einsum('ni,ji->nj',n,BASIS);ratio=np.exp(a[:,7:10].min(1)-np.median(a[:,7:10],axis=1));out[source]={'data':a,'q':local(a[:,:3]),'good':good,'normal':nl,'normalRatio':ratio,'indices':d[source+'Indices'],'visible':vis['seen']}
 return out,report

def diagnostics():
 out,_=get();fig,axs=plt.subplots(2,3,figsize=(19,11))
 for row,(source,o) in enumerate(out.items()):
  for col,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
   g=o['good'];p=o['q'][g];d=o['data'][g];c=np.clip(.5+.2820947918*d[:,11:14],0,1)
   axs[row,col].scatter(p[:,i],p[:,j],c=c,s=3);axs[row,col].set_aspect('equal');axs[row,col].grid();axs[row,col].set_title(source+f' {i}/{j} n{g.sum()}');axs[row,col].set_xlabel(['long','lateral','depth'][i]);axs[row,col].set_ylabel(['long','lateral','depth'][j])
 fig.tight_layout();fig.savefig(OUT/'supported-projections.png',dpi=180)
 for source,o in out.items():np.savez_compressed(OUT/f'{source}-full-supported.npz',**o)
 atlas({k:o['data'][o['good']] for k,o in out.items()},OUT/'supported-multiview.png',CENTER,.30,600,views=VIEWS,title='Original reliable observed source Gaussians before any coverage truncation')

REGIONS={'thumb':(.075,.119,.053,.080),'index':(.142,.21,.019,.044),'middle':(.153,.21,-.006,.014),'ring':(.138,.185,-.027,-.008),'pinky':(.105,.160,-.045,-.029)}
def ellipsoid_fit():
 """Bounded shared cap shape sensitivity; not a fit of opposite surfaces together."""
 D,report=get();records={};results=[]
 for name,(umin,umax,vmin,vmax) in REGIONS.items():
  records[name]={}
  for source,o in D.items():
   q=o['q'];sel=o['good']&(q[:,0]>umin)&(q[:,0]<umax)&(q[:,1]>vmin)&(q[:,1]<vmax)
   points=q[sel];points=points[points[:,0]>points[:,0].max()-.015]
   records[name][source]=points
 for angle in [-2,0,1,2,3,4,5]:
  rot=Rotation.from_rotvec(V*np.deg2rad(angle)).as_matrix();model=[]
  for name,r in records.items():
   front=r['front'];back=local(np.einsum('ij,nj->ni',rot,world(r['back'])-WRIST)+WRIST);points=np.r_[front,back];mean=points.mean(0)
   # Ellipsoid major axis follows the terminal finger slope in longitudinal/depth.
   start=np.r_[mean,[.012,.008,.006],-.3]
   lower=np.r_[mean-[.010,.006,.010],[.005,.004,.003],-.9];upper=np.r_[mean+[.010,.006,.010],[.025,.013,.013],.4]
   def resid(params,prior=True):
    center=params[:3];radii=params[3:6];theta=params[6];ct=np.cos(theta);st=np.sin(theta);q=points-center;localq=np.c_[q[:,0]*ct+q[:,2]*st,q[:,1],-q[:,0]*st+q[:,2]*ct];squared=np.sum((localq/radii)**2,axis=1);grad=2*np.sqrt(np.sum((localq/(radii*radii))**2,axis=1));errors=(squared-1)/np.maximum(grad,1e-5)
    return np.r_[errors,(radii[1]-.008)*.2] if prior else errors
   fit=least_squares(resid,start,bounds=(lower,upper),loss='soft_l1',f_scale=.001,max_nfev=800)
   errors=resid(fit.x,False);row={'digit':name,'angleDegrees':angle,'rms':float(np.sqrt(np.mean(errors**2))),'median':float(np.median(abs(errors))),'centerLocal':fit.x[:3].tolist(),'radii':fit.x[3:6].tolist(),'tilt':float(fit.x[6]),'n':[len(front),len(back)]};model.append(row)
  summary={'angleDegrees':angle,'rms':float(np.mean([r['rms'] for r in model])),'models':model};results.append(summary);print(angle,summary['rms'],[(m['digit'],round(m['rms'],5),np.round(m['radii'],4).tolist()) for m in model],flush=True)
 doc={'purpose':'Sensitivity check using bounded common ellipsoid terminalcap models; comparison alone does not prove real capcenter accuracy','models':results,'regionsLocalUV':REGIONS,'localAxes':BASIS.tolist(),'wrist':WRIST.tolist()};(OUT/'bounded-cap-ellipsoid-sensitivity.json').write_text(json.dumps(doc,indent=2)+'\n')

def digit_displacement(points,amount=1.):
 q=local(points)
 def ss(x):x=np.clip(x,0,1);return x*x*(3-2*x)
 ring=ss((q[:,0]-.115)/.050)*ss((q[:,1]+.034)/.007)*ss((-.006-q[:,1])/.006)
 pinky=ss((q[:,0]-.105)/.040)*ss((q[:,1]+.055)/.008)*ss((-.025-q[:,1])/.005)
 wr=np.maximum(1.,ring+pinky);ring/=wr;pinky/=wr
 delta=ring[:,None]*np.array([.00892,-.00069,-.00593])+pinky[:,None]*np.array([.00322,.00043,-.00141])
 return amount*np.einsum('ni,ij->nj',delta,BASIS)

def deform_digits(data,amount):
 delta=digit_displacement(data[:,:3],amount);changed=np.linalg.norm(delta,axis=1)>1e-12;out=data.copy();d=data[changed];p=d[:,:3].astype(float)
 J=np.broadcast_to(np.eye(3),(len(d),3,3)).copy();eps=1e-5
 for axis in range(3):
  step=np.zeros(3);step[axis]=eps;J[:,:,axis]+=(digit_displacement(p+step,amount)-digit_displacement(p-step,amount))/(2*eps)
 q=d[:,3:7].astype(float);q/=np.linalg.norm(q,axis=1)[:,None];R=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix();axes=np.einsum('nij,njk->nik',J,R)*np.exp(d[:,None,7:10]);vectors,sigma,_=np.linalg.svd(axes,full_matrices=False);flip=np.linalg.det(vectors)<0;vectors[flip,:,0]*=-1;quaternion=Rotation.from_matrix(vectors).as_quat();out[changed,:3]=p+delta[changed];out[changed,3:7]=np.c_[quaternion[:,3],quaternion[:,:3]];out[changed,7:10]=np.log(sigma)
 return out,{'affected':int(changed.sum()),'minimumJacobianDeterminant':float(np.linalg.det(J).min()),'maximumDisplacement':float(np.linalg.norm(delta,axis=1).max()),'maximumJacobianStretch':float(np.linalg.svd(J,compute_uv=False).max()),'amount':amount,'method':'Exploratory smoothly localized back-source ring/pinky cap-center pose correction; full covariance J Sigma J^T'}

def simulate_local(amount):
 from pipeline import BODY_PARTS,coverage_weights,attenuate,load_cleanup_masks
 from cleanup_masks import restore_surface_weights
 D,report=get();f=D['front']['data'];b,audit=deform_digits(D['back']['data'],amount);cfg=json.loads((BASE/'fusion-config.json').read_text());lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks'];meta={s:report['sources'][i] for i,s in enumerate(['front','back'])};masks,_=load_cleanup_masks(cfg['cleanupMasks'],meta,ROOT);restores=[]
 for rule in cfg.get('surfaceRestorations',[]):
  mm,_=load_cleanup_masks([rule['selection']],meta,ROOT);restores.append(mm)
 fw,bw=coverage_weights(f,b,lm,BODY_PARTS[6],cfg['fusion']);output=[];labels=[];sources=[];ids=[]
 for si,(source,dd,w) in enumerate([('front',f,fw),('back',b,bw)]):
  ix=D[source]['indices'];w,restored=restore_surface_weights(w,ix,[m[source] for m in restores],cfg.get('surfaceRestorations',[]));dd,k=attenuate(dd,w,cfg['fusion']['minWeight']);keep=~np.isin(ix[k],masks[source])|restored[k];output.append(dd[keep]);labels.append(np.full(keep.sum(),6,np.uint8));sources.append(np.full(keep.sum(),si,np.uint8));ids.append(ix[k][keep])
 data,l,s,_=load_fused(BASE);ix=np.load(BASE/'source-vertex-indices.npy');m=(l==5)&(data[:,0]>.59)
 return (np.concatenate([data[m],*output]),np.concatenate([l[m],*labels]),np.concatenate([s[m],*sources]),np.concatenate([ix[m],*ids])),audit

def local_trials():
 data,l,s,r=load_fused(BASE);m=np.isin(l,[5,6])&(data[:,0]>.59);captures={'V6 baseline':data[m]};evidence={}
 for amount in [.5,1.]:
  case,report=simulate_local(amount);name=f'local-cap-{amount:g}';np.savez_compressed(OUT/(name+'.npz'),data=case[0],labels=case[1],sources=case[2],indices=case[3]);captures[name]=case[0];evidence[name]=report
  render(case[0],[0,0,1],CENTER,.225,width=1250,height=850).save(OUT/(name+'-underside.png'))
 (OUT/'local-cap-trials.json').write_text(json.dumps(evidence,indent=2)+'\n');atlas(captures,OUT/'local-cap-before-after.png',CENTER,.29,550,views=VIEWS,title='Exploratory back ring/pinky pose correction / covariance correct / full coverage recalculated / not yet accepted')

def export_evidence():
 D,report=get();path=OUT/'bounded-cap-ellipsoid-sensitivity.json'
 if not path.exists():ellipsoid_fit()
 models=next(m['models'] for m in json.loads(path.read_text())['models'] if m['angleDegrees']==0)
 records=[];tr=report['parts'][6]['sourceToFrontRaw'];r0=np.array(tr['rotation']);t0=np.array(tr['translation'])
 for model in models:
  name=model['digit'];umin,umax,vmin,vmax=REGIONS[name];ct=np.cos(model['tilt']);st=np.sin(model['tilt']);center0=np.array(model['centerLocal']);fits=[]
  for depth_multiplier in [.8,1.,1.2]:
   radius=np.array(model['radii']);radius[2]*=depth_multiplier;pair={}
   for source,o in D.items():
    q=o['q'];mask=o['good']&(q[:,0]>umin)&(q[:,0]<umax)&(q[:,1]>vmin)&(q[:,1]<vmax);mask&=q[:,0]>q[mask,0].max()-.015;points=q[mask]
    def residual(center):
     d=points-center;d=np.c_[d[:,0]*ct+d[:,2]*st,d[:,1],-d[:,0]*st+d[:,2]*ct];level=np.sum((d/radius)**2,axis=1);grad=2*np.sqrt(np.sum((d/(radius*radius))**2,axis=1));return(level-1)/np.maximum(grad,1e-5)
    fit=least_squares(residual,center0,bounds=(center0-.015,center0+.015),loss='soft_l1',f_scale=.001)
    front_raw=world(fit.x[None,:])[0];native=front_raw if source=='front' else np.einsum('ij,j->i',r0.T,front_raw-t0)/tr['scale']
    pair[source]={'modeledCenterLocal':fit.x.tolist(),'modeledCenterFrontRaw':front_raw.tolist(),'modeledCenterNative':native.tolist(),'pointCount':len(points),'rmse':float(np.sqrt(np.mean(fit.fun**2))),'originalSourceIndices':o['indices'][mask].astype(int).tolist()}
   fits.append({'depthRadiusMultiplier':depth_multiplier,'backMinusFrontCenterLocal':(np.array(pair['back']['modeledCenterLocal'])-pair['front']['modeledCenterLocal']).tolist(),'fits':pair})
  records.append({'digit':name,'sharedShapeModel':model,'centerSensitivity':fits})
 document={'version':1,'status':'diagnostic trials; no recommended cap-derived rigid transform',
  'sourceHashes':{m['file']:m['sha256'] for m in report['sources']},'baseline':str(BASE.relative_to(ROOT)),
  'part':'left_hand','localFrame':{'origin':WRIST.tolist(),'axesLongitudinalLateralDepth':BASIS.tolist()},
  'sourceCounts':{s:len(o['data']) for s,o in D.items()},'observedQualitySupportCounts':{s:int(o['good'].sum()) for s,o in D.items()},
  'method':'Five corresponding observed terminal cap patches; bounded ellipsoid fit with independent source centers and shared shape, radius sensitivity; original Gaussian covariance and native visibility determine supporting rows.',
  'measurements':records,
  'limitations':['These are modeled cap centers, not manually observed physical landmarks. Partial cap coverage makes inferred center depth dependent on assumed shape.',
   'Naive independent sphere fits were rejected: changing assumed radius by .004 reversed estimated depth mismatch.',
   'A +3 degree whole-hand pitch collapses all five shared cap depth radii to the .003 lower bound and worsens average residual; visually hiding layering is not proof of geometric alignment.',
   'The full localized ring/pinky center correction creates new endpoint separation in palm/back views and max covariance stretch3.03; it is rejected.',
   'Half-strength localized correction provides only modest benefit; the independent shaft-circle audit gives better supported small longitudinal correction.'],
  'recommendation':'Prefer the independently measured common +.0033 longitudinal correction with about -.0004 depth from shaft-circle centerlines; judge fresh coverage multi-angle renders. Do not integrate cap-derived full pose corrections.',
  'productionChanges':False,'sourceToFrontRawUnchanged':tr,
  'proof':[str((OUT/n).relative_to(ROOT)) for n in ['supported-multiview.png','supported-projections.png','local-cap-before-after.png','local-cap-0.5-underside.png','local-cap-1-underside.png']],
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_left_tip_v7.py --ellipsoid --local-trials --evidence'}
 (OUT/'cap-center-evidence.json').write_text(json.dumps(document,indent=2)+'\n')
 print('Wrote',OUT/'cap-center-evidence.json',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--diagnostics',action='store_true');p.add_argument('--ellipsoid',action='store_true');p.add_argument('--local-trials',action='store_true');p.add_argument('--evidence',action='store_true');args=p.parse_args()
 if args.diagnostics:diagnostics()
 if args.ellipsoid:ellipsoid_fit()
 if args.local_trials:local_trials()
 if args.evidence:export_evidence()
