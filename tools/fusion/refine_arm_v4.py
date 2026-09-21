#!/usr/bin/env python3
"""Measure the right upper-arm axial collar and test one-DOF scan registration."""
import os,json,sys,argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mannequin-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import minimize_scalar,least_squares
from pipeline import ROOT,read_scan,rotate,dot,transform_gaussians,fit_rigid,sha256_file
from render_gaussians import atlas,load_fused
BASE=ROOT/'raw/fusion-work/refinement-v4/baseline';OUT=ROOT/'raw/fusion-work/refinement-v4/arm'

def context():
 lf=json.loads((BASE/'front-landmarks.json').read_text())['landmarks'];lb=json.loads((BASE/'back-landmarks.json').read_text())['landmarks'];fits=json.loads((BASE/'feature-transforms.json').read_text())['parts'];return lf,lb,fits

def unwrap():
 lf,lb,fits=context();R=np.array(fits['right_upper_arm']['sourceToFrontRaw']['rotation']);frontaxis=np.array(lf['right_wrist'])-lf['right_elbow'];frontaxis/=np.linalg.norm(frontaxis)
 result={};fig,axes=plt.subplots(2,2,figsize=(14,9))
 for row,(capture,lm) in enumerate([('front',lf),('back',lb)]):
  d,_=read_scan(ROOT/f'raw/mannequin_{capture}_119999.ply');center=np.array(lm['right_elbow']);u=frontaxis if capture=='front' else R.T@frontaxis
  v=np.array([0,1.,0]);v-=u*dot(v,u);v/=np.linalg.norm(v);w=np.cross(u,v);frame=np.array([u,v,w]);p=rotate(d[:,:3]-center,frame);radius=np.sqrt(np.sum(p[:,1:]**2,axis=1));good=(abs(p[:,0])<.09)&(radius>.026)&(radius<.085)&(d[:,10]>-1)&(np.exp(d[:,7:10].max(1))<.015);d=d[good];p=p[good]
  rgb=np.clip(.5+.28209479177387814*d[:,11:14],0,1);lum=dot(rgb,np.array([.2126,.7152,.0722]));angle=np.arctan2(p[:,2],p[:,1]);order=np.argsort(lum)[::-1]
  axes[row,0].scatter(p[order,0],np.rad2deg(angle[order]),c=rgb[order],s=2);axes[row,0].grid();axes[row,0].set_title(capture+' native circumference');axes[row,0].set_xlabel('axis station');axes[row,0].set_ylabel('angle');dark=lum<.24;axes[row,1].scatter(p[dark,0],np.rad2deg(angle[dark]),c=rgb[dark],s=2);axes[row,1].grid();axes[row,1].set_title('Dark boundary samples')
  np.savez_compressed(OUT/f'{capture}-collar-native.npz',data=d,local=p,angle=angle,center=center,frame=frame,rgb=rgb,lum=lum)
 fig.tight_layout();fig.savefig(OUT/'collar-unwrapped.png',dpi=170)

def measure_collar():
 results={};rng=np.random.default_rng(8);fig,axes=plt.subplots(2,3,figsize=(14,8))
 for row,capture in enumerate(['front','back']):
  c=np.load(OUT/f'{capture}-collar-native.npz');p=c['local'];angle=c['angle'];lum=c['lum'];good=(p[:,0]>.005)&(p[:,0]<.040)&(lum<.24);q=p[good];ang=angle[good];bins=np.floor((ang+np.pi)/(2*np.pi)*24).astype(int);best=None
  for _ in range(3000):
   idx=rng.choice(len(q),3,replace=False);design=np.c_[q[idx,1:],np.ones(3)]
   try:v=np.linalg.solve(design,q[idx,0])
   except np.linalg.LinAlgError:continue
   if np.linalg.norm(v[:2])>.40 or not .012<v[2]<.033:continue
   residual=abs(q[:,0]-dot(np.c_[q[:,1:],np.ones(len(q))],v));inside=residual<.0014;counts=np.bincount(bins[inside],minlength=24);score=np.minimum(counts,8).sum()
   if best is None or score>best[0]:best=(score,v,inside)
  _,v,inside=best
  for _ in range(5):
   v=least_squares(lambda v:(q[inside,0]-dot(np.c_[q[inside,1:],np.ones(sum(inside))],v)),v,loss='soft_l1',f_scale=.0006).x
   inside=abs(q[:,0]-dot(np.c_[q[:,1:],np.ones(len(q))],v))<.0018
  normal=np.r_[1.,-v[:2]];normal/=np.linalg.norm(normal);nativeaxis=c['frame'].T@normal;centerlocal=np.array([v[2],0,0]);nativecenter=c['center']+c['frame'].T@centerlocal
  # The center/radius of the full collar rim, in its own plane. Fit an ellipse
  # to radial ring support, constraining its offset to the previously observed arm center.
  radial=q[inside,1:];
  def radius_res(vv):return (np.sqrt(np.sum(((radial-vv[:2])/np.exp(vv[2:4]))**2,axis=1))-1)*.055
  fv=least_squares(radius_res,[0,0,np.log(.061),np.log(.057)],bounds=([-.025,-.025,np.log(.04),np.log(.04)],[.025,.025,np.log(.085),np.log(.085)]),loss='soft_l1',f_scale=.002).x
  centerlocal=np.array([v[2]+v[0]*fv[0]+v[1]*fv[1],fv[0],fv[1]]);nativecenter=c['center']+c['frame'].T@centerlocal
  pts=c['data'][good][inside,:3];out={'pivot':nativecenter.tolist(),'axis':nativeaxis.tolist(),'localPlane':v.tolist(),'radialCenter':fv[:2].tolist(),'radialRadii':np.exp(fv[2:]).tolist(),'inlierCount':int(sum(inside)),'planeMedianResidual':float(np.median(abs(q[inside,0]-dot(np.c_[q[inside,1:],np.ones(sum(inside))],v)))),'occupiedAngularBins':int(sum(np.bincount(bins[inside],minlength=24)>0))};results[capture]=out;print(capture,json.dumps(out),flush=True)
  np.savez_compressed(OUT/f'{capture}-collar-rim.npz',points=pts,pivot=nativecenter,axis=nativeaxis)
  for col,(xx,yy) in enumerate([(0,1),(0,2),(1,2)]):
   ax=axes[row,col];ax.scatter(c['data'][:,xx],c['data'][:,yy],c=c['rgb'],s=.4);ax.scatter(pts[:,xx],pts[:,yy],c='magenta',s=2);ax.scatter(nativecenter[xx],nativecenter[yy],c='lime',s=20);ax.set_aspect('equal');ax.grid();ax.set_title(capture+' '+ 'XYZ'[xx]+'/'+ 'XYZ'[yy])
 fig.tight_layout();fig.savefig(OUT/'collar-rim-fit.png',dpi=180);(OUT/'collar-measurements.json').write_text(json.dumps(results,indent=2))

def arm_data():
 from pipeline import clean_scan,BODY_PARTS
 lf,lb,fits=context();cfg=json.loads((BASE/'fusion-config.json').read_text());cache=OUT/'native-clean.npz'
 if cache.exists():c=np.load(cache);return c['front'],c['back'],c['fl'],c['bl'],lf,lb,fits,cfg
 records={}
 for name,lm in [('front',lf),('back',lb)]:
  data,_=read_scan(ROOT/cfg[name]);data,labels,_,indices=clean_scan(data,lm,cfg['filter'],BODY_PARTS,cfg.get('exclusions',{}).get(name,[]),cfg.get('segmentationOverrides',{}).get(name,[]),cfg['filter'].get('partOverrides',{}).get(name,{}));mask=np.isin(labels,[0,10,11,12]);records[name]=data[mask];records['fl' if name=='front' else 'bl']=labels[mask]
 np.savez_compressed(cache,**records);return records['front'],records['back'],records['fl'],records['bl'],lf,lb,fits,cfg

def axial_fits():
 from pipeline import BODY_PARTS,apply_transform,part_frame
 from refine_limbs import proxy,evaluate
 f,b,fl,bl,lf,lb,fits,cfg=arm_data();measure=json.loads((OUT/'collar-measurements.json').read_text());pivot=np.array(measure['front']['pivot']);axis=np.array(measure['front']['axis'])
 parent=fits['right_upper_arm']['sourceToFrontRaw'];rp=np.array(parent['rotation']);tp=np.array(parent['translation']);scale=parent['scale'];old=fits['right_forearm']['sourceToFrontRaw'];ro=np.array(old['rotation']);to=np.array(old['translation']);oldh=fits['right_hand']['sourceToFrontRaw'];rh=np.array(oldh['rotation']);th=np.array(oldh['translation'])
 # Use the measured collar as the material boundary; no invented intermediate hinge.
 for data,lab,name in [(f,fl,'front'),(b,bl,'back')]:
  pp=np.array(measure[name]['pivot']);aa=np.array(measure[name]['axis']);delta=data[:,:3]-pp;station=dot(delta,aa);rad=np.linalg.norm(delta-station[:,None]*aa,axis=1);m=np.isin(lab,[10,11])&(abs(station)<.11)&(rad<.11);lab[m]=np.where(station[m]<=0,10,11)
 lf['right_elbow']=measure['front']['pivot'];lb['right_elbow']=measure['back']['pivot']
 sets={}
 for part_i in [11,12]:
  part=BODY_PARTS[part_i];fp=proxy(f[fl==part_i],lf,part);bp=proxy(b[bl==part_i],lb,part);depth=part_frame(lf,part)[3];mask=abs(dot(fp[1],depth))<.90;sets[part_i]=(tuple(x[mask] for x in fp[:3]),bp[:3],depth)
 def trans(theta):
  q=Rotation.from_rotvec(axis*np.deg2rad(theta)).as_matrix();r=q@rp;t=q@(tp-pivot)+pivot;dh=r@ro.T;return r,t,dh@rh,dh@(th-to)+t
 def evaluate_transforms(r,t,hr,ht):
  metrics={};total=0
  for i,rr,tt in [(11,r,t),(12,hr,ht)]:
   (fp,fn,fc),(bp,bn,bc),depth=sets[i];m=apply_transform(bp,rr,tt,scale);n=rotate(bn,rr);valid=abs(dot(n,depth))<.90;ev=evaluate(fp,fn,fc,m[valid],n[valid],bc[valid]);metrics[BODY_PARTS[i][0]]=ev;total+=sum(x['cost'] for x in ev)*(1 if i==11 else .35)
  return total,metrics
 def cost(theta):
  return evaluate_transforms(*trans(theta))
 angles=np.arange(-50,20.1,2.5);scores=[]
 for a in angles:
  score,_=cost(a);scores.append(score);print('angle',a,'cost',score,flush=True)
 k=int(np.argmin(scores));fit=minimize_scalar(lambda x:cost(x)[0],bounds=(angles[max(0,k-1)],angles[min(len(angles)-1,k+1)]),method='bounded',options={'xatol':.02});theta=float(fit.x);r,t,hr,ht=trans(theta);score,metrics=cost(theta)
 result={'method':'measured axial collar, one rotational degree of freedom; zero fitted swing and zero fitted translation',
  'mechanicalConstraint':{'parentPart':'right_upper_arm','childPart':'right_forearm','pivotFrontRaw':pivot.tolist(),'axisFrontRaw':axis.tolist(),'twistDegrees':theta,'formula':'Rc=Q Rp; tc=Q(tp-pivot)+pivot; Q=axisAngle(axis,twist)','independentTranslation':[0,0,0],'independentSwingDegrees':0},
  'collarMeasurements':measure,'sourceToFrontRaw':{'scale':scale,'rotation':r.tolist(),'translation':t.tolist()},
  'handSourceToFrontRaw':{'scale':scale,'rotation':hr.tolist(),'translation':ht.tolist()},'handConstraint':'preserve baseline hand-to-forearm relative transform exactly',
  'parentSourceToFrontRaw':parent,'baselineForearm':old,'baselineHand':oldh,'fitCost':score,'metrics':metrics,'baselineMetricsAtCorrectedCollar':evaluate_transforms(ro,to,rh,th)[1],'angleSweep':{'degrees':angles.tolist(),'cost':scores},
  'segmentationRule':{'parts':['right_upper_arm','right_forearm'],'maximumAbsStation':.11,'maximumRadius':.11,'proximalPart':'right_upper_arm','distalPart':'right_forearm','planes':measure}}
 (OUT/'axial-fit.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'resegmented-native.npz',front=f,back=b,fl=fl,bl=bl)
 print('one-axis candidate',theta,score,flush=True)
 return result


def render_axial():
 from pipeline import BODY_PARTS,coverage_weights,attenuate
 f,b,fl,bl,lf,lb,fits,cfg=arm_data();c=np.load(OUT/'resegmented-native.npz');f,b,fl,bl=c['front'],c['back'],c['fl'],c['bl'];result=json.loads((OUT/'axial-fit.json').read_text());measure=result['collarMeasurements'];lf['right_elbow']=measure['front']['pivot'];lb['right_elbow']=measure['back']['pivot'];out=[];sources=[];labels=[]
 for i in [0,10,11,12]:
  part=BODY_PARTS[i];tr=result['sourceToFrontRaw'] if i==11 else result['handSourceToFrontRaw'] if i==12 else fits[part[0]]['sourceToFrontRaw'];ff=f[fl==i];bb=transform_gaussians(b[bl==i],np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);fw,bw=coverage_weights(ff,bb,lf,part,cfg['fusion']);ff,_=attenuate(ff,fw,cfg['fusion']['minWeight']);bb,_=attenuate(bb,bw,cfg['fusion']['minWeight']);out.extend([ff,bb]);sources.extend([np.zeros(len(ff),np.uint8),np.ones(len(bb),np.uint8)]);labels.extend([np.full(len(ff),i),np.full(len(bb),i)])
 data=np.concatenate(out);source=np.concatenate(sources);label=np.concatenate(labels);mask=data[:,0]<-.15;data=data[mask];source=source[mask];label=label[mask];np.savez_compressed(OUT/'candidate.npz',data=data,sources=source,labels=label)
 if not (OUT/'baseline.npz').exists():
  bd,bl,bs,_=load_fused(BASE);m=np.isin(bl,[0,10,11,12])&(bd[:,0]<-.15);np.savez_compressed(OUT/'baseline.npz',data=bd[m],labels=bl[m],sources=bs[m])
 baseline=np.load(OUT/'baseline.npz')['data'];views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Outer side',[-1,0,0]),('Inner side',[1,0,0]),('Axial end',[-.7,.15,.7]),('Oblique',[-1,-1,-.2])]
 atlas({'Baseline':baseline,'Axial only':data},OUT/'axial-collar-comparison.png',[-.37,-.037,.27],.43,560,views=views,title='Measured collar axis only; shoulder pad fixed, hand relative pose preserved')
 atlas({'Baseline':baseline,'Axial only':data},OUT/'axial-arm-comparison.png',[-.56,.0,.44],.98,550,views=views,title='Full right arm and wrist continuity under one-DOF collar constraint')
 # Keep native source quality visible: these rows do not use fusion attenuation.
 rows={'Front capture':f[np.isin(fl,[10,11,12])]}
 for name in ['Parent only','Axial only','Baseline forearm']:
  source_parts=[]
  for i,key in [(10,'parentSourceToFrontRaw'),(11,'sourceToFrontRaw'),(12,'handSourceToFrontRaw')]:
   if name=='Parent only':key='parentSourceToFrontRaw'
   elif name=='Baseline forearm':key={10:'parentSourceToFrontRaw',11:'baselineForearm',12:'baselineHand'}[i]
   tr=result[key];source_parts.append(transform_gaussians(b[bl==i],np.array(tr['rotation']),np.array(tr['translation']),tr['scale']))
  rows[name]=np.concatenate(source_parts)
 atlas(rows,OUT/'axial-sources.png',[-.55,0,.43],.9,700,views=[('Front',[0,-1,0]),('Outer',[-1,0,0]),('Back',[0,1,0])],title='Original source shapes; strict axial candidate compared with shoulder-parent and unconstrained baseline')

def evidence():
 """Export explicit mechanical linkage and measured source provenance."""
 r=json.loads((OUT/'axial-fit.json').read_text());c=r['mechanicalConstraint'];p=np.array(c['pivotFrontRaw']);axis=np.array(c['axisFrontRaw']);q=Rotation.from_rotvec(axis*np.deg2rad(c['twistDegrees'])).as_matrix()
 parent=r['parentSourceToFrontRaw'];rp=np.array(parent['rotation']);tp=np.array(parent['translation']);rr=np.array(r['sourceToFrontRaw']['rotation']);tt=np.array(r['sourceToFrontRaw']['translation'])
 assert np.max(abs(rr-q@rp))<1e-12
 assert np.max(abs(tt-(q@(tp-p)+p)))<1e-12
 ro=np.array(r['baselineForearm']['rotation']);to=np.array(r['baselineForearm']['translation']);rh=np.array(r['handSourceToFrontRaw']['rotation']);th=np.array(r['handSourceToFrontRaw']['translation']);rhos=np.array(r['baselineHand']['rotation']);thos=np.array(r['baselineHand']['translation'])
 assert np.max(abs(rr.T@rh-ro.T@rhos))<1e-12
 assert np.max(abs(rr.T@(th-tt)-ro.T@(thos-to)))<1e-12
 backp=np.array(r['collarMeasurements']['back']['pivot']);backaxis=np.array(r['collarMeasurements']['back']['axis']);s=parent['scale']
 r['validation']={'rotationFormulaMaxError':float(np.max(abs(rr-q@rp))),'translationFormulaMaxError':float(np.max(abs(tt-(q@(tp-p)+p)))),'handRelativeRotationMaxError':float(np.max(abs(rr.T@rh-ro.T@rhos))),'handRelativeTranslationMaxError':float(np.max(abs(rr.T@(th-tt)-ro.T@(thos-to)))),'parentCollarCenterResidual':float(np.linalg.norm(s*rp@backp+tp-p)),'baselineChildCollarCenterResidual':float(np.linalg.norm(s*ro@backp+to-p)),'constrainedChildCollarCenterResidual':float(np.linalg.norm(s*rr@backp+tt-p)),'parentCollarNormalDifferenceDegrees':float(np.rad2deg(np.arccos(np.clip(dot(rp@backaxis,axis),-1,1))))}
 r['version']=1;r['sourceHashes']=json.loads((BASE/'feature-transforms.json').read_text())['sourceHashes'];r['frame']='Source PLY native XYZ to front native XYZ. Distances use uncalibrated source units.'
 r['baseline']={'directory':str(BASE.relative_to(ROOT)),'files':{n:sha256_file(BASE/n) for n in ['feature-transforms.json','front-landmarks.json','back-landmarks.json','fusion-config.json']}}
 r['mechanicalConstraint']['pivotPrepared']=(p*np.array([1,-1,-1])).tolist();r['mechanicalConstraint']['axisPrepared']=(axis*np.array([1,-1,-1])).tolist()
 r['observations']=[
  'The dark circumferential collar is about 0.0215 source units distal to the previous capsule boundary. RANSAC and robust plane fitting select the thin dark seam across 24 angular sectors in each native scan; angular support prevents fitting a single visible edge.',
  'The shoulder pad fit is retained exactly. Its independent mapping puts the back native collar center 0.00135 source units from the measured front center; no fitted collar-center translation is introduced.',
  'The child has exactly one fitted degree of freedom: rotation about the measured front collar normal, directed from shoulder toward wrist. No swing, bend or independent translation is allowed.',
  'The right hand is carried through the forearm delta, preserving its baseline transform relative to the forearm. Original source wrist/hand geometry is not deformed.',
  'The large front training-panel border is mostly absent from the back capture. The precise twist is therefore selected by normal-compatible overlap under the mechanical constraint, not claimed as an independently observed panel-to-panel angle.',
  'The unconstrained baseline has slightly lower aggregate nearest-surface error, especially on the hand. The constrained fit is selected for the correct axial mechanism, substantially lower collar-center mismatch and smoother actual collar renders. Source contact-side corruption remains visible from the inner side.',
  'Six-angle full-covariance Gaussian renders were inspected for collar continuity, unchanged pad placement, panel silhouette and wrist continuity. Source-separated front/outer/back renders distinguish original corruption from fusion errors.'
 ]
 r['reproduce']={
  'command':'.venv-fusion/bin/python -B tools/fusion/refine_arm_v4.py --all',
  'stages':['--measure: read original scans, unwrap circular seam, fit native ring plane and ellipse center','--fit: use frozen baseline settings and search a single axial angle; carry the hand relative transform','--render: actual anisotropic Gaussian comparison against frozen v3 baseline','--evidence: verify mechanical equations and export portable evidence'],
  'tuning':'Adjust native seam selection in unwrap/measure_collar or the overlap search in axial_fits; re-run all. Do not independently edit the child transform matrix.'}
 r['reviewImages']=[str((OUT/n).relative_to(ROOT)) for n in ['collar-rim-fit.png','collar-unwrapped.png','axial-collar-comparison.png','axial-arm-comparison.png','axial-sources.png']]
 (OUT/'arm-feature-evidence.json').write_text(json.dumps(r,indent=2)+'\n')
 print(json.dumps(r['validation'],indent=2),flush=True)

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 for name in ['all','measure','fit','render','evidence']:parser.add_argument('--'+name,action='store_true')
 args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 hashes=json.loads((BASE/'feature-transforms.json').read_text())['sourceHashes'];cfg=json.loads((BASE/'fusion-config.json').read_text())
 for capture in ['front','back']:
  if sha256_file(ROOT/cfg[capture])!=hashes[capture]:raise ValueError('Source changed: '+capture)
 all_stages=args.all or not any(vars(args).values())
 if all_stages or args.measure:unwrap();measure_collar()
 if all_stages or args.fit:axial_fits()
 if all_stages or args.render:render_axial()
 if all_stages or args.evidence:evidence()

if __name__=='__main__':main()
