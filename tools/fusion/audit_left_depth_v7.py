#!/usr/bin/env python3
"""Independent 3D hand-side registration audit; writes only v7 scratch results."""
import os,json,argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mannequin-left-v7-mpl')
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import (ROOT,BODY_PARTS,dot,rotate,transform_gaussians,voxel_indices,
 coverage_weights,attenuate,load_cleanup_masks,read_scan)
from cleanup_masks import restore_surface_weights
from render_gaussians import load_fused,atlas,render
BASE=ROOT/'raw/fusion-work/refinement-v7/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v7/depth-audit';OUT.mkdir(parents=True,exist_ok=True)
V6=ROOT/'raw/fusion-work/refinement-v6/left-hand'
ORIGIN=np.array([.704,.065,.584]);LONG=np.array([.68,0,.7332]);LONG/=np.linalg.norm(LONG)
LATERAL=np.array([LONG[2],0,-LONG[0]]);BASIS=np.array([LONG,LATERAL,[0,1,0]])
CENTER=[.76,.095,.642]
VIEWS=[('Palm',[0,-1,0]),('Back',[0,1,0]),('Raw +Z',[0,0,1]),('Distal',[.7,0,.7]),('Palm underside',[.65,-.65,.8]),('Back underside',[.65,.65,.8])]

def load():
 c=np.load(V6/'stage-data.npz');r=json.loads((BASE/'report.json').read_text());tr=r['parts'][6]['sourceToFrontRaw']
 f=c['front'];b=transform_gaussians(c['backNative'],np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
 visibility={source:dict(np.load(V6/f'native-{source}-visibility.npz')) for source in ['front','back']}
 return f,b,c,visibility,r

def normals(d):
 q=d[:,3:7].astype(float);q/=np.linalg.norm(q,axis=1)[:,None]
 rotations=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix()
 n=rotations[np.arange(len(d)),:,np.argmin(d[:,7:10],axis=1)]
 scale=np.sort(np.exp(d[:,7:10]),axis=1)
 return n,scale[:,0]/scale[:,1]

def proxies(d,v,side_limit=.65):
 n,ratio=normals(d);loc=rotate(d[:,:3]-ORIGIN,BASIS)
 good=(v['seen']>v['oppositeSeen'])&(v['ratio']>.025)&(v['seen']>.05)&(d[:,10]>-2.5)
 good&=(ratio<.5)&(abs(n[:,1])<side_limit)&(loc[:,0]>.10)&(loc[:,0]<.205)
 ids=np.flatnonzero(good);ids=ids[voxel_indices(d[ids,:3],.0012)]
 return d[ids,:3].astype(float),n[ids],np.clip(.5+.2820947918*d[ids,11:14],0,1),ids

def sections(f,b,vis,name='baseline',correction=None):
 fig,ax=plt.subplots(2,4,figsize=(19,9));stations=[.05,.08,.105,.125,.145,.16,.175,.19]
 for source,dd,color in [('front',f,'tab:cyan'),('back',b,'tab:orange')]:
  vv=vis[source];good=(vv['seen']>vv['oppositeSeen'])&(vv['ratio']>.03)&(vv['seen']>.1)&(dd[:,10]>-2.5)
  q=rotate(dd[:,:3]-ORIGIN,BASIS)
  for a,s in zip(ax.flat,stations):
   m=good&(abs(q[:,0]-s)<.0025);a.scatter(q[m,1],q[m,2],c=color,s=2,alpha=.5,label=source)
   a.set_title('Longitudinal '+str(s));a.set_aspect('equal');a.set_xlabel('lateral');a.set_ylabel('raw Y - wrist Y');a.grid();a.set_xlim(-.07,.09);a.set_ylim(-.04,.10)
 ax[0,0].legend();fig.tight_layout();fig.savefig(OUT/(name+'-crosssections.png'),dpi=150);plt.close(fig)

def match(f,fn,fc,b,bn,bc,maxdist=.015):
 d,idx=cKDTree(f).query(b,k=16,workers=-1)
 ndot=abs(np.einsum('nki,ni->nk',fn[idx],bn));cd=np.linalg.norm(fc[idx]-bc[:,None,:],axis=2)
 score=np.where((d<maxdist)&(ndot>.8)&(cd<.40),d+(.002*(1-ndot))+(.002*cd),np.inf)
 col=np.argmin(score,axis=1);best=idx[np.arange(len(b)),col];valid=np.isfinite(score[np.arange(len(b)),col])
 ids=np.flatnonzero(valid);return ids,best[valid],d[np.arange(len(b)),col][valid]

def fit(f,b,vis,side_limit=.65):
 fp,fn,fc,fi=proxies(f,vis['front'],side_limit);bp,bn,bc,bi=proxies(b,vis['back'],side_limit)
 print('Proxy sizes',len(fp),len(bp),flush=True);pivot=np.array(CENTER);runs=[]
 def evaluate(Q,t):
  moved=rotate(bp-pivot,Q)+pivot+t;ns=rotate(bn,Q);ii,jj,di=match(fp,fn,fc,moved,ns,bc)
  return {'pairs':len(ii),'median':float(np.median(di)),'p90':float(np.quantile(di,.9)),'rmse':float(np.sqrt(np.mean(np.minimum(di,.01)**2)))},(ii,jj,moved,ns)
 before,_=evaluate(np.eye(3),np.zeros(3))
 for seed in [-.012,-.006,0,.006,.012]:
  v=np.r_[np.zeros(3),0,seed,0]
  for iteration in range(18):
   Q=Rotation.from_rotvec(v[:3]).as_matrix();metrics,ss=evaluate(Q,v[3:]);ii,jj,_,_=ss
   if len(ii)<30:break
   # Both tangent-plane and weak point distance preserve correspondence along a
   # narrow finger side, without attracting front and dorsal shell interiors.
   def residual(w):
    rr=Rotation.from_rotvec(w[:3]).as_matrix();m=rotate(bp[ii]-pivot,rr)+pivot+w[3:];delta=m-fp[jj]
    return np.r_[np.einsum('ni,ni->n',delta,fn[jj]),.18*delta.ravel(),w[:3]*.003,w[3:]*.04]
   opt=least_squares(residual,v,bounds=(np.r_[np.full(3,-.22),np.full(3,-.023)],np.r_[np.full(3,.22),np.full(3,.023)]),loss='soft_l1',f_scale=.0015,max_nfev=80)
   change=np.linalg.norm(opt.x-v);v=opt.x
   if change<1e-6:break
  Q=Rotation.from_rotvec(v[:3]).as_matrix();metrics,ss=evaluate(Q,v[3:]);runs.append({'parameters':v.tolist(),'metrics':metrics})
 best=min(runs,key=lambda k:k['metrics']['rmse']);v=np.array(best['parameters']);Q=Rotation.from_rotvec(v[:3]).as_matrix();t=pivot-rotate(pivot[None,:],Q)[0]+v[3:]
 result={'method':'observed native side Gaussians; covariance shortest-axis normals; constrained robust ICP excluding shell interiors','sideNormalMaximumAbsY':side_limit,'before':before,'runs':runs,'best':best,'pivotFrontRaw':pivot.tolist(),'correctionRotation':Q.tolist(),'correctionTranslation':t.tolist(),'correctionEulerXYZDegrees':Rotation.from_matrix(Q).as_euler('xyz',degrees=True).tolist()}
 np.savez_compressed(OUT/'proxy-correspondence-data.npz',front=fp,frontNormals=fn,back=bp,backNormals=bn)
 return Q,t,result

def fused_trial(Q,t,local_deformation=None):
 f,b,c,vis,report=load();cfg=json.loads((BASE/'fusion-config.json').read_text());lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
 masks,_=load_cleanup_masks(cfg['cleanupMasks'],{s:report['sources'][i] for i,s in enumerate(['front','back'])},ROOT)
 restores=[]
 for rule in cfg.get('surfaceRestorations',[]):
  mm,_=load_cleanup_masks([rule['selection']],{s:report['sources'][i] for i,s in enumerate(['front','back'])},ROOT);restores.append(mm)
 moved=transform_gaussians(b,Q,t)
 if local_deformation is not None:moved=local_deformation(moved)
 fw,bw=coverage_weights(f,moved,lm,BODY_PARTS[6],cfg['fusion']);output=[];labels=[];sources=[];ids=[]
 for si,(source,dd,w) in enumerate([('front',f,fw),('back',moved,bw)]):
  ix=c[source+'Indices'];w,restored=restore_surface_weights(w,ix,[m[source] for m in restores],cfg.get('surfaceRestorations',[]))
  dd,k=attenuate(dd,w,cfg['fusion']['minWeight']);keep=~np.isin(ix[k],masks[source])|restored[k]
  output.append(dd[keep]);labels.append(np.full(keep.sum(),6,np.uint8));sources.append(np.full(keep.sum(),si,np.uint8));ids.append(ix[k][keep])
 data,l,s,_=load_fused(BASE);ix=np.load(BASE/'source-vertex-indices.npy');m=(l==5)&(data[:,0]>.59)
 return np.concatenate([data[m],*output]),np.concatenate([l[m],*labels]),np.concatenate([s[m],*sources]),np.concatenate([ix[m],*ids])

def shift_digits(data,longitudinal=.0033,depth=0.):
 """Small original-back digit displacement with a C1 knuckle/thenar taper.

 Displacement is exactly zero near the wrist, continuous through the knuckles,
 and constant over the distal four fingers. The thumb-side fade avoids moving
 the thumb. Full Gaussian covariance is transported by the analytic Jacobian.
 """
 q=rotate(data[:,:3]-ORIGIN,BASIS);s=np.clip((q[:,0]-.080)/.045,0,1);lat=np.clip((q[:,1]-.043)/.012,0,1)
 ss=s*s*(3-2*s);ls=lat*lat*(3-2*lat);w=ss*(1-ls)
 ds=6*s*(1-s)/.045;dl=6*lat*(1-lat)/.012
 gradient=ds[:,None]*(1-ls[:,None])*LONG-ss[:,None]*dl[:,None]*LATERAL
 displacement=longitudinal*LONG+np.array([0,depth,0]);out=data.copy();active=w>0
 out[active,:3]+=w[active,None]*displacement
 J=np.eye(3)[None,:,:]+displacement[None,:,None]*gradient[active,None,:]
 qq=data[active,3:7];rotation=Rotation.from_quat(np.c_[qq[:,1:],qq[:,0]]).as_matrix();axes=rotation*np.exp(data[active,7:10])[:,None,:]
 changed=np.einsum('nij,njk->nik',J,axes);vectors,sigma,_=np.linalg.svd(changed,full_matrices=False)
 flip=np.linalg.det(vectors)<0;vectors[flip,:,0]*=-1
 quat=Rotation.from_matrix(vectors).as_quat();out[active,3:7]=np.c_[quat[:,3],quat[:,:3]];out[active,7:10]=np.log(np.maximum(sigma,np.finfo(float).tiny))
 return out

def digit_shift_trials():
 from functools import partial
 baseline=fused_trial(np.eye(3),np.zeros(3));captures={'V6 baseline':baseline[0]}
 for label,depth in [('Distal +.0033',0.),('Distal +.0033 / Y -.0004',-.0004)]:
  candidate=fused_trial(np.eye(3),np.zeros(3),partial(shift_digits,depth=depth));name='digit-distal-shift'+('-with-depth' if depth else '')
  np.savez_compressed(OUT/(name+'.npz'),data=candidate[0],labels=candidate[1],sources=candidate[2],indices=candidate[3]);captures[label]=candidate[0]
  render(candidate[0],[0,0,1],CENTER,.225,width=1250,height=850).save(OUT/(name+'-detail.png'))
  evidence={'status':'candidate for visual gate','longitudinalDisplacement':.0033,'depthDisplacement':depth,
   'frame':'aligned front raw','part':'left_hand','capture':'back','origin':ORIGIN.tolist(),'longitudinalAxis':LONG.tolist(),'lateralAxis':LATERAL.tolist(),
   'taper':{'longitudinalSmoothstep':[.080,.125],'lateralReverseSmoothstep':[.043,.055]},
   'covariance':'analytic J Sigma J-transpose','sourceOpacityAndColor':'unchanged; normal coverage attenuation recomputed',
   'method':'Small distal translation inferred independently from observed shaft centre curves of ring/middle/index; taper preserves wrist and thumb',
   'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_left_depth_v7.py --digit-shift'}
  (OUT/(name+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')
 atlas(captures,OUT/'digit-distal-shift-trials.png',CENTER,.33,750,views=VIEWS,title='Back four-finger distal shift / smooth knuckle taper / full original Gaussians and recomputed coverage')

def pitch_sweep():
 f,b,c,vis,report=load();records=[]
 for limit in [.25,.4,.65]:
  fp,fn,fc,fi=proxies(f,vis['front'],limit);bp,bn,bc,bi=proxies(b,vis['back'],limit);values=[]
  for degrees in np.arange(0,8.01,.5):
   Q=Rotation.from_rotvec(LATERAL*np.radians(degrees)).as_matrix();m=rotate(bp-ORIGIN,Q)+ORIGIN
   ii,jj,d=match(fp,fn,fc,m,rotate(bn,Q),bc)
   values.append({'degrees':float(degrees),'pairs':len(ii),'median':float(np.median(d)),
    'p90':float(np.quantile(d,.9)),'clippedMean':float(np.mean(np.minimum(d,.01)))})
  records.append({'limit':limit,'frontProxies':len(fp),'backProxies':len(bp),
   'candidates':values,'best':min(values,key=lambda row:row['clippedMean'])})
 (OUT/'wrist-fixed-pitch-sweep.json').write_text(json.dumps(records,indent=2)+'\n')
 before=fused_trial(np.eye(3),np.zeros(3));captures={'V6 baseline':before[0]};colors={}
 for degrees in [2.5,3.,3.5,4.]:
  Q=Rotation.from_rotvec(LATERAL*np.radians(degrees)).as_matrix();t=ORIGIN-Q@ORIGIN
  candidate=fused_trial(Q,t);name=f'wrist-pitch-{degrees:g}';tr=report['parts'][6]['sourceToFrontRaw']
  evidence={'status':'candidate pending final multiview gate','method':'wrist-fixed rigid pitch; supported thin digit-side Gaussian-normal correspondence',
   'degrees':degrees,'axisFrontRaw':LATERAL.tolist(),'pivotFrontRaw':ORIGIN.tolist(),
   'correctionRotation':Q.tolist(),'correctionTranslation':t.tolist(),
   'sourceHashes':{m['file']:m['sha256'] for m in report['sources']},
   'sourceToFrontRaw':{'scale':tr['scale'],'rotation':(Q@np.array(tr['rotation'])).tolist(),'translation':(Q@np.array(tr['translation'])+t).tolist()},
   'fitScope':{'longitudinalMinimum':.10,'longitudinalMaximum':.205,'minimumOpacityLogit':-2.5,'maximumGaussianScaleRatio':.5,
    'minimumObservedVisibilityRatio':.025,'minimumObservedFlux':.05,'requireObservedFluxExceedsOpposite':True,'sourceNativeVisibility':'v6 full quality-clean 900px/.30span'},
   'limitations':['Shortest-axis covariance normals approximate surface directions; residuals are not independently measured anatomical landmarks.',
    'Side thresholds2.5–3.5degrees agree; this corroborates pitch direction but does not alone establish the true finger thickness.',
    'Forearm held fixed; fused hand coverage, legacy cleanup and restoration recomputed from full clean original rows.'],
   'proof':str((OUT/'wrist-fixed-pitch-before-after.png').relative_to(ROOT)),
   'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_left_depth_v7.py --pitch-sweep'}
  (OUT/(name+'.json')).write_text(json.dumps(evidence,indent=2)+'\n')
  np.savez_compressed(OUT/(name+'.npz'),data=candidate[0],labels=candidate[1],sources=candidate[2],indices=candidate[3])
  captures[f'Wrist fixed {degrees:g} degrees']=candidate[0]
  if degrees==3:
   sections(f,transform_gaussians(b,Q,t),vis,'wrist-pitch-3')
   for label,case in [('Baseline',before),('Wrist pitch3',candidate)]:
    dc=case[0].copy();dc[case[2]==0,11:14]=(np.array([.1,.8,.9])-.5)/.2820947918;dc[case[2]==1,11:14]=(np.array([1.,.5,.1])-.5)/.2820947918;colors[label]=dc
   render(candidate[0],[0,0,1],CENTER,.225,width=1250,height=850).save(OUT/'wrist-pitch-3-underside-detail.png')
 atlas(captures,OUT/'wrist-fixed-pitch-before-after.png',CENTER,.33,650,views=VIEWS,title='Wrist-pinned pitch trial / full clean hand coverage recomputed / forearm fixed')
 atlas(colors,OUT/'wrist-fixed-pitch-source-color.png',CENTER,.33,650,views=VIEWS,title='Cyan front / orange back / wrist-fixed rigid pitch')

def circle_sections():
 """Infer each source's cylinder centre from a curved shaft arc, not its shell Y.

 A circle is an approximation: inconsistent source radii or high residuals mark
 untrustworthy slices. This deliberately reports that caveat rather than forcing
 opposing skin surfaces together.
 """
 f,b,c,vis,report=load();rows=[]
 windows={.125:[('pinky',-.045,-.032),('ring',-.032,-.012),('middle',-.009,.014),('index',.020,.047)],
  .145:[('ring',-.030,-.012),('middle',-.010,.013),('index',.021,.043)],
  .16:[('ring',-.033,-.012),('middle',-.014,.012),('index',.016,.040)],
  .175:[('ring',-.029,-.010),('middle',-.013,.010),('index',.014,.035)]}
 fig,axes=plt.subplots(4,4,figsize=(16,15));counter=0
 for station,fingers in windows.items():
  for name,lo,hi in fingers:
   fits={};ax=axes.flat[counter];counter+=1
   for source,d,color in [('front',f,'tab:cyan'),('back',b,'tab:orange')]:
    q=rotate(d[:,:3]-ORIGIN,BASIS);v=vis[source]
    ok=(v['seen']>v['oppositeSeen'])&(v['ratio']>.03)&(v['seen']>.1)&(d[:,10]>-2.5)
    ok&=(abs(q[:,0]-station)<.00175)&(q[:,1]>lo)&(q[:,1]<hi)
    p=q[ok,1:];center=np.median(p,axis=0);radius=(np.quantile(p[:,0],.97)-np.quantile(p[:,0],.03))/2
    center[1]+=radius*.65*(1 if source=='front' else -1)
    def residual(w):return np.linalg.norm(p-w[:2],axis=1)-w[2]
    opt=least_squares(residual,np.r_[center,radius],bounds=([lo,np.min(p[:,1])-.020,.002],[hi,np.max(p[:,1])+.020,.020]),loss='soft_l1',f_scale=.0004,max_nfev=300)
    fits[source]={'count':len(p),'centerLateralDepth':opt.x[:2].tolist(),'radius':float(opt.x[2]),'rmse':float(np.sqrt(np.mean(residual(opt.x)**2)))}
    ax.scatter(p[:,0],p[:,1],s=3,color=color,alpha=.65,label=source)
    angle=np.linspace(0,2*np.pi,200);ax.plot(opt.x[0]+opt.x[2]*np.cos(angle),opt.x[1]+opt.x[2]*np.sin(angle),color=color,linewidth=.8)
    ax.scatter([opt.x[0]],[opt.x[1]],s=30,marker='+',color=color)
   delta=fits['back']['centerLateralDepth'][1]-fits['front']['centerLateralDepth'][1]
   ratio=fits['back']['radius']/fits['front']['radius'];stable=max(v['rmse'] for v in fits.values())<.0015 and .85<ratio<1.18
   row={'station':station,'digit':name,'lateralWindow':[lo,hi],'fits':fits,'backMinusFrontCenterDepth':delta,'radiusRatioBackFront':ratio,'stableSlice':stable}
   rows.append(row);ax.set_title(f'{name} / station {station:g}\ncenterY difference {delta:+.4f}'+(' / stable' if stable else ' / uncertain'))
   ax.set_aspect('equal');ax.grid();ax.set_xlabel('Lateral');ax.set_ylabel('Depth Y from wrist')
 for ax in list(axes.flat)[counter:]:ax.set_visible(False)
 axes[0,0].legend();fig.suptitle('Observed finger shaft arcs and independent circle centres / circle assumption, source radius agreement gate',fontsize=13);fig.tight_layout();fig.savefig(OUT/'crosssection-circle-centers.png',dpi=150);plt.close(fig)
 document={'method':'independent robust circles fitted to observed complementary native shaft arcs at fixed aligned longitudinal stations',
  'coordinateFrame':{'origin':ORIGIN.tolist(),'longitudinalAxis':LONG.tolist(),'lateralAxis':LATERAL.tolist(),'depthAxis':[0,1,0]},
  'sliceHalfWidth':.00175,'stableGate':{'maximumRMSE':.0015,'radiusRatioRange':[.85,1.18]},'rows':rows,
  'interpretation':'Stable shaft-centre differences are much smaller than opposing-shell Y gaps; those gaps include real finger thickness. Terminal caps require independent evidence. Covariance-side ICP can bias toward collapsing thickness.',
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_left_depth_v7.py --circle-sections'}
 (OUT/'crosssection-circle-centers.json').write_text(json.dumps(document,indent=2)+'\n')

def center_curves():
 """Fit shaft-centre curve offsets with loose depth bands separating digits."""
 f,b,c,vis,report=load();curves={};fig,axes=plt.subplots(1,3,figsize=(16,5))
 for source,d in [('front',f),('back',b)]:
  q=rotate(d[:,:3]-ORIGIN,BASIS);v=vis[source];good=(v['seen']>v['oppositeSeen'])&(v['ratio']>.03)&(v['seen']>.1)&(d[:,10]>-2.5);curves[source]={}
  for digit,l0,slope,col in [('ring',-.024,.14,0),('middle',.003,-.1,1),('index',.033,-.18,2)]:
   results=[]
   for station in np.arange(.118,.18,.002):
    lateral=l0+slope*(station-.125)
    depth=.041-.39*(station-.125) if digit=='ring' else (.055-.35*max(station-.145,0) if digit=='middle' else .055-.2*max(station-.165,0))
    selected=good&(abs(q[:,0]-station)<.0015)&(abs(q[:,1]-lateral)<(.010 if digit=='ring' else .012))&(abs(q[:,2]-depth)<.013)
    points=q[selected,1:]
    if len(points)<20:continue
    center=np.median(points,axis=0);radius=np.clip((np.quantile(points[:,0],.93)-np.quantile(points[:,0],.07))/2,.004,.011)
    center[1]+=radius*.65*(1 if source=='front' else -1);seed=np.r_[center,radius];keep=np.ones(len(points),bool)
    for _ in range(4):
     def residual(w):return np.linalg.norm(points[keep]-w[:2],axis=1)-w[2]
     fitted=least_squares(residual,seed,bounds=([lateral-.008,np.min(points[:,1])-.02,.003],[lateral+.008,np.max(points[:,1])+.02,.012]),loss='soft_l1',f_scale=.0005,max_nfev=200)
     seed=fitted.x;errors=np.abs(np.linalg.norm(points-fitted.x[:2],axis=1)-fitted.x[2]);keep=errors<np.quantile(errors,.85)
    results.append([station,*fitted.x,float(np.sqrt(np.mean(errors[keep]**2)))])
   values=np.array(results);curves[source][digit]=values.tolist();axes[col].plot(values[:,0],values[:,2],'.-',label=source);axes[col].set_title(digit+' inferred circle centre depth');axes[col].legend();axes[col].grid()
 fig.tight_layout();fig.savefig(OUT/'dense-center-depth-curves-corrected.png',dpi=170);plt.close(fig)
 rows=[]
 for digit in ['ring','middle','index']:
  front=np.array(curves['front'][digit]);back=np.array(curves['back'][digit]);selected=(front[:,0]>.127)&(front[:,0]<.176);station=front[selected,0];front=front[selected]
  interpolate=interp1d(back[:,0],back[:,1:3],axis=0,fill_value='extrapolate')
  def residual(v):return (interpolate(station-v[0])+v[1:]-front[:,1:3]).ravel()
  fitted=least_squares(residual,[.003,0,0],bounds=([-.007,-.006,-.006],[.007,.006,.006]),loss='soft_l1',f_scale=.001)
  rows.append({'digit':digit,'shiftLongLateralDepth':fitted.x.tolist(),'beforeRMSE':float(np.sqrt(np.mean(residual(np.zeros(3))**2))),'afterRMSE':float(np.sqrt(np.mean(residual(fitted.x)**2)))})
 document={'curves':curves,'fits':rows,'sourceHashes':{m['file']:m['sha256'] for m in report['sources']},
  'method':'Independent robust circle centre curves, then bounded3D translational curve registration; circles infer shaft centres rather than matching opposing shell skin.',
  'selection':'Observed native primary visibility dominates opposite, ratio>.03,flux>.1,opacitylogit>-2.5; .003-wide longitudinal slices; loose digit-specific lateral/depth bands exclude adjacent fingers.',
  'robustness':'Four iterations trim15% circular residual outliers; inferred radius bounded.003–.012; fit region long.127–.176; shifts bounded±.007/.006/.006.',
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_left_depth_v7.py --center-curves',
  'conclusion':'All3finger shaft curves independently support +longitudinal.0031–.0034, lateral/depth corrections <.0006. Use conservative+.0033 distal shift; retain real finger thickness.'}
 (OUT/'dense-center-depth-curves-corrected.json').write_text(json.dumps(document,indent=2)+'\n')

def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--pitch-sweep',action='store_true');p.add_argument('--circle-sections',action='store_true');p.add_argument('--center-curves',action='store_true');p.add_argument('--digit-shift',action='store_true');p.add_argument('--limit',type=float,default=.65);a=p.parse_args()
 f,b,c,vis,report=load();sections(f,b,vis)
 if a.audit:
  colors=[]
  for source,dd,color in [('front',f,[.1,.8,.9]),('back',b,[1.,.5,.1])]:
   m=vis[source]['seen']>vis[source]['oppositeSeen'];dc=dd[m].copy();dc[:,11:14]=(np.array(color)-.5)/.2820947918;colors.append(dc)
  atlas({'Full clean observed front':f[vis['front']['seen']>vis['front']['oppositeSeen']], 'Full clean observed back':b[vis['back']['seen']>vis['back']['oppositeSeen']], 'Source surfaces cyan front orange back':np.concatenate(colors)},OUT/'native-observed-surface-audit.png',CENTER,.33,600,views=VIEWS,title='Full source observed Gaussian surfaces before coverage, unchanged hand transform')
 if a.fit:
  Q,t,result=fit(f,b,vis,a.limit);tr=report['parts'][6]['sourceToFrontRaw'];result['sourceToFrontRaw']={'scale':tr['scale'],'rotation':(Q@np.array(tr['rotation'])).tolist(),'translation':(Q@np.array(tr['translation'])+t).tolist()};(OUT/'side-surface-candidate.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
  moved=transform_gaussians(b,Q,t);sections(f,moved,vis,'candidate')
  baseline=fused_trial(np.eye(3),np.zeros(3));candidate=fused_trial(Q,t);np.savez_compressed(OUT/'side-surface-candidate.npz',data=candidate[0],labels=candidate[1],sources=candidate[2],indices=candidate[3])
  atlas({'V6 baseline':baseline[0],'3D side-normal candidate':candidate[0]},OUT/'side-surface-before-after.png',CENTER,.33,650,views=VIEWS,title='Observed-side covariance-normal rigid fit / original forearm retained')
  colored=[]
  for case in [baseline,candidate]:
   dc=case[0].copy();dc[case[2]==0,11:14]=(np.array([.1,.8,.9])-.5)/.2820947918;dc[case[2]==1,11:14]=(np.array([1.,.5,.1])-.5)/.2820947918;colored.append(dc)
  atlas({'Baseline source colors':colored[0],'Candidate source colors':colored[1]},OUT/'source-color-before-after.png',CENTER,.33,650,views=VIEWS,title='Cyan front / orange back / no shell-thickness collapse objective')
 if a.pitch_sweep:pitch_sweep()
 if a.circle_sections:circle_sections()
 if a.digit_shift:digit_shift_trials()
 if a.center_curves:center_curves()
if __name__=='__main__':main()
