#!/usr/bin/env python3
"""Read-only right dorsal wrist-pad continuity and hand-pose audit."""
import os,json,argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mannequin-wrist-v8-mpl')
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_fill_holes,binary_closing,binary_erosion,label
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import ROOT,transform_gaussians,rotate,dot,read_scan
from render_gaussians import render,atlas
BASE=ROOT/'raw/fusion-work/refinement-v8/baseline';OUT=ROOT/'raw/fusion-work/refinement-v8/pad-registration';OUT.mkdir(parents=True,exist_ok=True)
NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection'
P=np.array([-.697,.03,.597]);U=np.array([-.626,0,.7798]);U/=np.linalg.norm(U);V=np.array([U[2],0,-U[0]])

def report():return json.loads((BASE/'report.json').read_text())
def mapped(data,tr):return transform_gaussians(data,np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])

def hand_contours():
 r=report();rows={};data={};sourceids={}
 for source in ['front','back']:
  c=np.load(NATIVE/f'{source}-full-native.npz');sel=c['labels']==12;d=c['data'][sel];ids=c['indices'][sel]
  vis=np.load(ROOT/f'raw/fusion-work/refinement-v6/right-hand/{source}-full-visibility.npz');assert np.array_equal(ids,vis['indices'])
  good=(vis['seen']>vis['opposite_seen'])&(vis['observed_ratio']>.03)&(vis['seen']>.05);d=d[good]
  if source=='front':data['Front observed']=d
  else:
   data['Back current hand transform']=mapped(d,r['parts'][12]['sourceToFrontRaw'])
   data['Back shared forearm transform']=mapped(d,r['parts'][11]['sourceToFrontRaw'])
 center=np.array([-.754,.06,.669]);span=.28;size=950;edges={};fig,axes=plt.subplots(1,3,figsize=(18,6));metrics={}
 for name,d in data.items():
  white=d.copy();white[:,11:14]=(1-.5)/.2820947918;image=np.asarray(render(white,[0,-1,0],center,span,width=size,height=size));mask=image[:,:,0]>145;mask=binary_fill_holes(binary_closing(mask,iterations=2));labels,count=label(mask);counts=np.bincount(labels.ravel());counts[0]=0;mask=labels==np.argmax(counts);boundary=mask&~binary_erosion(mask);iy,ix=np.nonzero(boundary);xz=np.c_[ix+.5-size/2,iy+.5-size/2]*(span/size)+center[[0,2]];p=np.c_[xz[:,0],np.zeros(len(xz)),xz[:,1]];selected=dot(p-P,U)>.035;edges[name]=xz[selected]
  for ax in axes:ax.set_aspect('equal');ax.grid()
  axes[0].scatter(edges[name][:,0],edges[name][:,1],s=1,label=name)
 for col,name in enumerate(['Back current hand transform','Back shared forearm transform'],start=1):
  front=edges['Front observed'];back=edges[name];d=np.r_[cKDTree(front).query(back)[0],cKDTree(back).query(front)[0]]
  metrics[name]={'symmetricOutlineMean':float(np.mean(d)),'symmetricOutlineMedian':float(np.median(d)),'symmetricOutlineP90':float(np.quantile(d,.9)),'frontPoints':len(front),'backPoints':len(back)}
  axes[col].scatter(front[:,0],front[:,1],s=2,label='front');axes[col].scatter(back[:,0],back[:,1],s=2,label=name);axes[col].set_title(name)
 axes[0].set_title('Observed original finger silhouettes');[ax.legend(fontsize=7) for ax in axes];fig.tight_layout();fig.savefig(OUT/'right-hand-contour-gate.png',dpi=160);plt.close(fig)
 doc={'method':'same XZ finger outline from full quality-clean native observed surfaces, excluding wrist; opposing palm/dorsal planes are not paired','metrics':metrics,'scope':'projected contour gate only; does not validate depth by itself','sourceHashes':{m['file']:m['sha256'] for m in r['sources']}}
 (OUT/'right-hand-contour-gate.json').write_text(json.dumps(doc,indent=2)+'\n');print(json.dumps(doc,indent=2),flush=True)
 atlas(data,OUT/'right-hand-original-observed-poses.png',center,.34,650,views=[('Palm',[0,-1,0]),('Back',[0,1,0]),('Distal',[-.63,0,.78]),('Outer',[-1,0,0])],title='Original observed hand surfaces / back native hand kept rigid / no new filtering or fitting')

def continuity():
 r=report();features=json.loads((BASE/'feature-transforms.json').read_text())['parts'];source={m['file']:m['sha256'] for m in r['sources']}
 maskmeta=json.loads((NATIVE/'back-right-rigid-pad.json').read_text());assert maskmeta['sourceHashes']==source
 ids=np.load(ROOT/maskmeta['maskFile']);native=np.load(NATIVE/'back-full-native.npz');raw,meta=read_scan(ROOT/r['sources'][1]['file']);assert meta['sha256']==r['sources'][1]['sha256'];assert np.array_equal(raw[native['indices']],native['data'])
 selected=np.isin(native['indices'],ids);assert selected.sum()==len(ids);labels=native['labels'][selected];data=native['data'][selected];indices=native['indices'][selected]
 forearm=r['parts'][11]['sourceToFrontRaw'];hand=r['parts'][12]['sourceToFrontRaw'];Rf=np.array(forearm['rotation']);Rh=np.array(hand['rotation']);tf=np.array(forearm['translation']);th=np.array(hand['translation'])
 parent=mapped(data,forearm);old=parent.copy();old[labels==12]=mapped(data[labels==12],hand)
 relative=Rh@Rf.T;relative_translation=th-relative@tf
 pairs=np.load(NATIVE/'pad-cross-part-pairs.npz');fi=pairs['forearmSourceIndices'];hi=pairs['handSourceIndices'];fnew=mapped(raw[fi],forearm)[:,:3];hnew=mapped(raw[hi],forearm)[:,:3];hold=mapped(raw[hi],hand)[:,:3]
 dnew=np.linalg.norm(hnew-fnew,axis=1);dold=np.linalg.norm(hold-fnew,axis=1);displacement=np.linalg.norm(old[labels==12,:3]-parent[labels==12,:3],axis=1)
 counts={'right_forearm':int(sum(labels==11)),'right_hand':int(sum(labels==12))};metrics={'crossPartPairs':len(fi),'nativeSameRigidMedian':float(np.median(dnew)),'nativeSameRigidP90':float(np.quantile(dnew,.9)),
  'oldIndependentMedian':float(np.median(dold)),'oldIndependentP90':float(np.quantile(dold,.9)),'artificialHandPadShiftMedian':float(np.median(displacement)),'artificialHandPadShiftMaximum':float(displacement.max())}
 original_to_parent={'rotation':Rf.tolist(),'translation':tf.tolist(),'scale':forearm['scale']}
 old_to_parent={'rotation':(Rf@Rh.T).tolist(),'translation':(tf-Rf@Rh.T@th).tolist(),'scale':1.}
 document={'version':1,'status':'pad constraint accepted; whole-hand shared-transform trial rejected by distal depth gate; smooth endpoint transition under review',
  'sourceHashes':source,'baseline':str(BASE.relative_to(ROOT)),'picturedSide':'anatomical right','picturedFeature':'continuous dorsal rounded rectangular wrist pad with small yellow proximal opening',
  'frontSourceEvidence':'No reliable observed dorsal pad rim/corners survive in front source; visible original front contactsheet is corrupt. No paired-front-pad correspondence was invented.',
  'backSourceEvidence':'Original back source shows one uninterrupted pad across the current forearm/hand segmentation. All12,222 selected original pad rows belong to one physical surface.',
  'padSelection':str((NATIVE/'back-right-rigid-pad.json').relative_to(ROOT)),'padPartCounts':counts,'continuity':metrics,
  'oldRelativeHandForearm':{'rotation':relative.tolist(),'translation':relative_translation.tolist(),'rotationMagnitudeDegrees':float(np.degrees(Rotation.from_matrix(relative).magnitude())),
   'eulerXYZDegrees':Rotation.from_matrix(relative).as_euler('xyz',degrees=True).tolist()},
  'requiredPadSourceToFrontRaw':original_to_parent,'oldAlignedHandToPadCorrection':old_to_parent,
  'protectedMechanicalConstraint':features['right_forearm'].get('mechanicalConstraint'),
  'mechanicalScope':'Retain shoulder and right_forearm transforms exactly, including the accepted axial elbow constraint. Only back-hand/pad assignment changes; all pad means must use the unchanged forearm transform.',
  'wholeHandTrial':{'sourceToFrontRaw':original_to_parent,'status':'rejected as full-hand solution','reason':'Restores native pad continuity but moves distal source fingers in depth by up to.022, reintroducing separated tips despite improved XZ outline overlap.'},
  'recommendedEndpointBlend':{'proximalReference':'unchanged right_forearm transform','distalReference':'previous right_hand transform','frame':'forearm-aligned front raw','origin':P.tolist(),'longitudinalAxis':U.tolist(),
   'suggestedSmoothstepLongitudinal':[.065,.110],'jacobian':'(1-w)I+wA+(xOld-xParent) outer gradient(w)','requiredInvariant':'Every selected pad row remains exactly at the unchanged forearm mapping; distal fingers return exactly to previous hand mapping; positive deformation determinant; full J Sigma J-transpose covariance.'},
  'handContourGate':json.loads((OUT/'right-hand-contour-gate.json').read_text()) if (OUT/'right-hand-contour-gate.json').exists() else None,
  'proof':[str((OUT/n).relative_to(ROOT)) for n in ['right-identity.png','native-full-pad-source.png','right-hand-contour-gate.png','right-hand-original-observed-poses.png','back-pad-continuity.png']],
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/inspect_pad_v8.py --native --mask && .venv-fusion/bin/python -B tools/fusion/audit_wrist_pad_v8.py --hand-contours --continuity'}
 (OUT/'right-wrist-pad-evidence.json').write_text(json.dumps(document,indent=2)+'\n');np.savez_compressed(OUT/'back-pad-continuity.npz',old=old,shared=parent,labels=labels,indices=indices,forearmPairIndices=fi,handPairIndices=hi,oldPairDistances=dold,sharedPairDistances=dnew)
 colored=[]
 for d in [old,parent]:
  dd=d.copy();dd[:,11:14]=(np.where((labels==11)[:,None],[.2,.85,.95],[1.,.55,.12])-.5)/.2820947918;colored.append(dd)
 atlas({'Old independent transforms':old,'One native rigid pad':parent,'Old forearm cyan hand orange':colored[0],'Shared forearm mapping':colored[1]},OUT/'back-pad-continuity.png',[-.676,.055,.574],.29,750,views=[('Dorsal',[0,1,0]),('Thumb oblique',[-.45,1,0]),('Other oblique',[.5,1,0])],title='Same original back pad rows / only mapping changes / exact source continuity restored')
 print(json.dumps({'relativeRotationDegrees':document['oldRelativeHandForearm']['rotationMagnitudeDegrees'],'counts':counts,'continuity':metrics},indent=2),flush=True)

def transition_math():
 """Independent differential/covariance/end-constraint audit of the source field."""
 from right_wrist_transition import preserve_pad_align_fingers,displacement_and_jacobian
 native=np.load(NATIVE/'back-full-native.npz');d=native['data'];labels=native['labels'];ids=native['indices'];hand=labels==12
 padids=np.load(NATIVE/'back-right-rigid-pad.npy');pad=np.isin(ids,padids);pairs=np.load(NATIVE/'pad-cross-part-pairs.npz');r=report();results={}
 def covariance(data):
  q=data[:,3:7].astype(float);rotation=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix();factor=rotation*np.exp(data[:,None,7:10]);return np.einsum('nik,njk->nij',factor,factor)
 for end in [.11,.125]:
  trialdir=ROOT/'raw/fusion-work/refinement-v8/transition-trials';spec=json.loads((trialdir/f'transition-{end:g}.json').read_text())['parameters'];f=spec['referenceForearm'];h=spec['referenceHand']
  warped,audit=preserve_pad_align_fingers(d[hand],spec);delta,J,w=displacement_and_jacobian(d[hand,:3],spec)
  full=mapped(d,f);full[hand]=mapped(warped,f);parent=mapped(d,f);old=mapped(d[hand],h);distal=w==1
  # Compare actual source pair IDs across the existing part cut, not all pad
  # rows being present after opacity or visibility filtering.
  lookup={int(value):i for i,value in enumerate(ids)}
  fi=np.array([lookup[int(value)] for value in pairs['forearmSourceIndices']]);hi=np.array([lookup[int(value)] for value in pairs['handSourceIndices']]);pair_distance=np.linalg.norm(full[fi,:3]-full[hi,:3],axis=1)
  pc=covariance(parent[pad]);wc=covariance(full[pad]);dc=covariance(full[hand][distal]);oc=covariance(old[distal])
  # Finite differences on double-precision source positions avoid mistaking
  # float32 stored mean rounding for a Jacobian error.
  ratio=np.exp(d[hand,7:10].min(1)-d[hand,7:10].max(1));transition=np.flatnonzero((w>0)&(w<1));thin=transition[np.argsort(ratio[transition])[:50]]
  sampled=np.unique(np.r_[np.linspace(0,len(w)-1,70,dtype=int),thin]);points=d[hand][sampled,:3].astype(float);analytic=J[sampled];numeric=np.zeros_like(analytic);eps=1e-6
  for axis in range(3):
   step=np.zeros(3);step[axis]=eps
   dp=displacement_and_jacobian(points+step,spec)[0];dm=displacement_and_jacobian(points-step,spec)[0]
   numeric[:,:,axis]=(dp-dm)/(2*eps);numeric[:,axis,axis]+=1
  original_cov=covariance(d[hand][sampled]);expected=np.einsum('nij,njk,nlk->nil',analytic,original_cov,analytic);actual=covariance(warped[sampled])
  norm=np.linalg.norm(expected,axis=(1,2));covariance_relative=np.linalg.norm(actual-expected,axis=(1,2))/np.maximum(norm,1e-30)
  # Subsequent parent adjustment must carry the whole deformed source field.
  adjustment=Rotation.from_euler('xyz',[7.,-4.,9.],degrees=True).as_matrix();shift=np.array([.020,-.015,.007]);Rf=np.array(f['rotation']);tf=np.array(f['translation'])
  nextpose={'scale':f['scale'],'rotation':(adjustment@Rf).tolist(),'translation':(adjustment@tf+shift).tolist()}
  moved=mapped(warped,nextpose);reference=transform_gaussians(mapped(warped,f),adjustment,shift)
  output=np.load(trialdir/f'transition-{end:g}.npz');observed=(output['sources']==1)&np.isin(output['indices'],padids);outputpadids=output['indices'][observed];expected_indices=np.array([lookup[int(value)] for value in outputpadids]);output_pad_delta=np.linalg.norm(output['data'][observed,:3]-parent[expected_indices,:3],axis=1)
  outlookup={int(value):i for i,value in enumerate(output['indices']) if output['sources'][i]==1};retainedpairs=[(outlookup[int(a)],outlookup[int(b)]) for a,b in zip(pairs['forearmSourceIndices'],pairs['handSourceIndices']) if int(a) in outlookup and int(b) in outlookup]
  opairs=np.array(retainedpairs,int);od=np.linalg.norm(output['data'][opairs[:,0],:3]-output['data'][opairs[:,1],:3],axis=1)
  result={'sourceGeometry':{'padSourceRows':int(pad.sum()),'padHandRows':int(np.sum(pad&hand)),
   'padMaximumMeanDifference':float(np.abs(full[pad,:3]-parent[pad,:3]).max()),'padCovarianceMaximumDifference':float(np.abs(wc-pc).max()),
   'padNativeSourceFieldsExactlyUnchanged':bool(np.array_equal(warped[np.isin(ids[hand],padids)],d[hand][np.isin(ids[hand],padids)])),
   'distalSourceRows':int(distal.sum()),'distalMaximumMeanDifferenceVsOldHand':float(np.abs(full[hand][distal,:3]-old[distal,:3]).max()),'distalCovarianceMaximumDifferenceVsOldHand':float(np.abs(dc-oc).max()),
   'originalColorOpacityExact':bool(np.array_equal(warped[:,10:],d[hand][:,10:]))},
   'jacobian':{'samples':len(sampled),'thinSamples':len(thin),'minimumThinAxisRatio':float(ratio[thin].min()),'finiteDifferenceEpsilon':eps,'maximumElementError':float(np.abs(numeric-analytic).max()),
    'maximumRelativeTransportedCovarianceError':float(covariance_relative.max()),'minimumDeterminantFullHand':float(np.linalg.det(J).min())},
   'futureParentAdjustment':{'testedEulerXYZDegrees':[7,-4,9],'testedTranslation':shift.tolist(),'maximumMeanDifference':float(np.abs(moved[:,:3]-reference[:,:3]).max()),'maximumCovarianceDifference':float(np.abs(covariance(moved)-covariance(reference)).max()),
    'meaning':'Native support and endpoint references remain frozen; any subsequent parent rigid adjustment carries the entire field and both ends together.'},
   'sourcePairContinuity':{'pairs':len(fi),'median':float(np.median(pair_distance)),'p90':float(np.quantile(pair_distance,.9))},
   'actualTrialOutput':{'retainedPadRows':len(outputpadids),'maximumRetainedPadMeanDifference':float(output_pad_delta.max(initial=0)),
    'retainedCompleteCrossPartPairs':len(od),'pairMedian':float(np.median(od)),'pairP90':float(np.quantile(od,.9)),
    'note':'12,222 is the quality-clean evidence selection, not a claim all original pad rows survive fusion. Pair output metric includes only pairs whose two original IDs survive.'}}
  assert result['sourceGeometry']['padMaximumMeanDifference']==0
  assert result['sourceGeometry']['padCovarianceMaximumDifference']==0
  assert result['sourceGeometry']['padNativeSourceFieldsExactlyUnchanged']
  assert result['sourceGeometry']['distalMaximumMeanDifferenceVsOldHand']<2e-7
  assert result['jacobian']['maximumElementError']<1e-7
  assert result['jacobian']['maximumRelativeTransportedCovarianceError']<5e-6
  results[str(end)]=result;print(end,json.dumps(result,indent=2),flush=True)
 document={'version':1,'sourceHashes':{m['file']:m['sha256'] for m in r['sources']},'module':'tools/fusion/right_wrist_transition.py','results':results,
  'analyticalProof':'Let F(p)=sRf p+tf and H(p)=sRh p+th. Native delta=w[(Rf^T Rh-I)p+Rf^T(th-tf)/s]. Then F(p+delta)=(1-w)F(p)+wH(p). Its native Jacobian is I+w(A-I)+d outer gradient(w), where gradient(w)=smoothstepDerivative*sRf^T*u. Covariance transported by J Sigma J^T therefore matches the derivative of the same mean field.',
  'limits':'Geometry equality at the distal endpoint is mathematical; stored float32 mean/quaternion/logscale reconstruction differs by tiny rounding. Full fused opacity and row-retention are separate coverage/filter operations.',
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_wrist_pad_v8.py --transition-math'}
 (OUT/'right-wrist-transition-math-audit.json').write_text(json.dumps(document,indent=2)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--hand-contours',action='store_true');p.add_argument('--continuity',action='store_true');p.add_argument('--transition-math',action='store_true');a=p.parse_args()
 if a.hand_contours:hand_contours()
 if a.continuity:continuity()
 if a.transition_math:transition_math()
