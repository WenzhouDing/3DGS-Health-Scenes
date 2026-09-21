#!/usr/bin/env python3
"""Fine rigid-chain candidates around the frozen accepted V10 pose.

No production edits and no wrist/point deformation. The shoulder correction is
a small full 3D rotation about the paired pad centroid; the entire forearm and
hand share the same pose derived from the existing axial collar constraint.
Candidate acceptance requires matching observed seam support and visual review.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
from pipeline import ROOT,BODY_PARTS,apply_joint_boundaries,load_cleanup_masks
from refine_rigid_arm_v10 import parent_transform,child_transform,safeguards,generate
from render_gaussians import load_fused,atlas

BASE=ROOT/'raw/fusion-work/refinement-v11/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v11/registration'
NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection'
PAD=ROOT/'raw/fusion-work/feature-audit'

def context():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((BASE/'fusion-config.json').read_text());report=json.loads((BASE/'report.json').read_text());features=json.loads((BASE/'feature-transforms.json').read_text())['parts'];lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
 for rule in cfg['jointBoundaries']['front']:lm[rule['landmark']]=rule['pivot']
 native={s:dict(np.load(NATIVE/f'{s}-full-native.npz')) for s in ['front','back']};cache=np.load(ROOT/'raw/fusion-work/refinement-v4/arm/native-clean.npz');upper={}
 for source,key in [('front','fl'),('back','bl')]:
  landmarks=json.loads((BASE/f'{source}-landmarks.json').read_text())['landmarks'];labels,_=apply_joint_boundaries(cache[source][:,:3],cache[key].copy(),landmarks,cfg['jointBoundaries'][source],BODY_PARTS);upper[source]=cache[source][labels==10];assert len(upper[source])==report['parts'][10][source+'Before']
 masks,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],report['sources'])),ROOT);c=json.loads((ROOT/'raw/fusion-work/refinement-v4/arm/collar-measurements.json').read_text());f=np.load(PAD/'front-right-pad-ellipse.npz')['outline'];b=np.load(PAD/'back-right-pad-ellipse.npz')['outline']
 return cfg,report,features,lm,native,upper,masks,c,f,b

def transforms(x,ctx):
 parent=parent_transform(np.radians(x[:3]),ctx);child,constraint=child_transform(parent,ctx,x[3]);return parent,child,constraint

def center_targets(extended=False):
 audit=json.loads((ROOT/'raw/fusion-work/refinement-v10/gap-audit/shaft-center-audit.json').read_text());basis=np.array(audit['frame']['axesVYU']);origin=np.array(audit['frame']['origin']);reference=json.loads((ROOT/'raw/fusion-work/refinement-v10/baseline/report.json').read_text())['parts'][11]['sourceToFrontRaw'];curves={}
 for finger in ['index','middle','ring']:
  records=audit['sections'][finger];back=np.array([m['centerLocalVYU'] for m in records['back'] if m['accepted']]);front=np.array([m['centerLocalVYU'] for m in records['front'] if m['accepted'] and (extended or .134<m['station']<.156)]);curves[finger]=(np.einsum('ni,ij->nj',back,basis)+origin,front)
 return curves,basis,origin,reference

def center_errors(child,targets):
 curves,basis,origin,reference=targets;deltaR=np.array(child['rotation'])@np.array(reference['rotation']).T;deltat=np.array(child['translation'])-deltaR@reference['translation'];errors={}
 for finger,(points,target) in curves.items():
  moved=np.einsum('ni,ji->nj',points,deltaR)+deltat;local=np.einsum('ni,ji->nj',moved-origin,basis);local=local[np.argsort(local[:,2])];u=target[:,2];xy=interp1d(local[:,2],local[:,:2],axis=0,bounds_error=False,fill_value='extrapolate')(u);errors[finger]=xy-target[:,:2]
 return errors

def candidate_record(name,x,ctx,targets):
 parent,child,constraint=transforms(x,ctx);errors=center_errors(child,targets)
 return {'name':name,'status':'diagnostic candidate, not approved for production','baseline':str(BASE.relative_to(ROOT)),
  'incrementalShoulderRotvecRawDegrees':x[:3].tolist(),'incrementalAxialDegrees':float(x[3]),'totalAxialDegrees':constraint['twistDegrees'],
  'right_upper_arm':parent,'right_forearm':child,'right_hand':child,'mechanicalConstraint':constraint,'safeguards':safeguards(parent,child,ctx),
  'shaftCenterDiagnostics':{finger:{'medianDeltaLateral':float(np.median(e[:,0])),'medianDeltaDepth':float(np.median(e[:,1])),'medianSeparation':float(np.median(np.linalg.norm(e,axis=1))),'allSectionResidualsVY':e.tolist()}for finger,e in errors.items()},
  'limitations':'Circular shaft centers are inferred features. Extended terminal sections are less reliable; matching observed side/cap surfaces and true Gaussian views must decide acceptance. No matching of opposing skin planes.',
  'sourceHashes':{m['file']:m['sha256'] for m in ctx[1]['sources']}}

def fit():
 ctx=context();baseline,labels,_,_=load_fused(BASE);rows={'V10 baseline':baseline[np.isin(labels,[10,11,12])]};f,b=ctx[-2:];tree=cKDTree(f);c=ctx[-3];results={}
 for extended,name in [(False,'interior-four-dof'),(True,'extended-four-dof')]:
  targets=center_targets(extended)
  def residual(x):
   parent,child,_=transforms(x,ctx);r=np.array(parent['rotation']);t=np.array(parent['translation']);s=parent['scale'];m=s*np.einsum('ni,ji->nj',b,r)+t;center=s*r@np.array(c['back']['pivot'])+t;errors=center_errors(child,targets)
   return np.r_[np.concatenate(list(errors.values())).ravel()/.0015,(m-f[tree.query(m)[1]]).ravel()/np.sqrt(len(m))/.002,(center-c['front']['pivot'])/.004,x[:3]/.5,x[3]/5.]
  sol=least_squares(residual,np.zeros(4),bounds=([-.6,-.6,-.6,-3.],[.6,.6,.6,3.]),loss='soft_l1',f_scale=1.,max_nfev=250);x=sol.x;parent,child,_=transforms(x,ctx);record=candidate_record(name,x,ctx,targets);record['objectiveScales']={'shaftCenters':.0015,'padRim':.002,'collarCenter':.004,'incrementalShoulderPriorDegrees':.5,'incrementalAxialPriorDegrees':5.};record['fitCostBefore']=float(np.sum(residual(np.zeros(4))**2));record['fitCostAfter']=float(np.sum(residual(x)**2));candidate=generate(parent,child,ctx);np.savez_compressed(OUT/(name+'.npz'),**candidate);(OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');rows[name]=candidate['data'];results[name]=record;print(json.dumps(record),flush=True)
 (OUT/'center-fine-fit.json').write_text(json.dumps(results,indent=2)+'\n')
 views=[('Below',[0,0,1]),('Outer',[-1,.2,0]),('Dorsal',[-.45,1,0]),('Front',[0,-1,0])]
 atlas(rows,OUT/'center-fine-hand.png',[-.765,.045,.678],.30,850,views=views,title='V11 rigid fine candidates; preserve full source geometry and wrist pad')
 atlas(rows,OUT/'center-fine-collar.png',[-.359,-.038,.279],.33,700,views=views,title='Fixed collar axis/pivot; one rigid forearm and hand; upstream shoulder only')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--fit',action='store_true');args=parser.parse_args()
 if args.fit:fit()
