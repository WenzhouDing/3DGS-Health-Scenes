#!/usr/bin/env python3
"""Bounded native multi-angle right finger surface restoration experiments.

No geometry, covariance, colors, source file or active export is edited.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pipeline import ROOT,BODY_PARTS,transform_gaussians,coverage_weights,attenuate,load_cleanup_masks
from render_gaussians import atlas,render,load_fused
from refine_hand_v4 import visibility
BASE=ROOT/'raw/fusion-work/refinement-v10/baseline';NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection';OLD=ROOT/'raw/fusion-work/refinement-v6/right-hand';OUT=ROOT/'raw/fusion-work/refinement-v10/restoration';OUT.mkdir(parents=True,exist_ok=True)
ORIGIN=np.array([-.697,.03,.597]);LONG=np.array([-.626,0,.7798]);LONG/=np.linalg.norm(LONG)
CAMERAS=[[0,-1,0],[1,-.6,0],[-1,-.6,0],[0,-.6,1],[0,-.6,-1],[.7,-.6,.7],[-.7,-.6,.7],[.7,-.6,-.7],[-.7,-.6,-.7]]
CENTER=[-.755,.055,.665];VIEWS=[('Below / +Z',[0,0,1]),('Distal',[-.65,.05,.8]),('Palm',[0,-1,0]),('Back',[0,1,0]),('Palm oblique',[-.7,-1,.1]),('Back oblique',[.7,1,.1]),('Thumb edge',[1,0,0]),('Little-finger edge',[-1,0,0])]

def load():
 report=json.loads((BASE/'report.json').read_text());cfg=json.loads((BASE/'fusion-config.json').read_text());native={};ids={};aligned={};primary={}
 for si,s in enumerate(['front','back']):
  a=np.load(NATIVE/(s+'-full-native.npz'));m=a['labels']==12;native[s]=a['data'][m];ids[s]=a['indices'][m];primary[s]=dict(np.load(OLD/(s+'-full-visibility.npz')));assert np.array_equal(ids[s],primary[s]['indices']);assert len(ids[s])==report['parts'][12]['frontBefore' if s=='front' else 'backBefore']
  tr=report['parts'][12]['sourceToFrontRaw'];aligned[s]=native[s] if not si else transform_gaussians(native[s],np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
 return report,cfg,native,ids,aligned,primary

def compute_visibility():
 report,cfg,native,ids,aligned,primary=load()
 for source,data in native.items():
  result={};best_flux=np.zeros(len(data));best_ratio=np.zeros(len(data));opposite_flux=np.zeros(len(data))
  for i,camera in enumerate(CAMERAS):
   ratio,flux=visibility(data,camera,size=700,span=.30);result['observedRatio'+str(i)]=ratio;result['observedFlux'+str(i)]=flux;best_flux=np.maximum(best_flux,flux);best_ratio=np.maximum(best_ratio,ratio)
   other=np.array(camera,float);other[1]*=-1;_,opposite=visibility(data,other,size=700,span=.30);opposite_flux=np.maximum(opposite_flux,opposite)
   print(source,'hemisphere camera',i,flush=True)
  result.update(observedMaxFlux=best_flux,observedMaxRatio=best_ratio,oppositeMaxFlux=opposite_flux,indices=ids[source]);np.savez_compressed(OUT/(source+'-hemisphere-visibility.npz'),**result)

def selections():
 report,cfg,native,ids,aligned,primary=load();out={s:{} for s in native};stats={}
 for source,data in native.items():
  vis=np.load(OUT/(source+'-hemisphere-visibility.npz'));assert np.array_equal(vis['indices'],ids[source]);p=primary[source];scales=np.sort(np.exp(data[:,7:10]),axis=1);alpha=1/(1+np.exp(-np.clip(data[:,10],-40,40)));q=data[:,3:7].astype(float);q/=np.linalg.norm(q,axis=1)[:,None];R=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix();normal=R[np.arange(len(data)),:,np.argmin(data[:,7:10],axis=1)];lateral=abs(normal[:,1])<.7
  anchors=(p['observed_ratio']>.05)&(p['seen']>.1)&(p['seen']>p['opposite_seen'])&(scales[:,-1]<.01)&(alpha>.2);distance,_=cKDTree(data[anchors,:3]).query(data[:,:3]);long=np.einsum('ni,i->n',aligned[source][:,:3]-ORIGIN,LONG);region=long>.095
  rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1);lum=np.einsum('ni,i->n',rgb,[.2126,.7152,.0722]);skin=(rgb[:,1]>rgb[:,2]-.025)&(lum>.2)
  reliable=(vis['observedMaxRatio']>.04)&(vis['observedMaxFlux']>.10)&(scales[:,-1]<.01)&(alpha>.12)&region
  strict=reliable&lateral&(scales[:,0]<.6*scales[:,1])&(distance<.004)&(vis['observedMaxFlux']>.75*vis['oppositeMaxFlux'])&skin
  relaxed=reliable&lateral&(distance<.006)&(vis['observedMaxFlux']>.4*vis['oppositeMaxFlux'])&skin
  observed=region&(p['observed_ratio']>.05)&(p['seen']>.1)&(p['seen']>p['opposite_seen'])
  out[source]={'primaryCaps':observed,'connectedOblique':observed|strict,'relaxedOblique':observed|relaxed};stats[source]={'fullHand':len(data),'anchorPoints':int(anchors.sum()),'primaryCaps':int(observed.sum()),'strictObliqueAdditional':int((strict&~observed).sum()),'relaxedObliqueAdditional':int((relaxed&~observed).sum()),'oldV6RejectedRestoredStrict':int((strict&(p['seen']<p['opposite_seen'])).sum())}
 return out,stats

def trials():
 report,cfg,native,ids,aligned,primary=load();selection,stats=selections();baseline,bl,bs,_=load_fused(BASE);bi=np.load(BASE/'source-vertex-indices.npy');context=(bl==11)&(baseline[:,0]<-.61);hand=bl==12;mask,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],report['sources'])),ROOT);lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks'];weights=coverage_weights(aligned['front'],aligned['back'],lm,BODY_PARTS[12],cfg['fusion']);captures={'V9 baseline':baseline[context|hand]};evidence={'version':1,'sourceHashes':{m['file']:m['sha256'] for m in report['sources']},'counts':stats,'cameras':CAMERAS,'visibilitySize':700,'visibilitySpan':.30,'geometryChanges':False,'colorChanges':False,'sourceToFrontRaw':report['parts'][12]['sourceToFrontRaw'],'candidates':{}}
 for trial in ['primaryCaps','connectedOblique','relaxedOblique']:
  rows=[];labels=[];sources=[];indices=[];counts={};rules=[]
  for si,source in enumerate(native):
   restored=selection[source][trial];w=weights[si].copy();w[restored]=1.;dd,k=attenuate(aligned[source],w,cfg['fusion']['minWeight']);keep=(~np.isin(ids[source],mask[source])|restored)[k];dd=dd[keep];retained_ids=ids[source][k][keep];rows.append(dd);labels.append(np.full(len(dd),12,np.uint8));sources.append(np.full(len(dd),si,np.uint8));indices.append(retained_ids);existing=bi[(bs==si)&hand];chosen=np.unique(ids[source][restored]).astype(np.uint32);np.save(OUT/(trial+'-'+source+'.npy'),chosen)
   counts[source]={'restorationRows':len(chosen),'newlyRetainedRows':int(np.sum(~np.isin(retained_ids,existing))),'retainedHand':len(dd)}
   doc={'version':1,'purpose':'EXPERIMENT ONLY: observed native right finger curved surfaces with multiview support','sourceCapture':source,'sourceFile':report['sources'][si]['file'],'sourceSha256':report['sources'][si]['sha256'],'sourceHashes':evidence['sourceHashes'],'maskFile':str((OUT/(trial+'-'+source+'.npy')).relative_to(ROOT)),'uniqueSourceIndices':len(chosen),'part':'right_hand','maskMeaning':'sorted original source PLY vertex rows for surface restoration','trial':trial,'evidence':str((OUT/'restoration-evidence.json').relative_to(ROOT)),'reproduce':'.venv-fusion/bin/python -B tools/fusion/restore_right_caps_v10.py --visibility --trials'};jsonpath=OUT/(trial+'-'+source+'.json');jsonpath.write_text(json.dumps(doc,indent=2)+'\n');rules.append({'selection':str(jsonpath.relative_to(ROOT)),'part':'right_hand','minimumWeight':1.0})
  arrays={'data':np.concatenate([baseline[context],*rows]),'labels':np.concatenate([bl[context],*labels]),'sources':np.concatenate([bs[context],*sources]),'indices':np.concatenate([bi[context],*indices])};np.savez_compressed(OUT/(trial+'.npz'),**arrays);captures[trial]=arrays['data'];evidence['candidates'][trial]={'counts':counts,'configAdditions':{'surfaceRestorations':rules}};render(arrays['data'],[0,0,1],CENTER,.225,width=1200,height=800).save(OUT/(trial+'-below.png'));print(trial,counts,flush=True)
 (OUT/'restoration-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');atlas(captures,OUT/'restoration-trials.png',CENTER,.285,600,views=VIEWS,title='Right fingers / actual original Gaussian restoration only / strict rigid hand forearm / no new geometry')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--visibility',action='store_true');p.add_argument('--trials',action='store_true');a=p.parse_args()
 if a.visibility:compute_visibility()
 if a.trials:trials()
