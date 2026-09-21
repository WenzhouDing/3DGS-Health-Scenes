#!/usr/bin/env python3
"""Diagnostic reconstructed Gaussian coverage for missing lateral finger strips.

Means, opacity, color and source rows remain fixed. Existing covariance gains a
positive rank-one term along its circumferential tangent. This is reconstructed
coverage, not newly recovered scan evidence, and requires explicit visual QA.
Only confirmed +V index/middle/ring shaft gaps are considered. No pad, tips,
little finger, opposite side or rigid pose is changed.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pipeline import ROOT,transform_gaussians
from render_gaussians import atlas,load_fused
from inspect_finger_seam_v11 import BASIS,ORIGIN,covariance

BASE=ROOT/'raw/fusion-work/refinement-v11/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v11/registration/coverage-extension'
AUDIT=ROOT/'raw/fusion-work/refinement-v11/gap-audit'
NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection'

def smooth(x):
 x=np.clip(x,0.,1.);return x*x*(3-2*x)

def selection(data,labels,sources,ids):
 evidence=json.loads((AUDIT/'side-covariance-support.json').read_text());geo=np.load(AUDIT/'source-geometry.npz');local=transform_gaussians(data,BASIS,-BASIS@ORIGIN);C=covariance(local);_,eig=np.linalg.eigh(C);normal=eig[:,:,0]
 # Intersection of each measured surface tangent plane and a shaft section:
 # normal·tangent == 0, tangent longitudinal component == 0.
 tangent=np.c_[-normal[:,1],normal[:,0],np.zeros(len(data))];length=np.linalg.norm(tangent,axis=1);tangent/=np.maximum(length[:,None],1e-30)
 weights=np.zeros(len(data));finger_ids=np.full(len(data),-1,np.int8);supported=np.zeros(len(data),bool)
 for si,source in enumerate(['front','back']):supported|=(sources==si)&np.isin(ids,geo[source+'_ids'][geo[source+'_strong']])
 for fi,finger in enumerate(['index','middle','ring']):
  rows=[r for r in evidence['sections'] if r['finger']==finger and r['side']=='plusV'];stations=np.array([r['station']for r in rows]);order=np.argsort(stations);stations=stations[order]
  for si,source in enumerate(['front','back']):
   boundary=np.array([r[source+'BoundaryVY']for r in rows])[order];u=local[:,2];target=np.c_[np.interp(u,stations,boundary[:,0]),np.interp(u,stations,boundary[:,1])]
   distance=np.linalg.norm(local[:,:2]-target,axis=1)
   longitudinal=smooth((u-.111)/.011)*smooth((.174-u)/.009)
   neighborhood=smooth((.0045-distance)/.003)
   w=longitudinal*neighborhood
   valid=(labels==12)&(sources==si)&supported&(length>.75)&(abs(tangent[:,1])>.60)
   w[~valid]=0;take=w>weights;weights[take]=w[take];finger_ids[take]=fi
 weights[weights<.03]=0
 return local,C,normal,tangent,weights,finger_ids

def covariance_update(data,factors):
 C=covariance(data);changed=np.linalg.norm(factors,axis=1)>0;new=C[changed]+np.einsum('ni,nj->nij',factors[changed],factors[changed]);values,vectors=np.linalg.eigh(new)
 if np.any(values<=0):raise ValueError('Covariance addition lost positive definiteness')
 vectors[np.linalg.det(vectors)<0,:,0]*=-1
 q=Rotation.from_matrix(vectors).as_quat();out=data.copy();out[changed,3:7]=np.c_[q[:,3],q[:,:3]];out[changed,7:10]=.5*np.log(values)
 return out

def trials(amounts=(.0006,.0010,.0014),render_atlas=True):
 OUT.mkdir(parents=True,exist_ok=True);data,labels,sources,report=load_fused(BASE);ids=np.load(BASE/'source-vertex-indices.npy');chosen=np.isin(labels,[10,11,12]);data=data[chosen];labels=labels[chosen];sources=sources[chosen];ids=ids[chosen]
 local,C,normal,tangent,weights,fingers=selection(data,labels,sources,ids);selected=weights>0
 selection_report={'selectedRows':int(selected.sum()),'byFinger':{name:int(np.sum(selected&(fingers==i)))for i,name in enumerate(['index','middle','ring'])},'bySource':{name:int(np.sum(selected&(sources==i)))for i,name in enumerate(['front','back'])}}
 rows={'V10 rigid baseline':data};results={}
 for amount in amounts:
  local_factors=tangent*(amount*weights)[:,None];world_factors=np.einsum('ni,ij->nj',local_factors,BASIS);candidate=covariance_update(data,world_factors);name=f'tangent-sigma-{amount:g}'
  np.savez_compressed(OUT/(name+'.npz'),data=candidate,labels=labels,sources=sources,indices=ids)
  # Exact sparse source-addressed recipe, carried by subsequent rigid pose edits.
  bundles={}
  for si,source in enumerate(['front','back']):
   take=selected&(sources==si);order=np.argsort(ids[take]);selected_ids=ids[take][order];factor=world_factors[take][order]
   tr=report['parts'][12]['sourceToFrontRaw'] if si else {'scale':1.,'rotation':np.eye(3).tolist(),'translation':[0,0,0]}
   native_factor=np.einsum('ni,ji->nj',factor,np.array(tr['rotation']).T)/tr['scale'];n=np.load(NATIVE/f'{source}-full-native.npz');lookup_order=np.argsort(n['indices']);lookup=lookup_order[np.searchsorted(n['indices'][lookup_order],selected_ids)];assert np.array_equal(n['indices'][lookup],selected_ids)
   native_data=n['data'][lookup];guard=native_data[:,3:10];file=OUT/f'{name}-{source}-factors.npz';np.savez_compressed(file,indices=selected_ids.astype(np.uint32),factors=native_factor,original_parameters=guard)
   roundtrip=tr['scale']*np.einsum('ni,ji->nj',native_factor,np.array(tr['rotation']));assert np.max(abs(roundtrip-factor))<1e-12
   bundles[source]={'file':str(file.relative_to(ROOT)),'count':len(selected_ids),'maximumNativeFactorLength':float(np.linalg.norm(native_factor,axis=1).max(initial=0)),'frame':'original source','fields':{'indices':'original uint32 PLY rows','factors':'native rank-one covariance factor, add factors outer factors','original_parameters':'exact original source data columns 3:10 guard'}}
   metadata={'version':1,'sourceCapture':source,'sourceFile':report['sources'][si]['file'],'sourceSha256':report['sources'][si]['sha256'],'sourceHashes':{m['file']:m['sha256']for m in report['sources']},'part':'right_hand','repairFile':str(file.relative_to(ROOT)),'uniqueSourceIndices':len(selected_ids),'status':'diagnostic reconstructed coverage; not approved','description':'Sparse covariance-only circumferential footprint extension on proposed +V index/middle/ring side strips. Original means, alpha, color and rows unchanged.','candidateEvidence':str((OUT/(name+'.json')).relative_to(ROOT))}
   (OUT/f'{name}-{source}.json').write_text(json.dumps(metadata,indent=2)+'\n')
  variance=np.einsum('ni,nij,nj->n',tangent,C,tangent);increment=3*(np.sqrt(variance+(amount*weights)**2)-np.sqrt(variance))
  lateral_increment=3*(np.sqrt(C[:,0,0]+local_factors[:,0]**2)-np.sqrt(C[:,0,0]));longitudinal_increment=local_factors[:,2]**2;normal_increment=np.einsum('ni,ni->n',normal,local_factors)**2
  assert np.array_equal(candidate[:,:3],data[:,:3]);assert np.array_equal(candidate[:,10:],data[:,10:]);assert np.array_equal(candidate[~selected],data[~selected])
  record={'status':'diagnostic reconstructed coverage candidate; not approved','method':'Add positive rank-one covariance in measured local circumferential tangent; no mean or pose change','baseline':str(BASE.relative_to(ROOT)),'extraSigmaAligned':amount,'selection':selection_report,'sourceHashes':{m['file']:m['sha256']for m in report['sources']},'sourceFactorBundles':bundles,'maximumTangentialThreeSigmaFootprintIncrease':float(increment.max()),'medianSelectedTangentialThreeSigmaIncrease':float(np.median(increment[selected])),'maximumLateralThreeSigmaIncrease':float(lateral_increment.max()),'maximumLongitudinalVarianceIncrease':float(longitudinal_increment.max()),'maximumMeasuredNormalVarianceIncrease':float(normal_increment.max()),'meansOpacityColorAndRowsUnchanged':True,'minimumSelectedCovarianceEigenvalue':float(np.linalg.eigvalsh(covariance(candidate[selected])).min()),'selectionRecipe':{'fingers':['index','middle','ring'],'side':'plusV','longitudinalBounds':[.111,.174],'fadeInEnd':.122,'fadeOutStart':.165,'boundaryNeighborhoodOuterRadius':.0045,'boundaryNeighborhoodInnerRadius':.0015,'minimumNativeObservation':'seen > opposite_seen, seen > .05, observed_ratio > .03','minimumCrossSectionNormalLength':.75,'minimumAbsoluteTangentDepthComponent':.6,'existingRetainedRowsOnly':True},'evidence':'raw/fusion-work/refinement-v11/gap-audit/side-covariance-support.json','limitations':'This reconstructs missing Gaussian coverage. It is not recovered observed geometry. Requires actual grazing views and inter-finger silhouette/bridge review. No little-finger or terminal-tip support is inferred.','reproduce':'.venv-fusion/bin/python -B tools/fusion/extend_finger_support_v11.py --trials'}
  (OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');results[name]=record;rows[f'Reconstructed tangent sigma {amount:g}']=candidate;print(json.dumps(record),flush=True)
 aggregate=OUT/'coverage-extension-candidates.json';existing=json.loads(aggregate.read_text()) if aggregate.exists() else {};existing.update(results);aggregate.write_text(json.dumps(existing,indent=2)+'\n')
 views=[('Distal shallow palm',[-.626,-.15,.7798]),('Distal palm',[-.626,-.35,.7798]),('Below',[0,0,1]),('Outer',[-1,0,0]),('Front',[0,-1,0]),('Back',[0,1,0])]
 if render_atlas:atlas(rows,OUT/'coverage-extension-grazing.png',[-.765,.045,.678],.30,850,views=views,title='Reconstructed circumferential coverage only; original means, pose and alpha remain fixed')

def package():
 """Emit loader-compatible sparse metadata and verify native-frame roundtrip."""
 from gaussian_coverage import load_coverage_repairs,apply_coverage_repair
 report=json.loads((BASE/'report.json').read_text());proof={}
 for amount in [.0006,.0010,.0014,.0022]:
  name=f'tangent-sigma-{amount:g}';candidate_path=OUT/(name+'.npz')
  if not candidate_path.exists():continue
  candidate=np.load(candidate_path);paths=[]
  for si,source in enumerate(['front','back']):
   file=OUT/f'{name}-{source}-factors.npz';pack=dict(np.load(file))
   if 'quaternionAndLogScale' in pack:pack['original_parameters']=pack.pop('quaternionAndLogScale');np.savez_compressed(file,**pack)
   metadata={'version':1,'sourceCapture':source,'sourceFile':report['sources'][si]['file'],'sourceSha256':report['sources'][si]['sha256'],'sourceHashes':{m['file']:m['sha256']for m in report['sources']},'part':'right_hand','repairFile':str(file.relative_to(ROOT)),'uniqueSourceIndices':len(pack['indices']),'status':'diagnostic reconstructed coverage; not approved','description':'Covariance-only circumferential footprint extension on proposed +V shaft strips; no new observed geometry.','candidateEvidence':str((OUT/(name+'.json')).relative_to(ROOT))}
   path=OUT/f'{name}-{source}.json';path.write_text(json.dumps(metadata,indent=2)+'\n');paths.append(str(path.relative_to(ROOT)))
  repairs,_=load_coverage_repairs(paths,dict(zip(['front','back'],report['sources'])),ROOT);record={}
  for si,source in enumerate(['front','back']):
   n=np.load(NATIVE/f'{source}-full-native.npz');repair=repairs[source];corrected,count=apply_coverage_repair(n['data'],n['indices'],repair)
   unchanged_fields=np.r_[0:3,10:14];assert np.array_equal(corrected[:,unchanged_fields],n['data'][:,unchanged_fields])
   tr=report['parts'][12]['sourceToFrontRaw'] if si else {'scale':1.,'rotation':np.eye(3),'translation':np.zeros(3)}
   moved=transform_gaussians(corrected,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);take=(candidate['sources']==si)&np.isin(candidate['indices'],repair['indices']);target_ids=candidate['indices'][take];order=np.argsort(n['indices']);lookup=order[np.searchsorted(n['indices'][order],target_ids)];expected=covariance(moved[lookup]);actual=covariance(candidate['data'][take]);relative=np.linalg.norm(expected-actual,axis=(1,2))/np.linalg.norm(expected,axis=(1,2));assert relative.max()<3e-5
   native_C=covariance(n['data']);new_C=covariance(corrected);selected=np.isin(n['indices'],repair['indices']);delta=new_C[selected]-native_C[selected];minimum_delta_eigenvalue=float(np.linalg.eigvalsh(delta).min());record[source]={'count':count,'maximumNativeToAlignedCovarianceRelativeError':float(relative.max()),'minimumAddedCovarianceEigenvalueFloat32':minimum_delta_eigenvalue,'meansOpacityColorUnchanged':True,'minimumOutputCovarianceEigenvalue':float(np.linalg.eigvalsh(new_C[selected]).min())}
  proof[name]=record
 (OUT/'native-coverage-parity.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2),flush=True)

def angular_trials():
 """Extend only independently restored, observed sidewall seeds; diagnostic.

 The companion restoration supplies original source means, colors and native
 alpha. This function changes only their covariance, keeping measured normal
 and longitudinal variances fixed by a cross-section tangent rank-one update.
 """
 review=ROOT/'raw/fusion-work/refinement-v11/coverage-review';report=json.loads((BASE/'report.json').read_text());pack=dict(np.load(review/'angular-sidewall.npz'));original=dict(np.load(review/'baseline.npz'));data=pack['data'];labels=pack['labels'];sources=pack['sources'];ids=pack['indices'];selected=np.zeros(len(data),bool)
 for si,source in enumerate(['front','back']):selected|=(sources==si)&np.isin(ids,np.load(review/f'angular-sidewall-{source}-selected.npy'))&~np.isin(ids,original['indices'][original['sources']==si])
 assert selected.sum()==187 and np.all(labels[selected]==12)
 local=transform_gaussians(data,BASIS,-BASIS@ORIGIN);C=covariance(local);_,vectors=np.linalg.eigh(C);normal=vectors[:,:,0];tangent=np.c_[-normal[:,1],normal[:,0],np.zeros(len(data))];length=np.linalg.norm(tangent,axis=1);tangent/=np.maximum(length[:,None],1e-30)
 assert length[selected].min()>.55
 rows={'V10 baseline':np.load(review/'baseline.npz')['data'],'Restored native sidewall only':data};records={}
 for amount in [.0006,.0010,.0014,.0022]:
  name=f'angular-seed-sigma-{amount:g}';lf=tangent*(amount*selected)[:,None];wf=np.einsum('ni,ij->nj',lf,BASIS);assert np.isfinite(wf).all();candidate=covariance_update(data,wf);arrays={**pack,'data':candidate};np.savez_compressed(OUT/(name+'.npz'),**arrays);bundles={};paths=[]
  for si,source in enumerate(['front','back']):
   take=selected&(sources==si);order=np.argsort(ids[take]);chosen=ids[take][order];factor=wf[take][order];tr=report['parts'][12]['sourceToFrontRaw'] if si else {'scale':1.,'rotation':np.eye(3).tolist(),'translation':[0,0,0]};native_factor=factor@np.array(tr['rotation'])/tr['scale'];n=np.load(NATIVE/f'{source}-full-native.npz');ix=np.argsort(n['indices']);lookup=ix[np.searchsorted(n['indices'][ix],chosen)];assert np.array_equal(n['indices'][lookup],chosen)
   file=OUT/f'{name}-{source}-factors.npz';np.savez_compressed(file,indices=chosen.astype(np.uint32),factors=native_factor,original_parameters=n['data'][lookup,3:10]);meta={'version':1,'sourceCapture':source,'sourceFile':report['sources'][si]['file'],'sourceSha256':report['sources'][si]['sha256'],'sourceHashes':{m['file']:m['sha256']for m in report['sources']},'part':'right_hand','repairFile':str(file.relative_to(ROOT)),'uniqueSourceIndices':len(chosen),'status':'diagnostic reconstructed coverage; not approved','description':'Circumferential covariance extension on exact restored native +V index/middle/ring sidewall rows; requires companion original-source restoration. Original means, color, and native alpha retained.','candidateEvidence':str((OUT/(name+'.json')).relative_to(ROOT))};path=OUT/f'{name}-{source}.json';path.write_text(json.dumps(meta,indent=2)+'\n');paths.append(str(path.relative_to(ROOT)));bundles[source]={'metadata':str(path.relative_to(ROOT)),'count':len(chosen),'maximumNativeFactorLength':float(np.linalg.norm(native_factor,axis=1).max())}
  variance=np.einsum('ni,nij,nj->n',tangent,C,tangent);increase=3*(np.sqrt(variance+(amount*selected)**2)-np.sqrt(variance));lateral=3*(np.sqrt(C[:,0,0]+lf[:,0]**2)-np.sqrt(C[:,0,0]));assert np.array_equal(candidate[:,:3],data[:,:3]);assert np.array_equal(candidate[:,10:],data[:,10:]);assert np.array_equal(candidate[~selected],data[~selected])
  from gaussian_coverage import load_coverage_repairs,apply_coverage_repair
  repairs,_=load_coverage_repairs(paths,dict(zip(['front','back'],report['sources'])),ROOT);parity={}
  for si,source in enumerate(['front','back']):
   n=np.load(NATIVE/f'{source}-full-native.npz');repaired,count=apply_coverage_repair(n['data'],n['indices'],repairs[source]);tr=report['parts'][12]['sourceToFrontRaw'] if si else {'scale':1.,'rotation':np.eye(3),'translation':np.zeros(3)};moved=transform_gaussians(repaired,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);take=selected&(sources==si);ix=np.argsort(n['indices']);lookup=ix[np.searchsorted(n['indices'][ix],ids[take])];expected=covariance(moved[lookup]);actual=covariance(candidate[take]);relative=np.linalg.norm(expected-actual,axis=(1,2))/np.linalg.norm(expected,axis=(1,2));assert relative.max()<3e-5;parity[source]={'count':count,'maximumNativeToAlignedCovarianceRelativeError':float(relative.max())}
  record={'status':'diagnostic reconstructed coverage; not approved','units':'uncalibrated scan units','restorationSelection':{'originalRowsAtNativeAlpha':515,'previouslyRetainedRowsAtNativeAlpha':328,'newlyRetainedRows':187,'covarianceExtendedRows':187,'frontRestoredRows':265,'backRestoredRows':250},'extraSigmaAligned':amount,'selectedRows':187,'bySource':{'front':48,'back':139},'maximumTangentialThreeSigmaFootprintIncrease':float(increase.max()),'maximumLateralThreeSigmaFootprintIncrease':float(lateral.max()),'maximumLongitudinalVarianceIncrease':0,'maximumMeasuredNormalVarianceIncrease':float(np.max(np.einsum('ni,ni->n',normal,lf)**2)),'meansOpacityColorRowsUnchangedRelativeToNativeRestoration':True,'requiresNativeSourceRestoration':{s:str((review/f'angular-sidewall-{s}-selected.npy').relative_to(ROOT))for s in ['front','back']},'nativeSourceFactorBundles':bundles,'nativeFrameParity':parity,'selection':'Exact 187 original native sidewall observations; no further rows selected. All outside pad, tips and little finger.','minimumCrossSectionNormalLength':float(length[selected].min()),'reproduce':'.venv-fusion/bin/python -B tools/fusion/extend_finger_support_v11.py --angular'};(OUT/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n');records[name]=record;rows[f'Restored seed tangent sigma {amount:g}']=candidate;print(json.dumps(record),flush=True)
 (OUT/'angular-seed-evidence.json').write_text(json.dumps(records,indent=2)+'\n')
 views=[('Outer palm slight',[-1,-.1,.15]),('Inner palm slight',[1,-.1,.15]),('Below',[0,0,1]),('Distal palm',[-.626,-.15,.7798]),('Side +V',BASIS[0]),('Side -V',-BASIS[0])]
 atlas(rows,OUT/'angular-seed-grazing.png',[-.785,.055,.705],.235,760,views=views,title='Diagnostic reconstructed covariance on 187 restored native sidewall seeds / fixed means and native alpha')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--trials',action='store_true');parser.add_argument('--package',action='store_true');parser.add_argument('--angular',action='store_true');parser.add_argument('--arc-large',action='store_true');args=parser.parse_args()
 if args.trials:trials()
 if args.package:package()
 if args.angular:angular_trials()
 if args.arc_large:trials((.0022,),False)
