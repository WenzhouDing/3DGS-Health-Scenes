#!/usr/bin/env python3
"""Export observed left fingertip surface restorations from full native sources.

Writes only v7 scratch selections. Existing restoration rows remain untouched;
new selections are disjoint from every frozen baseline restoration. --raw-verify
rebuilds quality-clean hand membership directly from immutable PLY sources.
"""
import argparse,json
from pathlib import Path
import numpy as np
from pipeline import (ROOT,BODY_PARTS,read_scan,clean_scan,apply_joint_boundaries,
 assign_source_parts,load_cleanup_masks,transform_gaussians)
BASE=ROOT/'raw/fusion-work/refinement-v7/baseline'
OLD=ROOT/'raw/fusion-work/refinement-v6/left-hand'
OUT=ROOT/'raw/fusion-work/refinement-v7/cap-restoration'
ORIGIN=np.array([.704,.065,.584]);LONG=np.array([.68,0,.7332]);LONG/=np.linalg.norm(LONG)

def export(raw_verify=False):
 OUT.mkdir(parents=True,exist_ok=True);cfg=json.loads((BASE/'fusion-config.json').read_text());report=json.loads((BASE/'report.json').read_text());cached=np.load(OLD/'stage-data.npz');metadata={s:report['sources'][i] for i,s in enumerate(['front','back'])};hashes={m['file']:m['sha256'] for m in metadata.values()};previous=json.loads((OLD/'left-hand-v6-evidence.json').read_text());assert hashes==previous['sourceHashes'];assert previous['sourceToFrontRaw']==report['parts'][6]['sourceToFrontRaw']
 native={'front':cached['front'],'back':cached['backNative']};indices={s:cached[s+'Indices'] for s in ['front','back']};tr=report['parts'][6]['sourceToFrontRaw'];aligned={'front':native['front'],'back':transform_gaussians(native['back'],np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])}
 old={s:np.empty(0,np.uint32) for s in native}
 for rule in cfg.get('surfaceRestorations',[]):
  masks,_=load_cleanup_masks([rule['selection']],metadata,ROOT)
  for source in old:old[source]=np.unique(np.r_[old[source],masks[source]])
 checks={}
 if raw_verify:
  parts=[tuple(list(p[:4])+cfg.get('partRadii',{}).get(p[0],list(p[4:]))) for p in BODY_PARTS];cervical=json.loads((ROOT/cfg['cervicalSegmentation']).read_text())['cervicalSegmentation'] if cfg.get('cervicalSegmentation') else {}
  for source in native:
   raw,meta=read_scan(ROOT/cfg[source]);assert meta['sha256']==metadata[source]['sha256'];lm=json.loads((BASE/f'{source}-landmarks.json').read_text())['landmarks'];clean,labels,stats,ids=clean_scan(raw,lm,cfg['filter'],parts,cfg.get('exclusions',{}).get(source,[]),cfg.get('segmentationOverrides',{}).get(source,[]),cfg['filter'].get('partOverrides',{}).get(source,{}),cervical.get(source));labels,lm=apply_joint_boundaries(clean[:,:3],labels,lm,cfg.get('jointBoundaries',{}).get(source,[]),parts);sels=[]
   for rule in cfg.get('sourcePartAssignments',[]):
    masks,_=load_cleanup_masks([rule['selection']],metadata,ROOT);sels.append(masks[source])
   labels=assign_source_parts(labels,ids,sels,cfg.get('sourcePartAssignments',[]),parts);m=labels==6;assert np.array_equal(ids[m],indices[source]);assert np.array_equal(clean[m],native[source]);assert np.array_equal(raw[indices[source]],native[source]);checks[source]={'sourceSha256':meta['sha256'],'fullQualityCleanMembershipExact':True,'cachedRowsEqualOriginalPly':True,'count':int(m.sum())};print('Raw source membership verified',source,int(m.sum()),flush=True)
   del raw,clean,labels,ids
 outputs=[];stats={};proof={'sourceHashes':hashes,'baseline':str(BASE.relative_to(ROOT)),'sourceToFrontRaw':tr,'rawVerification':checks,'oldSelectionsPreserved':cfg.get('surfaceRestorations',[]),'geometryChanges':False,'colorChanges':False}
 current_ids=np.load(BASE/'source-vertex-indices.npy');current_sources=np.load(BASE/'capture-labels.npy');current_parts=np.load(BASE/'part-labels.npy')
 for si,source in enumerate(native):
  vis=np.load(OLD/f'native-{source}-visibility.npz');long=np.einsum('ni,i->n',aligned[source][:,:3]-ORIGIN,LONG);support=(long>.105)&(vis['seen']>vis['oppositeSeen'])&(vis['ratio']>.05)&(vis['seen']>.05);requested=np.unique(indices[source][support]).astype(np.uint32);added=np.setdiff1d(requested,old[source]).astype(np.uint32);assert len(np.intersect1d(added,old[source]))==0;union=np.union1d(old[source],added);assert np.all(np.isin(old[source],union));assert np.all(np.isin(union,indices[source]));name=f'left-hand-{source}-caps-v7';maskpath=OUT/(name+'.npy');np.save(maskpath,added)
  recipe={'method':'native observed-surface visible alpha flux from full quality-clean original hand Gaussians','nativeObservedCamera':[0,-1,0],'nativeOppositeCamera':[0,1,0],'visibilitySize':900,'visibilitySpan':.30,'cameraCenter':'mean full quality-clean native hand positions','projectionVariancePixels':.3,'sigmaCutoff':3,'minimumPixelOpacity':1/255,'opacity':'original source sigmoid opacity','frame':'frozen v7 aligned front raw','origin':ORIGIN.tolist(),'longitudinalAxis':LONG.tolist(),'minimumLongitudinal':.105,'minimumObservedVisibilityRatio':.05,'minimumObservedVisibleAlphaFlux':.05,'observedFluxMustExceedOppositeFlux':True,'lateralRestriction':None,'excludeAllExistingRestorationRows':True}
  doc={'version':1,'purpose':'Restore observed left finger curved surfaces removed by straight source coverage plane; added rows disjoint from existing restorations','sourceCapture':source,'sourceFile':metadata[source]['file'],'sourceSha256':metadata[source]['sha256'],'sourceHashes':hashes,'maskFile':str(maskpath.relative_to(ROOT)),'uniqueSourceIndices':len(added),'part':'left_hand','maskMeaning':'sorted original source PLY vertex rows; surface restoration, not deletion','algorithm':recipe,'baseline':str(BASE.relative_to(ROOT)),'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_left_caps_v7.py --raw-verify','sourceCache':'raw/fusion-work/refinement-v6/left-hand/stage-data.npz','sourceVisibilityCache':str((OLD/f'native-{source}-visibility.npz').relative_to(ROOT))};maskpath.with_suffix('.json').write_text(json.dumps(doc,indent=2)+'\n');outputs.append({'selection':str(maskpath.with_suffix('.json').relative_to(ROOT)),'part':'left_hand','minimumWeight':1.0});retained=current_ids[(current_sources==si)&(current_parts==6)];stats[source]={'fullCleanHand':len(native[source]),'observedCapPredicate':len(requested),'oldRestorationRows':len(old[source]),'addedRestorationRows':len(added),'unionRestorationRows':len(union),'addedRowsAbsentFromBaseline':int(np.sum(~np.isin(added,retained))),'addedRowsAlreadyRetainedAtPartialOrFullWeight':int(np.sum(np.isin(added,retained)))};print(source,stats[source],flush=True)
 proof.update({'counts':stats,'configAdditions':{'surfaceRestorations':outputs},'newSelectionsDisjointFromOld':True,'requestedSurfacesPreservedByUnion':True,'selectionRecipe':recipe,'note':'These are opacity/coverage restorations of observed original surfaces, not new or reconstructed geometry. Existing old back3964 remain active unchanged.'});evidence=OUT/'left-cap-restoration-v7-evidence.json';evidence.write_text(json.dumps(proof,indent=2)+'\n');print('Evidence',evidence,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw-verify',action='store_true');a=p.parse_args();export(a.raw_verify)
