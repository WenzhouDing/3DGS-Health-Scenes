#!/usr/bin/env python3
"""Independent left-hand source registration experiments; never edits production."""
import os,json,argparse
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/mannequin-left-v6-mpl')
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares,minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pipeline import (ROOT,BODY_PARTS,read_scan,rotate,dot,apply_transform,
 transform_gaussians,clean_scan,coverage_weights,attenuate,load_cleanup_masks,
 apply_joint_boundaries,assign_source_parts)
from render_gaussians import load_fused,atlas
from render_gaussians import render
from scipy.ndimage import binary_fill_holes,binary_erosion,binary_closing,label as image_label
BASE=ROOT/'raw/fusion-work/refinement-v6/baseline';OUT=ROOT/'raw/fusion-work/refinement-v6/left-hand';OUT.mkdir(parents=True,exist_ok=True)
VIEWS=[('Palm',[0,-1,0]),('Back',[0,1,0]),('Fingers toward wrist',[.65,.1,.8]),('Palm from below',[.65,-.65,.8]),('Back from below',[.65,.65,.8]),('Outer edge',[1,0,0]),('Thumb edge',[-1,0,0]),('Nearly flat below',[.3,.04,1])]
CENTER=[.76,.095,.642]

def context():
 cfg=json.loads((BASE/'fusion-config.json').read_text());lf=json.loads((BASE/'front-landmarks.json').read_text())['landmarks'];lb=json.loads((BASE/'back-landmarks.json').read_text())['landmarks'];r=json.loads((BASE/'report.json').read_text());return cfg,lf,lb,r

def load():
 cache=OUT/'hand-baseline.npz';cfg,lf,lb,report=context()
 if cache.exists():return dict(np.load(cache)),cfg,lf,lb,report
 data,labels,sources,_=load_fused(BASE);ids=np.load(BASE/'source-vertex-indices.npy');m=np.isin(labels,[5,6])&(data[:,0]>.59);saved={'data':data[m],'labels':labels[m],'sources':sources[m],'ids':ids[m]}
 for i,name in enumerate(['front','back']):
  raw,meta=read_scan(ROOT/cfg[name]);expected=report['sources'][i];assert meta['sha256']==expected['sha256'];sel=(saved['labels']==6)&(saved['sources']==i);saved[name+'Native']=raw[saved['ids'][sel]];saved[name+'Indices']=saved['ids'][sel]
 np.savez_compressed(cache,**saved);return saved,cfg,lf,lb,report

def audit():
 c,cfg,lf,lb,r=load();d=c['data'];l=c['labels'];s=c['sources'];tr=r['parts'][6]['sourceToFrontRaw'];back=transform_gaussians(c['backNative'],np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);rows={'Fused v5':d,'Front contribution':d[s==0],'Back contribution':d[s==1],'Original front rows':c['frontNative'],'Original back rows':back}
 atlas(rows,OUT/'underside-source-audit.png',CENTER,.33,600,views=VIEWS,title='Left hand underside audit: same views, full anisotropic Gaussians; original opacity rows below')
 u=np.array([.68,0,.7332]);u/=np.linalg.norm(u);v=np.array([u[2],0,-u[0]]);w=np.array([0,1.,0]);basis=np.array([u,v,w]);p=np.array(lf['left_wrist']);fig,axes=plt.subplots(2,3,figsize=(16,8))
 for row,(name,dd) in enumerate([('front',c['frontNative']),('back',back)]):
  good=(dd[:,10]>0)&(np.exp(dd[:,7:10].max(1))<.01);q=rotate(dd[good,:3]-p,basis);rgb=np.clip(.5+.2820947918*dd[good,11:14],0,1)
  for col,(a,b) in enumerate([(0,1),(0,2),(1,2)]):
   ax=axes[row,col];ax.scatter(q[:,a],q[:,b],c=rgb,s=.8);ax.grid();ax.set_aspect('equal');ax.set_title(name+' '+['long','lateral','depth'][a]+'/'+['long','lateral','depth'][b])
 fig.tight_layout();fig.savefig(OUT/'source-projections.png',dpi=170)

def silhouette_fit():
 c,cfg,lf,lb,report=load();tr=report['parts'][6]['sourceToFrontRaw'];r0=np.array(tr['rotation']);t0=np.array(tr['translation']);scale=tr['scale'];back=transform_gaussians(c['backNative'],r0,t0,scale);size=800;span=.32;center=np.array(CENTER);u=np.array([.68,.7332]);u/=np.linalg.norm(u);wrist=np.array(lf['left_wrist'])[[0,2]];edges={};fig,axes=plt.subplots(1,3,figsize=(14,6))
 for k,d in [('front',c['frontNative']),('back',back)]:
  white=d.copy();white[:,11:14]=(1-.5)/.28209479177387814
  im=np.asarray(render(white,[0,-1,0],center,span,width=size,height=size));mask=im[:,:,0]>145;mask=binary_fill_holes(binary_closing(mask,iterations=2));labels,n=image_label(mask);counts=np.bincount(labels.ravel());counts[0]=0;mask=labels==counts.argmax();boundary=mask&~binary_erosion(mask);iy,ix=np.nonzero(boundary);points=np.c_[ix+.5-size/2,iy+.5-size/2]*(span/size)+center[[0,2]];points=points[dot(points-wrist,u)>.040];voxel=np.floor(points/.001).astype(int);_,ind=np.unique(voxel,axis=0,return_index=True);points=points[ind];edges[k]=points
  axes[0].scatter(points[:,0],points[:,1],s=2,label=k)
  from PIL import Image
  Image.fromarray(np.uint8(mask)*255).save(OUT/(k+'-silhouette.png'))
 fp,bp=edges['front'],edges['back'];pivot=fp.mean(0);tree=cKDTree(fp)
 def unpack(v,pts=bp):
  ang=v[0];mat=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]]);return rotate(pts-pivot,mat)+pivot+v[1:]
 def residual(v):
  moved=unpack(v);di,ind=tree.query(moved);d2,j=cKDTree(moved).query(fp);return np.r_[(moved-fp[ind]).ravel(),(moved[j]-fp).ravel()]
 runs=[]
 for degrees in [-10,-5,0,5,10]:
  s=least_squares(residual,[np.deg2rad(degrees),0,0],bounds=([np.deg2rad(-15),-.020,-.020],[np.deg2rad(15),.020,.020]),loss='soft_l1',f_scale=.002,max_nfev=150);runs.append((float(np.mean(np.minimum(residual(s.x)**2,.0001))),s.x))
 score,v=min(runs,key=lambda x:x[0]);moved=unpack(v);angle=v[0];q=Rotation.from_rotvec([0,-angle,0]).as_matrix();p3=np.array([pivot[0],0,pivot[1]]);dt=np.array([v[1],0,v[2]]);r=q@r0;t=q@(t0-p3)+p3+dt;metrics={}
 for key,pts in [('before',bp),('after',moved)]:
  dist=np.r_[tree.query(pts)[0],cKDTree(pts).query(fp)[0]];metrics[key]={'mean':float(np.mean(dist)),'median':float(np.median(dist)),'p90':float(np.quantile(dist,.9))}
 axes[1].scatter(fp[:,0],fp[:,1],s=2,label='front');axes[1].scatter(moved[:,0],moved[:,1],s=2,label='back aligned');axes[2].scatter(bp[:,0],bp[:,1],s=2,label='before');axes[2].scatter(moved[:,0],moved[:,1],s=2,label='after')
 for ax in axes:ax.set_aspect('equal');ax.legend();ax.grid()
 fig.tight_layout();fig.savefig(OUT/'silhouette-fit.png',dpi=160)
 result={'method':'same physical projected finger outline; rotation about view-depth Y plus XZ translation only; native shell thickness/depth retained','adjustmentDegreesY':float(-np.degrees(angle)),'translationXZ':v[1:].tolist(),'pivotFrontRaw':p3.tolist(),'sourceToFrontRaw':{'scale':scale,'rotation':r.tolist(),'translation':t.tolist()},'metrics':metrics,'edgeCounts':{k:len(p) for k,p in edges.items()}}
 (OUT/'silhouette-fit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2));data=c['data'].copy();sel=(c['labels']==6)&(c['sources']==1);data[sel]=transform_gaussians(data[sel],q,dt+p3-q@p3);np.savez_compressed(OUT/'silhouette-trial.npz',data=data,labels=c['labels'],sources=c['sources'],indices=c['ids']);atlas({'V5 baseline':c['data'],'Projected outline fit':data},OUT/'silhouette-trial.png',CENTER,.33,650,views=VIEWS+[('Feet up',[0,0,1]),('Distal along hand',[.7,0,.7])],title='Left hand: retain shell depth while matching same finger outlines; exploratory fixed-opacity comparison')
def export_observed_caps(skip_render=False):
 """Rebuild accepted v6 hand selections from immutable PLY rows, not old output.

 The old planar coverage split erased curved fingertip skin. Restore only clean
 back rows visibly supported by the original observed surface, and discard
 primary-hidden layers of either color. Geometry and source opacity are original;
 the only opacity operation is the pipeline's existing optical-density weight.
 """
 from refine_hand_v4 import visibility
 c,cfg,lf,lb,report=load()
 parts=[tuple(list(p[:4])+cfg.get('partRadii',{}).get(p[0],list(p[4:]))) for p in BODY_PARTS]
 hid=next(i for i,p in enumerate(parts) if p[0]=='left_hand')
 native={};indices={};metadata={};stats={}
 cervical={}
 if cfg.get('cervicalSegmentation'):
  cervical=json.loads((ROOT/cfg['cervicalSegmentation']).read_text())['cervicalSegmentation']
 for si,source in enumerate(['front','back']):
  raw,meta=read_scan(ROOT/cfg[source]);metadata[source]=meta
  assert meta['sha256']==report['sources'][si]['sha256'],'Source differs from frozen baseline'
  landmarks=lf if source=='front' else lb
  clean,labels,statistics,ids=clean_scan(raw,landmarks,cfg['filter'],parts,
   cfg.get('exclusions',{}).get(source,[]),cfg.get('segmentationOverrides',{}).get(source,[]),
   cfg['filter'].get('partOverrides',{}).get(source,{}),cervical.get(source))
  labels,landmarks=apply_joint_boundaries(clean[:,:3],labels,landmarks,cfg.get('jointBoundaries',{}).get(source,[]),parts)
  # Existing source ownership rules are arm-only. Apply them nevertheless so the
  # export reproduces pipeline membership and fails if that assumption changes.
  selection_arrays=[]
  for rule in cfg.get('sourcePartAssignments',[]):
   document=json.loads((ROOT/rule['selection']).read_text())
   assert document['sourceHashes']=={m['file']:m['sha256'] for m in report['sources']}
   selection_arrays.append(np.load(ROOT/document['maskFile']) if document['sourceCapture']==source else np.empty(0,np.uint32))
  labels=assign_source_parts(labels,ids,selection_arrays,cfg.get('sourcePartAssignments',[]),parts)
  selected=labels==hid;native[source]=clean[selected];indices[source]=ids[selected]
  stats[source]={'sourceQuality':statistics,'fullCleanHand':int(selected.sum())}
  if source=='front':lf=landmarks
  else:lb=landmarks
  del raw,clean,labels,ids
  print('Full clean hand',source,len(native[source]),flush=True)
 hashes={m['file']:m['sha256'] for m in metadata.values()}
 old_masks,_=load_cleanup_masks(cfg.get('cleanupMasks',[]),metadata,ROOT)
 tr=report['parts'][hid]['sourceToFrontRaw']
 aligned={'front':native['front'],'back':transform_gaussians(native['back'],np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])}
 fw,bw=coverage_weights(aligned['front'],aligned['back'],lf,parts[hid],cfg['fusion'])
 weights={'front':fw,'back':bw};vis={};rejected={};restored={}
 origin=np.asarray(lf['left_wrist']);long_axis=np.array([.68,0,.7332]);lateral_axis=np.array([.7332,0,-.68])
 for source in ['front','back']:
  ratio,seen=visibility(native[source],direction=(0,-1,0),size=900,span=.30)
  opposite_ratio,opposite_seen=visibility(native[source],direction=(0,1,0),size=900,span=.30)
  vis[source]={'ratio':ratio,'seen':seen,'oppositeRatio':opposite_ratio,'oppositeSeen':opposite_seen}
  rejected[source]=opposite_seen>seen
  restored[source]=np.zeros(len(seen),bool)
  if source=='back':
   relative=aligned[source][:,:3]-origin
   region=(dot(relative,long_axis)>.105)&(dot(relative,lateral_axis)<.015)
   restored[source]=region&(seen>opposite_seen)&(ratio>.05)&(seen>.05)
  assert not np.any(rejected[source]&restored[source])
  np.savez_compressed(OUT/f'native-{source}-visibility.npz',**vis[source])
 recipe={'method':'front-to-back visible alpha flux from full quality-clean native hand Gaussians',
  'nativeObservedCamera':[0,-1,0],'nativeOppositeCamera':[0,1,0],
  'cameraCenter':'mean full quality-clean native hand positions','visibilitySize':900,'visibilitySpan':.30,
  'projectionVariancePixels':.3,'sigmaCutoff':3,'minimumPixelOpacity':1/255,
  'opacity':'original source sigmoid opacity','cleanupCriterion':'opposite visible alpha flux > observed visible alpha flux',
  'colorRestriction':None}
 def write_selection(source,selection,name,purpose,algorithm):
  selected=np.unique(indices[source][selection]).astype(np.uint32);path=OUT/(name+'.npy');np.save(path,selected)
  doc={'version':1,'purpose':purpose,'sourceCapture':source,'sourceFile':metadata[source]['file'],
   'sourceSha256':metadata[source]['sha256'],'sourceHashes':hashes,'baseline':str(BASE.relative_to(ROOT)),
   'maskFile':str(path.relative_to(ROOT)),'uniqueSourceIndices':len(selected),'part':'left_hand',
   'maskMeaning':'sorted original source PLY vertex rows','algorithm':algorithm,
   'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_left_hand_v6.py --export',
   'proof':str((OUT/'accepted-before-after.png').relative_to(ROOT))}
  path.with_suffix('.json').write_text(json.dumps(doc,indent=2)+'\n');return str(path.with_suffix('.json').relative_to(ROOT))
 cleanup=[]
 for source in ['front','back']:
  cleanup.append(write_selection(source,rejected[source],f'left-hand-v6-{source}',
   'Remove original hand layers seen more strongly from the unobserved source side; all colors',recipe))
 restoration_recipe={**recipe,'frame':'existing aligned front raw','origin':origin.tolist(),
  'longitudinalAxis':long_axis.tolist(),'lateralAxis':lateral_axis.tolist(),
  'minimumLongitudinal':.105,'maximumLateral':.015,'minimumObservedVisibilityRatio':.05,
  'minimumObservedVisibleAlphaFlux':.05,'observedFluxMustExceedOppositeFlux':True}
 restoration=write_selection('back',restored['back'],'left-back-tip-restoration',
  'Restore observed back fingertip skin truncated by the straight hand coverage plane',restoration_recipe)
 # Assemble from precoverage clean source arrays; this also simulates every old
 # cleanup mask. The restored rows are exempt from the union after attenuation.
 rows=[];labs=[];captures=[];source_ids=[]
 for si,source in enumerate(['front','back']):
  ids=indices[source];w=weights[source].copy();r=restored[source];w[r]=1.
  dd,coverage_keep=attenuate(aligned[source],w,cfg['fusion']['minWeight'])
  remove_old=np.isin(ids,old_masks[source]);remove_new=rejected[source]
  keep=(~(remove_old|remove_new)|r)[coverage_keep]
  dd=dd[keep];kept_ids=ids[coverage_keep][keep]
  baseline_ids=c['ids'][(c['labels']==hid)&(c['sources']==si)]
  assert np.all(np.isin(ids[r],kept_ids))
  stats[source].update({'baselineRetained':len(baseline_ids),'restorationSelected':int(r.sum()),
   'restorationNewlyRetained':int(np.sum(r&~np.isin(ids,baseline_ids))),
   'newMaskSelectedFullClean':int(remove_new.sum()),
   'newMaskPreviouslyRetained':int(np.sum(remove_new&np.isin(ids,baseline_ids))),
   'retained':len(dd),'geometryUnchanged':True,
   'baselineRowsRemoved':int(np.sum(~np.isin(baseline_ids,kept_ids))),
   'newRowsRetained':int(np.sum(~np.isin(kept_ids,baseline_ids)))})
  rows.append(dd);labs.append(np.full(len(dd),hid,np.uint8));captures.append(np.full(len(dd),si,np.uint8));source_ids.append(kept_ids)
 hand=np.concatenate(rows);handlabels=np.concatenate(labs);handsources=np.concatenate(captures);handids=np.concatenate(source_ids)
 np.savez_compressed(OUT/'candidate-hand.npz',data=hand,labels=handlabels,sources=handsources,indices=handids)
 context_mask=c['labels']!=hid
 final=np.concatenate([c['data'][context_mask],hand]);final_labels=np.r_[c['labels'][context_mask],handlabels]
 final_sources=np.r_[c['sources'][context_mask],handsources];final_indices=np.r_[c['ids'][context_mask],handids]
 np.savez_compressed(OUT/'candidate.npz',data=final,labels=final_labels,sources=final_sources,indices=final_indices)
 # Experimental masks were generated using a cached full-quality selection. This
 # comparison proves the raw-PLY rerun recreated the accepted masks exactly.
 trial=OUT/'All-color-dominance.npz'
 if trial.exists():
  previous=np.load(trial)
  assert np.array_equal(previous['frontRemoved'],indices['front'][rejected['front']])
  assert np.array_equal(previous['backRemoved'],indices['back'][rejected['back']])
  assert np.array_equal(previous['backRestore'],indices['back'][restored['back']])
 evidence={'version':1,'status':'accepted local candidate; production integration verified separately',
  'sourceHashes':hashes,'baseline':str(BASE.relative_to(ROOT)),'part':'left_hand','method':'native observed-surface recovery and unobserved-side rejection',
  'sourceToFrontRaw':tr,'geometryChanges':False,'colorChanges':False,
  'configAdditions':{'cleanupMasks':cleanup,'surfaceRestorations':[{'selection':restoration,'part':'left_hand','minimumWeight':1.0}]},
  'counts':stats,'cause':'Straight hand coverage plane removed real curved back finger caps; comparing only surviving shells falsely suggested articulated finger pose differences.',
  'selectionRecipe':restoration_recipe,
  'rejectedAlternatives':['Rigid silhouette correction moved already sound thumb/index and was unnecessary after full clean source inspection.',
   'Individual finger deformation was abandoned before implementation when full quality-clean original caps were found.',
   'Extra observed front-cap restoration gave negligible visual improvement and was omitted.',
   'Back restoration minimumWeight0.6 and1 produced near-identical renders;1 preserves original observed opacity.'],
  'review':'Six-angle anisotropic trial, including raw+Z and distal(.7,0,.7), preserves palm/back without new visible holes; caps improve and bright underside layering is reduced. Real creases and native scan softness remain.',
  'invariants':['No change to wrist, thumb/index pose, hand rigid transform, arm/cable, or original scans.',
   'Restoration is restricted to already quality-cleaned left_hand rows; no invalid/isolated/sparse-contact rescue.',
   'Full source opacity used for visibility; complementary palm/dorsal thickness is preserved.',
   'Selections recomputed from original PLYs and frozen baseline settings match accepted trial exactly.'],
  'proof':[str((OUT/n).relative_to(ROOT)) for n in ['accepted-before-after.png','accepted-feet-up-detail.png','all-color-dominance-trials.png','full-vs-kept-silhouettes.png']],
  'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_left_hand_v6.py --export'}
 (OUT/'left-hand-v6-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 print(json.dumps(stats,indent=2),flush=True)
 if not skip_render:
  views=[('Palm',[0,-1,0]),('Back',[0,1,0]),('Feet up / raw +Z',[0,0,1]),('Distal along fingers',[.7,0,.7]),
   ('Palm underside',[.65,-.65,.8]),('Back underside',[.65,.65,.8]),('Outer edge',[1,0,0]),('Thumb edge',[-1,0,0])]
  atlas({'V5 baseline':c['data'],'Observed caps + source dominance':final},OUT/'accepted-before-after.png',CENTER,.33,650,views=views,
   title='Left hand / original Gaussian geometry / restored observed caps and removed unobserved layers')
  render(final,[0,0,1],CENTER,.225,width=1250,height=850).save(OUT/'accepted-feet-up-detail.png')

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',action='store_true');p.add_argument('--silhouette',action='store_true');p.add_argument('--export',action='store_true');p.add_argument('--skip-render',action='store_true');a=p.parse_args()
 if a.audit:audit()
 if a.silhouette:silhouette_fit()
 if a.export:export_observed_caps(a.skip_render)
