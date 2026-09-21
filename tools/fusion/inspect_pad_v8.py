#!/usr/bin/env python3
"""Independent wrist-pad source/ownership inspection on frozen v8 baseline."""
import json,argparse,os
os.environ.setdefault("MPLCONFIGDIR","/tmp/pad-v8-mpl")
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from pipeline import (ROOT,read_scan,transform_gaussians,BODY_PARTS,clean_scan,apply_joint_boundaries,assign_source_parts,load_cleanup_masks)
from render_gaussians import atlas,load_fused,render
BASE=ROOT/'raw/fusion-work/refinement-v8/baseline';OUT=ROOT/'raw/fusion-work/refinement-v8/pad-inspection';OUT.mkdir(parents=True,exist_ok=True)
VIEWS=[('Front',[0,-1,0]),('Back',[0,1,0]),('Lateral',[1,0,0]),('Medial',[-1,0,0]),('Back oblique',[.4,1,.2]),('Under',[0,0,1])]

def initial():
 d,l,s,r=load_fused(BASE);indices=np.load(BASE/'source-vertex-indices.npy');ids={p['id']:i for i,p in enumerate(r['parts'])};rows={}
 for side,sign in [('left',1),('right',-1)]:
  selection=np.isin(l,[ids[side+'_forearm'],ids[side+'_hand']])&(sign*d[:,0]>.52);dd=d[selection].copy();dd[:,0]+=1.44 if side=='right' else 0.;center=[.72,.04,.58];rows[side]=dd
  np.savez_compressed(OUT/(side+'-baseline.npz'),data=d[selection],labels=l[selection],sources=s[selection],indices=indices[selection])
 atlas(rows,OUT/'side-identification.png',[.72,.055,.58],.42,600,views=VIEWS,title='Identify screenshot side / right translated for shared framing; original covariance preserved')

def full_native():
 cfg=json.loads((BASE/'fusion-config.json').read_text());report=json.loads((BASE/'report.json').read_text());metadata={s:report['sources'][i] for i,s in enumerate(['front','back'])};parts=[tuple(list(p[:4])+cfg.get('partRadii',{}).get(p[0],list(p[4:]))) for p in BODY_PARTS];cervical=json.loads((ROOT/cfg['cervicalSegmentation']).read_text())['cervicalSegmentation'] if cfg.get('cervicalSegmentation') else {}
 for source in ['front','back']:
  raw,meta=read_scan(ROOT/cfg[source]);assert meta['sha256']==metadata[source]['sha256'];lm=json.loads((BASE/f'{source}-landmarks.json').read_text())['landmarks'];clean,labels,stats,ids=clean_scan(raw,lm,cfg['filter'],parts,cfg.get('exclusions',{}).get(source,[]),cfg.get('segmentationOverrides',{}).get(source,[]),cfg['filter'].get('partOverrides',{}).get(source,{}),cervical.get(source));labels,lm=apply_joint_boundaries(clean[:,:3],labels,lm,cfg.get('jointBoundaries',{}).get(source,[]),parts);selections=[]
  for rule in cfg.get('sourcePartAssignments',[]):
   masks,_=load_cleanup_masks([rule['selection']],metadata,ROOT);selections.append(masks[source])
  labels=assign_source_parts(labels,ids,selections,cfg.get('sourcePartAssignments',[]),parts);m=np.isin(labels,[11,12]);np.savez_compressed(OUT/f'{source}-full-native.npz',data=clean[m],labels=labels[m],indices=ids[m]);print(source,'saved fullquality original forearm+hand',int(m.sum()),flush=True)
  del raw,clean,labels,ids

ORIGIN=np.array([-.697,.03,.597]);U=np.array([-.626,0,.7798]);U/=np.linalg.norm(U);V=np.array([U[2],0,-U[0]]);BASIS=np.array([U,V,[0,1.,0]])
def coords(data):return np.einsum('ni,ji->nj',data[:,:3]-ORIGIN,BASIS)
def pad_region(data):
 q=coords(data);return(q[:,0]>-.16)&(q[:,0]<.07)&(abs(q[:,1])<.067)&(abs(q[:,2])<.09)
def source_views():
 report=json.loads((BASE/'report.json').read_text());baseline=np.load(OUT/'right-baseline.npz');rows={};b=baseline['data'];sel=pad_region(b);rows['Current fused']=b[sel];label=baseline['labels'][sel];capture=baseline['sources'][sel];c=rows['Current fused'].copy();c[:,11:14]=(np.where((label==11)[:,None],np.array([.2,.85,.95]),np.array([1.,.55,.12]))-.5)/.2820947918;rows['Current part forearm cyan hand orange']=c
 for source in ['front','back']:
  native=np.load(OUT/f'{source}-full-native.npz');data=native['data'];labels=native['labels'];current=data.copy();by_parent={}
  if source=='back':
   for label in [11,12]:
    tr=report['parts'][label]['sourceToFrontRaw'];dd=transform_gaussians(data,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);by_parent[label]=dd;current[labels==label]=dd[labels==label]
  else:by_parent={11:data.copy(),12:data.copy()}
  selection=pad_region(current);current=current[selection];labels=labels[selection];rows[source+' native with current part transforms']=current;c=current.copy();c[:,11:14]=(np.where((labels==11)[:,None],np.array([.2,.85,.95]),np.array([1.,.55,.12]))-.5)/.2820947918;rows[source+' part ownership']=c
  if source=='back':
   for label in [11,12]:rows['back all moved by '+str(label)]=by_parent[label][selection]
 views=[('Dorsal',[0,1,0]),('Dorsal thumb oblique',[-.45,1,0]),('Dorsal other edge',[.5,1,0]),('Radial edge',[-1,0,0]),('Palmar',[0,-1,0])]
 atlas(rows,OUT/'pad-source-ownership.png',[-.676,.047,.574],.275,640,views=views,title='RIGHT dorsal wrist pad / complete original quality-clean sources / orange hand cyan forearm')
 np.savez_compressed(OUT/'source-views.npz',**{k.replace(' ','_'):v for k,v in rows.items()})


PAD_POLYGON=np.array([[-.123,.011],[-.122,.020],[-.115,.028],[-.106,.034],[-.088,.030],[-.058,.026],[-.020,.023],[.020,.023],[.041,.023],[.050,.017],[.056,.008],[.058,-.008],[.054,-.029],[.050,-.038],[.041,-.040],[.021,-.030],[-.016,-.021],[-.050,-.019],[-.084,-.015],[-.104,-.011],[-.115,-.004]])
def pad_mask():
 from matplotlib.path import Path as PolygonPath
 from scipy.spatial import cKDTree
 report=json.loads((BASE/'report.json').read_text());native=np.load(OUT/'back-full-native.npz');tr=report['parts'][11]['sourceToFrontRaw'];forearm=transform_gaussians(native['data'],np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);q=coords(forearm)
 inside=PolygonPath(PAD_POLYGON).contains_points(q[:,:2],radius=.002)
 depth_lower=.004+np.clip((q[:,0]+.12)/.18,0,1)*.016
 selected=inside&(q[:,2]>depth_lower)&(q[:,2]<.075)
 labels=native['labels'];indices=native['indices'];counts={report['parts'][label]['id']:int(np.sum(selected&(labels==label))) for label in [11,12]};assert min(counts.values())>100
 chosen=np.unique(indices[selected]).astype(np.uint32);assert len(chosen)==int(selected.sum());np.save(OUT/'back-right-rigid-pad.npy',chosen)
 old=forearm.copy();hand=report['parts'][12]['sourceToFrontRaw'];handdata=transform_gaussians(native['data'],np.array(hand['rotation']),np.array(hand['translation']),hand['scale']);old[labels==12]=handdata[labels==12];difference=np.linalg.norm(old[:,:3]-forearm[:,:3],axis=1)
 # Neighbour pairs from the same original continuous rim/surface on opposite
 # sides of label boundary prove a segmentation-induced discontinuity.
 ii=np.flatnonzero(selected&(labels==11));jj=np.flatnonzero(selected&(labels==12));di,neighbour=cKDTree(forearm[jj,:3]).query(forearm[ii,:3]);close=di<.005;left=ii[close];right=jj[neighbour[close]];native_distance=np.linalg.norm(forearm[left,:3]-forearm[right,:3],axis=1);old_distance=np.linalg.norm(old[left,:3]-old[right,:3],axis=1)
 meta={'version':1,'purpose':'Evidence-only continuous native dorsal RIGHT wrist pad; original rows crossing forearm/hand segmentation; not an opacity/deletion mask','sourceCapture':'back','sourceFile':report['sources'][1]['file'],'sourceSha256':report['sources'][1]['sha256'],'sourceHashes':{m['file']:m['sha256'] for m in report['sources']},'maskFile':str((OUT/'back-right-rigid-pad.npy').relative_to(ROOT)),'uniqueSourceIndices':len(chosen),'part':'right_wrist_pad','maskMeaning':'sorted original source PLY vertex rows','baseline':str(BASE.relative_to(ROOT)),
 'algorithm':{'frame':'back source mapped with frozen right_forearm sourceToFrontRaw','referenceTransform':tr,'origin':ORIGIN.tolist(),'axesLongitudinalLateralDepth':BASIS.tolist(),'polygonLocalUV':PAD_POLYGON.tolist(),'polygonExpansion':.002,'depthLower':'0.004 + clamp((U+0.12)/0.18,0,1)*0.016','depthUpper':.075,'selectionSource':'Full original quality-clean right forearm+hand; hashguarded original rows'},
 'currentPartCounts':counts,'continuity':{'crossPartCloseNeighbourPairs':len(left),'maximumNativePairDistance':.005,'nativePairDistanceMedian':float(np.median(native_distance)),'nativePairDistanceP90':float(np.quantile(native_distance,.9)),'oldIndependentPartPairDistanceMedian':float(np.median(old_distance)),'oldIndependentPartPairDistanceP90':float(np.quantile(old_distance,.9)),'oldPadHandRowsRelativeDisplacementMedian':float(np.median(difference[selected&(labels==12)])),'oldPadHandRowsRelativeDisplacementMaximum':float(difference[selected&(labels==12)].max())},
 'reproduce':'.venv-fusion/bin/python -B tools/fusion/inspect_pad_v8.py --native --views --mask','frontPadEvidence':'Front original contains contact-side sparse phantom surfaces here; no reliable matching dorsal perimeter. Only back mask is exported.',
 'proof':str((OUT/'pad-mask-proof.png').relative_to(ROOT))}
 (OUT/'back-right-rigid-pad.json').write_text(json.dumps(meta,indent=2)+'\n');np.savez_compressed(OUT/'pad-cross-part-pairs.npz',forearmSourceIndices=indices[left],handSourceIndices=indices[right],originalAlignedForearm=forearm[left,:3],originalAlignedHand=forearm[right,:3],oldSplitForearm=old[left,:3],oldSplitHand=old[right,:3]);print(json.dumps({'count':len(chosen),'parts':counts,'continuity':meta['continuity']},indent=2),flush=True)
 tint=forearm.copy();tint[selected,11:14]=(np.array([.08,.9,.85])-.5)/.2820947918;rows={'Native single rigid forearm transform':forearm[pad_region(forearm)],'Evidence pad cyan':tint[pad_region(forearm)],'Pad only':forearm[selected],'Old split pad only':old[selected]};atlas(rows,OUT/'pad-mask-proof.png',[-.676,.047,.574],.275,640,views=VIEWS,title='RIGHT original pad evidence / fixed continuous surface / source-index mask')


def verify_pad(directory):
 directory=Path(directory);d,l,s,report=load_fused(directory);indices=np.load(directory/'source-vertex-indices.npy');native=np.load(OUT/'back-full-native.npz');mask=np.load(OUT/'back-right-rigid-pad.npy');document=json.loads((OUT/'back-right-rigid-pad.json').read_text());assert document['sourceHashes']=={m['file']:m['sha256'] for m in report['sources']};selected=(s==1)&np.isin(indices,mask);selected_ids=indices[selected];order=np.argsort(native['indices']);lookup=np.searchsorted(native['indices'][order],selected_ids);assert np.array_equal(native['indices'][order][lookup],selected_ids);raw=native['data'][order[lookup]];tr=report['parts'][11]['sourceToFrontRaw'];expected=transform_gaussians(raw,np.array(tr['rotation']),np.array(tr['translation']),tr['scale']);actual=d[selected];position_error=np.linalg.norm(actual[:,:3]-expected[:,:3],axis=1)
 def covariance(a):
  quat=a[:,3:7].astype(float);quat/=np.linalg.norm(quat,axis=1)[:,None];R=Rotation.from_quat(np.c_[quat[:,1:],quat[:,0]]).as_matrix();axes=R*np.exp(a[:,None,7:10]);return np.einsum('nik,njk->nij',axes,axes)
 ec=covariance(expected);ac=covariance(actual);covariance_error=np.linalg.norm(ac-ec,axis=(1,2))/np.maximum(np.linalg.norm(ec,axis=(1,2)),1e-30);assert position_error.max()<2e-7;assert covariance_error.max()<2e-5
 proof={'directory':str(directory.relative_to(ROOT)),'sourceHashes':document['sourceHashes'],'retainedPadRows':int(selected.sum()),'parts':{report['parts'][label]['id']:int(np.sum(l[selected]==label)) for label in [11,12]},'meanMaximumDeviationFromOriginalSingleForearmRigidTransform':float(position_error.max()),'covarianceMaximumRelativeDeviation':float(covariance_error.max()),'allRetainedPadMeansAndCovarianceRemainRigid':True,'tolerances':{'position':2e-7,'relativeCovariance':2e-5},'reproduce':'.venv-fusion/bin/python -B tools/fusion/inspect_pad_v8.py --verify-dir '+str(directory.relative_to(ROOT))}
 (OUT/('verification-'+directory.name+'.json')).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--initial',action='store_true');p.add_argument('--native',action='store_true');p.add_argument('--views',action='store_true');p.add_argument('--mask',action='store_true');p.add_argument('--verify-dir',type=Path);a=p.parse_args()
 if a.initial:initial()
 if a.native:full_native()
 if a.views:source_views()
 if a.mask:pad_mask()
 if a.verify_dir:verify_pad(a.verify_dir if a.verify_dir.is_absolute() else ROOT/a.verify_dir)
