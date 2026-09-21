#!/usr/bin/env python3
"""Fixed-V10-pose right-hand coverage ablation; diagnostic outputs only.

Original means/covariances/colors are immutable. Multi-angle native visibility
was measured in V10 before alignment and is reusable under a new rigid pose.
No candidate here is accepted without the new-pose grazing-angle visual gate.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pipeline import ROOT,BODY_PARTS,transform_gaussians,coverage_weights,attenuate,load_cleanup_masks
from render_gaussians import load_fused,atlas,render

BASE=ROOT/'raw/fusion-work/refinement-v11/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v11/coverage-review'
NATIVE=ROOT/'raw/fusion-work/refinement-v8/pad-inspection'
VIS=ROOT/'raw/fusion-work/refinement-v6/right-hand'
MULTI=ROOT/'raw/fusion-work/refinement-v10/restoration'
ORIGIN=np.array([-.697,.03,.597]);LONG=np.array([-.626,0,.7798]);LONG/=np.linalg.norm(LONG)
CENTER=[-.785,.055,.705]
VIEWS=[('Below +Z',[0,0,1]),('Outer raw-X',[-1,0,0]),('Distal',[-.626,0,.78]),('Finger side A',[.78,0,.626]),('Finger side B',[-.78,0,-.626]),('Side B back grazing',[-.78,.15,-.626]),('Side B palm grazing',[-.78,-.15,-.626]),('Distal palm grazing',[-.626,-.15,.78])]

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 cfg=json.loads((BASE/'fusion-config.json').read_text());report=json.loads((BASE/'report.json').read_text());lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
 for rule in cfg['jointBoundaries']['front']:lm[rule['landmark']]=rule['pivot']
 tr=report['parts'][12]['sourceToFrontRaw'];assert tr==report['parts'][11]['sourceToFrontRaw']
 masks,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],report['sources'])),ROOT)
 native={};aligned={};ids={};selection={};stats={}
 for si,source in enumerate(['front','back']):
  z=np.load(NATIVE/f'{source}-full-native.npz');hand=z['labels']==12;data=z['data'][hand];native[source]=data;ids[source]=z['indices'][hand]
  assert len(data)==report['parts'][12][source+'Before']
  p=np.load(VIS/f'{source}-full-visibility.npz');v=np.load(MULTI/f'{source}-hemisphere-visibility.npz')
  np.testing.assert_array_equal(ids[source],p['indices']);np.testing.assert_array_equal(ids[source],v['indices'])
  aligned[source]=data if si==0 else transform_gaussians(data,np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
  station=np.einsum('ni,i->n',aligned[source][:,:3]-ORIGIN,LONG);assert np.isfinite(station).all()
  scales=np.sort(np.exp(data[:,7:10]),axis=1);alpha=1/(1+np.exp(-np.clip(data[:,10],-40,40)))
  q=data[:,3:7].astype(float);q/=np.linalg.norm(q,axis=1)[:,None];r=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix();normal=r[np.arange(len(data)),:,np.argmin(data[:,7:10],axis=1)]
  anchors=(p['observed_ratio']>.05)&(p['seen']>.1)&(p['seen']>p['opposite_seen'])&(scales[:,-1]<.01)&(alpha>.2)
  distance=cKDTree(data[anchors,:3]).query(data[:,:3])[0]
  rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1);lum=np.einsum('ni,i->n',rgb,[.2126,.7152,.0722]);assert np.isfinite(lum).all();skin=(rgb[:,1]>rgb[:,2]-.025)&(lum>.2)
  region=station>.105
  observed=region&(p['observed_ratio']>.05)&(p['seen']>.1)&(p['seen']>p['opposite_seen'])
  strict=region&(v['observedMaxRatio']>.04)&(v['observedMaxFlux']>.10)&(scales[:,-1]<.01)&(alpha>.12)&(abs(normal[:,1])<.7)&(scales[:,0]<.6*scales[:,1])&(distance<.004)&(v['observedMaxFlux']>.75*v['oppositeMaxFlux'])&skin
  selection[source]={'strong-observed-caps':observed,'connected-oblique':observed|strict,'full-native-distal-diagnostic':region}
  stats[source]={'qualityCleanRows':len(data),'sourceHash':report['sources'][si]['sha256'],'strongObservedSelected':int(observed.sum()),'strictObliqueAdditional':int((strict&~observed).sum())}
  np.savez_compressed(OUT/f'{source}-native-analysis.npz',data=data,aligned=aligned[source],indices=ids[source],station=station,primaryObserved=observed,connectedOblique=observed|strict,normal=normal)
 weights=coverage_weights(aligned['front'],aligned['back'],lm,BODY_PARTS[12],cfg['fusion'])
 baseline,bl,bs,_=load_fused(BASE);bi=np.load(BASE/'source-vertex-indices.npy');context=(bl==11)&(baseline[:,0]<-.61);hand=bl==12
 captures={'V10 baseline':baseline[context|hand]};evidence={'status':'diagnostic trials awaiting visual gate','baselineRevision':report['parameters']['revision'],'sourceToFrontRaw':tr,'meansOrCovarianceOrColorChanges':False,'selectionStatistics':stats,'candidates':{},'cameras':VIEWS,'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py'}
 for name in ['confidence-cleaned-full-alpha','strong-observed-caps','connected-oblique','full-native-distal-diagnostic']:
  rows=[baseline[context]];labels=[bl[context]];sources=[bs[context]];indices=[bi[context]];counts={}
  for si,source in enumerate(['front','back']):
   restored=np.zeros(len(ids[source]),bool) if name=='confidence-cleaned-full-alpha' else selection[source][name]
   w=weights[si].copy()
   if name=='confidence-cleaned-full-alpha':w[:]=1
   else:w[restored]=1
   data,k=attenuate(aligned[source],w,cfg['fusion']['minWeight']);keep=(~np.isin(ids[source],masks[source])|restored)[k]
   ii=ids[source][k][keep];rows.append(data[keep]);labels.append(np.full(len(ii),12,np.uint8));sources.append(np.full(len(ii),si,np.uint8));indices.append(ii)
   existing=bi[(bs==si)&hand];counts[source]={'retainedHandRows':len(ii),'newlyRetainedRows':int(np.sum(~np.isin(ii,existing))),'alphaRestorationSelections':int(restored.sum())}
   np.save(OUT/f'{name}-{source}-selected.npy',np.unique(ids[source][restored]).astype(np.uint32))
  arrays=dict(data=np.concatenate(rows),labels=np.concatenate(labels),sources=np.concatenate(sources),indices=np.concatenate(indices));np.savez_compressed(OUT/(name+'.npz'),**arrays);captures[name]=arrays['data'];evidence['candidates'][name]={'counts':counts,'geometryUnchanged':True,'diagnosticOnly':True}
  print(name,counts,flush=True)
 np.savez_compressed(OUT/'baseline.npz',data=baseline[context|hand],labels=bl[context|hand],sources=bs[context|hand],indices=bi[context|hand])
 atlas(captures,OUT/'fixed-pose-coverage-grazing.png',CENTER,.235,740,views=VIEWS,title='Fixed accepted V10 rigid pose / opacity and original source-row selection ablation / no geometry changes')
 for name,data in captures.items():
  file='baseline' if name=='V10 baseline' else name
  for view,direction in [('below',[0,0,1]),('outer',[-1,0,0]),('side-back',[-.78,.15,-.626])]:
   render(data,direction,CENTER,.21,width=1400,height=1000).save(OUT/f'{file}-{view}.png')
 (OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
def grazing_review():
 names=['baseline','confidence-cleaned-full-alpha','strong-observed-caps','connected-oblique','full-native-distal-diagnostic']
 captures={name:np.load(OUT/(name+'.npz'))['data'] for name in names}
 views=[('Outer palm slight',[-1,-.1,.15]),('Inner palm slight',[1,-.1,.15]),('Outer palm more',[-1,-.3,.15]),('Inner palm more',[1,-.3,.15]),('Outer distal grazing',[-1,-.05,.35]),('Inner distal grazing',[1,-.05,.35])]
 atlas(captures,OUT/'palm-leaning-grazing-trials.png',CENTER,.235,760,views=views,title='Current V10 pose / same original surfaces / finger side seams from palm-leaning angles')
 for name,data in captures.items():
  for vi,(view,direction) in enumerate(views):
   render(data,direction,CENTER,.21,width=1400,height=1000).save(OUT/f'{name}-palm-grazing-{vi}.png')
 evidence=json.loads((OUT/'evidence.json').read_text());evidence['additionalPalmLeaningCameras']=views;(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
def sector_review(broad=False):
 trial_name="broad-sidewall" if broad else "angular-sidewall"
 evidence=json.loads((OUT/'evidence.json').read_text());tr=evidence['sourceToFrontRaw'];report=json.loads((BASE/'report.json').read_text());cfg=json.loads((BASE/'fusion-config.json').read_text())
 fits=json.loads((ROOT/'raw/fusion-work/refinement-v10/gap-audit/resliced-shaft-center-audit.json').read_text())['candidates']['center-fit']['sections']
 V=np.array([LONG[2],0,-LONG[0]])
 baseline=dict(np.load(OUT/'baseline.npz'));restored={};stats={}
 for si,source in enumerate(['front','back']):
  a=np.load(OUT/f'{source}-native-analysis.npz');d=a['data'];aligned=a['aligned'];ids=a['indices'];normal=a['normal'];normal=normal if not si else np.einsum('ni,ji->nj',normal,np.array(tr['rotation']))
  q=np.column_stack([np.einsum('ni,i->n',aligned[:,:3]-ORIGIN,V),aligned[:,1]-ORIGIN[1],a['station']])
  vv=np.load(MULTI/f'{source}-hemisphere-visibility.npz');p=np.load(VIS/f'{source}-full-visibility.npz')
  sigma=np.sort(np.exp(d[:,7:10]),axis=1);alpha=1/(1+np.exp(-np.clip(d[:,10],-40,40)));rgb=np.clip(.5+.28209479177387814*d[:,11:14],0,1);lum=np.einsum('ni,i->n',rgb,[.2126,.7152,.0722])
  anchor=(p['observed_ratio']>.05)&(p['seen']>.1)&(p['seen']>p['opposite_seen'])&(sigma[:,-1]<.01)&(alpha>.2);dist=cKDTree(aligned[anchor,:3]).query(aligned[:,:3])[0]
  mask=np.zeros(len(d),bool);count={}
  for finger in ['index','middle','ring']:
   records=sorted([f for f in fits if f['finger']==finger and all(f['fits'][s]['accepted'] for s in ['front','back'])],key=lambda f:f['station'])
   stations=np.array([f['station'] for f in records]);centers=np.array([np.mean([f['fits'][s]['centerVY'] for s in ['front','back']],axis=0) for f in records]);radii=np.array([np.mean([f['fits'][s]['radius'] for s in ['front','back']]) for f in records])
   cv=np.interp(q[:,2],stations,centers[:,0]);cy=np.interp(q[:,2],stations,centers[:,1]);radius=np.interp(q[:,2],stations,radii)
   dv=q[:,0]-cv;dy=q[:,1]-cy;rad=np.sqrt(dv*dv+dy*dy);angle=np.arctan2(dy,dv)
   radial=(dv[:,None]*V+dy[:,None]*np.array([0,1,0]))/np.maximum(rad[:,None],1e-10)
   facing=abs(np.einsum('ni,ni->n',normal,radial))
   lo,hi={'index':[-.060,-.0235],'middle':[-.0235,.0055],'ring':[.0055,.0325]}[finger]
   chosen=(q[:,2]>.12)&(q[:,2]<.165)&(q[:,0]>lo)&(q[:,0]<hi)&(abs(angle)<np.radians(60))&(abs(rad-radius)<(.0035 if broad else .0025))&(facing>(.35 if broad else .55))&(sigma[:,-1]<.01)&(alpha>(.025 if broad else .08))&(sigma[:,0]<.8*sigma[:,1])&(dist<.006)
   if not broad:chosen&=(lum>.22)&(rgb[:,2]<rgb[:,1]+.025)&(vv['observedMaxFlux']>.05)&(vv['observedMaxRatio']>.015)
   mask|=chosen;count[finger]=int(chosen.sum())
  existing=baseline['indices'][baseline['sources']==si];selected=ids[mask];stats[source]={'selectedPerFinger':count,'selected':int(mask.sum()),'missingFromBaseline':int((~np.isin(selected,existing)).sum())}
  restored[source]=(aligned,ids,mask);np.save(OUT/f'{trial_name}-{source}-selected.npy',np.unique(selected).astype(np.uint32))
 rows=[baseline['data']];labels=[baseline['labels']];sources=[baseline['sources']];indices=[baseline['indices']]
 # Replace the selected retained rows as well, restoring native opacity exactly.
 for si,source in enumerate(['front','back']):
  d,ids,mask=restored[source];present=(baseline['sources']==si)&np.isin(baseline['indices'],ids[mask]);rows[0]=rows[0].copy();
  # Original source order is maintained through index lookup; only opacity can change on these rows.
  if present.any():
   pos=np.searchsorted(ids,baseline['indices'][present]);rows[0][present,10]=d[pos,10]
  new=mask&~np.isin(ids,baseline['indices'][baseline['sources']==si]);rows.append(d[new]);labels.append(np.full(new.sum(),12,np.uint8));sources.append(np.full(new.sum(),si,np.uint8));indices.append(ids[new])
 arrays=dict(data=np.concatenate(rows),labels=np.concatenate(labels),sources=np.concatenate(sources),indices=np.concatenate(indices));np.savez_compressed(OUT/(trial_name+'.npz'),**arrays)
 evidence['broadSidewallTrial' if broad else 'angularSidewallTrial']={'status':'low-confidence original-source restored coverage; diagnostic only' if broad else 'diagnostic only','counts':stats,'sectionEvidence':'raw/fusion-work/refinement-v10/gap-audit/resliced-shaft-center-audit.json','angleAroundPositiveV':[-60,60],'stationRange':[.12,.165],'radialTolerance':.0035 if broad else .0025,'normalRadialAlignmentMinimum':.35 if broad else .55,'nativeVisibilityRequired':not broad,'meansCovarianceColorUnchanged':True};(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 views=[('Outer palm slight',[-1,-.1,.15]),('Inner palm slight',[1,-.1,.15]),('Finger side positiveV',V),('Finger side negativeV',-V),('Palm',[0,-1,0]),('Back',[0,1,0])]
 atlas({'V10 baseline':baseline['data'],trial_name:arrays['data']},OUT/(trial_name+'-review.png'),CENTER,.235,850,views=views,title='Original native +V sidewall sector rows only / unchanged shape / measured shaft centers')
 for view,direction in [('outer',[-1,-.1,.15]),('inner',[1,-.1,.15])]:render(arrays['data'],direction,CENTER,.21,width=1400,height=1000).save(OUT/f'{trial_name}-{view}.png')
 print(json.dumps(stats),flush=True)
def stabilize_new_sidewall_colors(seed_name='angular-sidewall',all_selected=False):
 color_prefix=seed_name+('-all-color' if all_selected else '-color')
 from appearance_colors import load_color_overrides,apply_color_values
 import inspect_finger_seam_v11 as audit
 baseline=np.load(OUT/'baseline.npz');seed=np.load(OUT/(seed_name+'.npz'));arrays={k:seed[k].copy() for k in seed.files};report=json.loads((BASE/'report.json').read_text());hashes={m['file']:m['sha256'] for m in report['sources']};meta_paths=[];statistics={};allowed_color=np.zeros(len(seed['data']),bool)
 for si,source in enumerate(['front','back']):
  source_meta=report['sources'][si];a=np.load(OUT/f'{source}-native-analysis.npz');native=a['data'];aligned=a['aligned'];ids=a['indices'];q=audit.local(aligned)[:,:3];vis=np.load(VIS/f'{source}-full-visibility.npz')
  new=(seed['sources']==si)&(np.isin(seed['indices'],np.load(OUT/f'{seed_name}-{source}-selected.npy')) if all_selected else ~np.isin(seed['indices'],baseline['indices'][baseline['sources']==si]));positions=np.flatnonzero(new);allowed_color[positions]=True;selected_ids=seed['indices'][positions];sort=np.argsort(selected_ids);positions=positions[sort];selected_ids=selected_ids[sort];native_positions=np.searchsorted(ids,selected_ids);np.testing.assert_array_equal(ids[native_positions],selected_ids)
  rgb=np.clip(.5+.28209479177387814*native[:,11:14],0,1);lum=np.einsum('ni,i->n',rgb,[.2126,.7152,.0722]);sigma=np.exp(native[:,7:10]).max(1);alpha=1/(1+np.exp(-np.clip(native[:,10],-40,40)))
  trusted=np.isin(ids,baseline['indices'][baseline['sources']==si])&(vis['seen']>vis['opposite_seen'])&(vis['observed_ratio']>.03)&(alpha>.18)&(sigma<.008)&(lum>.20)&(lum<.85)&(rgb[:,2]<rgb[:,1]+.025)
  if all_selected:trusted&=~np.isin(ids,selected_ids)
  dc=native[native_positions,11:14].copy();original=dc.copy();donor_counts=[];neighbor_max=[];changed=0;skipped=0
  for j,row in enumerate(native_positions):
   band=next((b for b in audit.SUPPORT_BANDS.values() if b[0]<q[row,0]<b[1]),None)
   if band is None:raise ValueError('New sidewall row outside digit support bands')
   candidate=trusted&(q[:,0]>band[0])&(q[:,0]<band[1])&(abs(q[:,2]-q[row,2])<.008)
   donor=np.flatnonzero(candidate);dist=np.linalg.norm(aligned[donor,:3]-aligned[row,:3],axis=1);order=np.argsort(dist);donor=donor[order];dist=dist[order];within=dist<.005
   if within.sum()<8:within=dist<.008
   donor=donor[within][:36];dist=dist[within][:36]
   if len(donor)<4:
    skipped+=1;donor_counts.append(len(donor));neighbor_max.append(float(dist.max()) if len(dist) else 0.);continue
   colors=rgb[donor];median=np.median(colors,axis=0);mad=np.maximum(1.4826*np.median(abs(colors-median),axis=0),.015)
   robust=np.max(abs(colors-median)/mad,axis=1)<3;colors=colors[robust] if robust.sum()>=4 else colors;median=np.median(colors,axis=0);mad=np.maximum(1.4826*np.median(abs(colors-median),axis=0),.015)
   difference=abs(rgb[row]-median);outlier=(difference.max()>.035) and (np.max(difference/mad)>2.5)
   if outlier:dc[j]=(median-.5)/.28209479177387814;changed+=1
   donor_counts.append(len(colors));neighbor_max.append(float(dist.max()))
  np.testing.assert_array_equal(seed['data'][positions,11:14],original)
  arrays['data'][positions,11:14]=dc
  packed=OUT/f'{color_prefix}-{source}.npz';np.savez_compressed(packed,indices=selected_ids.astype(np.uint32),f_dc=dc.astype(np.float32),original_f_dc=original.astype(np.float32))
  document={'version':1,'purpose':'DIAGNOSTIC ONLY: stabilize color outliers on selected original finger sidewall rows only','sourceCapture':source,'sourceFile':source_meta['file'],'sourceSha256':source_meta['sha256'],'sourceHashes':hashes,'uniqueSourceIndices':len(selected_ids),'overrideFile':str(packed.relative_to(ROOT)),'method':'Same-source, same-finger trusted observed neighbors within .005 raw units (expand to .008 only if fewer than8); at most36 nearest, robust median and MAD; replace only >.035RGB and >2.5MAD outliers. No geometry, covariance or opacity changes.','selectedRowsExcludedFromDonors':all_selected,'actuallyChangedColors':changed,'unchangedInsufficientDonors':skipped,'maximumNeighborDistance':max(neighbor_max),'minimumDonorCount':min(donor_counts),'evidence':str((OUT/'evidence.json').relative_to(ROOT)),'reproduce':'.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py '+('--stabilize-all ' if all_selected else '--stabilize-seed ')+seed_name}
  path=OUT/f'{color_prefix}-{source}.json';path.write_text(json.dumps(document,indent=2)+'\n');meta_paths.append(str(path.relative_to(ROOT)));statistics[source]=document
  # Validate production loader and original-color guard against full native rows.
 overrides,_=load_color_overrides(meta_paths,dict(zip(['front','back'],report['sources'])),ROOT)
 for source in ['front','back']:
  a=np.load(OUT/f'{source}-native-analysis.npz');apply_color_values(a['data'][:,11:14],a['indices'],overrides[source])
 np.testing.assert_array_equal(arrays['data'][:,:11],seed['data'][:,:11])
 np.testing.assert_array_equal(arrays['data'][~allowed_color],seed['data'][~allowed_color])
 np.savez_compressed(OUT/(color_prefix+'.npz'),**arrays)
 evidence=json.loads((OUT/'evidence.json').read_text());evidence[color_prefix+'Trial']={'status':'diagnostic only','metadata':meta_paths,'statistics':statistics,'unchangedOutsideSelectedRows':True,'selectedRows':int(sum(d['uniqueSourceIndices'] for d in statistics.values())),'meansCovarianceAlphaUnchanged':True};(OUT/'evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
 compare_external(OUT/(color_prefix+'.npz'))
 print(json.dumps({source:{'selected':statistics[source]['uniqueSourceIndices'],'changed':statistics[source]['actuallyChangedColors'],'maximumNeighborDistance':statistics[source]['maximumNeighborDistance']} for source in statistics}),flush=True)

def combine_color_covariance(candidate_path):
 from appearance_colors import load_color_overrides,apply_color_values
 report=json.loads((BASE/'report.json').read_text());paths=[str((OUT/f'angular-sidewall-color-{source}.json').relative_to(ROOT)) for source in ['front','back']];overrides,_=load_color_overrides(paths,dict(zip(['front','back'],report['sources'])),ROOT)
 seed=np.load(candidate_path);arrays={k:seed[k].copy() for k in seed.files}
 for si,source in enumerate(['front','back']):
  m=arrays['sources']==si;arrays['data'][m,11:14],_=apply_color_values(arrays['data'][m,11:14],arrays['indices'][m],overrides[source])
 np.testing.assert_array_equal(arrays['data'][:,:11],seed['data'][:,:11])
 path=OUT/('color-'+Path(candidate_path).stem+'.npz');np.savez_compressed(path,**arrays);compare_external(path);print(path,flush=True)

def fix_index_color_patch():
 import colorsys
 from PIL import Image
 import inspect_finger_seam_v11 as audit
 from appearance_colors import load_color_overrides,apply_color_values
 seed=np.load(OUT/'broad-sidewall-all-color.npz');arrays={k:seed[k].copy() for k in seed.files};q=audit.local(seed['data']);lo,hi=audit.SUPPORT_BANDS['index'];take=(seed['labels']==12)&(q[:,0]>lo)&(q[:,0]<hi)&(q[:,2]>.105)&(q[:,2]<.198);rows=np.flatnonzero(take);d=q[take]
 center=np.array([(lo+hi)/2,.03,.15]);direction=np.array([1.,-.05,0]);direction/=np.linalg.norm(direction);right=np.cross(direction,[0,0,-1]);right/=np.linalg.norm(right);vertical=np.cross(right,direction);basis=np.array([right,-vertical]);scale=750/.11
 tile=render(d,direction,center,.11,width=750,height=750);hsv=np.asarray(tile.convert('HSV')).astype(float)/255;mask=(hsv[:,:,0]>.13)&(hsv[:,:,0]<.5)&(hsv[:,:,1]>.5)&(hsv[:,:,2]>.10);yy,xx=np.where(mask);assert len(xx)>0
 Image.fromarray((mask*255).astype(np.uint8)).save(OUT/'index-green-pixel-mask.png')
 xyz=d[:,:3].astype(float)-center;xy=np.einsum('ni,ji->nj',xyz,basis)*scale+375;qq=d[:,3:7].astype(float);qq/=np.linalg.norm(qq,axis=1)[:,None];axes=np.einsum('ij,njk->nik',basis,Rotation.from_quat(qq[:,[1,2,3,0]]).as_matrix())*np.exp(d[:,7:10])[:,None,:];cov=np.einsum('nik,njk->nij',axes,axes)*scale**2;cov[:,0,0]+=.3;cov[:,1,1]+=.3;det=cov[:,0,0]*cov[:,1,1]-cov[:,0,1]**2;inv=np.c_[cov[:,1,1],-cov[:,0,1],cov[:,0,0]]/det[:,None];alpha=1/(1+np.exp(-np.clip(d[:,10].astype(float),-40,40)));order=np.argsort(-np.einsum('ni,i->n',xyz,direction),kind='stable');trans=np.ones(len(xx));flux=np.zeros(len(d))
 for j in order:
  dx=xx+.5-xy[j,0];dy=yy+.5-xy[j,1];power=-.5*(inv[j,0]*dx*dx+2*inv[j,1]*dx*dy+inv[j,2]*dy*dy);a=np.minimum(.99,alpha[j]*np.exp(np.maximum(power,-100)));a[(power< -4.5)|(a<1/255)]=0;flux[j]=np.sum(trans*a);trans*=1-a
 rgb=np.clip(.5+.28209479177387814*d[:,11:14],0,1);hs=np.array([colorsys.rgb_to_hsv(*c) for c in rgb]);bad=(flux>.05)&(hs[:,0]>.11)&(hs[:,0]<.5)&(hs[:,1]>.55)&(d[:,2]>.115)&(d[:,2]<.135)
 chosen=np.flatnonzero(bad);assert len(chosen)>0;print('green patch contributors',len(chosen),[(int(seed['sources'][rows[j]]),int(seed['indices'][rows[j]]),float(flux[j])) for j in chosen],flush=True)
 report=json.loads((BASE/'report.json').read_text());changes=[];native={source:np.load(OUT/f'{source}-native-analysis.npz') for source in ['front','back']}
 for j in chosen:
  row=rows[j];si=int(seed['sources'][row]);other=['back','front'][si];a=native[other];oq=audit.local(a['aligned']);vis=np.load(VIS/f'{other}-full-visibility.npz');orrgb=np.clip(.5+.28209479177387814*a['data'][:,11:14],0,1);olum=np.einsum('ni,i->n',orrgb,[.2126,.7152,.0722]);oldselected=np.load(OUT/f'broad-sidewall-{other}-selected.npy')
  plausible=(oq[:,0]>lo)&(oq[:,0]<hi)&(abs(oq[:,2]-d[j,2])<.01)&(vis['seen']>vis['opposite_seen'])&(vis['observed_ratio']>.03)&(olum>.20)&(olum<.8)&(orrgb[:,0]>orrgb[:,1]*1.04)&(orrgb[:,1]>orrgb[:,2]*1.03)&~np.isin(a['indices'],oldselected)
  donor=np.flatnonzero(plausible);dist=np.linalg.norm(a['aligned'][donor,:3]-seed['data'][row,:3],axis=1);order=np.argsort(dist);donor=donor[order];dist=dist[order];keep=dist<.012;donor=donor[keep][:24];dist=dist[keep][:24];assert len(donor)>=4,(int(seed['indices'][row]),len(donor))
  median=np.median(orrgb[donor],axis=0);oldrgb=rgb[j];replacement=oldrgb+np.clip(median-oldrgb,-.25,.25);arrays['data'][row,11:14]=(replacement-.5)/.28209479177387814
  changes.append({'source':['front','back'][si],'originalRow':int(seed['indices'][row]),'localVYU':d[j,:3].tolist(),'greenMaskVisibleAlphaFlux':float(flux[j]),'oldRGB':oldrgb.tolist(),'replacementRGB':replacement.tolist(),'donorSource':other,'donorOriginalRows':a['indices'][donor].astype(int).tolist(),'maximumDonorDistance':float(dist.max()),'maximumRGBChannelChange':float(abs(replacement-oldrgb).max())})
 np.testing.assert_array_equal(arrays['data'][:,:11],seed['data'][:,:11]);outside=np.ones(len(seed['data']),bool);outside[rows[chosen]]=False;np.testing.assert_array_equal(arrays['data'][outside],seed['data'][outside])
 commands=['.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py --broad-sector','.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py --stabilize-all broad-sidewall','.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py --fix-green-patch']
 paths=[]
 for si,source in enumerate(['front','back']):
  old=np.load(OUT/f'broad-sidewall-all-color-{source}.npz');patchids=np.array([c['originalRow'] for c in changes if c['source']==source],np.uint32);ids=np.union1d(old['indices'],patchids).astype(np.uint32);a=native[source];pos=np.searchsorted(a['indices'],ids);np.testing.assert_array_equal(a['indices'][pos],ids);orig=a['data'][pos,11:14].copy()
  present=np.flatnonzero(seed['sources']==si);order=np.argsort(seed['indices'][present]);present=present[order];pos=np.searchsorted(seed['indices'][present],ids);np.testing.assert_array_equal(seed['indices'][present][pos],ids);dc=arrays['data'][present[pos],11:14]
  packed=OUT/f'broad-sidewall-final-color-{source}.npz';np.savez_compressed(packed,indices=ids,f_dc=dc,original_f_dc=orig)
  meta=json.loads((OUT/f'broad-sidewall-all-color-{source}.json').read_text());meta.update(purpose='Final diagnostic color stabilization on exact restored sidewall rows plus traced index green-patch contributors',uniqueSourceIndices=len(ids),overrideFile=str(packed.relative_to(ROOT)),actuallyChangedColors=int(np.any(dc!=orig,axis=1).sum()),targetedGreenPatchRows=len(patchids),reproduce='\n'.join(commands),reproduceCommands=commands,greenPatchMethod='Exactly traced back-capture contributors use plausible front-capture same-finger donors within about .0115 uncalibrated scan units; hard limit .012, maximum RGB-channel correction .25.',greenPatchEvidence=str((OUT/'green-patch-evidence.json').relative_to(ROOT)));path=OUT/f'broad-sidewall-final-color-{source}.json';path.write_text(json.dumps(meta,indent=2)+'\n');paths.append(str(path.relative_to(ROOT)))
 overrides,_=load_color_overrides(paths,dict(zip(['front','back'],report['sources'])),ROOT)
 for source in ['front','back']:apply_color_values(native[source]['data'][:,11:14],native[source]['indices'],overrides[source])
 np.savez_compressed(OUT/'broad-sidewall-final.npz',**arrays)
 evidence={'method':'Trace exact front-to-back alpha contributions to index green pixels in fixed canonical +V grazing camera; replace only offending source-row colors from plausible same-finger opposite-capture neighbors. Original guards preserved in merged overrides.','distanceUnits':'uncalibrated scan units','reproduceCommands':commands,'greenPixelCount':len(xx),'pixelBBox':[int(xx.min()),int(yy.min()),int(xx.max()),int(yy.max())],'changes':changes,'maximumAllowedNeighborDistance':.012,'maximumAllowedRGBChannelChange':.25,'allMeansCovarianceOpacityUnchanged':True,'allOtherColorsUnchangedFromAllSelectedColorTrial':True,'metadata':paths};(OUT/'green-patch-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n');compare_external(OUT/'broad-sidewall-final.npz');print(json.dumps({'tracedRows':len(changes),'metadata':paths}),flush=True)

def final_review():
 baseline=np.load(OUT/'baseline.npz');final=np.load(OUT/'broad-sidewall-final.npz')
 views=[('Below',[0,0,1]),('Outer finger side',[-1,-.1,.15]),('Inner finger side',[1,-.1,.15]),('Outer more palm',[-1,-.3,.15]),('Inner more palm',[1,-.3,.15]),('Palm',[0,-1,0]),('Back',[0,1,0]),('Dorsal low',[0,.1,1])]
 atlas({'V10 baseline':baseline['data'],'Restored original sidewalls + local color stabilization':final['data']},OUT/'final-whole-hand-review.png',[-.755,.055,.665],.30,850,views=views,title='Fixed rigid hand and forearm / original sidewall Gaussian shapes / before and final candidate')
 for name,data in [('baseline',baseline['data']),('final',final['data'])]:
  for vi,(view,direction) in enumerate(views[:5]):render(data,direction,CENTER,.235,width=1400,height=1000).save(OUT/f'{name}-gate-{vi}.png')
 print('final gallery ready',flush=True)

def compare_external(candidate_path):
 # Reuse the independent audit's exact canonical partitions/camera, with all
 # output redirected to this task-owned directory; no shared file mutation.
 import inspect_finger_seam_v11 as audit
 audit.OUT=OUT
 baseline=np.load(OUT/'baseline.npz');take=baseline['labels']==12
 sources={'baseline':{'fusedLocal':audit.local(baseline['data'][take])}}
 audit.compare_isolated(sources,Path(candidate_path))

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--grazing-only',action='store_true');parser.add_argument('--sector-only',action='store_true');parser.add_argument('--compare-external',type=Path);parser.add_argument('--stabilize-colors',action='store_true');parser.add_argument('--color-covariance',action='append',type=Path);parser.add_argument('--broad-sector',action='store_true');parser.add_argument('--stabilize-seed');parser.add_argument('--stabilize-all');parser.add_argument('--fix-green-patch',action='store_true');parser.add_argument('--final-review',action='store_true');args=parser.parse_args()
 if args.final_review:final_review()
 elif args.fix_green_patch:fix_index_color_patch()
 elif args.stabilize_all:stabilize_new_sidewall_colors(args.stabilize_all,True)
 elif args.broad_sector:sector_review(True)
 elif args.stabilize_seed:stabilize_new_sidewall_colors(args.stabilize_seed)
 elif args.color_covariance:
  for path in args.color_covariance:combine_color_covariance(path)
 elif args.stabilize_colors:stabilize_new_sidewall_colors()
 elif args.compare_external:compare_external(args.compare_external)
 elif args.sector_only:sector_review()
 else:
  if not args.grazing_only:main()
  grazing_review()
