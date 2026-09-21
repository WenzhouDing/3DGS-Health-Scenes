#!/usr/bin/env python3
"""Local, source-addressed refinement of remaining left-hand dark splat artifacts."""
import argparse,json,sys,warnings
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from numba import njit
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import ROOT, BODY_PARTS, FIELDS, read_scan, transform_gaussians
from render_gaussians import load_fused,atlas,render
BASE=ROOT/'raw/fusion-work/refinement-v4/baseline';OUT=ROOT/'raw/fusion-work/refinement-v4/hand';OUT.mkdir(exist_ok=True)
VIEWS=[('Palm',[0,-1,0]),('Back',[0,1,0]),('Palm oblique',[-.65,-1,.1]),('Back oblique',[.65,1,.1]),('Thumb side',[-1,.15,0]),('Finger side',[1,.15,0])]
CENTER=[.759,.095,.647];SPAN=.265

def load():
 cache=OUT/'baseline-hand.npz';report=json.loads((BASE/'report.json').read_text());ids={p['id']:i for i,p in enumerate(report['parts'])}
 if cache.exists():c=np.load(cache);return c['data'],c['labels'],c['sources'],c['indices'],report
 data,labels,sources,report=load_fused(BASE);indices=np.load(BASE/'source-vertex-indices.npy');keep=np.isin(labels,[ids['left_hand'],ids['left_forearm']])&(data[:,0]>.635)
 data,labels,sources,indices=data[keep],labels[keep],sources[keep],indices[keep];np.savez_compressed(cache,data=data,labels=labels,sources=sources,indices=indices)
 return data,labels,sources,indices,report

def surface_statistics(native, opacity=.0, radius=.004, quantile=.2):
 core=native[(native[:,10]>opacity)&(np.exp(native[:,7:10].max(1))<.010)]
 dist,nn=cKDTree(core[:,[0,2]]).query(native[:,[0,2]],k=32,workers=-1)
 depth=np.where(dist<radius,core[nn,1],np.nan)
 with warnings.catch_warnings():
  warnings.simplefilter('ignore',RuntimeWarning)
  surface=np.nanquantile(depth,quantile,axis=1)
 return native[:,1]-surface

@njit
def _visibility_raster(xy,inverse,radius,alpha,width,height):
 trans=np.ones((height,width),np.float32);seen=np.zeros(len(xy));total=np.zeros(len(xy))
 for i in range(len(xy)):
  cx,cy=xy[i];rx,ry=radius[i];aa,ab,bb=inverse[i]
  for y in range(max(0,int(np.floor(cy-ry))),min(height-1,int(np.ceil(cy+ry)))+1):
   for x in range(max(0,int(np.floor(cx-rx))),min(width-1,int(np.ceil(cx+rx)))+1):
    dx=x+.5-cx;dy=y+.5-cy;power=-.5*(aa*dx*dx+2*ab*dx*dy+bb*dy*dy)
    if power < -4.5:continue
    a=min(.99,alpha[i]*np.exp(power))
    if a<1/255:continue
    seen[i]+=trans[y,x]*a;total[i]+=a;trans[y,x]*=1-a
 return seen/(total+1e-12),seen

def visibility(data,direction=(0,-1,0),size=850,span=.30):
 direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
 right=np.cross(direction,[0,0,-1]);right/=np.linalg.norm(right);vertical=np.cross(right,direction);basis=np.array([right,-vertical])
 xyz=data[:,:3].astype(float);xyz-=np.mean(xyz,axis=0)
 projected=np.einsum('ni,ji->nj',xyz,basis);scale=size/span
 q=data[:,3:7].astype(float);q/=np.linalg.norm(q,axis=1)[:,None]
 axes=np.einsum('ij,njk->nik',basis,Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix())*np.exp(data[:,7:10])[:,None,:]
 cov=np.einsum('nik,njk->nij',axes,axes)*scale**2;cov[:,0,0]+=.3;cov[:,1,1]+=.3
 det=cov[:,0,0]*cov[:,1,1]-cov[:,0,1]**2
 inverse=np.c_[cov[:,1,1],-cov[:,0,1],cov[:,0,0]]/det[:,None]
 radii=3*np.sqrt(np.c_[cov[:,0,0],cov[:,1,1]])
 xy=projected*scale+size/2;alpha=1/(1+np.exp(-np.clip(data[:,10].astype(float),-40,40)))
 order=np.argsort(-np.einsum('ni,i->n',xyz,direction),kind='stable')
 ratio,seen=_visibility_raster(np.ascontiguousarray(xy[order]),np.ascontiguousarray(inverse[order]),np.ascontiguousarray(radii[order]),np.ascontiguousarray(alpha[order]),size,size)
 reverse=np.argsort(order);return ratio[reverse],seen[reverse]

def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',default='final',choices=['audit','peel','visibility','dominance','final']);p.add_argument('--skip-render',action='store_true');args=p.parse_args()
 d,l,s,idx,report=load();hid={p['id']:i for i,p in enumerate(report['parts'])}['left_hand']
 if args.mode=='audit':
  atlas({'Current both':d,'Front contribution':d[s==0],'Back contribution':d[s==1]},OUT/'source-audit.png',CENTER,SPAN,size=850,views=VIEWS,title='Left-hand actual Gaussian closeups, immutable v3 baseline');return
 excess=np.zeros(len(d));native=np.zeros_like(d)
 for si,source in enumerate(['front','back']):
  m=(s==si)&(l==hid)
  # Read and hash original sources rather than trusting a stale workspace cache.
  original,metadata=read_scan(ROOT/report['sources'][si]['file'])
  if metadata['sha256']!=report['sources'][si]['sha256']:raise ValueError('Source scan differs from immutable review baseline')
  native[m]=original[idx[m]]
  excess[m]=surface_statistics(native[m]);print(source,'excess',np.nanquantile(excess[m],[0,.1,.5,.9,.99,1]),flush=True)
 if args.mode in ['visibility','dominance','final']:
  ratios=np.ones(len(d));seen=np.ones(len(d))
  opposite_ratio=np.ones(len(d));opposite_seen=np.ones(len(d))
  for si,source in enumerate(['front','back']):
   m=(s==si)&(l==hid);ratios[m],seen[m]=visibility(native[m]);print(source,'ratio',np.quantile(ratios[m],[0,.1,.5,.9,1]),flush=True)
   if args.mode in ['dominance','final']:opposite_ratio[m],opposite_seen[m]=visibility(native[m],direction=(0,1,0))
  rgb=np.clip(.5+.28209479177387814*d[:,11:14],0,1);lum=np.einsum('ni,i->n',rgb,[.2126,.7152,.0722])
  np.savez_compressed(OUT/f'{args.mode}.npz',ratios=ratios,seen=seen,lum=lum,opposite_ratio=opposite_ratio,opposite_seen=opposite_seen)
  captures={'Current':d}
  if args.mode=='final':
   # Only reject dark layers whose visible alpha contribution is dominated by
   # the unobserved source side. This preserves primary-view skin and creases.
   rejected=(l==hid)&(opposite_seen>2*seen)&(lum<.38)
   longitudinal=(d[:,0]-.704)*.68+(d[:,2]-.584)*.7332
   rejected|=(l==hid)&(longitudinal>.10)&(opposite_seen>seen)&(lum<.40)
   keep=~rejected
   hashes={meta['file']:meta['sha256'] for meta in report['sources']}
   counts={}
   for si,source in enumerate(['front','back']):
    removed=np.unique(idx[(s==si)&rejected]).astype(np.uint32)
    mask=OUT/f'hand-v4-{source}.npy';np.save(mask,removed);counts[source]=len(removed)
    evidence={
     'version':1,'purpose':'remove residual dark hand and finger layers visible mainly from the unobserved side of each capture',
     'sourceCapture':source,'sourceFile':report['sources'][si]['file'],'sourceSha256':report['sources'][si]['sha256'],'sourceHashes':hashes,
     'baseline':str(BASE.relative_to(ROOT)),'maskFile':str(mask.relative_to(ROOT)),'uniqueSourceIndices':len(removed),
     'maskMeaning':'sorted original source PLY vertex rows','applyStage':'after coverage confidence weighting and opacity attenuation','part':'left_hand',
     'algorithm':{'method':'integrated front-to-back alpha contribution from actual anisotropic Gaussian rasterization',
      'nativeObservedCamera':[0,-1,0],'nativeUnobservedCamera':[0,1,0],'cameraCenter':'mean original native positions of baseline hand contribution',
      'imageSize':850,'span':.30,'projectionVariancePixels':.3,'sigmaCutoff':3,'minPixelOpacity':1/255,
      'opacity':'original source sigmoid opacity','primaryCriterion':'unobserved visible alpha flux > 2 * observed flux and DC luminance < 0.38',
      'fingerCriterion':'unobserved visible alpha flux > observed flux and DC luminance < 0.40',
      'fingerRegion':{'frame':'aligned front raw','originXZ':[.704,.584],'axisXZ':[.68,.7332],'minimumDot':.10},
      'luminanceWeights':[.2126,.7152,.0722],'scope':'baseline left_hand labels only; no forearm rows'},
     'rejectedTrials':'Tighter native depth shells barely affected dark finger layers; global visibility ratio 0.12 and distal dominance 0.3 caused skin pitting. No color replacement or geometry changes were used.',
     'proof':str((OUT/'recommended-before-after.png').relative_to(ROOT)),
     'detailProof':str((OUT/'recommended-fingers.png').relative_to(ROOT)),
     'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_hand_v4.py --mode final',
    }
    mask.with_suffix('.json').write_text(json.dumps(evidence,indent=2)+'\n')
   np.savez_compressed(OUT/'recommended.npz',data=d[keep],labels=l[keep],sources=s[keep],indices=idx[keep],keep=keep)
   print('FINAL',counts,'retained hand',int(((l==hid)&keep).sum()),flush=True)
   if not args.skip_render:
    atlas({'Before':d,'Dark layers removed':d[keep]},OUT/'recommended-before-after.png',CENTER,SPAN,size=850,views=VIEWS,title='Left palm and back / original Gaussian rows only / no recoloring')
    atlas({'Before':d,'Dark layers removed':d[keep]},OUT/'recommended-fingers.png',[.790,.108,.686],.16,size=950,views=[VIEWS[i] for i in [0,1,3,5]],title='Finger surface detail / original Gaussian rows only / no recoloring')
   return
  if args.mode=='dominance':
   for threshold in [5,2,1]:
    keep=(l!=hid)|(opposite_seen<=threshold*seen)|(lum>.38)
    captures[f'Opposite dominance {threshold}']=d[keep];np.savez_compressed(OUT/f'dominance-{threshold}.npz',keep=keep)
    print('dominance',threshold,'removed',(~keep).sum(),flush=True)
   atlas(captures,OUT/'dominance-trials.png',CENTER,SPAN,size=650,views=VIEWS,title='Remove dark splats seen mainly from the unobserved side');return
  for threshold in [.005,.03,.12]:
   keep=(l!=hid)|(ratios>=threshold)|(lum>.4)
   captures[f'Hidden dark ratio {threshold:.3f}']=d[keep]
   np.savez_compressed(OUT/f'visibility-{threshold:.3f}.npz',keep=keep)
   print('visibility',threshold,'removed',(~keep).sum(),flush=True)
  atlas(captures,OUT/'visibility-trials.png',CENTER,SPAN,size=650,views=VIEWS,title='Observed camera visibility / dark hidden layers / actual Gaussians');return
 captures={'Current':d}
 for threshold in [.008,.004,.002]:
  keep=(l!=hid)|~np.isfinite(excess)|(excess<=threshold)
  captures[f'Native peel {threshold:.3f}']=d[keep]
  np.savez_compressed(OUT/f'peel-{threshold:.3f}.npz',keep=keep)
  print('peel',threshold,'removed',(~keep).sum(),flush=True)
 atlas(captures,OUT/'peel-trials.png',CENTER,SPAN,size=650,views=VIEWS,title='Local observed-shell tolerance trials / actual Gaussians')

if __name__=='__main__':main()
