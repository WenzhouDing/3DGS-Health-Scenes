#!/usr/bin/env python3
"""Reproduce skull/cervical-cuff alignment from a molded ear feature.

Use --rerender to recreate annotated orthographic Gaussian source images;
--render also saves a multiview fused head/cuff atlas. Source PLYs stay intact.
The frozen baseline camera transform in head-feature-evidence.json prevents
cumulative corrections if the main fused result changes.
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import (ROOT, BODY_PARTS, apply_transform, dot, read_scan,
                      clean_scan, transform_gaussians, coverage_weights,
                      attenuate, part_frame)
OUT=ROOT/'raw/fusion-work/head-refinement'
OUT.mkdir(parents=True,exist_ok=True)

def cervical_labels(points, labels, specification, parts=BODY_PARTS):
 """Respect visible skull and cuff seams instead of capsule competition there.

 specification is a single source entry from head-feature-evidence.json. The back
 reference transform only defines segmentation volumes; it does not move data.
 """
 ids={p[0]:i for i,p in enumerate(parts)};out=labels.copy();tr=specification['referenceTransform']
 p=apply_transform(points,np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
 skull=specification['skullPlane'];cuff=specification['cuffPlane']
 above=dot(p-np.array(skull['point']),np.array(skull['normal']))<0
 below=dot(p-np.array(cuff['point']),np.array(cuff['normal']))>0
 center=np.array(specification['cuffRadialCenter']);radii=np.array(specification['cuffRadialRadii'])
 radial=np.sum(((p[:,:2]-center)/radii)**2,axis=1)<1
 out[(out==ids['head'])&~above]=ids['neck']
 out[radial&above&np.isin(out,[ids['head'],ids['neck']])]=ids['head']
 out[radial&~above&~below]=ids['neck']
 out[(out==ids['neck'])&below]=ids['torso']
 return out


def fit_ear_image_registration(front_image,back_image,baseline_scale,scale):
 """Fix pitch using a real molded feature; compensate capture exposure only.

 Input images are side (-X) Gaussian renders, 1000 square, center
 [0,-.04,-.135], span .49. Other rigid degrees of freedom stay at the reviewed
 baseline; the 3D ear canal then fixes translation. Scale remains global.
 """
 from scipy.ndimage import map_coordinates
 from scipy.optimize import minimize
 from PIL import Image
 f=np.asarray(Image.open(front_image),dtype=float)/255;b=np.asarray(Image.open(back_image),dtype=float)/255
 y,x=np.mgrid[480:558,454:548];offset=np.stack([x-490,y-515],axis=2).reshape(-1,2);color=f[y,x].reshape(-1,3)
 mask=(offset[:,0]>-25)&(np.sum((offset/[43,38])**2,axis=1)<1);offset=offset[mask];color=color[mask];relative=scale/baseline_scale
 def error(v):
  angle,dx,dy=v;a=np.deg2rad(angle);R=np.array([[np.cos(a),-np.sin(a)],[np.sin(a),np.cos(a)]])
  q=np.einsum('ni,ji->nj',offset,R)/relative+[493+dx,582+dy]
  sampled=np.array([map_coordinates(b[:,:,i],q.T[::-1],order=1) for i in range(3)]).T
  design=np.c_[sampled,np.ones(len(sampled))];exposure=np.linalg.lstsq(design,color,rcond=None)[0]
  residual=np.linalg.norm(np.einsum('ni,ij->nj',design,exposure)-color,axis=1)
  return float(np.minimum(residual,.15).mean())
 fit=minimize(error,[28,0,0],method='Powell',bounds=[(0,45),(-8,8),(-8,8)],options={'xtol':1e-5,'ftol':1e-9})
 return {'pitchDegrees':float(fit.x[0]),'backPixelOffset':fit.x[1:].tolist(),'residual':float(fit.fun),'baselineResidual':error([0,0,0]),'renderRelativeScale':relative,'sweep':[[a,error([a,0,0])] for a in range(0,41,2)],'optimizerConverged':bool(fit.success),'feature':'Molded right ear canal and surrounding pinna; projected hair region excluded'}


def regenerate_feature_fit(scale=.928,rerender=False):
 """Reproduce the fixed-feature head/cuff fit and native seam definitions."""
 from render_gaussians import render
 path=ROOT/'tools/fusion/head-feature-evidence.json';record=json.loads(path.read_text());base=record['baselineRenderTransform'];Rbase=np.array(base['rotation']);tbase=np.array(base['translation']);sbase=base['scale']
 if rerender or not (OUT/'front-right.png').exists() or not (OUT/'back-right.png').exists():
  cfg=json.loads((ROOT/'tools/fusion/fusion-config.json').read_text())
  for name in ['front','back']:
   data,meta=read_scan(ROOT/cfg[name])
   if meta['sha256']!=record['sourceHashes'][name]:raise ValueError('Feature annotations belong to a different source: '+name)
   lm=json.loads((ROOT/cfg[name+'Landmarks']).read_text())['landmarks'];region=(data[:,2]<(.10 if name=='front' else -.30))&(abs(data[:,0])<.19)
   data,_,_,_=clean_scan(data[region],lm,cfg['filter'],BODY_PARTS)
   if name=='back':data=transform_gaussians(data,Rbase,tbase,sbase)
   render(data,[-1,0,0],[0,-.04,-.135],.49,width=1000,height=1000).save(OUT/f'{name}-right.png')
 reg=fit_ear_image_registration(OUT/'front-right.png',OUT/'back-right.png',sbase,scale)
 target=np.array(record['frontEarCenter']);pixel=np.array([493,582])+reg['backPixelOffset']
 baseline_ear=np.array([-.09523778,-.04-(pixel[0]-500)*.49/1000,-.135+(pixel[1]-500)*.49/1000])
 native=Rbase.T@(baseline_ear-tbase)/sbase;R=Rotation.from_euler('x',reg['pitchDegrees'],degrees=True).as_matrix()@Rbase;t=target-scale*R@native
 transform={'scale':scale,'rotation':R.tolist(),'translation':t.tolist()};record.update({'backNativeEarCenter':native.tolist(),'finalPitchCorrectionDegrees':reg['pitchDegrees'],'finalScale':scale,'sourceToFrontRaw':transform,'finalPhotometricRegistration':reg})
 for name in ['front','back']:
  spec=record['cervicalSegmentation'][name];RR=np.eye(3) if name=='front' else R;tt=np.zeros(3) if name=='front' else t;ss=1 if name=='front' else scale
  spec['referenceTransform']={'scale':ss,'rotation':RR.tolist(),'translation':tt.tolist()}
  for key in ['skull','cuff']:
   plane=spec[key+'Plane'];spec['native'+key.title()+'Plane']={'point':(RR.T@(np.array(plane['point'])-tt)/ss).tolist(),'normal':(RR.T@np.array(plane['normal'])).tolist()}
 path.write_text(json.dumps(record,indent=2)+'\n')
 fits={'version':1,'sourceHashes':record['sourceHashes'],'parts':{}}
 for name in ['head','neck']:
  fits['parts'][name]={'sourceToFrontRaw':transform,'method':'Molded ear canal + pinna feature guided rigid fit' if name=='head' else 'Complete upper cervical cuff follows skull','reason':'Reviewed through actual anisotropic Gaussian renders from both sides, front, rear and oblique; sleeve is segmented at visible seam planes','fitEvidence':{'feature':'Molded right ear canal and surrounding pinna','pitchDegrees':reg['pitchDegrees'],'photometricResidualBefore':reg['baselineResidual'],'photometricResidualAfter':reg['residual'],'sourceFeature':native.tolist(),'frontFeature':target.tolist(),'commonScale':scale,'cervicalSegmentation':'tools/fusion/head-feature-evidence.json'}}
 (OUT/'final-head-transforms.json').write_text(json.dumps(fits,indent=2)+'\n')
 (OUT/'proposed-transforms.json').write_text(json.dumps(fits['parts'],indent=2)+'\n')
 return fits


def render_feature_fit(fits):
 """Render complete retained skull/cuff Gaussians from six fixed cameras."""
 from render_gaussians import atlas
 cfg=json.loads((ROOT/'tools/fusion/fusion-config.json').read_text())
 evidence=json.loads((ROOT/'tools/fusion/head-feature-evidence.json').read_text())
 lf=json.loads((ROOT/cfg['frontLandmarks']).read_text())['landmarks'];sources={}
 for name in ['front','back']:
  data,meta=read_scan(ROOT/cfg[name])
  if meta['sha256']!=evidence['sourceHashes'][name]:raise ValueError('Feature source changed: '+name)
  lm=json.loads((ROOT/cfg[name+'Landmarks']).read_text())['landmarks']
  region=(data[:,2]<(.12 if name=='front' else -.26))&(abs(data[:,0])<.20)
  data,labels,_,_=clean_scan(data[region],lm,cfg['filter'],BODY_PARTS)
  labels=cervical_labels(data[:,:3],labels,evidence['cervicalSegmentation'][name])
  sources[name]=(data,labels)
 pieces={'front':[],'back':[]}
 for i,part in enumerate(BODY_PARTS):
  if part[0] not in ('head','neck'):continue
  f=sources['front'][0][sources['front'][1]==i];b=sources['back'][0][sources['back'][1]==i]
  tr=fits['parts'][part[0]]['sourceToFrontRaw'];b=transform_gaussians(b,np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
  fw,bw=coverage_weights(f,b,lf,part,cfg['fusion'])
  # Contact-side source ghosts must not be rescued after seam reassignment.
  a,_,_,axis,_=part_frame(lf,part);bw=1/(1+np.exp(np.clip(-dot(b[:,:3]-a,axis)/cfg['fusion']['feather'],-40,40)))
  ff,_=attenuate(f,fw,cfg['fusion']['minWeight']);bb,_=attenuate(b,bw,cfg['fusion']['minWeight'])
  pieces['front'].append(ff);pieces['back'].append(bb)
 front=np.concatenate(pieces['front']);back=np.concatenate(pieces['back'])
 return atlas({'Front retained':front,'Back retained':back,'Fused':np.r_[front,back]},OUT/'head-cuff-feature-fit.png',[0,-.01,-.135],.49,size=600,title='Molded ear feature alignment; explicit skull/cuff seams; actual Gaussian rendering')


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--scale',type=float,default=.928);parser.add_argument('--rerender',action='store_true');parser.add_argument('--render',action='store_true');args=parser.parse_args()
 result=regenerate_feature_fit(args.scale,args.rerender)
 if args.render:render_feature_fit(result)
 print(json.dumps(result,indent=2))
