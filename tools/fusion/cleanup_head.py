#!/usr/bin/env python3
"""Audit conservative scalp fusion edits using immutable v2 alignment and real splats."""
import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import *
from render_gaussians import atlas,render,load_fused
BASE=ROOT/'raw/fusion-work/cleanup-v3/baseline';OUT=ROOT/'raw/fusion-work/cleanup-v3/head';OUT.mkdir(exist_ok=True)
VIEWS=[('Front',[0,-1,0]),('Back',[0,1,0]),('Right',[-1,0,0]),('Left',[1,0,0]),('High rear',[-.6,1,-.65]),('Crown',[.3,.25,-1])]

def load():
 cache=OUT/'head-sources.npz'
 cfg=json.loads((BASE/'fusion-config.json').read_text());rep=json.loads((BASE/'report.json').read_text());lf=json.loads((ROOT/cfg['frontLandmarks']).read_text())['landmarks'];lb=json.loads((ROOT/cfg['backLandmarks']).read_text())['landmarks'];tr=next(p['sourceToFrontRaw'] for p in rep['parts'] if p['id']=='head');part=next(p for p in BODY_PARTS if p[0]=='head')
 if cache.exists():c=np.load(cache);return c['front'],c['back'],c['baseline'],lf,cfg,tr
 data,l,s,r=load_fused(BASE);headid=next(i for i,p in enumerate(r['parts']) if p['id']=='head');baseline=data[l==headid];result={'baseline':baseline}
 cervical=json.loads((ROOT/cfg['cervicalSegmentation']).read_text())['cervicalSegmentation']
 for name,lm in [('front',lf),('back',lb)]:
  d=np.load(ROOT/'raw/fusion-work'/f'{name}.npy',mmap_mode='r');roi=(abs(d[:,0])<.23)&(d[:,2]<( .12 if name=='front' else -.25));d=np.array(d[roi]);d,l,stats,idx=clean_scan(d,lm,cfg['filter'],BODY_PARTS,cfg['exclusions'].get(name,[]),(),None,cervical[name]);d=d[l==2]
  if name=='back':d=transform_gaussians(d,np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
  result[name]=d
 np.savez_compressed(cache,**result)
 return result['front'],result['back'],baseline,lf,cfg,tr

def fused(f,b,lf,cfg,options):
 settings=json.loads(json.dumps(cfg['fusion']));settings['partOverrides']['head'].update(options);part=next(p for p in BODY_PARTS if p[0]=='head');fw,bw=coverage_weights(f,b,lf,part,settings);ff,_=attenuate(f,fw,settings['minWeight']);bb,_=attenuate(b,bw,settings['minWeight']);return np.r_[ff,bb]

from hair_shape import DEFAULT_FULLNESS, hair_displacement, hair_jacobian, deform_head_hair

def experiment_fullness():
 f,b,baseline,lf,cfg,tr=load();part=next(p for p in BODY_PARTS if p[0]=='head');_,_,_,y,_=part_frame(lf,part);a=np.array(lf[part[2]]);depth=dot(b[:,:3]-a,y);primary=(b[:,1]>.035)&(b[:,2]<-.13);bw=1/(1+np.exp(np.clip(-depth/cfg['fusion']['feather'],-40,40)))
 audit={'posteriorPrimaryHairSamples':int(primary.sum()),'posteriorHairWeightQuantiles':np.quantile(bw[primary],[0,.01,.5,.99,1]).tolist()}
 captures={'Current':baseline,'Back favored .02':fused(f,b,lf,cfg,{'depthBias':-.02})}
 for amount in [.008,.014,.022]:
  ff,r1=deform_head_hair(f,{'amount':amount});bb,r2=deform_head_hair(b,{'amount':amount});out=fused(ff,bb,lf,cfg,{});captures[f'Hair fullness {amount}']=out;np.save(OUT/f'fullness-{amount}.npy',out);audit[str(amount)]={'front':r1,'back':r2}
 (OUT/'fullness-trials.json').write_text(json.dumps(audit,indent=2)+'\n');atlas(captures,OUT/'fullness-trials.png',[0,.015,-.17],.40,size=590,views=VIEWS,title='Conservative hair-only appearance correction; rigid ears/face unchanged, Gaussian covariance warped')


def experiment_restore_primary_hair():
 """Control: restore original sparse posterior hair without moving any means."""
 f,b,baseline,lf,cfg,tr=load();lb=json.loads((ROOT/cfg['backLandmarks']).read_text())['landmarks']
 cervical=json.loads((ROOT/cfg['cervicalSegmentation']).read_text())['cervicalSegmentation']['back']
 raw=np.load(ROOT/'raw/fusion-work/back.npy',mmap_mode='r');roi=(abs(raw[:,0])<.22)&(raw[:,2]<-.39);d=np.array(raw[roi])
 labels,distance=segment_points(d[:,:3],lb,BODY_PARTS,cervical_spec=cervical)
 alpha=1/(1+np.exp(-np.clip(d[:,10],-40,40)));largest=np.exp(d[:,7:10].max(1))
 good=(labels==2)&(distance<1.8)&(alpha>.005)&(largest<.04)&(largest>1.5e-5)
 d=transform_gaussians(d[good],np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
 primary=(d[:,1]>.035)&(d[:,2]<-.13);old=(b[:,1]>.035)&(b[:,2]<-.13)
 candidate=fused(f,np.r_[b[~old],d[primary]],lf,cfg,{})
 np.save(OUT/'restored-primary-hair.npy',candidate)
 report={'originalPrimaryRetained':int(old.sum()),'restoredPrimaryBeforeFusion':int(primary.sum()),'minimumOpacity':.005,'maxScale':.04,'restorationBoundsFrontRaw':{'minimumY':.035,'maximumZ':-.13}}
 (OUT/'restore-primary-hair.json').write_text(json.dumps(report,indent=2)+'\n')
 atlas({'Current':baseline,'Restore sparse primary hair':candidate},OUT/'restore-primary-hair.png',[0,.015,-.17],.40,size=620,views=VIEWS,title='Restore original low-opacity primary posterior hair, without neighbor-density rejection')


if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--trials',action='store_true');parser.add_argument('--restore',action='store_true');args=parser.parse_args()
 if args.trials:experiment_fullness()
 elif args.restore:experiment_restore_primary_hair()
 else:
  f,b,baseline,lf,cfg,tr=load();atlas({'Current fused':baseline,'Front complete cleaned':f,'Back complete cleaned':b},OUT/'source-flatness-audit.png',[0,.015,-.17],.40,size=620,views=VIEWS,title='Current rigid pose; actual full Gaussian covariances; complete source shells before fusion weighting')
