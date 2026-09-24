#!/usr/bin/env python3
"""Bounded captured corner repair on the accepted pass4 ambulance baseline.

Only original iPhone opacity is suppressed. Reference geometry, covariance,
colors and all directional coefficients are copied unchanged. Actual mixed
scene visibility is traced so newly exposed old layers are diagnosed rather
than guessed from source point centers. Outputs use the assembler NPZ contract.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT,columns,read_ply,sha256_file,write_ply
from repair_surfaces import WORLD_FROM_RAW

GROUPS={'rear-corners': {'description': 'Observed rear padded crossbar and rounded right rear corner '
                                 'cover, with original clock volume and accepted surrounding '
                                 'repairs protected.',
                  'cameras': ['ceiling-rear', 'rear-wall', 'right-wall'],
                  'protected_boxes': [{'id': 'original-clock-and-face',
                                       'bounds': [[-30, 0.754, -0.268], [-1.2, 0.984, -0.032]]}],
                  'boxes': [{'id': 'rear-crossbar',
                             'bounds': [[-1.85, 0.725, -0.775], [-1.24, 0.985, 0.445]],
                             'feather': 0.022},
                            {'id': 'rounded-rear-right-cover',
                             'bounds': [[-1.36, 0.475, 0.42], [-0.88, 0.995, 1.03]],
                             'feather': 0.024}],
                  'surfaces': [{'id': 'rear-crossbar-front',
                                'axis': 0,
                                'uv_axes': [2, 1],
                                'uv_bounds': [[-0.72, 0.748], [0.405, 0.94]],
                                'plane': [0.093, 0.04, -1.44],
                                'behind_sign': -1,
                                'deep_bounds': [-30, -1.22],
                                'grid': [41, 9]},
                               {'id': 'rounded-cover-front',
                                'axis': 0,
                                'uv_axes': [2, 1],
                                'uv_bounds': [[0.475, 0.52], [0.96, 0.93]],
                                'plane': [0.20597509, 0.054804985, -1.18756658],
                                'behind_sign': -1,
                                'deep_bounds': [-30, -0.88],
                                'grid': [21, 17]}]},
 'front-header': {'description': 'Fixed front padded header, vent surround and upper right corner '
                                 'cover; original metal grille, red net and its foreground '
                                 'attachments protected.',
                  'cameras': ['front-wall', 'ceiling-front', 'right-grazing'],
                  'preserve_saturated': True,
                  'protected_boxes': [{'id': 'original-vent-grille',
                                       'bounds': [[1.56, 0.682, -0.012], [1.775, 0.807, 0.23]]},
                                      {'id': 'foreground-net-and-hooks',
                                       'bounds': [[0.5, 0.59, 0.18], [1.3, 1.08, 0.585]]}],
                  'boxes': [{'id': 'front-top-header',
                             'bounds': [[1.59, 0.858, -1.13], [2.04, 1.0, 0.55]],
                             'feather': 0.016},
                            {'id': 'front-vent-surround',
                             'bounds': [[1.6, 0.498, -0.14], [2.12, 0.885, 0.54]],
                             'feather': 0.018},
                            {'id': 'front-right-upper-corner',
                             'bounds': [[0.68, 0.607, 0.44], [1.76, 0.996, 0.945]],
                             'feather': 0.022}],
                  'surfaces': [{'id': 'front-top-header',
                                'axis': 0,
                                'uv_axes': [2, 1],
                                'uv_bounds': [[-1.04, 0.886], [0.49, 0.975]],
                                'plane': [0.097, -0.015, 1.782],
                                'behind_sign': 1,
                                'deep_bounds': [1.6, 30],
                                'grid': [41, 7]},
                               {'id': 'front-vent-surround',
                                'axis': 0,
                                'uv_axes': [2, 1],
                                'uv_bounds': [[-0.1, 0.53], [0.49, 0.852]],
                                'plane': [0.09, 0.0, 1.68],
                                'behind_sign': 1,
                                'deep_bounds': [1.61, 30],
                                'grid': [23, 15]},
                               {'id': 'front-right-upper-corner',
                                'axis': 2,
                                'uv_axes': [0, 1],
                                'uv_bounds': [[0.73, 0.645], [1.69, 0.939]],
                                'plane': [-0.09, 0.03, 0.723],
                                'behind_sign': 1,
                                'deep_bounds': [0.48, 30],
                                'grid': [29, 13]}]},
 'front-vertical': {'description': 'Visible fixed vertical corner cover beside front red net, '
                                   'bounded above monitor bag. Preserve foreground hooks, '
                                   'saturated restraints, bag and accepted bench.',
                    'cameras': ['equipment-bag', 'right-grazing', 'ceiling-front'],
                    'preserve_saturated': True,
                    'protected_boxes': [{'id': 'foreground-net-hardware',
                                         'bounds': [[0.5, 0.025, 0.43], [0.85, 0.68, 0.645]]}],
                    'boxes': [{'id': 'front-vertical-cover',
                               'bounds': [[0.65, -0.045, 0.602], [0.96, 0.697, 0.92]],
                               'feather': 0.02}],
                    'surfaces': [{'id': 'front-vertical-cover-front',
                                  'axis': 0,
                                  'uv_axes': [2, 1],
                                  'uv_bounds': [[0.645, -0.012], [0.87, 0.66]],
                                  'plane': [0.1, 0.01, 0.657],
                                  'behind_sign': 1,
                                  'deep_bounds': [0.66, 30],
                                  'grid': [17, 29]}]},
 'cabinet-end': {'description': 'Upper fixed black cabinet-end cover visible beside glazing in '
                                'ceiling-rear. Lower straps and the glass-side border stay outside '
                                'the patch.',
                 'cameras': ['ceiling-rear', 'left-wall', 'left-seats'],
                 'preserve_saturated': True,
                 'protected_boxes': [{'id': 'glass-side-border',
                                      'bounds': [[-30, -1, -2], [30, 1.1, -1.17]]}],
                 'boxes': [{'id': 'upper-cabinet-end-cover',
                            'bounds': [[-0.5, 0.34, -1.17], [-0.16, 0.82, -0.86]],
                            'feather': 0.023}],
                 'surfaces': [{'id': 'upper-cabinet-end-cover',
                               'axis': 0,
                               'uv_axes': [2, 1],
                               'uv_bounds': [[-1.14, 0.372], [-0.889, 0.785]],
                               'plane': [0.1, 0.0, -0.22],
                               'behind_sign': -1,
                               'deep_bounds': [-30, -0.17],
                               'grid': [19, 25]}]}}


def smoothstep(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)


def positions(v):
 return np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float))


def base_weights(p,group):
 w=np.zeros(len(p),np.float32);report=[]
 for box in group['boxes']:
  lo,hi=np.asarray(box['bounds']);edge=np.minimum(p-lo,hi-p).min(1);bw=smoothstep(edge/box['feather']);w=np.maximum(w,bw.astype(np.float32))
  report.append({**box,'selected_count':int((bw>0).sum())})
 for protected in group.get('protected_boxes',[]):
  plo,phi=np.asarray(protected['bounds']);w[((p>plo)&(p<phi)).all(1)]=0
 return w,report


def trace(v,p,group,primary_rows=None):
 scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float));alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
 cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in group['cameras']]
 selected_all=np.zeros(len(v),bool);records=[]
 for camera in cameras:
  eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward);right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=-np.cross(right,forward);V=np.array([right,down,forward])
  q=np.einsum('ij,nj->ni',V,p-eye);z=q[:,2];safe=np.maximum(z,.02);f=375/np.tan(np.radians(camera['fov'])/2)
  screen=np.c_[500+f*q[:,0]/safe,375+f*q[:,1]/safe];radius=4*f*scales.max(1)/safe+2
  for surface in group['surfaces']:
   ax=surface['axis'];axes=surface['uv_axes'];lo,hi=np.asarray(surface['uv_bounds']);nu,nv=surface['grid'];coef=np.asarray(surface['plane'])
   uv=np.array(np.meshgrid(np.linspace(lo[0],hi[0],nu),np.linspace(lo[1],hi[1],nv))).reshape(2,-1).T
   points=np.empty((len(uv),3));points[:,axes]=uv;points[:,ax]=np.einsum('ij,j->i',uv,coef[:2])+coef[2]
   for protected in group.get('protected_boxes',[]):
    plo,phi=np.asarray(protected['bounds']);points=points[~((points>plo)&(points<phi)).all(1)]
   projected=np.einsum('ij,nj->ni',V,points-eye);pixels=np.c_[500+f*projected[:,0]/projected[:,2],375+f*projected[:,1]/projected[:,2]]
   ok=(projected[:,2]>.02)&(pixels[:,0]>=0)&(pixels[:,0]<1000)&(pixels[:,1]>=0)&(pixels[:,1]<750);pixels=pixels[ok]
   if not len(pixels):continue
   low,high=pixels.min(0),pixels.max(0);overlap=(z>.02)&(screen[:,0]+radius>low[0])&(screen[:,0]-radius<high[0])&(screen[:,1]+radius>low[1])&(screen[:,1]-radius<high[1])
   ids=np.flatnonzero(overlap);ids=ids[np.argsort(z[ids])]
   Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@WORLD_FROM_RAW,Q)
   J=np.zeros((len(ids),2,3));J[:,0,0]=f/z[ids];J[:,1,1]=f/z[ids]
   J[:,0,2]=-f*np.clip(q[ids,0]/z[ids],-1.3*500/f,1.3*500/f)/z[ids];J[:,1,2]=-f*np.clip(q[ids,1]/z[ids],-1.3*375/f,1.3*375/f)/z[ids]
   B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C)
   det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2;splat_radius=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-splat_radius)/(1024/1080*750),0,1)
   residual=p[ids,ax]-np.einsum('ij,j->i',p[ids][:,axes],coef[:2])-coef[2];dlo,dhi=surface['deep_bounds']
   eligible=(surface['behind_sign']*residual>.025)&(p[ids,ax]>dlo)&(p[ids,ax]<dhi)&(np.abs(p[ids])<40).all(1)
   if 'candidate_bounds' in group:
    candidate_lo,candidate_hi=np.asarray(group['candidate_bounds']);eligible&=((p[ids]>candidate_lo)&(p[ids]<candidate_hi)).all(1)
   if 'candidate_rgb_min' in group:
    rgb=.5+.28209479177387814*columns(v[ids],['f_dc_0','f_dc_1','f_dc_2']);eligible&=(rgb.min(1)>group['candidate_rgb_min'])&(np.ptp(rgb,axis=1)<group['candidate_chroma_max'])
   for protected in group.get('protected_boxes',[]):
    protected_lo,protected_hi=np.asarray(protected['bounds']);eligible&=~((p[ids]>protected_lo)&(p[ids]<protected_hi)).all(1)
   if group.get('preserve_saturated'):
    rgb=.5+.28209479177387814*columns(v[ids],['f_dc_0','f_dc_1','f_dc_2']);eligible&=np.ptp(rgb,axis=1)<.25
   if primary_rows is not None:eligible&=ids<primary_rows
   picked=np.zeros(len(ids),bool);coverage=[]
   for pixel in pixels:
    d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d);a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)));T=np.r_[1,np.cumprod(1-a[:-1])];visible=a*T
    picked|=(visible>.003)&eligible;coverage.append(float(visible.sum()))
   selected_all[ids[picked]]=True
   record={'camera':camera['id'],'surface':surface['id'],'rays':len(pixels),'contributors':int(picked.sum()),'median_scene_opacity':float(np.median(coverage))};records.append(record);print(record,flush=True)
 return np.flatnonzero(selected_all),records


def attenuate(v,multipliers):
 out=v.copy();a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));a=np.clip(a*multipliers,1e-8,1-1e-8);out['opacity']=np.log(a/(1-a));return out


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--group',choices=GROUPS,required=True);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass4/combined-v1');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
 if args.out.resolve()==args.baseline.resolve():raise ValueError('Accepted baseline is immutable')
 base_report=json.loads((args.baseline/'report.json').read_text());src=Path(base_report['iphone']);refsrc=Path(base_report['reference']);phone,_,_=read_ply(src);ref,_,_=read_ply(refsrc)
 if sha256_file(src)!=base_report['iphone_sha256'] or sha256_file(refsrc)!=base_report['reference_sha256']:raise ValueError('Baseline source hash mismatch')
 if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):raise ValueError('Reference SH3 required')
 bs=np.load(args.baseline/'selections.npz');baseline_pm=np.ones(len(phone));baseline_rm=np.zeros(len(ref));baseline_pm[bs['iphone_indices']]=bs['iphone_multipliers'];baseline_rm[bs['reference_indices']]=bs['reference_multipliers']
 group=GROUPS[args.group]
 for context in group.get('context_components',[]):
  context_path=ROOT/context;cs=np.load(context_path/'iphone-opacity-selection.npz');cr=np.load(context_path/'reference-selection.npz')
  baseline_pm[cs['indices']]=np.minimum(baseline_pm[cs['indices']],cs['multipliers']);baseline_rm[cr['indices']]=np.maximum(baseline_rm[cr['indices']],cr['multipliers'])
 pp=positions(phone);rp=positions(ref);pw,pr=base_weights(pp,group);rw,rr=base_weights(rp,group)
 if group.get('preserve_saturated'):
  for data,weight in [(phone,pw),(ref,rw)]:
   rgb=.5+.28209479177387814*columns(data,['f_dc_0','f_dc_1','f_dc_2']);weight[np.ptp(rgb,axis=1)>.25]=0
 if group.get('source_only'):rid=np.empty(0,dtype=np.int64);reference_trace=[]
 else:rid,reference_trace=trace(ref,rp,group);rw[rid]=1
 composite_rw=np.maximum(baseline_rm,rw);cri=np.flatnonzero(composite_rw>0);composite_ref=attenuate(ref[cri],composite_rw[cri]);extra=np.empty(len(composite_ref),dtype=phone.dtype)
 for name in phone.dtype.names:extra[name]=composite_ref[name]
 rounds=[];additional=[]
 for iteration in range(4):
  composite_pm=np.minimum(baseline_pm,1-pw);cpi=np.flatnonzero(composite_pm<1);current=phone.copy();current[cpi]=attenuate(phone[cpi],composite_pm[cpi])
  scene=np.concatenate([current,extra]);picked,trace_report=trace(scene,np.concatenate([pp,rp[cri]]),group,len(phone));new=picked[pw[picked]<1-1e-6]
  rounds.append({'iteration':iteration,'new_original_rows':len(new),'traces':trace_report});print('Hybrid pass',iteration,'additional',len(new),flush=True)
  if not len(new):break
  pw[new]=1;additional.extend(new.tolist())
 pi=np.flatnonzero(pw>0);ri=np.flatnonzero(rw>0);args.out.mkdir(parents=True,exist_ok=True)
 np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rw[ri]);np.save(args.out/'additional-cloud-indices.npy',np.asarray(additional,dtype=np.int64));np.save(args.out/'reference-contributor-indices.npy',rid)
 patch=attenuate(ref[ri],rw[ri]);write_ply(args.out/'reference-corners.ply',patch)
 composite_pm=np.minimum(baseline_pm,1-pw);cpi=np.flatnonzero(composite_pm<1);current=phone.copy();current[cpi]=attenuate(phone[cpi],composite_pm[cpi]);assembled=args.out/'assembled';assembled.mkdir(exist_ok=True);write_ply(assembled/'iphone.ply',current);write_ply(assembled/'reference-patches.ply',composite_ref)
 checks={'iphone_only_opacity_changed':all(np.array_equal(current[n],phone[n]) for n in phone.dtype.names if n!='opacity'),'reference_geometry_covariance_DC_SH_exact':all(np.array_equal(patch[n],ref[n][ri]) for n in ref.dtype.names if n!='opacity'),'accepted_phone_opacity_not_restored':bool(np.all(composite_pm<=baseline_pm)),'accepted_reference_support_retained':bool(np.all(composite_rw>=baseline_rm))}
 if not all(checks.values()):raise RuntimeError(checks)
 report={'status':'Unreviewed incremental corner candidate','group':args.group,'description':group['description'],'iphone':str(src),'iphone_sha256':sha256_file(src),'reference':str(refsrc),'reference_sha256':sha256_file(refsrc),'accepted_baseline':str(args.baseline),'baseline_report_sha256':sha256_file(args.baseline/'report.json'),'baseline_selections_sha256':sha256_file(args.baseline/'selections.npz'),'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'iphone_regions':pr,'reference_regions':rr,'trace_surfaces':group['surfaces'],'reference_trace':reference_trace,'hybrid_passes':rounds,'checks':checks,'review_cameras':group['cameras'],'reference_unchanged_fields':'All except feathered opacity','limitations':['Static-region alignment is not exact ground truth.','Deep radiance support is not physical surface geometry.','Surrounding glass, fixtures, floor and accepted repairs require visual regression review.'],'remote_publish':False}
 report['protected_regions']=group.get('protected_boxes',[])
 report['source_only']=group.get('source_only',False)
 report['context_components']=group.get('context_components',[])
 report['attribution_candidate_guards']={key:group[key] for key in ['candidate_bounds','candidate_rgb_min','candidate_chroma_max'] if key in group}
 report['incremental_effect']={'iphone_new_selected_rows':int(np.sum(baseline_pm[pi]==1)),
  'iphone_strengthened_rows':int(np.sum((1-pw[pi])<baseline_pm[pi])),
  'reference_new_rows':int(np.sum(baseline_rm[ri]==0)),
  'reference_strengthened_rows':int(np.sum(rw[ri]>baseline_rm[ri]))}
 shutil.copyfile(__file__,args.out/'generator.py');report['generator_sha256']=sha256_file(args.out/'generator.py')
 (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['group','iphone_changed_rows','reference_added_rows','checks']},indent=2),flush=True)


if __name__=='__main__':main()
