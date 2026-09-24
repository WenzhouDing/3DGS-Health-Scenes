#!/usr/bin/env python3
"""Bounded area-preserving tangent-shape repair for the supply-cabinet wall.

The user identified white fibers, not genuine laminate grain. The imported
reference has a median tangent aspect ratio 5.83 vs 2.10 in the denser phone
material. Convert long thin on-surface ellipses into compact grains, preserving
the tangent area, smallest scale, position, color, SH and alpha of every splat.
This is an appearance repair, not an exact captured-covariance transfer.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W

RECIPE={'uv_bounds':[[-1.38,-.34],[-.36,.135]],'plane':[-.08237276,.02981509,-.92014449],
        'feather':.03,'depth_bounds':[-.022,.022],'depth_feather':.005,
        'live_alpha_min':.01,'max_tangent_aspect':2.2,'long_sigma_min':.002,
        'normal_alignment_min':.80,'preserved':'All fields except scale_0,scale_1,scale_2; tangent area and minimum axis scale unchanged.',
        'exclusions':'Cabinet glass/metal lip above y=.135; Stryker instruction labels left of x=-1.38; foreground stretcher and hardware outside the fitted surface band.'}


def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)


def apply(v):
 p=np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float));uv=p[:,:2];lo,hi=np.asarray(RECIPE['uv_bounds']);cf=np.asarray(RECIPE['plane']);d=p[:,2]-np.einsum('ij,j->i',uv,cf[:2])-cf[2];edge=np.minimum(uv-lo,hi-uv).min(1);a,b=RECIPE['depth_bounds'];weight=smooth(edge/RECIPE['feather'])*smooth(np.minimum(d-a,b-d)/RECIPE['depth_feather']);alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));ids=np.flatnonzero((weight>0)&(alpha>RECIPE['live_alpha_min']));sc=columns(v[ids],['scale_0','scale_1','scale_2']).astype(float);order=np.argsort(sc,axis=1);sr=np.take_along_axis(sc,order,axis=1);ratio=np.exp(sr[:,2]-sr[:,1]);smax=np.exp(sr[:,2]);Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',W,Q);normal=Q[np.arange(len(ids)),:,order[:,0]];plane_n=np.array([-cf[0],-cf[1],1.]);plane_n/=np.linalg.norm(plane_n);alignment=abs(np.einsum('ni,i->n',normal,plane_n));valid=(ratio>RECIPE['max_tangent_aspect'])&(smax>RECIPE['long_sigma_min'])&(alignment>RECIPE['normal_alignment_min']);ids=ids[valid];sc=sc[valid];order=order[valid];sr=sr[valid];ratio=ratio[valid];alignment=alignment[valid]
 # Linear interpolation in log sigma preserves the product of tangent scales
 # at every feather position, not just the fully repaired interior.
 delta=.5*(sr[:,2]-sr[:,1]-np.log(RECIPE['max_tangent_aspect']))*weight[ids];sc[np.arange(len(ids)),order[:,2]]-=delta;sc[np.arange(len(ids)),order[:,1]]+=delta
 out=v.copy()
 for i,n in enumerate(['scale_0','scale_1','scale_2']):out[n][ids]=sc[:,i]
 after=np.sort(np.exp(sc),axis=1);before=np.exp(sr);quant=[.01,.1,.5,.9,.99]
 report={'selected_rows':len(ids),'world_bounds':[p[ids].min(0).tolist(),p[ids].max(0).tolist()],'before_aspect_quantiles':np.quantile(ratio,quant).tolist(),'after_aspect_quantiles':np.quantile(after[:,2]/after[:,1],quant).tolist(),'before_maxsigma_quantiles':np.quantile(before[:,2],quant).tolist(),'after_maxsigma_quantiles':np.quantile(after[:,2],quant).tolist(),'minimum_axis_alignment_quantiles':np.quantile(alignment,quant).tolist(),'tangent_area_ratio_max_error':float(np.max(abs(after[:,2]*after[:,1]/(before[:,2]*before[:,1])-1))),'unchanged_nonscale_fields':all(np.array_equal(out[n],v[n]) for n in v.dtype.names if not n.startswith('scale_')),'unselected_all_fields_exact':bool(np.array_equal(out[np.setdiff1d(np.arange(len(v)),ids)],v[np.setdiff1d(np.arange(len(v)),ids)]))}
 if not report['unchanged_nonscale_fields'] or not report['unselected_all_fields_exact']:raise RuntimeError(report)
 return out,ids,report


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--out',type=Path,required=True);ap.add_argument('--aspect',type=float,default=2.2);args=ap.parse_args();RECIPE['max_tangent_aspect']=args.aspect
 if args.out.resolve()==args.baseline.resolve():raise ValueError('Baseline is immutable')
 args.out.mkdir(parents=True,exist_ok=True);reports={};changes={};hashes={}
 for kind,filename in [('iphone','iphone.ply'),('reference','reference-patches.ply')]:
  src=args.baseline/filename;v,_,_=read_ply(src);out,ids,rep=apply(v);write_ply(args.out/filename,out);reports[kind]=rep;changes[kind+'_indices']=ids;changes[kind+'_scale_before']=columns(v[ids],['scale_0','scale_1','scale_2']);changes[kind+'_scale_after']=columns(out[ids],['scale_0','scale_1','scale_2']);hashes[kind]={'input_sha256':sha256_file(src),'output_sha256':sha256_file(args.out/filename),'row_count':len(v)};print(kind,rep,flush=True)
 np.savez_compressed(args.out/'changes.npz',**changes);shutil.copyfile(__file__,args.out/'generator.py');report={'status':'Unreviewed bounded tangent-covariance appearance repair','baseline':str(args.baseline),'recipe':RECIPE,'sources':hashes,'results':reports,'changed_properties':['scale_0','scale_1','scale_2'],'added_rows':0,'deleted_rows':0,'index_mapping':'Every output row retains its accepted pass5 baseline row index. Reference baseline indices map to original capture using baseline selections.npz reference_indices.','generator_sha256':sha256_file(args.out/'generator.py'),'review_required':['wall-material-close','wall-material-grazing','left-wall','mattress-top'],'limitations':['Area conservation does not guarantee identical depth-sorted transmittance; verify coverage and contrast in GPU views.','The correction changes captured covariance; geometry/color/SH/opacity stay exact.','No repainting, synthetic grain or new surfaces.'],'remote_publish':False};(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
