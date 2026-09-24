#!/usr/bin/env python3
"""Isolated covariance-only diagnostic for traced coarse lower-wall support.

Keep original centers, DC/SH, opacity and tangent covariance. Blend the full
world covariance toward P C P + (.0008)^2 n n^T inside the measured lower-wall
region, retaining the existing sloped-bottom and label protection guards.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W


def smooth(t):
    t=np.clip(t,0,1)
    return t*t*(3-2*t)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--source-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists(): parser.error('Diagnostic output must be new.')
    source_report=json.loads(args.source_report.read_text())
    inherited=source_report['recipe']
    v,_,_=read_ply(args.source)
    p=np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))
    cf=np.asarray(inherited['plane'])
    n=np.array([-cf[0],-cf[1],1.]);n/=np.linalg.norm(n)
    P=np.eye(3)-np.outer(n,n)
    lo=np.array([-.84,-.44]);hi=np.array([-.48,-.32])
    uv=p[:,:2]
    local_edge=np.minimum(uv-lo,hi-uv).min(1)
    depth=p[:,2]-np.einsum('ij,j->i',uv,cf[:2])-cf[2]
    uvlo,uvhi=np.array(inherited['uv_bounds'])
    inherited_edge=np.minimum(uv-uvlo,uvhi-uv).min(1)
    bottom_slope,bottom_offset=inherited['bottom_edge_y_from_x1']
    inherited_edge=np.minimum(inherited_edge,p[:,1]-(bottom_slope*p[:,0]+bottom_offset))
    label_lo,label_hi=np.array(inherited['label_protection_uv'])
    label_clearance=np.linalg.norm(np.maximum(np.maximum(label_lo-uv,uv-label_hi),0),axis=1)
    depth_lo,depth_hi=inherited['depth_bounds']
    weight=(smooth(local_edge/.015)*smooth(inherited_edge/inherited['feather'])
            *smooth(label_clearance/inherited['label_feather'])
            *smooth(np.minimum(depth-depth_lo,depth_hi-depth)/inherited['depth_feather']))
    alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
    candidates=np.flatnonzero((weight>0)&(alpha>inherited['live_alpha_min']))
    block=v[candidates]
    axes=np.einsum('ij,njk->nik',W,Rotation.from_quat(columns(block,['rot_1','rot_2','rot_3','rot_0']).astype(float)).as_matrix())
    scales=np.exp(columns(block,['scale_0','scale_1','scale_2']).astype(float))
    scaled=axes*scales[:,None,:]
    cov=np.einsum('nik,njk->nij',scaled,scaled)
    normal_sigma=np.sqrt(np.einsum('i,nij,j->n',n,cov,n))
    tangent_cov=np.einsum('ij,njk,lk->nil',P,cov,P)
    tangent_major=np.sqrt(np.maximum(np.linalg.eigvalsh(tangent_cov)[:,-1],0))
    selected=(normal_sigma>.003)|(tangent_major>.015)
    ids=candidates[selected]
    before=cov[selected]
    before_sigma=normal_sigma[selected]
    target=tangent_cov[selected]+.0008**2*np.outer(n,n)
    w=weight[ids]
    final=(1-w[:,None,None])*before+w[:,None,None]*target
    eigen,rotation=np.linalg.eigh(final)
    rotation[np.linalg.det(rotation)<0,:,0]*=-1
    quaternion=Rotation.from_matrix(np.einsum('ij,njk->nik',W.T,rotation)).as_quat()
    out=v.copy()
    for i,f in enumerate(['scale_0','scale_1','scale_2']):out[f][ids]=.5*np.log(np.maximum(eigen[:,i],1e-12))
    for i,f in enumerate(['rot_1','rot_2','rot_3','rot_0']):out[f][ids]=quaternion[:,i]
    # Independently measure the serialized float32 covariance representation.
    qa=np.einsum('ij,njk->nik',W,Rotation.from_quat(columns(out[ids],['rot_1','rot_2','rot_3','rot_0']).astype(float)).as_matrix())
    ss=np.exp(columns(out[ids],['scale_0','scale_1','scale_2']).astype(float))
    aa=qa*ss[:,None,:];actual_cov=np.einsum('nik,njk->nij',aa,aa)
    after_sigma=np.sqrt(np.einsum('i,nij,j->n',n,actual_cov,n))
    tangent_error=float(np.max(np.abs(np.einsum('ij,njk,lk->nil',P,actual_cov-before,P))))
    covariance_fields={'scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'}
    untouched=np.ones(len(v),bool);untouched[ids]=False
    full=w>.999999
    checks={'all_centers_DC_SH_opacity_exact':all(np.array_equal(out[f],v[f]) for f in v.dtype.names if f not in covariance_fields),
            'all_unselected_records_byte_exact':out[untouched].tobytes()==v[untouched].tobytes(),
            'tangential_covariance_preserved':tangent_error<1e-8,
            'full_strength_normal_sigma_0008':bool(np.allclose(after_sigma[full],.0008,atol=2e-8)),
            'all_covariances_positive':bool(np.all(np.linalg.eigvalsh(actual_cov)>0)),
            'all_values_finite':all(bool(np.isfinite(out[f]).all()) for f in v.dtype.names),
            'all_selected_inside_requested_uv':bool(((uv[ids]>lo)&(uv[ids]<hi)).all()),
            'all_selected_inside_depth_band':bool(np.all(np.abs(depth[ids])<.022)),
            'no_additions_or_removals':len(v)==len(out)}
    if not all(checks.values()):raise RuntimeError(checks)
    args.out.mkdir(parents=True)
    write_ply(args.out/'native.ply',out)
    np.savez_compressed(args.out/'changes.npz',reference_original_indices=ids,
                        covariance_before=before,covariance_after=actual_cov,
                        normal_sigma_before=before_sigma,normal_sigma_after=after_sigma,
                        full_covariance_weight=w)
    shutil.copy2(__file__,args.out/'generator.py')
    report={'status':'Unreviewed isolated covariance-only native lower-wall diagnostic',
            'source':str(args.source.resolve()),'source_sha256':sha256_file(args.source),
            'source_report':str(args.source_report.resolve()),'source_report_sha256':sha256_file(args.source_report),
            'recipe':{'uv_bounds':[lo.tolist(),hi.tolist()],'local_feather':.015,
                      'plane':cf.tolist(),'depth_bounds':[-.022,.022],
                      'inherited_guards':inherited,
                      'selection':'normal sigma > .003 OR projected tangent major sigma > .015; independent of aspect/alignment',
                      'target_normal_sigma':.0008,
                      'covariance':'Feathered blend toward P C P + .0008^2 n n^T'},
            'counts':{'selected':len(ids),'full_strength':int(full.sum()),
                      'normal_sigma_decreased':int(np.sum(after_sigma<before_sigma-1e-9)),
                      'normal_sigma_increased':int(np.sum(after_sigma>before_sigma+1e-9)),
                      'added':0,'removed':0},
            'normal_sigma_before_q0_q50_q95_q100':np.quantile(before_sigma,[0,.5,.95,1]).tolist(),
            'normal_sigma_after_q0_q50_q95_q100':np.quantile(after_sigma,[0,.5,.95,1]).tolist(),
            'maximum_tangential_covariance_error':tangent_error,
            'checks':checks,
            'limitations':['This targets coarse normal support; coarse tangential shading can remain.',
                           'Feather-edge normal variance is deliberately blended, not uniformly capped.',
                           'Specified .0008 normal target can slightly increase initially thinner selected rows; no positions or radiance fields change.',
                           'Full native proof diagnostic only; visual review required before any hybrid integration.'],
            'output_sha256':sha256_file(args.out/'native.ply'),
            'changes_sha256':sha256_file(args.out/'changes.npz'),
            'generator_sha256':sha256_file(args.out/'generator.py'),'remote_publish':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
