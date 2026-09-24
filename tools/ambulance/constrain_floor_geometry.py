#!/usr/bin/env python3
"""Constrain provenance-confirmed captured aisle material to its measured floor.

This is a geometry trial, not a global collision model. Every change is indexed
against the frozen pass5 PLY pair. No colors, SH, or visible opacities are edited.
Only confirmed floor donor centers/covariance are constrained, with source
bed/wheels/cables unchanged. Already hidden source floor rows receive explicit
physical-deletion IDs for the parent compositor; trial row counts stay stable.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,write_ply,columns,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
from transfer_reference_floor import world,smooth

REGION={'xz_bounds':[[-1.50,-.045],[1.48,.60]],'edge_feather':.025,
        'eligible_height':[-.10,.225],'preserved_center_half_height':.012,
        'relocated_center_backing_height':-.012,'maximum_normal_sigma':.0025,
        'unchanged_support_half_height':.020,
        'bench_face_z_from_xy1':[-.08432874,.02431246,.38086325],
        'bench_face_minimum_front_clearance':.045}


def floor_fit(v,p,floor_mask):
    scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float))
    Q=Rotation.from_quat(columns(v,['rot_1','rot_2','rot_3','rot_0']).astype(float)).as_matrix();Q=np.einsum('ij,njk->nik',W,Q)
    n=np.array([.006536127350277606,1,.011240657756362868]);n/=np.linalg.norm(n)
    alignment=np.abs(np.einsum('i,ni->n',n,Q[np.arange(len(v)),:,scales.argmin(1)]))
    a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));height=p[:,1]+.006536127350277606*p[:,0]+.011240657756362868*p[:,2]+.7765111297658739
    lo,hi=np.asarray(REGION['xz_bounds']);uv=p[:,[0,2]]
    keep=floor_mask&(np.abs(height)<.03)&(a>.5)&(alignment>.92)&(scales.max(1)<.035)&(scales.min(1)<.002)&((uv>lo)&(uv<hi)).all(1)
    q=p[keep];X=np.c_[q[:,0],q[:,2],np.ones(len(q))]
    if len(q)<100:raise ValueError('Insufficient measured floor support')
    coef=np.linalg.lstsq(X,q[:,1],rcond=None)[0]
    for _ in range(8):
        r=q[:,1]-np.einsum('ij,j->i',X,coef);mad=1.4826*np.median(abs(r-np.median(r)));w=1/np.maximum(1,abs(r)/(2*max(mad,.0005)));coef=np.linalg.lstsq(X*w[:,None],q[:,1]*w,rcond=None)[0]
    r=q[:,1]-np.einsum('ij,j->i',X,coef)
    return coef,{'anchor_count':len(q),'plane_y_from_xz1':coef.tolist(),'residual_absolute_quantiles':np.quantile(abs(r),[.5,.9,.95,.99,1]).tolist(),'anchor_baseline_reference_indices':np.flatnonzero(keep)},Q,scales


def metrics(p,Q,s,coef):
    n=np.array([-coef[0],1,-coef[1]]);norm=np.linalg.norm(n);n/=norm;d=(p[:,1]-np.einsum('ij,j->i',p[:,[0,2]],coef[:2])-coef[2])/norm
    sigma=np.sqrt(np.sum((np.einsum('i,nij->nj',n,Q)*s)**2,axis=1));return d,sigma,n


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    if args.baseline.resolve()==args.out.resolve():raise ValueError('Baseline overwrite forbidden')
    phone,_,_=read_ply(args.baseline/'iphone.ply');ref,_,_=read_ply(args.baseline/'reference-patches.ply');baseids=np.load(args.baseline/'selections.npz')['reference_indices'];floor=np.load(ROOT/'raw/ambulance-cleanup/pass4/floor-v4/reference-selection.npz')['indices'];floor_mask=np.isin(baseids,floor);p=world(ref);coef,fit,Q,s=floor_fit(ref,p,floor_mask);d,sigma,n=metrics(p,Q,s,coef)
    lo,hi=np.asarray(REGION['xz_bounds']);edge=np.minimum(p[:,[0,2]]-lo,hi-p[:,[0,2]]).min(1);bench=np.asarray(REGION['bench_face_z_from_xy1']);bench_depth=np.einsum('ij,j->i',p[:,:2],bench[:2])+bench[2];edge=np.minimum(edge,bench_depth-p[:,2]-REGION['bench_face_minimum_front_clearance']);weight=smooth(edge/REGION['edge_feather']);eligible=floor_mask&(weight>0)&(d>REGION['eligible_height'][0])&(d<REGION['eligible_height'][1]);near_bad=eligible&((abs(d)+3*sigma)>REGION['unchanged_support_half_height']);deep_bad=floor_mask&(weight>0)&(d<=REGION['eligible_height'][0])&(d+3*sigma>.020);bad=near_bad|deep_bad;ids=np.flatnonzero(bad);w=weight[ids]
    # Keep supported small floor roughness. Only detached center layers move.
    target_d=d[ids].copy();move=(abs(target_d)>REGION['preserved_center_half_height'])&near_bad[ids];target_d[move]=REGION['relocated_center_backing_height'];newp=p[ids]-(w*(d[ids]-target_d))[:,None]*n
    C=np.einsum('nik,njk->nij',Q[ids]*s[ids,None,:],Q[ids]*s[ids,None,:]);P=np.eye(3)-np.outer(n,n);thin=np.minimum(sigma[ids],REGION['maximum_normal_sigma']);deep_local=deep_bad[ids];thin[deep_local]=np.minimum(sigma[ids][deep_local],(-.005-d[ids][deep_local])/3);targetC=np.einsum('ij,njk,lk->nil',P,C,P)+thin[:,None,None]**2*np.outer(n,n);newC=(1-w[:,None,None])*C+w[:,None,None]*targetC
    eigen,R=np.linalg.eigh(newC);eigen=np.maximum(eigen,1e-12);R[np.linalg.det(R)<0,:,0]*=-1;nativeR=np.einsum('ij,njk->nik',W.T,R);quat=Rotation.from_matrix(nativeR).as_quat();out=ref.copy();raw=np.einsum('ij,nj->ni',W.T,newp)
    for i,name in enumerate(['x','y','z']):out[name][ids]=raw[:,i]
    for i,name in enumerate(['scale_0','scale_1','scale_2']):out[name][ids]=.5*np.log(eigen[:,i])
    for i,name in enumerate(['rot_1','rot_2','rot_3','rot_0']):out[name][ids]=quat[:,i]
    changed=np.zeros(len(ref),bool)
    for name in ref.dtype.names:changed|=out[name]!=ref[name]
    changedids=np.flatnonzero(changed);geometric_fields=['x','y','z','scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3'];p2=world(out);s2=np.exp(columns(out,['scale_0','scale_1','scale_2']).astype(float));Q2=Rotation.from_quat(columns(out,['rot_1','rot_2','rot_3','rot_0']).astype(float)).as_matrix();Q2=np.einsum('ij,njk->nik',W,Q2);d2,sigma2,_=metrics(p2,Q2,s2,coef)
    # These rows were already opacity-floor hidden by reviewed floor-v4. The
    # manifest asks the final compositor to compact them, not call them removed
    # merely because a visual opacity changed in an earlier pass.
    oldfloor=np.load(ROOT/'raw/ambulance-cleanup/pass4/floor-v4/iphone-opacity-selection.npz')['indices'];pa=1/(1+np.exp(-np.clip(phone['opacity'].astype(float),-40,40)));deletephone=oldfloor[pa[oldfloor]<=1.00001e-8]
    args.out.mkdir(parents=True,exist_ok=True);write_ply(args.out/'iphone.ply',phone);write_ply(args.out/'reference-patches.ply',out)
    np.savez_compressed(args.out/'changes.npz',phone_changed_indices=np.empty(0,np.int64),reference_changed_indices=changedids,phone_delete_indices=deletephone,reference_delete_indices=np.empty(0,np.int64),reference_original_capture_indices=baseids[changedids],reference_center_moved_indices=ids[move&(w>0)],floor_anchor_reference_indices=fit.pop('anchor_baseline_reference_indices'))
    def summary(mask):
        return {'count':int(mask.sum()),'center_height_before':np.quantile(d[mask],[0,.5,.95,1]).tolist(),'center_height_after':np.quantile(d2[mask],[0,.5,.95,1]).tolist(),'normal_sigma_before':np.quantile(sigma[mask],[0,.5,.95,1]).tolist(),'normal_sigma_after':np.quantile(sigma2[mask],[0,.5,.95,1]).tolist(),'absolute_three_sigma_extent_before':np.quantile(abs(d[mask])+3*sigma[mask],[.5,.95,1]).tolist(),'absolute_three_sigma_extent_after':np.quantile(abs(d2[mask])+3*sigma2[mask],[.5,.95,1]).tolist()}
    core=near_bad&(weight>=1-1e-10);deep_core=deep_bad&(weight>=1-1e-10);report={'status':'Unreviewed bounded floor geometry candidate','baseline':str(args.baseline),'baseline_phone_sha256':sha256_file(args.baseline/'iphone.ply'),'baseline_reference_sha256':sha256_file(args.baseline/'reference-patches.ply'),'region':REGION,'floor_fit':fit,'counts':{'phone_geometrically_changed':0,'reference_geometrically_changed':len(changedids),'reference_centers_moved':int(np.sum(move&(w>0))),'reference_covariances_constrained':len(ids),'deep_backing_normal_covariances_capped':int(deep_bad.sum()),'already_invisible_floor_phone_rows_marked_for_physical_deletion':len(deletephone),'rows_physically_compacted_in_trial':0},'core_geometry':summary(core),'all_changed_geometry':summary(bad),'deep_backing_geometry':{'count':int(deep_bad.sum()),'full_strength_count':int(deep_core.sum()),'max_upper_three_sigma_before':float(np.max((d+3*sigma)[deep_core])) if deep_core.any() else None,'max_upper_three_sigma_after':float(np.max((d2+3*sigma2)[deep_core])) if deep_core.any() else None},'maximum_center_displacement':float(np.linalg.norm(p2-p,axis=1).max()),'checks':{'reference_color_SH_opacity_exact':all(np.array_equal(out[f],ref[f]) for f in ref.dtype.names if f not in geometric_fields),'all_fields_finite':all(np.isfinite(out[f]).all().item() for f in out.dtype.names),'unselected_reference_byte_exact':out[~changed].tobytes()==ref[~changed].tobytes(),'phone_byte_exact':sha256_file(args.out/'iphone.ply')==sha256_file(args.baseline/'iphone.ply'),'core_three_sigma_extent_leq_002':bool(np.all(abs(d2[core])+3*sigma2[core]<=.020001)),'deep_core_three_sigma_below_floor':bool(np.all((d2+3*sigma2)[deep_core]<=-.004999))},'collision_limitations':['Distances are scene units, not calibrated metric distances.','Only provenance-confirmed aisle-floor donors in the listed bounded slab are constrained.','Source bed, wheels and cable records are byte-identical; their capture geometry is not claimed collision-safe.','Deep remote radiance backing and material outside this ROI remain; whole-scene GS occupancy is not certified.','Actual final physical deletion requires the compositor to apply the provided deletion indices.','A separate validated collision proxy may still be required; 3-sigma bounds describe the Gaussian support convention, not a watertight mesh.'],'remote_publish':False}
    if not all(report['checks'].values()):raise RuntimeError(report['checks'])
    shutil.copyfile(__file__,args.out/'generator.py');report['generator_sha256']=sha256_file(args.out/'generator.py');report['changes_sha256']=sha256_file(args.out/'changes.npz');(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
