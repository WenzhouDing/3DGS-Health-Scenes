"""Register independent Insta360 capture to original iPhone cabin architecture.

No active viewer or cleanup asset is modified. Similarity fitting uses the fixed
cabin shell and excludes its movable interior equipment. The reported mapping
is raw Insta PLY -> existing iPhone viewer world; native Insta cameras can be
mapped inversely so the untouched SOG retains its directional SH appearance.
"""
from pathlib import Path
import argparse,json,sys,subprocess
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_mannequin import read_ply,sha256_file
F=np.diag([-1.,-1.,1.])
IPHONE_WORLD_FROM_RAW=Rotation.from_euler('xyz',[-78.243,.463,-4.499],degrees=True).as_matrix()@F


def shell(p):
    # Interior transport bays, stretcher, loose straps and bags are excluded.
    room=(p[:,0]>-1.83)&(p[:,0]<1.86)&(p[:,1]>-.89)&(p[:,1]<1.11)&(p[:,2]>-1.55)&(p[:,2]<1.12)
    v=p[:,2]+.085*p[:,0]
    fixed=(p[:,0]<-1.45)|(p[:,0]>1.44)|(p[:,1]<-.68)|(p[:,1]>.77)|(v<-.77)|(v>.74)
    return room&fixed


def voxel_points(p,c,a,cell):
    order=np.argsort(-a,kind='stable');key=np.floor(p[order]/cell).astype(np.int64)
    _,idx=np.unique(key,axis=0,return_index=True);take=order[idx]
    return p[take],c[take]


def load_cloud(path,iphone):
    vertices,_,_=read_ply(path)
    p=np.column_stack([vertices[k] for k in ['x','y','z']]).astype(float)
    c=np.clip(.5+.28209479177387814*np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)]),0,1)
    a=1/(1+np.exp(-np.clip(vertices['opacity'],-40,40)))
    s=np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]))
    quat=np.column_stack([vertices[f'rot_{i}'] for i in range(4)])
    valid=np.isfinite(p).all(1)&np.isfinite(c).all(1)&np.isfinite(quat).all(1)&(np.linalg.norm(quat,axis=1)>0)&(a>.52)&(s.max(1)<(.035 if iphone else .14))
    p=p[valid];c=c[valid];a=a[valid]
    if iphone:
        p=np.einsum('ij,nj->ni',IPHONE_WORLD_FROM_RAW,p);m=shell(p)
        return voxel_points(p[m],c[m],a[m],.008)
    initial=np.einsum('ij,nj->ni',np.diag([1.,-1.,-1.]),p)*.27+np.array([1.70,-.80,-.12])
    m=shell(initial)
    # Work in raw source coordinates; the voxel size is converted approximately.
    return voxel_points(p[m],c[m],a[m],.008/.27)


def similarity(source,target,weight=None):
    if weight is None:weight=np.ones(len(source))
    weight=weight/weight.sum();mu_s=np.einsum('n,ni->i',weight,source);mu_t=np.einsum('n,ni->i',weight,target)
    a=source-mu_s;b=target-mu_t;cov=np.einsum('n,ni,nj->ij',weight,b,a)
    U,S,Vt=np.linalg.svd(cov);D=np.ones(3)
    if np.linalg.det(U@Vt)<0:D[-1]=-1
    R=(U*D)@Vt;scale=float(np.sum(S*D)/np.einsum('n,ni,ni->',weight,a,a));t=mu_t-scale*(R@mu_s)
    return scale,R,t


def align(source,source_color,target,target_color,initial=None,iterations=90):
    tree=cKDTree(target);s,R,t=initial or (.27,np.diag([1.,-1.,-1.]),np.array([1.70,-.80,-.12]))
    history=[]
    for it in range(iterations):
        p=s*np.einsum('ij,nj->ni',R,source)+t
        dist,near=tree.query(p,k=4,workers=4)
        color=np.linalg.norm(source_color[:,None]-target_color[near],axis=2)
        chosen=np.argmin(dist+.025*np.minimum(color,.7),axis=1);rows=np.arange(len(p));dist=dist[rows,chosen];near=near[rows,chosen]
        cutoff=max(.027,.15*(.95**it));ok=(dist<cutoff)&(color[rows,chosen]<.60)
        # Huber weighting keeps a few capture-specific protrusions from pulling
        # the architecture toward bags, open doors or reflection billboards.
        residual=dist[ok];weight=np.minimum(1,.018/np.maximum(residual,1e-9))
        ns,nR,nt=similarity(source[ok],target[near[ok]],weight)
        movement=abs(ns-s)+np.linalg.norm(nt-t)+np.linalg.norm(nR-R)
        s,R,t=ns,nR,nt
        history.append({'iteration':it,'inliers':int(ok.sum()),'cutoff':float(cutoff),'median_match_error':float(np.median(residual)),'scale':s,'movement':float(movement)})
        if it>45 and movement<1e-7:break
    p=s*np.einsum('ij,nj->ni',R,source)+t;dist,near=tree.query(p,k=1,workers=4)
    reverse,_=cKDTree(p).query(target,k=1,workers=4)
    return s,R,t,{'history':history,'distance_units':'iPhone viewer scene units; not calibrated meters','quantile_levels':[.25,.5,.75,.9,.95],'source_to_target_quantiles':np.quantile(dist,[.25,.5,.75,.9,.95]).tolist(),'target_to_source_quantiles':np.quantile(reverse,[.25,.5,.75,.9,.95]).tolist(),'source_within_0_02':float(np.mean(dist<.02)),'source_within_0_05':float(np.mean(dist<.05)),'target_within_0_02':float(np.mean(reverse<.02)),'target_within_0_05':float(np.mean(reverse<.05))}


def export_native(report,out):
    """Export with the converter's tested SH3 rotation, including format flips."""
    from render_review import resolve_converter
    s=float(report['scale']);R=np.asarray(report['rotation']);t=np.asarray(report['translation'])
    # The converter both reads and writes PLY through F. Desired raw output is
    # A.T @ (s R raw + t), so its user transform must include these two flips.
    user_rotation=F@IPHONE_WORLD_FROM_RAW.T@R@F
    user_translation=F@IPHONE_WORLD_FROM_RAW.T@t
    euler=Rotation.from_matrix(user_rotation).as_euler('xyz',degrees=True)
    output=out/'insta-aligned-iphone-raw.ply'
    cmd=['node',str(resolve_converter(None)),report['source'],'-N','-r',','.join(map(str,euler)),'-s',str(s),'-t',','.join(map(str,user_translation)),str(output),'-w']
    print(json.dumps({'export_command':cmd}),flush=True);subprocess.run(cmd,check=True)
    report['aligned_reference']={'path':str(output),'frame':'original iPhone raw PLY coordinates','full_degree3_sh_preserved':True,'converter':'@playcanvas/splat-transform 3.4.2','command':cmd,'raw_output_rotation':(IPHONE_WORLD_FROM_RAW.T@R).tolist(),'raw_output_translation':(IPHONE_WORLD_FROM_RAW.T@t).tolist(),'sha256':sha256_file(output)}
    return report


def validate_alignment(report,cache,out):
    """Refit disjoint spatial subsets to measure transform stability.

    These trials do not overwrite the delivered mapping or aligned PLY. A small
    per-region error alone does not imply that capture-specific contents agree.
    """
    d=np.load(cache);source,sc,target,tc=d['source'],d['source_rgb'],d['target'],d['target_rgb']
    s=float(report['scale']);R=np.asarray(report['rotation']);t=np.asarray(report['translation'])
    p=s*np.einsum('ij,nj->ni',R,source)+t
    rng=np.random.default_rng(20260922);tree=cKDTree(target);results={}
    masks={'without_rear':p[:,0]>-1.35,'without_front':p[:,0]<1.30,'without_left_wall':p[:,2]+.085*p[:,0]>-.65,'without_right_wall':p[:,2]+.085*p[:,0]<.65,'without_roof':p[:,1]<.72,'without_floor':p[:,1]>-.65}
    for name,mask in masks.items():
        available=np.flatnonzero(mask);ids=rng.choice(available,min(32000,len(available)),replace=False)
        jitter=Rotation.from_rotvec(rng.normal(0,.015,3)).as_matrix()
        init=(s*(1+rng.uniform(-.012,.012)),jitter@R,t+rng.normal(0,.015,3))
        ns,nR,nt,metrics=align(source[ids],sc[ids],target,tc,init,85)
        mapped=ns*np.einsum('ij,nj->ni',nR,source)+nt
        drift=np.linalg.norm(mapped-p,axis=1);heldout=~mask
        residual,_=tree.query(mapped[heldout],workers=4)
        results[name]={'training_points':len(ids),'heldout_points':int(heldout.sum()),'scale':ns,'scale_delta_fraction':ns/s-1,'rotation_delta_degrees':float(np.degrees(Rotation.from_matrix(nR@R.T).magnitude())),'translation_delta':(nt-t).tolist(),'all_points_transform_drift_quantiles':np.quantile(drift,[.5,.9,.99]).tolist(),'heldout_nearest_error_quantiles':np.quantile(residual,[.5,.9,.95]).tolist(),'final_iteration':metrics['history'][-1]}
        print(json.dumps({'stability_trial':name,**results[name]}),flush=True)
    result={'method':'Six spatial leave-region-out refits, each using at most 32000 points and a perturbed initial transform. Delivered mapping is unchanged.','distance_units':'iPhone viewer scene units, not calibrated meters','trials':results,'max_99_percent_transform_drift':max(v['all_points_transform_drift_quantiles'][2] for v in results.values())}
    (out/'stability-validation.json').write_text(json.dumps(result,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/alignment');ap.add_argument('--export-only',action='store_true');ap.add_argument('--validate-only',action='store_true');ap.add_argument('--export',action='store_true');args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    if args.export_only:
        report_path=args.out/'reference-alignment.json';report=json.loads(report_path.read_text())
        report=export_native(report,args.out);report_path.write_text(json.dumps(report,indent=2)+'\n');return
    if args.validate_only:
        report=json.loads((args.out/'reference-alignment.json').read_text());validate_alignment(report,args.out/'fixed-shell-clouds.npz',args.out);return
    src=ROOT/'raw/ambulance_exp_insta360.ply';dst=ROOT/'raw/ambulance_exp11_boot_sharp.ply'
    cache=args.out/'fixed-shell-clouds.npz'
    if cache.exists():d=np.load(cache);source,sc,target,tc=d['source'],d['source_rgb'],d['target'],d['target_rgb']
    else:
        source,sc=load_cloud(src,False);target,tc=load_cloud(dst,True);np.savez(cache,source=source,source_rgb=sc,target=target,target_rgb=tc)
    print('clouds',len(source),len(target),flush=True)
    s,R,t,metrics=align(source,sc,target,tc)
    world_matrix=np.eye(4);world_matrix[:3,:3]=s*R;world_matrix[:3,3]=t
    native_to_iphone=np.eye(4);native_to_iphone[:3,:3]=s*R@F;native_to_iphone[:3,3]=t
    report={'version':1,'source':str(src),'target':str(dst),'source_sha256':sha256_file(src),'target_sha256':sha256_file(dst),'mapping':'iphone_viewer_world = scale * rotation @ insta_raw_xyz + translation','scale':s,'rotation':R.tolist(),'translation':t.tolist(),'matrix':world_matrix.tolist(),'native_insta_viewer_to_iphone_viewer':native_to_iphone.tolist(),'iphone_raw_to_viewer_rotation':IPHONE_WORLD_FROM_RAW.tolist(),'source_fixed_cloud_points':len(source),'target_fixed_cloud_points':len(target),'metrics':metrics,'limitations':['Captures are independent; movable contents and glass reflections are not registration targets.','Metrics are in iPhone scene units; metric-to-meter calibration is not established.']}
    if args.export:report=export_native(report,args.out)
    (args.out/'reference-alignment.json').write_text(json.dumps(report,indent=2)+'\n');np.savez(args.out/'aligned-reference-cloud.npz',xyz=s*np.einsum('ij,nj->ni',R,source)+t,rgb=sc,target_xyz=target,target_rgb=tc)
    print(json.dumps({'scale':s,'rotation':R.tolist(),'translation':t.tolist(),'metrics':{k:v for k,v in metrics.items() if k!='history'}},indent=2),flush=True)

if __name__=='__main__':main()
