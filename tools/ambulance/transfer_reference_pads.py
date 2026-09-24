#!/usr/bin/env python3
"""Trial-only replacement of fixed pad interiors with registered real splats.

No positions, covariance, color, or spherical harmonics are synthesized. Only
opacity is changed. The original iPhone scan is the starting point. Explicit
trial recipes select either material interiors or whole padded objects, and
the contributor mode retains measured off-shell radiance support as well.
All outputs require perspective review; earlier crop-only trials were rejected.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT, columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW


PATCHES = [
    {'id': 'right-upper-pad', 'axis': 2, 'uv_axes': [0, 1], 'uv_bounds': [[-.77, .46], [.48, .685]],
     'plane_z_from_xy1': [-.08291, .06183, .78042],
     'depth_residual_bounds': [-.15, .25], 'feather': .035},
    {'id': 'right-lower-pad', 'axis': 2, 'uv_axes': [0, 1], 'uv_bounds': [[-.77, -.017], [.48, .244]],
     'plane_z_from_xy1': [-.08171, -.06387, .79002],
     'depth_residual_bounds': [-.025, .18], 'feather': .035,
     'front_depth_feather': .008},
]
BENCH = {'id': 'exposed-central-bench', 'axis': 1, 'uv_axes': [0, 2],
         'uv_bounds': [[-.40, .43], [.18, .70]],
         'plane_z_from_xy1': [-.001115, .008680, -.305487],
         'depth_residual_bounds': [-.055, .047], 'feather': .035}
WHOLE_PADS = [
    {'id': 'right-upper-pad-whole', 'axis': 2, 'uv_axes': [0, 1],
     'uv_bounds': [[-.86, .425], [.57, .715]],
     'depth_world_bounds': [.60, 1.12], 'feather': .020},
    {'id': 'right-lower-pad-whole', 'axis': 2, 'uv_axes': [0, 1],
     'uv_bounds': [[-.86, -.06], [.57, .28]],
     'depth_world_bounds': [.53, 1.12], 'feather': .020},
]


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def weights(v, patches):
    p = np.einsum('ij,nj->ni', WORLD_FROM_RAW,
                  columns(v, ['x', 'y', 'z']).astype(np.float64))
    total = np.zeros(len(v), dtype=np.float32)
    report = []
    for recipe in patches:
        lo, hi = np.asarray(recipe['uv_bounds'])
        uv = p[:, recipe['uv_axes']]
        edge = np.minimum(uv - lo, hi - uv).min(axis=1)
        if 'depth_world_bounds' in recipe:
            depth = p[:, recipe['axis']]
            dmin, dmax = recipe['depth_world_bounds']
        else:
            coef = np.asarray(recipe['plane_z_from_xy1'])
            depth = p[:, recipe['axis']] - np.einsum('ij,j->i', uv, coef[:2]) - coef[2]
            dmin, dmax = recipe['depth_residual_bounds']
        w = smoothstep(edge / recipe['feather'])
        # Captured black materials contain view-dependent colored coefficients.
        # A DC-neutral filter leaves colored iPhone ghosts behind and removes
        # useful Insta highlight support. Geometry is the selection criterion.
        w *= (depth > dmin) & (depth < dmax)
        if 'depth_world_bounds' in recipe:
            w *= smoothstep(np.minimum(depth-dmin,dmax-depth)/.02)
            # The red equipment bag reaches into the lower-pad box at the rear.
            # Preserve that capture-specific foreground; it is not pad material.
            if recipe['id']=='right-lower-pad-whole':
                bag=(p[:,0]<-.38)&(p[:,1]<.16)&(p[:,2]<.78)
                w[bag]=0
        if recipe.get('front_depth_feather'):
            w *= smoothstep((depth-dmin)/recipe['front_depth_feather'])
        # Keep only bounded actual pad centers; no remote support billboards.
        w *= (np.abs(p) < 2).all(axis=1)
        selected = np.flatnonzero(w > 0)
        total = np.maximum(total, w.astype(np.float32))
        report.append({**recipe, 'selected_count': len(selected),
                       'world_bounds': [p[selected].min(0).tolist(), p[selected].max(0).tolist()],
                       'depth_quantiles_05_50_95': np.quantile(depth[selected], [.05, .5, .95]).tolist(),
                       'center_alpha_quantiles_05_50_95': np.quantile(1/(1+np.exp(-np.clip(v['opacity'][selected], -40, 40))), [.05, .5, .95]).tolist()})
    return total, report


def attenuate(records, multipliers):
    out = records.copy()
    alpha = 1/(1+np.exp(-np.clip(out['opacity'].astype(float), -40, 40)))
    alpha = np.clip(alpha*multipliers, 1e-8, 1-1e-8)
    out['opacity'] = np.log(alpha/(1-alpha))
    return out


def visible_backing(v, base_weights, primary_rows=None):
    """Collect actual deep radiance support seen through a bounded pad silhouette.

    This is a diagnostic opacity trace using full perspective covariance, not
    a physical-surface assertion. Deep centers are intentionally retained only
    when they contribute appreciably to sampled pad pixels in reviewed views.
    """
    p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float))
    scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float))
    alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
    cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in ['pads-close','right-wall','right-grazing']]
    backing=np.zeros(len(v),dtype=bool); stats=[]
    for camera in cameras:
        eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);up=np.cross(right,forward);V=np.array([right,up,forward])
        c=np.einsum('ij,nj->ni',V,p-eye);z=c[:,2];zs=np.maximum(z,.02)
        f=375/np.tan(np.radians(camera['fov'])/2)
        screen=np.column_stack([500+f*c[:,0]/zs,375-f*c[:,1]/zs]);radius=4*f*scales.max(1)/zs+2
        for patch,plane in zip(WHOLE_PADS,PATCHES):
            lo,hi=np.asarray(patch['uv_bounds']);xs=np.linspace(lo[0]+.025,hi[0]-.025,25);ys=np.linspace(lo[1]+.020,hi[1]-.020,7)
            uv=np.array(np.meshgrid(xs,ys)).reshape(2,-1).T;coef=np.asarray(plane['plane_z_from_xy1']);world=np.column_stack([uv,np.einsum('ij,j->i',uv,coef[:2])+coef[2]])
            q=np.einsum('ij,nj->ni',V,world-eye);pixels=np.column_stack([500+f*q[:,0]/q[:,2],375-f*q[:,1]/q[:,2]])
            low,high=pixels.min(0),pixels.max(0)
            box=(z>.02)&(screen[:,0]+radius>low[0])&(screen[:,0]-radius<high[0])&(screen[:,1]+radius>low[1])&(screen[:,1]-radius<high[1])
            ids=np.flatnonzero(box);order=np.argsort(z[ids]);ids=ids[order]
            quat=columns(v[ids],['rot_1','rot_2','rot_3','rot_0']);Q=Rotation.from_quat(quat).as_matrix();Q=np.einsum('ij,njk->nik',V@WORLD_FROM_RAW,Q)
            J=np.zeros((len(ids),2,3));J[:,0,0]=f/z[ids];J[:,0,2]=-f*np.clip(c[ids,0]/z[ids],-1.3*500/f,1.3*500/f)/z[ids];J[:,1,1]=-f/z[ids];J[:,1,2]=f*np.clip(c[ids,1]/z[ids],-1.3*375/f,1.3*375/f)/z[ids]
            B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];cov=np.einsum('nik,njk->nij',B,B);cov[:,0,0]+=.3;cov[:,1,1]+=.3;inv=np.linalg.inv(cov)
            det=cov[:,0,0]*cov[:,1,1]-cov[:,0,1]**2;mid=(cov[:,0,0]+cov[:,1,1])/2
            splat_radius=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)))
            fade=np.clip((2048/1080*750-splat_radius)/(1024/1080*750),0,1)
            depth=p[ids,2]-np.einsum('ij,j->i',p[ids,:2],coef[:2])-coef[2]
            eligible=(depth>.025)&(p[ids,2]<5.5)&(p[ids,2]>.64)&(abs(p[ids,0])<7)&(abs(p[ids,1])<5)
            if primary_rows is not None: eligible &= ids < primary_rows
            picked=np.zeros(len(ids),dtype=bool);covered=[]
            for pixel in pixels:
                delta=screen[ids]-pixel-.5;power=np.einsum('ni,nij,nj->n',delta,inv,delta)
                a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(-.5*power)-np.exp(-4.5)))
                trans=np.r_[1,np.cumprod(1-a[:-1])];w=a*trans
                picked|=(w>.003)&eligible
                covered.append(float(w.sum()))
            backing[ids[picked]]=True
            stats.append({'camera':camera['id'],'pad':patch['id'],'sampled_rays':len(pixels),'contributors':int(picked.sum()),'full_scene_alpha_median':float(np.median(covered))})
            print(f"Trace {camera['id']} {patch['id']}: {picked.sum()} backing contributors",flush=True)
    result=base_weights.copy();result[backing]=1
    added=backing&(base_weights<1)
    return result,{'selection':'full perspective covariance opacity contributions > .003 on sampled pad interior rays; centers behind the pad, max world z 5.5','deep_contributor_count':int(backing.sum()),'additional_or_strengthened_rows':int(added.sum()),'world_bounds':[p[backing].min(0).tolist(),p[backing].max(0).tolist()] if backing.any() else None,'views':stats},np.flatnonzero(backing)


def refine_hybrid(base,out):
    """Remove measured original clouds exposed by the already constructed hybrid."""
    prior=json.loads((base/'report.json').read_text())
    phone,_,_=read_ply(base/'iphone.ply');phone=phone.copy();ref,_,_=read_ply(base/'reference-pads.ply')
    selection=np.load(base/'iphone-opacity-selection.npz');pw=np.zeros(len(phone),np.float32);pw[selection['indices']]=1-selection['multipliers']
    extra=np.empty(len(ref),dtype=phone.dtype)
    for name in phone.dtype.names: extra[name]=ref[name]
    traces=[];new_ids=[]
    for iteration in range(5):
        scene=np.concatenate([phone,extra]);base_weights=np.r_[pw,np.zeros(len(extra),np.float32)]
        candidate,trace,selected=visible_backing(scene,base_weights,len(phone));candidate=candidate[:len(phone)]
        changed=np.flatnonzero(candidate>pw+1e-6);trace['iteration']=iteration;trace['new_original_rows']=len(changed);traces.append(trace)
        print(f'Hybrid peel {iteration}: {len(changed)} newly exposed original cloud rows',flush=True)
        if not len(changed): break
        phone[changed]=attenuate(phone[changed],np.zeros(len(changed)));pw=candidate;new_ids.extend(changed.tolist())
    ids=np.flatnonzero(pw>0);np.savez_compressed(out/'iphone-opacity-selection.npz',indices=ids,multipliers=1-pw[ids])
    write_ply(out/'iphone.ply',phone);write_ply(out/'reference-pads.ply',ref)
    shutil.copyfile(base/'reference-selection.npz',out/'reference-selection.npz')
    np.save(out/'additional-cloud-indices.npy',np.asarray(new_ids,dtype=np.int64))
    report={**prior,'status':'Unreviewed attribution-driven hybrid correction',
            'base_candidate':str(base),'iphone_changed_rows':len(ids),
            'hybrid_trace_new_rows':len(new_ids),'hybrid_trace_passes':traces,
            'note':'Reference is byte-identical in all vertex fields to V4; only original off-pad support visibly exposed in the actual hybrid is additionally suppressed.'}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iphone', type=Path, default=ROOT/'raw/ambulance_exp11_boot_sharp.ply')
    ap.add_argument('--reference', type=Path, default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply')
    ap.add_argument('--out', type=Path, default=ROOT/'raw/ambulance-cleanup/pass3/trials/pads')
    ap.add_argument('--upper-only', action='store_true')
    ap.add_argument('--include-bench', action='store_true')
    ap.add_argument('--whole-pads', action='store_true')
    ap.add_argument('--visible-backing', action='store_true')
    ap.add_argument('--hybrid-from', type=Path)
    ap.add_argument('--write-trial-phone', action='store_true')
    args = ap.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    if args.hybrid_from:
        refine_hybrid(args.hybrid_from,args.out);return
    phone, _, _ = read_ply(args.iphone); ref, _, _ = read_ply(args.reference)
    if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):
        raise ValueError('Aligned reference must retain degree-three SH')
    available = WHOLE_PADS if args.whole_pads else PATCHES
    patches = available[:1] if args.upper_only else available
    if args.include_bench: patches = patches + [BENCH]
    pw, pr = weights(phone, patches); rw, rr = weights(ref, patches)
    backing_report=None
    if args.visible_backing:
        if not args.whole_pads: raise ValueError('--visible-backing requires --whole-pads')
        pw,pbr,pbi=visible_backing(phone,pw);rw,rbr,rbi=visible_backing(ref,rw)
        np.savez_compressed(args.out/'deep-backing-indices.npz',iphone_indices=pbi,reference_indices=rbi)
        backing_report={'iphone':pbr,'reference':rbr}
    ids = np.flatnonzero(pw > 0); selected = np.flatnonzero(rw > 0)
    np.savez_compressed(args.out/'iphone-opacity-selection.npz', indices=ids,
                        multipliers=1-pw[ids])
    patch = attenuate(ref[selected], rw[selected])
    write_ply(args.out/'reference-pads.ply', patch)
    np.savez_compressed(args.out/'reference-selection.npz', indices=selected, multipliers=rw[selected])
    if args.write_trial_phone:
        out = phone.copy(); out[ids] = attenuate(phone[ids], 1-pw[ids])
        write_ply(args.out/'iphone.ply', out)
    report = {'status': 'Unreviewed actual-reference transfer trial; no active viewer modified',
              'iphone': str(args.iphone), 'iphone_sha256': sha256_file(args.iphone),
              'reference': str(args.reference), 'reference_sha256': sha256_file(args.reference),
              'iphone_changed_rows': len(ids), 'reference_added_rows': len(selected),
              'iphone_regions': pr, 'reference_regions': rr,
              'iphone_only_changed_field': 'opacity',
              'reference_unchanged_fields': 'all except feathered opacity',
              'mode': 'whole static padded objects' if args.whole_pads else 'interior material transfer',
              'configuration':{'whole_pads':args.whole_pads,'visible_backing':args.visible_backing,
                               'include_bench':args.include_bench,'upper_only':args.upper_only},
              'backing_support':backing_report,
              'exclusions': ['Movable bags and bench outside verified exposed center',
                             'Other scene objects outside the explicit boxes'],
              'review_cameras': ['pads-close', 'right-wall', 'right-grazing']}
    (args.out/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
