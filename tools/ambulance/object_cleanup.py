"""Manual correction of object-surface Gaussian shells in the iPhone ambulance.

Unlike global opacity filtering, this projects misplaced material support back
to an observed surface and compresses the covariance normal to it. It preserves
each Gaussian's in-plane covariance, saturated straps, small opaque detail, and
the prior manually added backing. Operates in memory without writing the source.
"""
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from repair_surfaces import WORLD_FROM_RAW

SH_C0 = 0.28209479177387814


def _plane(uv, w):
    a = np.column_stack([uv, np.ones(len(uv))])
    keep = np.ones(len(w), bool)
    for _ in range(6):
        coef = np.linalg.lstsq(a[keep], w[keep], rcond=None)[0]
        r = w - np.sum(a * coef, axis=1)
        mad = np.median(np.abs(r[keep] - np.median(r[keep])))
        updated = np.abs(r - np.median(r[keep])) < max(.008, 2.5 * 1.4826 * mad)
        if updated.sum() < 50:
            break
        keep = updated
    return coef, keep


def apply(vertices, config=None):
    """Return the original prefix plus repair backing and explicit change report."""
    if config is None:
        config = json.loads(Path(__file__).with_name('object-surfaces.json').read_text())
    elif isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    out = vertices.copy()
    p = np.einsum('ij,nj->ni', WORLD_FROM_RAW, np.column_stack([vertices[k] for k in ['x','y','z']]).astype(np.float64))
    rgb = np.clip(.5 + SH_C0 * np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)]), 0, 1)
    alpha = 1 / (1 + np.exp(-np.clip(vertices['opacity'], -35, 35)))
    sigma = np.exp(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]).astype(np.float64))
    largest = sigma.max(1)
    chroma = np.ptp(rgb, axis=1)
    dark = (rgb.max(1) < .48) & (chroma < .13)
    blue = (rgb[:,2] > .48) & (rgb[:,1] > .3) & (rgb[:,1]-rgb[:,0] > .08) & (rgb[:,2]-rgb[:,0] > .16)
    prior_backing = (np.abs(sigma[:,0]-.0064)<1e-6)&(np.abs(sigma[:,1]-.0064)<1e-6)&(np.abs(sigma[:,2]-.0007)<1e-7)
    all_changed = np.zeros(len(vertices),bool)
    patch_reports = []
    additions=[]
    for patch in config['patches']:
        if not patch.get('enabled',True):
            patch_reports.append({'id':patch['id'],'modified_count':0,'review_note':patch['review_note']});continue
        axis=patch['axis']; axes=patch['uv_axes']; lo,hi=np.asarray(patch['uv_bounds']); dlo,dhi=patch['depth_bounds']
        uv=p[:,axes];w=p[:,axis]
        domain=((uv>=lo)&(uv<=hi)).all(1)&(w>=dlo)&(w<=dhi)
        material=dark if patch['material']=='dark' else blue
        seed_dlo,seed_dhi=patch.get('seed_depth_bounds',[dlo,dhi])
        seed_domain=domain&(w>=seed_dlo)&(w<=seed_dhi)
        seed_color=material&(rgb.max(1)<patch.get('seed_max_channel',1.))
        seeds=np.flatnonzero(seed_domain&seed_color&(alpha>.72)&(largest<.045)&~prior_backing)
        pr={'id':patch['id'],'candidate_count':int(domain.sum()),'support_count':len(seeds),'modified_count':0,
            'manual_bounds':{'axis':axis,'uv_axes':axes,'uv_bounds':patch['uv_bounds'],'depth_bounds':[dlo,dhi],
                             'seed_depth_bounds':[seed_dlo,seed_dhi]}}
        if len(seeds)<80:
            pr['skipped']='fewer than 80 reliable material samples';patch_reports.append(pr);continue
        coef,inliers=_plane(uv[seeds],w[seeds])
        if patch['model']=='plane':
            seeds=seeds[inliers]
            if np.linalg.norm(coef[:2])>.65:
                pr['skipped']='fitted slope inconsistent with pad orientation';patch_reports.append(pr);continue
        if len(seeds)<50:
            pr['skipped']='insufficient surface inliers';patch_reports.append(pr);continue
        # Do not change red straps, yellow/green fittings, or opaque small text.
        color_eligible=(chroma<.16)|material
        object_exclusions=np.zeros(len(vertices),bool)
        for box in patch.get('exclude_boxes',[]):
            object_exclusions |= ((p>=box[0])&(p<=box[1])).all(1)
        for strip in patch.get('protect_uv_strips',[]):
            object_exclusions |= (uv[:,0]>=strip[0])&(uv[:,0]<=strip[1])
        protected_small=(alpha>.92)&(largest<.007) if patch.get('protect_small_details',True) else np.zeros(len(vertices),bool)
        candidates=np.flatnonzero(domain&color_eligible&~prior_backing&~object_exclusions&(largest>.003)&~protected_small)
        if not len(candidates):
            pr['skipped']='no diffuse support in selected object';patch_reports.append(pr);continue
        tree=cKDTree(uv[seeds]); dist,nb=tree.query(uv[candidates],k=24,workers=2)
        valid=dist[:,0]<patch['support_distance']
        ids=candidates[valid];dist=dist[valid];nb=nb[valid]
        if not len(ids):
            pr['skipped']='no nearby observed surface';patch_reports.append(pr);continue
        suv=uv[seeds][nb];sw=w[seeds][nb]
        if patch.get('geometry_mode')=='color_only':
            ci=np.flatnonzero(domain&~object_exclusions&(largest>.005)&(rgb.max(1)<.75)&(chroma<.35)&~prior_backing)
            if len(ci):
                cd,cn=tree.query(uv[ci],k=24,workers=2)
                eligible=cd[:,0]<patch['support_distance'];ci=ci[eligible];cn=cn[eligible]
                local=np.median(rgb[seeds][cn],axis=1);global_rgb=np.median(rgb[seeds],axis=0)
                target=.35*local+.65*global_rgb
                target=np.maximum(target,patch.get('material_floor',.14))
                target=.2*target+.8*target.mean(1,keepdims=True)
                corrected=.15*rgb[ci]+.85*target
                for i in range(3):out[f'f_dc_{i}'][ci]=(corrected[:,i]-.5)/SH_C0
                all_changed[ci]=True
                pr.update({'modified_count':len(ci),'geometry_modified_count':0,'material_color_corrected_count':len(ci),
                           'observed_material_rgb_median':global_rgb.tolist(),'model':'material color only',
                           'world_bounds':[p[ci].min(0).tolist(),p[ci].max(0).tolist()] if len(ci) else None})
            patch_reports.append(pr);continue
        if patch['model']=='plane':
            residual=w[seeds]-np.sum(np.column_stack([uv[seeds],np.ones(len(seeds))])*coef,axis=1)
            expected=np.sum(np.column_stack([uv[ids],np.ones(len(ids))])*coef,axis=1)+np.clip(np.median(residual[nb],axis=1),-.006,.006)
            slopes=np.broadcast_to(coef[:2],(len(ids),2)).copy()
        else:
            # Robust local depth regression keeps the mattress folds and chair curvature.
            median=np.median(sw,axis=1);mad=np.median(np.abs(sw-median[:,None]),axis=1)
            weights=np.exp(-(dist/.035)**2)*(np.abs(sw-median[:,None])<np.maximum(.01,3*mad)[:,None])
            weights/=np.maximum(weights.sum(1,keepdims=True),1e-12)
            umean=np.sum(suv*weights[:,:,None],axis=1);wmean=np.sum(sw*weights,axis=1)
            du=suv-umean[:,None,:];dw=sw-wmean[:,None]
            cov=np.einsum('nki,nkj,nk->nij',du,du,weights)+np.eye(2)[None,:,:]*2e-6
            cross=np.einsum('nki,nk,nk->ni',du,dw,weights)
            slopes=np.linalg.solve(cov,cross[:,:,None])[:,:,0]
            slope_size=np.linalg.norm(slopes,axis=1)
            slopes*=np.minimum(1,.9/np.maximum(slope_size,1e-10))[:,None]
            expected=wmean+np.sum((uv[ids]-umean)*slopes,axis=1)
        normal=np.zeros((len(ids),3));normal[:,axis]=1;normal[:,axes]=-slopes
        normal/=np.linalg.norm(normal,axis=1,keepdims=True)
        q=np.column_stack([vertices[f'rot_{i}'][ids] for i in [1,2,3,0]])
        raw_rot=Rotation.from_quat(q).as_matrix()
        world_rot=np.einsum('ij,njk->nik',WORLD_FROM_RAW,raw_rot)
        covariance=np.einsum('nik,nk,njk->nij',world_rot,sigma[ids]**2,world_rot)
        normal_variance=np.einsum('ni,nij,nj->n',normal,covariance,normal)
        normal_sigma=np.sqrt(np.maximum(normal_variance,1e-20))
        signed=patch['front_sign']*(w[ids]-expected)/np.sqrt(1+np.sum(slopes**2,axis=1))
        material_rgb=np.median(rgb[seeds][nb],axis=1)
        pale=(rgb[ids].mean(1)>material_rgb.mean(1)+.07)&(chroma[ids]<.13)
        # Half a centimetre of thickness is visible at a grazing angle. Restrict
        # correction to front support; backing behind the material remains intact.
        affected=(signed>-.003)&(signed<patch['max_projection'])&((signed>.004)|(normal_sigma>.0045))
        affected &= (material[ids]|pale)
        if patch.get('geometry_mode')=='haze_only':
            affected &= (normal_sigma>patch.get('minimum_geometry_thickness',.009))&(signed>patch.get('minimum_front_offset',.006))&(pale|(alpha[ids]<.80))
        ids=ids[affected];normal=normal[affected];covariance=covariance[affected]
        expected=expected[affected];signed=signed[affected];normal_sigma=normal_sigma[affected]
        material_rgb=material_rgb[affected];pale=pale[affected];slopes=slopes[affected]
        if not len(ids):
            pr['skipped']='selected surface already geometrically thin';patch_reports.append(pr);continue
        # Preserve in-plane covariance with a 2D eigendecomposition; set only its
        # normal extent to the measured material shell thickness.
        tangent=np.zeros_like(normal);tangent[:,axes[0]]=1;tangent[:,axis]=slopes[:,0]
        tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
        bitangent=np.cross(normal,tangent)
        basis=np.stack([tangent,bitangent],axis=2)
        cov2=np.einsum('nji,njk,nkl->nil',basis,covariance,basis)
        values,vectors=np.linalg.eigh(cov2)
        inplane=np.einsum('nij,njk->nik',basis,vectors)
        frame=np.concatenate([inplane,normal[:,:,None]],axis=2)
        negative=np.linalg.det(frame)<0;frame[negative,:,1]*=-1
        raw_frame=np.einsum('ij,njk->nik',WORLD_FROM_RAW.T,frame)
        qnew=Rotation.from_matrix(raw_frame).as_quat()[:,[3,0,1,2]]
        new_sigma=np.column_stack([np.sqrt(np.maximum(values,1e-16)),np.minimum(normal_sigma,patch['normal_sigma'])])
        # Limit displacement to this surface's normal, leaving its footprint fixed.
        center=p[ids].copy();delta=(w[ids]-expected)/np.sqrt(1+np.sum(slopes**2,axis=1))
        center-=normal*delta[:,None]
        raw=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,center)
        for i,k in enumerate(['x','y','z']):out[k][ids]=raw[:,i]
        for i in range(3):out[f'scale_{i}'][ids]=np.log(new_sigma[:,i])
        for i in range(4):out[f'rot_{i}'][ids]=qnew[:,i]
        recolor=pale&(signed>patch.get('minimum_front_offset',.006))&(normal_sigma>patch.get('minimum_geometry_thickness',.0045))
        if not patch.get('recolor_opaque_haze',False):recolor &= alpha[ids]<.92
        color_ids=ids[recolor]
        corrected=.25*rgb[color_ids]+.75*material_rgb[recolor]
        for i in range(3):out[f'f_dc_{i}'][color_ids]=(corrected[:,i]-.5)/SH_C0
        all_changed[ids]=True
        pr.update({'support_count':len(seeds),'modified_count':len(ids),'recolored_count':len(color_ids),
                   'plane_coefficients':coef.tolist(),'model':patch['model'],
                   'normal_world_median':np.median(normal,axis=0).tolist(),
                   'observed_material_rgb_median':np.median(material_rgb,axis=0).tolist(),
                   'normal_sigma_before_median':float(np.median(normal_sigma)),
                   'normal_sigma_after_median':float(np.median(new_sigma[:,2])),
                   'normal_sigma_before_p95':float(np.quantile(normal_sigma,.95)),
                   'normal_sigma_after_p95':float(np.quantile(new_sigma[:,2],.95)),
                   'normal_offset_before_p95':float(np.quantile(np.abs(signed),.95)),
                   'normal_offset_after_p95':float(np.quantile(np.abs(np.sum((center-p[ids])*normal,axis=1)+delta),.95)),
                   'maximum_center_displacement':float(np.max(np.linalg.norm(center-p[ids],axis=1))),
                   'world_bounds':[center.min(0).tolist(),center.max(0).tolist()]})
        patch_reports.append(pr)
        if patch.get('smooth_color'):
            # Material-only interiors are intentionally neutralized: faint blue,
            # brown and purple ghosts are capture contamination, not upholstery.
            color_ids=np.flatnonzero(domain&~object_exclusions&(largest>.005)&(rgb.max(1)<.75)&(chroma<.35)&~prior_backing)
            if len(color_ids):
                cd,cn=tree.query(uv[color_ids],k=24,workers=2)
                eligible=cd[:,0]<patch['support_distance'];color_ids=color_ids[eligible];cn=cn[eligible]
                local=np.median(rgb[seeds][cn],axis=1)
                global_rgb=np.median(rgb[seeds],axis=0)
                target=.35*local+.65*global_rgb
                target=np.maximum(target,patch.get('material_floor',.14))
                # Preserve gentle light falloff, discard view-dependent stain colors.
                neutral=np.mean(target,axis=1,keepdims=True)
                target=.2*target+.8*neutral
                corrected=.15*rgb[color_ids]+.85*target
                for i in range(3):out[f'f_dc_{i}'][color_ids]=(corrected[:,i]-.5)/SH_C0
                all_changed[color_ids]=True;pr['material_color_corrected_count']=len(color_ids)
        if patch.get('backing'):
            # Dense, thin backing at the fitted material shell replaces the
            # accidental coverage previously supplied by thick fog Gaussians.
            spacing=.008
            gu=np.arange(lo[0]+.018,hi[0]-.018,spacing);gv=np.arange(lo[1]+.018,hi[1]-.018,spacing)
            grid=np.stack(np.meshgrid(gu,gv),axis=-1).reshape(-1,2)
            gd,gn=tree.query(grid,k=24,workers=2)
            supported=gd[:,0]<patch['support_distance'];grid=grid[supported];gn=gn[supported]
            if len(grid):
                gres=w[seeds]-np.sum(np.column_stack([uv[seeds],np.ones(len(seeds))])*coef,axis=1)
                dep=np.sum(np.column_stack([grid,np.ones(len(grid))])*coef,axis=1)+np.clip(np.median(gres[gn],axis=1),-.004,.004)-patch['front_sign']*.004
                pts=np.zeros((len(grid),3));pts[:,axes]=grid;pts[:,axis]=dep
                mat=np.median(rgb[seeds][gn],axis=1);mat=.3*mat+.7*np.median(rgb[seeds],axis=0)
                mat=np.maximum(mat,patch.get('material_floor',.2));mat=.2*mat+.8*mat.mean(1,keepdims=True)
                n=np.zeros(3);n[axis]=1;n[axes]=-coef[:2];n/=np.linalg.norm(n)
                t=np.zeros(3);t[axes[0]]=1;t[axis]=coef[0];t/=np.linalg.norm(t)
                b=np.cross(n,t);frame=WORLD_FROM_RAW.T@np.column_stack([t,b,n])
                quat=Rotation.from_matrix(frame).as_quat()[[3,0,1,2]]
                add=np.zeros(len(grid),dtype=vertices.dtype);rpts=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,pts)
                for i,k in enumerate(['x','y','z']):add[k]=rpts[:,i]
                for i in range(3):add[f'f_dc_{i}']=(mat[:,i]-.5)/SH_C0;add[f'scale_{i}']=np.log(.0065 if i<2 else .0006)
                for i in range(4):add[f'rot_{i}']=quat[i]
                edge=np.minimum(grid-lo,hi-grid).min(1);aa=.66*np.clip(edge/.045,0,1)
                add['opacity']=np.log(aa/(1-aa));additions.append(add);pr['backing_added_count']=len(add)
                pr['backing_world_bounds']=[pts.min(0).tolist(),pts.max(0).tolist()]
                pr['backing_material_rgb_median']=np.median(mat,axis=0).tolist()
                pr['backing_sigma']=[.0065,.0065,.0006]
    touchup_reports=[]
    for touchup in config.get('touchups',[]):
        op=np.einsum('ij,nj->ni',WORLD_FROM_RAW,np.column_stack([out[k] for k in ['x','y','z']]).astype(np.float64))
        oc=.5+SH_C0*np.column_stack([out[f'f_dc_{i}'] for i in range(3)])
        tl,th=np.asarray(touchup['world_bounds'])
        ti=np.flatnonzero(((op>=tl)&(op<=th)).all(1)&(oc.mean(1)<touchup['maximum_luma'])&~prior_backing)
        for i in range(3):out[f'f_dc_{i}'][ti]=(touchup['rgb'][i]-.5)/SH_C0
        all_changed[ti]=True;touchup_reports.append(dict(touchup,affected_count=len(ti)))
    changed_indices=np.flatnonzero(all_changed)
    property_counts={name:int(np.count_nonzero(vertices[name][changed_indices]!=out[name][changed_indices])) for name in vertices.dtype.names}
    if additions:out=np.concatenate([out,*additions])
    return out,{'recipe_version':config['version'],'input_count':len(vertices),'output_count':len(out),
                'added_count':len(out)-len(vertices),
                'affected_count':len(changed_indices),'affected_indices':changed_indices.tolist(),
                'changed_properties':property_counts,'prior_backing_preserved':bool(np.array_equal(vertices[prior_backing],out[:len(vertices)][prior_backing])),
                'patches':patch_reports,'touchups':touchup_reports}
