"""Normalize reviewed architectural Gaussian sheets without removing coverage.

Native PLY properties are changed only inside manually bounded architectural
regions. A robust source-supported sheet constrains centers and covariance;
2D tangential covariance and captured DC appearance are preserved. This repairs
opaque, broad and needle-shaped shells that opacity filtering cannot fix.
Missing roof, fascia and hatch coverage receives explicit local Gaussian
supports after distant background billboards are suppressed; those manually
reviewed materials and inferred support dimensions are recorded in the recipe.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

WORLD_FROM_RAW = Rotation.from_euler('xyz',[-78.243,.463,-4.499],degrees=True).as_matrix() @ np.diag([-1.,-1.,1.])


def _design(uv):
    return np.column_stack([uv,np.ones(len(uv))])


def _predict(uv, coef):
    return np.einsum('ni,i->n',_design(uv),coef)


def _fit(uv, depth, max_slope, seed):
    """Deterministic RANSAC followed by trimmed source point regression."""
    rng=np.random.default_rng(seed)
    if len(uv)>24000:
        take=rng.choice(len(uv),24000,replace=False);u=uv[take];d=depth[take]
    else:u=uv;d=depth
    A=_design(u);best=None;score=-1
    for _ in range(320):
        ids=rng.choice(len(u),3,replace=False)
        try:c=np.linalg.solve(A[ids],d[ids])
        except np.linalg.LinAlgError:continue
        if np.linalg.norm(c[:2])>max_slope:continue
        residual=abs(d-np.einsum('ni,i->n',A,c));count=int((residual<.006).sum())
        if count>score:score=count;best=c
    if best is None:raise ValueError('No stable plane fit')
    for _ in range(5):
        residual=depth-_predict(uv,best);inlier=abs(residual)<.009
        if inlier.sum()<40:break
        best=np.linalg.lstsq(_design(uv[inlier]),depth[inlier],rcond=None)[0]
    residual=depth-_predict(uv,best);inlier=abs(residual)<.009
    return best,inlier,float(np.median(abs(residual[inlier])))


def apply(vertices, config=None):
    """Return original rows in order plus append-only supports and a region audit."""
    if config is None:config=json.loads(Path(__file__).with_name('architectural-surfaces.json').read_text())
    if isinstance(config,(str,Path)):config=json.loads(Path(config).read_text())
    out=vertices.copy();names=vertices.dtype.names
    p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,np.column_stack([vertices[k] for k in ['x','y','z']]).astype(float))
    rgb=np.clip(.5+.28209479177387814*np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)]),0,1)
    luma=rgb.mean(1);chroma=np.ptp(rgb,axis=1)
    alpha=1/(1+np.exp(-np.clip(vertices['opacity'],-40,40)))
    sigma=np.exp(np.clip(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]),-35,5))
    largest=sigma.max(1)
    changed=np.zeros(len(vertices),bool);moved=np.zeros(len(vertices),bool);altered_opacity=np.zeros(len(vertices),bool)
    regions=[];region_ids=[]
    for number,recipe in enumerate(config['regions']):
        axis=recipe['axis'];axes=recipe['uv_axes'];lo,hi=np.asarray(recipe['uv_bounds'],float);dlo,dhi=recipe['depth_bounds'];uv=p[:,axes]
        domain=((uv>=lo)&(uv<=hi)).all(1)&(p[:,axis]>=dlo)&(p[:,axis]<=dhi)
        ids=np.flatnonzero(domain)
        report={'id':recipe['id'],'domain_count':int(len(ids)),'corrected_count':0,'moved_count':0,'opacity_changed_count':0}
        if len(ids)<150:report['skipped']='insufficient region geometry';regions.append(report);continue
        q=np.column_stack([vertices[f'rot_{i}'][ids] for i in [1,2,3,0]]).astype(float)
        norm=np.linalg.norm(q,axis=1);valid=norm>1e-12;q[~valid]=[0,0,0,1]
        raw_rotation=Rotation.from_quat(q).as_matrix();world_rotation=np.einsum('ij,njk->nik',WORLD_FROM_RAW,raw_rotation)
        shortest=world_rotation[np.arange(len(ids)),:,np.argmin(sigma[ids],axis=1)]
        min_luma=recipe.get('min_luma',.42)
        anchor=(alpha[ids]>.55)&(chroma[ids]<.24)&(luma[ids]>max(min_luma,.45))&(largest[ids]<(.16 if recipe.get('ceiling') else .045))&(np.abs(shortest[:,axis])>(.50 if recipe.get('max_slope',.6)>1 else .78))&valid
        if recipe.get('ceiling'):anchor&=(luma[ids]<.91)
        if recipe.get('require_sloped_normal'):anchor&=(abs(shortest[:,1])>.40)&(p[ids,1]>recipe.get('fit_y_min',.80))
        seed_ids=ids[anchor]
        if len(seed_ids)<80:report['skipped']='insufficient credible sheet anchors';report['anchor_count']=int(len(seed_ids));regions.append(report);continue
        try:coef,inliers,fit_error=_fit(uv[seed_ids],p[seed_ids,axis],recipe.get('max_slope',.60),number+71)
        except ValueError as exc:report['skipped']=str(exc);regions.append(report);continue
        normal=np.zeros(3);normal[axis]=1;normal[axes]=-coef[:2];normal/=np.linalg.norm(normal)
        tangent=np.zeros(3);tangent[axes[0]]=1;tangent[axis]=coef[0];tangent/=np.linalg.norm(tangent)
        bitangent=np.cross(normal,tangent);bitangent/=np.linalg.norm(bitangent)
        basis=np.column_stack([tangent,bitangent,normal])
        C=np.einsum('nik,njk->nij',world_rotation*sigma[ids,None,:],world_rotation*sigma[ids,None,:])
        local=np.einsum('ij,njk,kl->nil',basis.T,C,basis)
        normal_before=np.sqrt(np.maximum(local[:,2,2],0))
        signed=(p[ids,axis]-_predict(uv[ids],coef))*normal[axis]
        close=abs(signed)<recipe.get('snap_distance',.030)
        # Fine raised fittings retain their physical depth. Wider splats with
        # support leaving the sheet are eligible even at near-opaque alpha.
        compact_detail=(largest[ids]<.0035)&(abs(signed)>.0045)
        material=(chroma[ids]<.24)&(luma[ids]>min_luma)
        if recipe.get('ceiling'):
            # Preserve white lamp disks. Their darker gasket splats have tiny
            # tangent support and are retained by the compact-detail guard.
            material &= luma[ids]<.935
        detached=(abs(signed)<.008)|((luma[ids]>.40)&(largest[ids]>.0045))
        needle=(normal_before>.005)&(largest[ids]>.008)
        eligible=close&material&valid&~compact_detail&(detached|needle)
        target_sigma=recipe.get('normal_sigma',.0009)
        eligible&=(normal_before>target_sigma*1.3)|(abs(signed)>.002)
        pos=np.flatnonzero(eligible);selected=ids[pos]
        if not len(pos):report['skipped']='sheet already thin';regions.append(report);continue
        # Edge feather avoids creating boundaries where neighboring sheet fits
        # meet. Tangential covariance remains source-derived, including broad
        # ceiling Gaussians that bridge genuinely textureless areas.
        edge=np.minimum(uv[selected]-lo,hi-uv[selected]).min(1)
        feather=np.clip(edge/.018,0,1);feather=feather*feather*(3-2*feather)
        delta=signed[pos]*feather*.82
        newp=p[selected]-delta[:,None]*normal
        rawp=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,newp)
        for k,key in enumerate(['x','y','z']):out[key][selected]=rawp[:,k]
        ct=local[pos,:2,:2];eigenvalue,eigenvector=np.linalg.eigh(ct);eigenvalue=np.maximum(eigenvalue,1e-14)
        frames=np.empty((len(pos),3,3));frames[:,:,:2]=np.einsum('ij,njk->nik',basis[:,:2],eigenvector);frames[:,:,2]=normal
        # Eigh may choose a left-handed tangent pair; flip one axis without
        # changing the covariance before converting to a proper quaternion.
        det=np.linalg.det(frames);frames[det<0,:,0]*=-1
        rawframes=np.einsum('ij,njk->nik',WORLD_FROM_RAW.T,frames)
        quat=Rotation.from_matrix(rawframes).as_quat()[:,[3,0,1,2]]
        new_sigma_normal=np.minimum(normal_before[pos],np.maximum(target_sigma,normal_before[pos]*(1-feather)))
        scales=np.column_stack([np.sqrt(eigenvalue),new_sigma_normal])
        for k in range(3):out[f'scale_{k}'][selected]=np.log(scales[:,k])
        for k in range(4):out[f'rot_{k}'][selected]=quat[:,k]
        # A broad pale off-sheet layer is redundant illumination, not paint.
        # Keep its coverage but lower its contribution after flattening.
        ghost=(abs(signed[pos])>.018)&(luma[selected]>.62)&(largest[selected]>.014)
        gi=selected[ghost];ga=alpha[gi]*.40
        out['opacity'][gi]=np.log(np.maximum(ga,1e-8)/(1-ga));altered_opacity[gi]=True
        changed[selected]=True;moved[selected[abs(delta)>1e-6]]=True;region_ids.append((recipe['id'],selected))
        report.update({'anchor_count':int(len(seed_ids)),'fit_inlier_count':int(inliers.sum()),'fit_median_error':fit_error,
            'plane_coefficients':coef.tolist(),'normal_world':normal.tolist(),'corrected_count':int(len(selected)),
            'moved_count':int((abs(delta)>1e-6).sum()),'opacity_changed_count':int(len(gi)),
            'normal_sigma_before_quantiles':np.quantile(normal_before[pos],[.5,.9,.99]).tolist(),
            'normal_sigma_after_quantiles':np.quantile(new_sigma_normal,[.5,.9,.99]).tolist(),
            'center_residual_before_quantiles':np.quantile(abs(signed[pos]),[.5,.9,.99]).tolist(),
            'center_residual_after_quantiles':np.quantile(abs(signed[pos]-delta),[.5,.9,.99]).tolist(),
            'max_center_displacement':float(abs(delta).max()),'uv_bounds':[lo.tolist(),hi.tolist()],
            'depth_bounds':[dlo,dhi]})
        regions.append(report)
    # The original reconstruction used distant opaque billboards as accidental
    # room surfaces. Replace the missing local roof/hatch support before removing
    # those physically misplaced sheets, including ones with high opacity.
    supports=[]
    outside=(p[:,0]<-1.8)|(p[:,0]>1.9)|(p[:,1]<-.9)|(p[:,1]>1.1)|(p[:,2]<-1.5)|(p[:,2]>1.05)
    distant=np.flatnonzero(outside&(largest>.15))
    if len(distant) and config.get('roof_reconstruction') and config.get('metal_hatch_support'):
        out['opacity'][distant]=-18.0;changed[distant]=True;altered_opacity[distant]=True
    else:
        # Do not remove false-depth coverage without its reviewed replacements.
        distant=np.empty(0,dtype=np.int64)
    roof=config.get('roof_reconstruction')
    reconstruction=[]
    if roof:
        a,b,c=roof['plane_y_from_xz'];u=p[:,0];v=p[:,2]+.085*p[:,0]
        lo,hi=np.asarray(roof['uv_bounds'],float)
        roomroof=(u>=lo[0])&(u<=hi[0])&(v>=lo[1])&(v<=hi[1])&(p[:,1]>.870)&(p[:,1]<1.08)
        lamp=np.zeros(len(p),bool)
        for lx,lz in roof['lamp_centers_xz']:
            lamp|=(p[:,0]-lx)**2+(p[:,2]-lz)**2<roof['lamp_preserve_radius']**2
        rail=abs(v-roof['rail_v_center'])<roof['rail_half_width']
        keep_hardware=lamp|rail|(chroma>.24)
        ghostroof=np.flatnonzero(roomroof&~keep_hardware)
        # Keep a small contribution from captured paint; remove floating smoke
        # and dark needles while the new local sheet supplies the surface.
        out['opacity'][ghostroof]=np.log(np.maximum(alpha[ghostroof]*.10,1e-8)/(1-np.maximum(alpha[ghostroof]*.10,1e-8)))
        changed[ghostroof]=True;altered_opacity[ghostroof]=True
        step=roof['spacing'];U,V=np.meshgrid(np.arange(lo[0]+step/2,hi[0],step),np.arange(lo[1]+step/2,hi[1],step))
        U=U.ravel();V=V.ravel();Z=V-.085*U;Y=a*U+b*Z+c
        points=np.column_stack([U,Y,Z]);normal=np.array([-a,1,-b]);normal/=np.linalg.norm(normal)
        tangent=np.array([1,a,0]);tangent/=np.linalg.norm(tangent);bitangent=np.cross(normal,tangent)
        frame=WORLD_FROM_RAW.T@np.column_stack([tangent,bitangent,normal]);quat=Rotation.from_matrix(frame).as_quat()[[3,0,1,2]]
        native=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,points);rec=np.zeros(len(points),dtype=vertices.dtype)
        for k,key in enumerate(['x','y','z']):rec[key]=native[:,k]
        edge=np.minimum(np.column_stack([U,V])-lo,hi-np.column_stack([U,V])).min(1);fade=np.clip(edge/.025,0,1)
        al=np.clip(roof['opacity']*fade,.03,roof['opacity']);rec['opacity']=np.log(al/(1-al))
        for k in range(3):
            rec[f'f_dc_{k}']=(roof['rgb'][k]-.5)/.28209479177387814
            rec[f'scale_{k}']=np.log(step*.86 if k<2 else .00065)
        for k in range(4):rec[f'rot_{k}']=quat[k]
        supports.append(rec)
        reconstruction.append({'id':'physical-ceiling-sheet','added_count':len(rec),'world_bounds':[points.min(0).tolist(),points.max(0).tolist()],'plane_y_from_xz':[a,b,c],'rgb':roof['rgb'],'normal_sigma':.00065,'normal_world':normal.tolist(),'suppressed_local_roof_ghosts':int(len(ghostroof)),'preserved_lamp_disks':8,'source':roof['plane_source']})
    hatch=config.get('metal_hatch_support')
    if hatch:
        lo,hi=np.asarray(hatch['uv_bounds']);step=hatch['spacing'];Z,Y=np.meshgrid(np.arange(lo[0]+step/2,hi[0],step),np.arange(lo[1]+step/2,hi[1],step));Z=Z.ravel();Y=Y.ravel();X=np.full(len(Z),hatch['depth_x']);points=np.c_[X,Y,Z]
        native=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,points);rec=np.zeros(len(points),dtype=vertices.dtype)
        for k,key in enumerate(['x','y','z']):rec[key]=native[:,k]
        normal=np.array([1.,0,0]);tangent=np.array([0.,1,0]);bitangent=np.cross(normal,tangent);quat=Rotation.from_matrix(WORLD_FROM_RAW.T@np.c_[tangent,bitangent,normal]).as_quat()[[3,0,1,2]]
        for k in range(3):rec[f'f_dc_{k}']=(hatch['rgb'][k]-.5)/.28209479177387814;rec[f'scale_{k}']=np.log(step*.80 if k<2 else .00065)
        for k in range(4):rec[f'rot_{k}']=quat[k]
        edge=np.minimum(np.c_[Z,Y]-lo,hi-np.c_[Z,Y]).min(1);fade=np.clip(edge/hatch.get('edge_feather',.02),0,1);fade=fade*fade*(3-2*fade);ha=np.clip(hatch['opacity']*fade,.025,hatch['opacity']);rec['opacity']=np.log(ha/(1-ha));supports.append(rec)
        reconstruction.append({'id':'metal-hatch-backing','added_count':len(rec),'world_bounds':[points.min(0).tolist(),points.max(0).tolist()],'rgb':hatch['rgb'],'normal_sigma':.00065,'normal_world':normal.tolist()})
    for fascia in config.get('fascia_backing',[]):
        measured=next((r for r in regions if r['id']==fascia['fit_region'] and 'plane_coefficients' in r),None)
        if measured is None:continue
        a,b,c=measured['plane_coefficients'];lo,hi=np.asarray(fascia['uv_bounds'],float);step=fascia['spacing'];X,Y=np.meshgrid(np.arange(lo[0]+step/2,hi[0],step),np.arange(lo[1]+step/2,hi[1],step));X=X.ravel();Y=Y.ravel();Z=a*X+b*Y+c+fascia['behind_sign']*.012;points=np.c_[X,Y,Z]
        native=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,points);rec=np.zeros(len(points),dtype=vertices.dtype)
        normal=np.array([-a,-b,1.]);normal/=np.linalg.norm(normal);tangent=np.array([1.,0,a]);tangent/=np.linalg.norm(tangent);bitangent=np.cross(normal,tangent);quat=Rotation.from_matrix(WORLD_FROM_RAW.T@np.c_[tangent,bitangent,normal]).as_quat()[[3,0,1,2]]
        for k,key in enumerate(['x','y','z']):rec[key]=native[:,k]
        for k in range(3):rec[f'f_dc_{k}']=(fascia['rgb'][k]-.5)/.28209479177387814;rec[f'scale_{k}']=np.log(step*.88 if k<2 else .00065)
        for k in range(4):rec[f'rot_{k}']=quat[k]
        edge=np.minimum(np.c_[X,Y]-lo,hi-np.c_[X,Y]).min(1);fade=np.clip(edge/.018,0,1);fa=np.clip(fascia['opacity']*fade,.03,fascia['opacity']);rec['opacity']=np.log(fa/(1-fa));supports.append(rec)
        reconstruction.append({'id':fascia['id'],'added_count':len(rec),'world_bounds':[points.min(0).tolist(),points.max(0).tolist()],'rgb':fascia['rgb'],'normal_sigma':.00065,'normal_world':normal.tolist(),'source_plane_z_from_xy':[a,b,c]})
    indices=np.flatnonzero(changed)
    if supports:out=np.concatenate([out,*supports])
    report={'recipe_version':config.get('version',1),'input_count':len(vertices),'output_count':len(out),'added_count':len(out)-len(vertices),'reconstructed_surfaces':reconstruction,'distant_billboards_suppressed':int(len(distant)),
        'changed_count':int(changed.sum()),'moved_count':int(moved.sum()),'opacity_changed_count':int(altered_opacity.sum()),
        'changed_properties':['x','y','z','rot_0','rot_1','rot_2','rot_3','scale_0','scale_1','scale_2','opacity'],
        'changed_indices':indices.tolist(),'regions':regions,'preserved':'DC colors, semantic labels, SH coefficients, source row order; tangential covariance'}
    return out,report
