"""Conservative, reproducible manual repair of measured ambulance upholstery.

The bounded recipes are intentionally semantic: no geometry is inferred outside
observed cushion/pad interiors. Original attributes are preserved except opacity
attenuation of a bright diffuse veil demonstrably ahead of a fitted dark pad.
New Gaussian backing is generated only where measured surface coverage is weak.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

SH_C0 = 0.28209479177387814
WORLD_FROM_RAW = Rotation.from_euler('xyz', [-78.243, .463, -4.499], degrees=True).as_matrix() @ np.diag([-1., -1., 1.])


def _rounded_mask(uv, lo, hi, radius):
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    q = np.maximum(np.abs(uv - center) - (half - radius), 0)
    return ((uv >= lo) & (uv <= hi)).all(1) & (np.sum(q*q, axis=1) <= radius*radius)


def _fit_plane(uv, w):
    """Robust depth regression; reject volumetric splats before fitting backing."""
    A = np.column_stack([uv, np.ones(len(uv))])
    coef = np.linalg.lstsq(A, w, rcond=None)[0]
    keep = np.ones(len(w), bool)
    for _ in range(6):
        residual = w - np.einsum('ni,i->n', A, coef)
        center = np.median(residual[keep])
        mad = np.median(np.abs(residual[keep] - center))
        keep = np.abs(residual - center) < max(.008, 2.5 * 1.4826 * mad)
        if keep.sum() < 40:
            break
        coef = np.linalg.lstsq(A[keep], w[keep], rcond=None)[0]
    return coef, keep


def repair(vertices, config=None):
    """Return (structured PLY vertices, report), retaining source rows in order."""
    if config is None:
        config = json.loads(Path(__file__).with_name('surfaces.json').read_text())
    elif isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    required = ['x','y','z','opacity'] + [f'{prefix}_{i}' for prefix, n in [('scale',3),('rot',4),('f_dc',3)] for i in range(n)]
    if not vertices.dtype.names or any(k not in vertices.dtype.names for k in required):
        raise ValueError('repair requires standard structured Gaussian PLY vertices')
    xyz_raw = np.column_stack([vertices[k] for k in ['x','y','z']]).astype(np.float64)
    xyz = np.einsum('ij,nj->ni', WORLD_FROM_RAW, xyz_raw)
    color = np.clip(.5 + SH_C0 * np.column_stack([vertices[f'f_dc_{i}'] for i in range(3)]), 0, 1)
    alpha = 1 / (1 + np.exp(-np.clip(vertices['opacity'], -40, 40)))
    sigma = np.exp(np.clip(np.column_stack([vertices[f'scale_{i}'] for i in range(3)]), -40, 10))
    neutral_dark = (color.max(1) < .48) & (np.ptp(color, axis=1) < .105)
    credible = neutral_dark & (alpha > .28) & (sigma.max(1) < .060) & np.isfinite(xyz).all(1)
    appended = []
    out = vertices.copy()
    changed = np.zeros(len(out), bool)
    reports = []
    spacing = float(config.get('spacing', .008))
    for patch in config['patches']:
        axis = int(patch['axis']); axes = patch['uv_axes']
        lo, hi = np.asarray(patch['uv_bounds'], float)
        dlo, dhi = patch['depth_bounds']; sign = patch['backing_sign']
        uv = xyz[:, axes]; w = xyz[:, axis]
        domain = _rounded_mask(uv, lo, hi, patch['corner_radius'])
        bound = domain & (w >= dlo) & (w <= dhi)
        seed_indices = np.flatnonzero(bound & credible)
        pr = {'id': patch['id'], 'axis': axis, 'uv_axes': axes, 'uv_bounds': [lo.tolist(),hi.tolist()],
              'depth_bounds': [dlo,dhi], 'support_count': int(len(seed_indices)), 'added_count': 0, 'attenuated_count': 0}
        if len(seed_indices) < 80:
            pr['skipped'] = 'insufficient dark surface support'; reports.append(pr); continue
        coef, fit_keep = _fit_plane(uv[seed_indices], w[seed_indices])
        seed_indices = seed_indices[fit_keep]
        if len(seed_indices) < 40:
            pr['skipped'] = 'insufficient inliers after robust surface fit'; reports.append(pr); continue
        if np.linalg.norm(coef[:2]) > .65:
            pr['skipped'] = 'unstable fitted surface orientation'; reports.append(pr); continue
        seed_uv = uv[seed_indices]; seed_w = w[seed_indices]
        tree = cKDTree(seed_uv)
        uu = np.arange(lo[0] + spacing/2, hi[0], spacing)
        vv = np.arange(lo[1] + spacing/2, hi[1], spacing)
        grid = np.stack(np.meshgrid(uu,vv), axis=-1).reshape(-1,2)
        grid = grid[_rounded_mask(grid, lo, hi, patch['corner_radius'])]
        dist, near = tree.query(grid, k=min(32,len(seed_indices)), workers=1)
        nearest = dist[:,0]
        valid = nearest < patch['max_support_distance']
        grid = grid[valid]; dist = dist[valid]; near = near[valid]; nearest = nearest[valid]
        if not len(grid):
            pr['skipped'] = 'no supported gap cells'; reports.append(pr); continue
        global_depth = np.einsum('ni,i->n', np.column_stack([grid,np.ones(len(grid))]), coef)
        seed_residual = seed_w - np.einsum('ni,i->n', np.column_stack([seed_uv,np.ones(len(seed_uv))]), coef)
        weights = np.exp(-.5*(dist/.09)**2)
        local_residual = np.sum(seed_residual[near]*weights,axis=1)/np.maximum(weights.sum(1),1e-20)
        depth = global_depth + np.clip(local_residual,-.010,.010) + sign*patch.get('backing_offset',.016)
        depth = np.clip(depth, dlo+.002, dhi-.002)
        # Coverage is measured on splats lying close to this fitted material shell.
        qraw = np.column_stack([vertices[f'rot_{i}'][seed_indices] for i in range(4)])
        qraw /= np.maximum(np.linalg.norm(qraw,axis=1,keepdims=True),1e-20)
        matrices = Rotation.from_quat(qraw[:,[1,2,3,0]]).as_matrix()
        world_rot = np.einsum('ij,njk->nik',WORLD_FROM_RAW,matrices)
        variances = np.einsum('nij,nj->ni',world_rot**2,sigma[seed_indices]**2)
        uv_sigma = np.sqrt(variances[:,axes])
        delta = grid[:,None,:]-seed_uv[near]
        sig = np.clip(uv_sigma[near],.0004,.030)
        power = -.5*np.sum((delta/sig)**2,axis=2)
        near_shell = np.abs(seed_w[near]-depth[:,None]) < .025
        density = np.sum(alpha[seed_indices][near]*np.exp(np.maximum(power,-30))*near_shell,axis=1)
        needs = density < 1.8
        grid = grid[needs];depth = depth[needs];near = near[needs];nearest = nearest[needs];density = density[needs]
        normal = np.zeros(3);normal[axis]=1;normal[axes]=-coef[:2];normal/=np.linalg.norm(normal)
        tangent = np.zeros(3);tangent[axes[0]]=1;tangent[axis]=coef[0];tangent/=np.linalg.norm(tangent)
        bitangent = np.cross(normal,tangent);bitangent/=np.linalg.norm(bitangent)
        raw_frame = WORLD_FROM_RAW.T @ np.column_stack([tangent,bitangent,normal])
        q = Rotation.from_matrix(raw_frame).as_quat()[[3,0,1,2]]
        add = np.zeros(len(grid),dtype=vertices.dtype)
        points=np.zeros((len(grid),3));points[:,axes]=grid;points[:,axis]=depth
        raw=np.einsum('ij,nj->ni',WORLD_FROM_RAW.T,points)
        for i,k in enumerate(['x','y','z']): add[k]=raw[:,i]
        # Median neighborhood material color avoids bringing the pale veil into holes.
        local_rgb = np.median(color[seed_indices][near],axis=1)
        material = np.quantile(color[seed_indices],patch.get('material_quantile',.70),axis=0)
        # The material itself is smooth; large local color swings are capture
        # noise on these untextured panels. Preserve only a soft local variation.
        rgb = np.clip(.20*local_rgb+.80*material,patch.get('material_floor',.20),.40)
        for i in range(3):
            add[f'f_dc_{i}']=(rgb[:,i]-.5)/SH_C0
            add[f'scale_{i}']=np.log(spacing*.80 if i<2 else .0007)
        for i in range(4): add[f'rot_{i}']=q[i]
        edge_distance = np.minimum(grid-lo,hi-grid).min(1)
        feather = np.clip(edge_distance/patch.get('edge_feather',.035),0,1)
        feather = feather*feather*(3-2*feather)
        base_alpha = patch.get('opacity_base',.43)
        fill_alpha = np.clip((base_alpha-.10*density)*feather,.025,base_alpha)
        add['opacity']=np.log(fill_alpha/(1-fill_alpha))
        for i,k in enumerate(['nx','ny','nz']):
            if k in add.dtype.names: add[k]=(WORLD_FROM_RAW.T @ normal)[i]
        appended.append(add)
        # Semantic veil reduction is limited to faint neutral white blobs in front
        # of dark pad interiors; bright buckles and near-opaque texture are retained.
        local_domain = _rounded_mask(uv,lo+.015,hi-.015,max(.005,patch['corner_radius']-.005))
        idx = np.flatnonzero(local_domain & (alpha<.50) & (color.min(1)>.52) & (np.ptp(color,axis=1)<.09) & (sigma.max(1)>.009))
        if len(idx):
            expected = np.einsum('ni,i->n', np.column_stack([uv[idx],np.ones(len(idx))]), coef)
            ahead = sign*(expected-w[idx])
            idx=idx[(ahead>.015)&(ahead<.11)]
            if len(idx):
                reduced=alpha[idx]*.20
                out['opacity'][idx]=np.log(reduced/(1-reduced));changed[idx]=True
        pr.update({'support_count':int(len(seed_indices)), 'added_count':int(len(add)), 'attenuated_count':int(len(idx)),
                   'plane_coefficients':coef.tolist(),'normal_world':normal.tolist(),
                   'tangent_axes_world':[tangent.tolist(),bitangent.tolist()],
                   'world_bounds':[points.min(0).tolist(),points.max(0).tolist()] if len(points) else None,
                   'seed_distance_max':float(nearest.max()) if len(nearest) else 0,
                   'sigma_tangent':spacing*.80,'sigma_normal':.0007,
                   'color_min':rgb.min(0).tolist() if len(rgb) else None,'color_max':rgb.max(0).tolist() if len(rgb) else None,
                   'mean_existing_density_at_additions':float(density.mean()) if len(density) else 0})
        reports.append(pr)
    if appended: out = np.concatenate([out,*appended])
    return out, {'input_count':len(vertices),'added_count':len(out)-len(vertices),'output_count':len(out),
                 'attenuated_count':int(changed.sum()),'patches':reports,'recipe_version':config.get('version',1)}
