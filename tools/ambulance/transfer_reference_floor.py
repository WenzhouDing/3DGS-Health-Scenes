#!/usr/bin/env python3
"""Bounded, capture-backed aisle-floor trial layered onto accepted pass3.

Original captures and accepted assets are read-only. Source rows retain every
property except selected opacity; appended reference rows retain geometry,
covariance and SH3. A fitted plane limits selection and protects lifted objects;
it is never used to invent, flatten or repaint a surface.
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT, columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W

FLOOR = {'id': 'right-exposed-aisle', 'axis': 1, 'uv_axes': [0, 2],
         'uv_bounds': [[-1.60, -.08], [1.58, .66]],
         'plane_y_from_xz1': [-.006536127350277606, -.011240657756362868, -.7765111297658739],
         'material_residual_bounds': [-.10, .018], 'feather': .045,
         'deep_y_bounds': [-5.5, -.69], 'sample_grid': [49, 15]}

# Manually traced from both opposite downward views of the original. These
# could be loose cable or shadows; the absence in a separate capture does not
# justify removing them. Preserve an original narrow floor corridor underneath.
CURVE_GUARDS = [[[1.0052,-.0338],[.8492,.0139],[.6802,.0613],
                 [.4985,.1268],[.4921,.1859],[.5386,.2446]],
                [[-.2794,.3966],[-.1359,.3583],[.0184,.3021],
                 [.1102,.2542],[.1960,.1936],[.2437,.1431],
                 [.2236,.0703],[.2027,-.0014]]]


def smooth(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def world(v):
    return np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))


def plane_depth(p):
    coef=np.asarray(FLOOR['plane_y_from_xz1'])
    return p[:,1]-np.einsum('ij,j->i',p[:,[0,2]],coef[:2])-coef[2]


def curve_distance(p):
    uv=p[:,[0,2]];distance=np.full(len(p),np.inf)
    for curve in CURVE_GUARDS:
        for a,b in zip(np.asarray(curve)[:-1],np.asarray(curve)[1:]):
            delta=b-a;t=np.clip(np.einsum('ij,j->i',uv-a,delta)/np.dot(delta,delta),0,1)
            d=np.linalg.norm(uv-a-t[:,None]*delta,axis=1);distance=np.minimum(distance,d)
    return distance


def curve_guard(v,p):
    # The two photographs show dark lines, not a bright tread corridor. Full
    # corridor retention created a pale arc in V2, so retain only dark captured
    # support along the manually traced line. No color is changed or fabricated.
    rgb=.5+.28209479177387814*columns(v,['f_dc_0','f_dc_1','f_dc_2'])
    dark=rgb.mean(1)<.39
    return (1-smooth((curve_distance(p)-.006)/.014))*(np.abs(plane_depth(p))<.10)*dark


def material_weight(p):
    uv=p[:,[0,2]];lo,hi=np.asarray(FLOOR['uv_bounds']);edge=np.minimum(uv-lo,hi-uv).min(1)
    d=plane_depth(p);dmin,dmax=FLOOR['material_residual_bounds']
    weight=smooth(edge/FLOOR['feather'])*smooth((d-dmin)/.025)*smooth((dmax-d)/.008)
    return weight.astype(np.float32)


def cameras(extra):
    fixed=json.loads(Path(__file__).with_name('pass2-cameras.json').read_text())
    chosen=[c for c in fixed if c['id'] in ['mattress-top','mattress-grazing','bench-grazing']]
    if extra and extra.exists():chosen+=json.loads(extra.read_text())
    return chosen


def backing(v,p,cams,primary_rows=None):
    """Trace visible floor support with verified PlayCanvas opacity math.

    The source/ref support selections follow their measured pixel contributors.
    Bounded neutral medium/broad original veil slightly above the floor can also
    qualify; black wheels, fine detail and higher fixtures remain protected.
    """
    scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float))
    alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
    coef=np.asarray(FLOOR['plane_y_from_xz1']);lo,hi=np.asarray(FLOOR['uv_bounds'])
    nu,nv=FLOOR['sample_grid'];uv=np.array(np.meshgrid(np.linspace(lo[0]+.04,hi[0]-.04,nu),np.linspace(lo[1]+.025,hi[1]-.025,nv))).reshape(2,-1).T
    points=np.empty((len(uv),3));points[:,[0,2]]=uv;points[:,1]=np.einsum('ij,j->i',uv,coef[:2])+coef[2]
    selected_all=np.zeros(len(v),bool);stats=[]
    for camera in cams:
        eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);V=np.array([right,-np.cross(right,forward),forward])
        view=np.einsum('ij,nj->ni',V,p-eye);depth=view[:,2];safe=np.maximum(depth,.02)
        f=375/np.tan(np.radians(camera['fov'])/2);screen=np.c_[500+f*view[:,0]/safe,375+f*view[:,1]/safe];radius=4*f*scales.max(1)/safe+2
        q=np.einsum('ij,nj->ni',V,points-eye);pixels=np.c_[500+f*q[:,0]/q[:,2],375+f*q[:,1]/q[:,2]]
        visible=(q[:,2]>.02)&(pixels[:,0]>=0)&(pixels[:,0]<1000)&(pixels[:,1]>=0)&(pixels[:,1]<750);pixels=pixels[visible]
        if not len(pixels):continue
        low,high=pixels.min(0),pixels.max(0)
        overlap=(depth>.02)&(screen[:,0]+radius>low[0])&(screen[:,0]-radius<high[0])&(screen[:,1]+radius>low[1])&(screen[:,1]-radius<high[1])
        ids=np.flatnonzero(overlap);ids=ids[np.argsort(depth[ids])]
        Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@W,Q)
        J=np.zeros((len(ids),2,3));J[:,0,0]=f/depth[ids];J[:,1,1]=f/depth[ids]
        J[:,0,2]=-f*np.clip(view[ids,0]/depth[ids],-1.3*500/f,1.3*500/f)/depth[ids]
        J[:,1,2]=-f*np.clip(view[ids,1]/depth[ids],-1.3*375/f,1.3*375/f)/depth[ids]
        B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C)
        det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2
        exact_radius=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)));fade=np.clip((2048/1080*750-exact_radius)/(1024/1080*750),0,1)
        # Below-sheet support is eligible; this excludes original lifted cables,
        # casters and rails even if their projected ellipses overlap floor rays.
        residual=plane_depth(p[ids]);dlo,dhi=FLOOR['deep_y_bounds']
        eligible=(residual<-.015)&(p[ids,1]>dlo)&(p[ids,1]<dhi)&(np.abs(p[ids])<7).all(1)
        if primary_rows is None:
            # Native full-scene pixel traces demonstrate that smooth captured
            # floor appearance also uses supports slightly above the sheet.
            # Keeping only its thin surface exaggerates the visible crisscross
            # reconstruction lines. Include these actual near-floor contributors
            # with their unchanged covariance/color/SH, never synthetic tint.
            rgb=.5+.28209479177387814*columns(v[ids],['f_dc_0','f_dc_1','f_dc_2'])
            near_support=(residual>.018)&(residual<.225)&(scales[ids].max(1)>.015)
            near_support&=(np.ptp(rgb,axis=1)<.22)
            near_support&=(p[ids,0]>-1.55)&(p[ids,0]<1.53)&(p[ids,2]>-.03)&(p[ids,2]<.62)
            eligible|=near_support
        if primary_rows is not None:
            # Pixel attribution in V2 identified the retained veil at +.03 to
            # +.20 above the actual floor. Only neutral medium/broad supports
            # observed on exposed-floor rays qualify, excluding black wheels,
            # colored gear, small detailed splats, and higher lifted fixtures.
            rgb=.5+.28209479177387814*columns(v[ids],['f_dc_0','f_dc_1','f_dc_2'])
            floor_veil=(residual>.02)&(residual<.225)&(scales[ids].max(1)>.015)
            floor_veil&=(rgb.min(1)>.19)&(rgb.mean(1)>.28)&(np.ptp(rgb,axis=1)<.22)
            floor_veil&=(p[ids,0]>-1.55)&(p[ids,0]<1.53)&(p[ids,2]>-.03)&(p[ids,2]<.62)
            eligible|=floor_veil
            eligible&=ids<primary_rows
            eligible&=curve_guard(v[ids],p[ids])<1e-6
        chosen=np.zeros(len(ids),bool);coverage=[];primary_mass=[]
        for pixel in pixels:
            d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d)
            a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)));T=np.r_[1,np.cumprod(1-a[:-1])];weight=a*T
            chosen|=(weight>.003)&eligible;coverage.append(float(weight.sum()))
            if primary_rows is not None:primary_mass.append(float(weight[eligible].sum()))
        selected_all[ids[chosen]]=True
        record={'camera':camera['id'],'rays':len(pixels),'contributors':int(chosen.sum()),'alpha_median':float(np.median(coverage))}
        if primary_mass:record['eligible_original_mass_median']=float(np.median(primary_mass))
        stats.append(record);print(record,flush=True)
    ids=np.flatnonzero(selected_all)
    return ids,{'method':'Full perspective covariance and front-to-back alpha attribution >.003 on exposed-floor rays; geometry/covariance/SH unchanged',
                'eligible_depth':'Below observed floor by .015; reference also includes actual neutral medium/broad support +.018 to +.225. Original hybrid also includes attributed neutral medium/broad veil +.02 to +.225. Raised support is bounded to the exposed aisle.',
                'count':len(ids),'world_bounds':[p[ids].min(0).tolist(),p[ids].max(0).tolist()] if len(ids) else None,'views':stats}


def attenuate(v,m):
    out=v.copy();a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));a=np.clip(a*m,1e-8,1-1e-8);out['opacity']=np.log(a/(1-a));return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iphone',type=Path,default=ROOT/'raw/ambulance_exp11_boot_sharp.ply')
    ap.add_argument('--reference',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply')
    ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/final-bench')
    ap.add_argument('--out',type=Path,default=ROOT/'raw/ambulance-cleanup/pass4/floor-v1')
    ap.add_argument('--extra-cameras',type=Path,default=ROOT/'raw/ambulance-cleanup/pass4/review-cameras.json')
    ap.add_argument('--peels',type=int,default=3)
    ap.add_argument('--reuse-source-selection',type=Path,help='Reuse a reviewed source selection folder when testing reference coverage alone')
    args=ap.parse_args()
    if args.out.resolve()==args.baseline.resolve():raise ValueError('Accepted baseline overwrite forbidden')
    args.out.mkdir(parents=True,exist_ok=True);source,_,_=read_ply(args.iphone);reference,_,_=read_ply(args.reference)
    baseline=json.loads((args.baseline/'report.json').read_text());ph,rh=sha256_file(args.iphone),sha256_file(args.reference)
    if baseline['iphone_sha256']!=ph or baseline['reference_sha256']!=rh:raise ValueError('Baseline capture hashes differ')
    base=np.load(args.baseline/'selections.npz');base_pm=np.ones(len(source));base_rm=np.zeros(len(reference))
    base_pm[base['iphone_indices']]=base['iphone_multipliers'];base_rm[base['reference_indices']]=base['reference_multipliers']
    pp,rp=world(source),world(reference);pw=material_weight(pp);rw=material_weight(rp);cams=cameras(args.extra_cameras)
    guarded=curve_guard(source,pp)
    pw*=1-guarded
    if args.reuse_source_selection:
        prior=json.loads((args.reuse_source_selection/'report.json').read_text())
        if prior['iphone_sha256']!=ph or prior['reference_sha256']!=rh:raise ValueError('Reused selection source hashes differ')
        cached=np.load(args.reuse_source_selection/'iphone-opacity-selection.npz')
        pw[:]=0;pw[cached['indices']]=1-cached['multipliers']
    ri_deep,ref_trace=backing(reference,rp,cams);rw[ri_deep]=1
    all_rm=np.maximum(base_rm,rw);all_ri=np.flatnonzero(all_rm>0);patch=attenuate(reference[all_ri],all_rm[all_ri])
    current=source.copy();combined_pm=np.minimum(base_pm,1-pw);pi=np.flatnonzero(combined_pm<1);current[pi]=attenuate(source[pi],combined_pm[pi])
    extra=np.empty(len(patch),dtype=source.dtype)
    for name in source.dtype.names:extra[name]=patch[name]
    traces=[];additional=[]
    for iteration in range(args.peels):
        scene=np.concatenate([current,extra]);positions=np.concatenate([pp,rp[all_ri]])
        ids,trace=backing(scene,positions,cams,len(source));ids=ids[pw[ids]<1];trace['iteration']=iteration;trace['new_source_rows']=len(ids);traces.append(trace)
        print('Actual hybrid floor trace',iteration,len(ids),'original rows',flush=True)
        if not len(ids):break
        pw[ids]=1;current[ids]=attenuate(current[ids],np.zeros(len(ids)));additional.extend(ids.tolist())
    pi=np.flatnonzero(pw>0);ri=np.flatnonzero(rw>0)
    np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi])
    np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rw[ri])
    np.savez_compressed(args.out/'floor-contributors.npz',reference_indices=ri_deep,iphone_indices=np.asarray(additional,dtype=np.int64))
    write_ply(args.out/'reference-floor.ply',attenuate(reference[ri],rw[ri]))
    write_ply(args.out/'iphone.ply',current);write_ply(args.out/'reference-patches.ply',patch)
    checks={'original_nonopacity_fields_exact':all(np.array_equal(current[n],source[n]) for n in source.dtype.names if n!='opacity'),
            'reference_nonopacity_fields_exact':all(np.array_equal(patch[n],reference[n][all_ri]) for n in reference.dtype.names if n!='opacity'),
            'baseline_sog_unchanged':sha256_file(args.baseline/'hybrid.sog')==baseline['browser_export']['sha256']}
    if not all(checks.values()):raise RuntimeError(checks)
    report={'status':'Unreviewed floor trial layered onto accepted final-bench; local only',
            'iphone':str(args.iphone),'iphone_sha256':ph,'reference':str(args.reference),'reference_sha256':rh,
            'baseline':str(args.baseline),'baseline_report_sha256':sha256_file(args.baseline/'report.json'),
            'regions':[FLOOR],'original_curve_guards':{'polylines_xz':CURVE_GUARDS,'full_width':.012,'feather_width':.014,'depth_half_width':.10,'maximum_mean_dc_rgb':.39,'guarded_rows':int((guarded>0).sum())},
            'attributed_above_floor_veil':{'residual_bounds':[.02,.225],'minimum_max_sigma':.015,'minimum_rgb_channel':.19,'minimum_rgb_mean':.28,'maximum_rgb_chroma':.22,'xz_bounds':[[-1.55,-.03],[1.53,.62]],'method':'Only records with actual hybrid visible contribution >.003 on floor rays are selected.'},
            'reference_near_floor_support':{'residual_bounds':[.018,.225],'minimum_max_sigma':.015,'maximum_rgb_chroma':.22,'xz_bounds':[[-1.55,-.03],[1.53,.62]],'method':'Actual native visible contribution >.003 on exposed-floor rays; unchanged capture covariance/DC/SH.'},
            'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),
            'combined_reference_rows':len(all_ri),'reference_trace':ref_trace,'hybrid_traces':traces,'checks':checks,
            'source_selection_reused_from':str(args.reuse_source_selection) if args.reuse_source_selection else None,
            'source_selection_reused_sha256':sha256_file(args.reuse_source_selection/'iphone-opacity-selection.npz') if args.reuse_source_selection else None,
            'review_cameras':[c['id'] for c in cams],
            'limitations':['Only right exposed aisle selected; floor beneath equipment and left side not claimed repaired.',
                           'Raised cable, wheel, stretcher and bench details remain original; near-surface haze above selection may remain.',
                           'Deep captured radiance support is retained when observed; not a claim of physical floor shell geometry.']}
    shutil.copyfile(__file__,args.out/'generator.py');report['generator_sha256']=sha256_file(args.out/'generator.py')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
