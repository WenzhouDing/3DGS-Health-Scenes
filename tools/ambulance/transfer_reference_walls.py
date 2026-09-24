#!/usr/bin/env python3
"""Transfer bounded, observed static paint from registered Insta360 splats.

Diagnostic only until reviewed. Original iPhone records stay in their original
order and only selected opacity values change. The patch retains the reference
positions, covariance, DC colors and all rotated directional SH coefficients.
Glass contents, worktop equipment, screens, harness and bedding are excluded.
"""
import argparse,json
from pathlib import Path
import numpy as np

from cleanup import ROOT,columns,read_ply,sha256_file,write_ply
from repair_surfaces import WORLD_FROM_RAW

PATCHES=[
    {'id':'left-lower-white-cabinet','axis':2,'uv_axes':[0,1],
     # The lower photographed panel is poorly constrained in the reference;
     # exclude its y≈-.41 colored ghosts rather than importing more support.
     'uv_bounds':[[-1.38,-.34],[-.36,.135]],
     'plane_depth_from_uv1':[-.08237276,.02981509,-.92014449],
     'phone_depth_residual_bounds':[-.025,.080],
     'reference_depth_residual_bounds':[-.020,.020],
     'feather':.03,'depth_feather':.006,'color_filter':False},
    {'id':'rear-lower-door-paint','axis':0,'uv_axes':[2,1],
     'uv_bounds':[[-.67,-.605],[.64,-.36]],
     'plane_depth_from_uv1':[.09023278,.00961142,-1.57076640],
     'phone_depth_residual_bounds':[-.025,.075],
     'reference_depth_residual_bounds':[-.020,.020],
     'feather':.03,'depth_feather':.006},
]


def smoothstep(v):
    v=np.clip(v,0,1);return v*v*(3-2*v)


def weights(vertices,patches,reference=False):
    p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(vertices,['x','y','z']).astype(float))
    rgb=.5+.28209479177387814*columns(vertices,['f_dc_0','f_dc_1','f_dc_2'])
    # No brightness cut: the actual white laminate contains dark grain. Only
    # saturated capture-specific material is protected from replacement.
    neutral=np.ptp(rgb,axis=1)<.28
    total=np.zeros(len(vertices),np.float32);report=[]
    for region in patches:
        uv=p[:,region['uv_axes']];lo,hi=np.array(region['uv_bounds']);coef=np.array(region['plane_depth_from_uv1'])
        edge=np.minimum(uv-lo,hi-uv).min(1)
        depth=p[:,region['axis']]-np.einsum('ij,j->i',uv,coef[:2])-coef[2]
        dlo,dhi=region['reference_depth_residual_bounds' if reference else 'phone_depth_residual_bounds']
        w=smoothstep(edge/region['feather'])
        if region.get('color_filter',True):w*=neutral
        w*=smoothstep(np.minimum(depth-dlo,dhi-depth)/region['depth_feather'])
        total=np.maximum(total,w.astype(np.float32));ids=np.flatnonzero(w>0)
        report.append({**region,'selected_count':len(ids),'bounds_world':[p[ids].min(0).tolist(),p[ids].max(0).tolist()] if len(ids) else None,
            'depth_residual_quantiles':np.quantile(depth[ids],[.01,.5,.99]).tolist() if len(ids) else None})
    return total,report


def attenuate(v,multiplier):
    out=v.copy();a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
    a=np.clip(a*multiplier,1e-8,1-1e-8);out['opacity']=np.log(a/(1-a));return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--iphone',type=Path,default=ROOT/'raw/ambulance_exp11_boot_sharp.ply')
    ap.add_argument('--reference',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply')
    ap.add_argument('--out',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/trials/walls')
    ap.add_argument('--include-rear',action='store_true',help='Rejected V1 rear transfer, retained only for diagnosis')
    ap.add_argument('--write-trial-phone',action='store_true')
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    phone,_,_=read_ply(args.iphone);ref,_,_=read_ply(args.reference)
    if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):raise ValueError('Reference must retain all degree-three SH')
    patches=PATCHES if args.include_rear else PATCHES[:1]
    pw,pr=weights(phone,patches);rw,rr=weights(ref,patches,reference=True)
    ids=np.flatnonzero(pw>0);refs=np.flatnonzero(rw>0)
    np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=ids,multipliers=1-pw[ids])
    np.savez_compressed(args.out/'reference-selection.npz',indices=refs,multipliers=rw[refs])
    patch=attenuate(ref[refs],rw[refs]);write_ply(args.out/'reference-walls.ply',patch)
    if args.write_trial_phone:
        out=phone.copy();out[ids]=attenuate(phone[ids],1-pw[ids]);write_ply(args.out/'iphone.ply',out)
        assert all(np.array_equal(out[n],phone[n]) for n in phone.dtype.names if n!='opacity')
    assert all(np.array_equal(patch[n],ref[n][refs]) for n in ref.dtype.names if n!='opacity')
    report={'status':'Unreviewed actual-reference transfer; no active asset modified',
        'iphone':str(args.iphone),'iphone_sha256':sha256_file(args.iphone),
        'reference':str(args.reference),'reference_sha256':sha256_file(args.reference),
        'iphone_changed_rows':len(ids),'reference_added_rows':len(refs),
        'iphone_regions':pr,'reference_regions':rr,
        'iphone_only_changed_field':'opacity','reference_unchanged_fields':'all except feathered opacity',
        'plane_source':'Robust fit of neutral bright registered reference centers; fit chooses the material band only, never moves centers',
        'exclusions':['Glass panels and contents above white cabinet','Screen, telephone, worktop gear','Bedding and movable bags','Rear windows, latch/handle region, red seams and bottom warning stripes'],
        'review_cameras':['left-wall','left-grazing','rear-wall','mattress-top']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
