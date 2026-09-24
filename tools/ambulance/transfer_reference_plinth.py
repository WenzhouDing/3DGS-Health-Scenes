#!/usr/bin/env python3
"""Actual captured material for the static bench base facing the aisle.

The plane describes the manually identified face; it limits opacity selection
and does not flatten or synthesize geometry. Yellow/red foreground straps are
protected. Review alongside the separately selected horizontal floor.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cleanup import ROOT, columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW
from transfer_reference_seat import attenuate, backing, smoothstep

SURFACE={'id':'bench-base-aisle-face','axis':2,'uv_axes':[0,1],
         'uv_bounds':[[-1.17,-.77],[.91,-.37]],
         'plane_depth_from_uv1':[-.08432874,.02431246,.38086325],
         'behind_sign':1,'deep_bounds':[.29,4.5],'sample_grid':[37,13]}


def world(v):
    return np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float))


def material(v,p):
    lo,hi=np.asarray(SURFACE['uv_bounds']);uv=p[:,:2]
    w=smoothstep(np.minimum(uv-lo,hi-uv).min(1)/.035)
    coef=np.asarray(SURFACE['plane_depth_from_uv1']);d=p[:,2]-np.einsum('ij,j->i',uv,coef[:2])-coef[2]
    w*=smoothstep((d+.065)/.02)*smoothstep((.22-d)/.03)
    rgb=.5+.28209479177387814*columns(v,['f_dc_0','f_dc_1','f_dc_2'])
    saturated=(rgb.max(1)-rgb.min(1)>.18)&(rgb.max(1)>.35)
    # Saturated foreground restraints differ between captures. Neutral face
    # material and black built-in hardware remain eligible.
    protect=saturated&(d<.025)
    w[protect]=0
    return w.astype(np.float32),protect


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--refine-base',type=Path,help='Apply the exact manually attributed V1 haze corrections')
    args=ap.parse_args()
    baseline=ROOT/'raw/ambulance-cleanup/pass3/final-bench'
    phone_path=ROOT/'raw/ambulance_exp11_boot_sharp.ply';ref_path=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply'
    args.out.mkdir(parents=True,exist_ok=True)
    if args.refine_base:
        if args.refine_base.resolve()==args.out.resolve():ap.error('Keep the base trial immutable')
        base=args.refine_base;current,_,_=read_ply(base/'iphone.ply');current=current.copy()
        # Exact hybrid rays identify these original Gaussians as colored or
        # gray overlays. They are not the native dark face or metal hardware.
        # Row2231388 deliberately overrides the broad saturation guard: it is
        # a blue haze fragment on this neutral face, not a red/yellow restraint.
        manual=np.array([2231388,2131270,2846723,2937227],dtype=np.int64)
        current[manual]=attenuate(current[manual],np.zeros(len(manual)))
        selection=np.load(base/'iphone-opacity-selection.npz');pw=np.zeros(len(current),np.float32)
        pw[selection['indices']]=1-selection['multipliers'];pw[manual]=1;ids=np.flatnonzero(pw>0)
        np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=ids,multipliers=1-pw[ids])
        shutil.copyfile(base/'reference-selection.npz',args.out/'reference-selection.npz')
        shutil.copyfile(base/'reference-patches.ply',args.out/'reference-patches.ply');write_ply(args.out/'iphone.ply',current)
        shutil.copyfile(__file__,args.out/'generator.py')
        report=json.loads((base/'report.json').read_text())
        report.update({'status':'Unreviewed manually attributed plinth overlay correction','base':str(base),
                       'manual_rows':manual.tolist(),'manual_trace':'plinth-audit/hybrid-plinth-v1-floor-front-down-pixels.json',
                       'iphone_changed_rows':len(ids),'generator_sha256':sha256_file(args.out/'generator.py')})
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return
    phone,_,_=read_ply(phone_path);ref,_,_=read_ply(ref_path)
    p,r=world(phone),world(ref);pw,protected=material(phone,p);rw,rprotected=material(ref,r)
    cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in ['mattress-top','bench-grazing']]
    cameras+=json.loads((ROOT/'raw/ambulance-cleanup/pass4/review-cameras.json').read_text())
    rw,rtrace,_=backing(ref,r,rw,[SURFACE],cameras=cameras);rw[rprotected]=0
    prior=np.load(baseline/'selections.npz');pm=np.ones(len(phone));rm=np.zeros(len(ref))
    pm[prior['iphone_indices']]=prior['iphone_multipliers'];rm[prior['reference_indices']]=prior['reference_multipliers']
    rm=np.maximum(rm,rw);ri=np.flatnonzero(rm>0);rp=attenuate(ref[ri],rm[ri])
    extra=np.empty(len(rp),dtype=phone.dtype)
    for name in phone.dtype.names:extra[name]=rp[name]
    current=phone.copy();mul=np.minimum(pm,1-pw);changed=np.flatnonzero(mul<1);current[changed]=attenuate(phone[changed],mul[changed])
    traces=[]
    for iteration in range(4):
        scene=np.concatenate([current,extra]);positions=np.concatenate([p,r[ri]])
        w,trace,_=backing(scene,positions,np.r_[pw,np.zeros(len(extra))],[SURFACE],len(phone),cameras)
        w=w[:len(phone)];w[protected]=0;ids=np.flatnonzero(w>pw+1e-6)
        trace.update({'iteration':iteration,'new_phone_rows':len(ids)});traces.append(trace)
        if not len(ids):break
        current[ids]=attenuate(current[ids],np.zeros(len(ids)));pw=w
    pi=np.flatnonzero(pw>0);newri=np.flatnonzero(rw>0)
    np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi])
    np.savez_compressed(args.out/'reference-selection.npz',indices=newri,multipliers=rw[newri])
    write_ply(args.out/'iphone.ply',current);write_ply(args.out/'reference-patches.ply',rp)
    shutil.copyfile(__file__,args.out/'generator.py')
    report={'status':'Unreviewed actual bench-base transfer','iphone':str(phone_path),'iphone_sha256':sha256_file(phone_path),
            'reference':str(ref_path),'reference_sha256':sha256_file(ref_path),'surface':SURFACE,
            'iphone_changed_rows':len(pi),'reference_added_rows':len(newri),'protected_saturated_foreground_rows':int(protected.sum()),
            'reference_trace':rtrace,'hybrid_traces':traces,'generator_sha256':sha256_file(args.out/'generator.py'),
            'review_cameras':[c['id'] for c in cameras],
            'limitations':['Horizontal floor is a separate patch.','Deep support is captured radiance, not a recovered physical shell.','Foreground straps, cushion and bags must pass preservation checks.']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
