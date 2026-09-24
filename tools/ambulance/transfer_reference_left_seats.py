#!/usr/bin/env python3
"""Bounded capture transfer for the left seat's two fixed backrests.

Keeps captured geometry and SH, includes measured deeper material support, and
can suppress newly exposed original contributions in a composed hybrid.
The loose cushion straps and cabinet contents remain outside this recipe.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from cleanup import ROOT, columns, read_ply, sha256_file
from repair_surfaces import WORLD_FROM_RAW
from transfer_reference_seat import attenuate, backing, smoothstep

BOXES = [
    {'id': 'left-upper-backrest', 'bounds': [[-.355,.350,-1.65],[.315,.64,-1.08]], 'feather': .025},
    {'id': 'left-lower-backrest-and-anchors', 'bounds': [[-.355,-.10,-1.65],[.315,.34,-1.04]], 'feather': .025},
]
SURFACES = [
    {'id': 'left-upper-material', 'axis': 2, 'uv_axes': [0,1],
     'uv_bounds': [[-.27,.40],[.23,.585]], 'plane_depth_from_uv1': [-.10,.05,-1.34],
     'behind_sign': -1, 'deep_bounds': [-5.5,-1.22], 'sample_grid': [19,9]},
    {'id': 'left-lower-material', 'axis': 2, 'uv_axes': [0,1],
     'uv_bounds': [[-.27,-.04],[.23,.195]], 'plane_depth_from_uv1': [-.1,.05,-1.31],
     'behind_sign': -1, 'deep_bounds': [-5.5,-1.22], 'sample_grid': [19,11]},
]


def positions(v):
    return np.einsum('ij,nj->ni', WORLD_FROM_RAW, columns(v,['x','y','z']).astype(float))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--base-patch',type=Path)
    ap.add_argument('--hybrid',type=Path)
    ap.add_argument('--manual-phone-row',type=int,choices=[1245007,3804476],action='append',default=[],
                    help='Exact source row confirmed by a saved hybrid pixel attribution')
    ap.add_argument('--manual-fleck-cluster',action='store_true',
                    help='Suppress the measured duplicate white-point cluster behind the lower pad')
    args=ap.parse_args()
    phone_path=ROOT/'raw/ambulance_exp11_boot_sharp.ply'
    ref_path=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply'
    cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in ['left-seats','left-wall','left-grazing','cabinet-counter']]
    args.out.mkdir(parents=True,exist_ok=True)
    if args.manual_phone_row or args.manual_fleck_cluster:
        if not args.base_patch or args.base_patch.resolve()==args.out.resolve():
            ap.error('Manual attribution requires a separate base-patch and output')
        source,_,_=read_ply(phone_path)
        manual=np.asarray(args.manual_phone_row,dtype=np.int64)
        if args.manual_fleck_cluster:
            p=positions(source);rgb=.5+.28209479177387814*columns(source,['f_dc_0','f_dc_1','f_dc_2'])
            cluster=(np.linalg.norm(p-[-.6376,.0764,-1.9163],axis=1)<.006)&(rgb.min(1)>.4)
            manual=np.r_[manual,np.flatnonzero(cluster)]
        manual=np.unique(manual)
        if np.any(manual<0) or np.any(manual>=len(source)):
            ap.error('Manual source row out of range')
        prior=json.loads((args.base_patch/'report.json').read_text())
        sel=np.load(args.base_patch/'iphone-opacity-selection.npz')
        pm=np.ones(len(source),np.float32);pm[sel['indices']]=sel['multipliers'];pm[manual]=0
        ids=np.flatnonzero(pm<1)
        np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=ids,multipliers=pm[ids])
        shutil.copyfile(args.base_patch/'reference-selection.npz',args.out/'reference-selection.npz')
        shutil.copyfile(__file__,args.out/'generator.py')
        report={**prior,'status':'Unreviewed exact-row manual attribution correction',
                'base_patch':str(args.base_patch),'iphone_changed_rows':len(ids),
                'manual_original_rows':manual.tolist(),'manual_original_world_centers':positions(source[manual]).tolist(),
                'manual_attribution':'Saved left-seats-v3/v4-counter-pixel-contributors.json identifies a white duplicate-point cluster behind the lower pad, centered at[-.6376,.0764,-1.9163]. Row1245007 contributed20.06% at pixel70,319; after its removal row4672283 contributes21.79% in the surrounding footprint. The optional cluster mask is a .006-radius ball containing only pale source records; row3804476 is a separately attributed faint contributor.',
                'generator_sha256':sha256_file(args.out/'generator.py'),
                'review_cameras':[c['id'] for c in cameras]}
        (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True);return
    if args.base_patch:
        if not args.hybrid or args.out.resolve()==args.base_patch.resolve():
            ap.error('Hybrid refinement needs separate base-patch and output folders')
        prior=json.loads((args.base_patch/'report.json').read_text())
        phone,_,_=read_ply(args.hybrid/'iphone.ply');phone=phone.copy()
        ref,_,_=read_ply(args.hybrid/'reference-patches.ply')
        extra=np.empty(len(ref),dtype=phone.dtype)
        for name in phone.dtype.names: extra[name]=ref[name]
        sel=np.load(args.base_patch/'iphone-opacity-selection.npz')
        pw=np.zeros(len(phone),np.float32);pw[sel['indices']]=1-sel['multipliers']
        traces=[]
        for iteration in range(4):
            scene=np.concatenate([phone,extra])
            w,trace,ids=backing(scene,positions(scene),np.r_[pw,np.zeros(len(extra))],SURFACES,len(phone),cameras)
            w=w[:len(phone)];ids=np.flatnonzero(w>pw+1e-6)
            trace.update({'iteration':iteration,'new_phone_rows':len(ids)});traces.append(trace)
            if not len(ids): break
            phone[ids]=attenuate(phone[ids],np.zeros(len(ids)));pw=w
        ids=np.flatnonzero(pw>0)
        np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=ids,multipliers=1-pw[ids])
        shutil.copyfile(args.base_patch/'reference-selection.npz',args.out/'reference-selection.npz')
        report={**prior,'status':'Unreviewed actual-hybrid attribution refinement','hybrid':str(args.hybrid),
                'iphone_changed_rows':len(ids),'hybrid_traces':traces}
    else:
        results=[]
        for label,path in [('iphone',phone_path),('reference',ref_path)]:
            v,_,_=read_ply(path);p=positions(v);w=np.zeros(len(v),np.float32);regions=[]
            for box in BOXES:
                lo,hi=np.asarray(box['bounds']);bw=smoothstep(np.minimum(p-lo,hi-p).min(1)/box['feather'])
                w=np.maximum(w,bw);regions.append({**box,'rows':int((bw>0).sum())})
            w,trace,ids=backing(v,p,w,SURFACES,cameras=cameras)
            selected=np.flatnonzero(w>0)
            np.savez_compressed(args.out/('iphone-opacity-selection.npz' if label=='iphone' else 'reference-selection.npz'),
                                indices=selected,multipliers=1-w[selected] if label=='iphone' else w[selected])
            results.append({'source':label,'selected_rows':len(selected),'regions':regions,'trace':trace})
        report={'status':'Unreviewed bounded fixed-left-backrests transfer','iphone':str(phone_path),'iphone_sha256':sha256_file(phone_path),
                'reference':str(ref_path),'reference_sha256':sha256_file(ref_path),'regions':results,'surfaces':SURFACES,
                'exclusions':['Loose cushion straps','Seat cushion outside the backrest boxes','Cabinet contents','Bedding'],
                'review_cameras':[c['id'] for c in cameras]}
    report['review_cameras']=[c['id'] for c in cameras]
    shutil.copyfile(__file__,args.out/'generator.py')
    report['generator_sha256']=sha256_file(args.out/'generator.py')
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__': main()
