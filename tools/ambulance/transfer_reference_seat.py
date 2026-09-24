#!/usr/bin/env python3
"""One reviewed trial of actual registered front-seat appearance/support.

Selects the fixed seat's observed records and measured deeper contributors.
No position, covariance, color or SH coefficient is generated or changed.
Original neighboring equipment is retained; the whole captured seat, including
its existing logo and harness, must pass same-view review before integration.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT,columns,read_ply,sha256_file,write_ply
from repair_surfaces import WORLD_FROM_RAW

BOXES=[
    {'id':'backrest-and-harness','bounds':[[1.265,-.27,-.85],[1.655,.50,-.29]],'feather':.02},
    {'id':'cushion-and-lip','bounds':[[.975,-.465,-.86],[1.43,-.235,-.29]],'feather':.02},
]
PATCHES=[
    {'id':'backrest','axis':0,'uv_axes':[2,1],
     'uv_bounds':[[-.745,-.19],[-.37,.375]],
     'plane_depth_from_uv1':[.10282582,.15635052,1.55029359],
     'behind_sign':1,'deep_bounds':[1.27,5.5],'sample_grid':[11,19]},
    {'id':'cushion-top','axis':1,'uv_axes':[0,2],
     'uv_bounds':[[1.105,-.745],[1.37,-.36]],
     'plane_depth_from_uv1':[.12156235,-.01223687,-.46314444],
     'behind_sign':-1,'deep_bounds':[-3.,-.24],'sample_grid':[9,13]},
    {'id':'cushion-lip','axis':0,'uv_axes':[2,1],
     'uv_bounds':[[-.745,-.415],[-.36,-.325]],
     'plane_depth_from_uv1':[.20938654,.99880679,1.53046372],
     'behind_sign':1,'deep_bounds':[.975,5.5],'sample_grid':[15,5]},
]
CROWN=[
    {'id':'attributed-crown-background','axis':0,'uv_axes':[2,1],
     'uv_bounds':[[-.68,.39],[-.43,.46]],
     'plane_depth_from_uv1':[.10282582,.15635052,1.55029359],
     'behind_sign':1,'deep_bounds':[1.64,2.1],'sample_grid':[13,7],
     'center_bounds':[[1.64,.30,-.90],[2.1,.62,-.20]]},
]


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def base_weights(p):
    weights=np.zeros(len(p),np.float32);regions=[]
    for box in BOXES:
        lo,hi=np.asarray(box['bounds']);edge=np.minimum(p-lo,hi-p).min(1)
        w=smoothstep(edge/box['feather']);weights=np.maximum(weights,w.astype(np.float32))
        regions.append({**box,'selected_count':int((w>0).sum())})
    return weights,regions


def backing(v,p,base,patches=None,primary_rows=None,cameras=None):
    scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float))
    alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
    if cameras is None:
        cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in ['front-seat','front-wall','left-grazing']]
    picked_all=np.zeros(len(v),bool);stats=[]
    for camera in cameras:
        eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=-np.cross(right,forward);V=np.array([right,down,forward])
        view=np.einsum('ij,nj->ni',V,p-eye);depth=view[:,2];safe=np.maximum(depth,.02)
        f=375/np.tan(np.radians(camera['fov'])/2)
        screen=np.c_[500+f*view[:,0]/safe,375+f*view[:,1]/safe]
        radius=4*f*scales.max(1)/safe+2
        for patch in patches or PATCHES:
            uv_axes=patch['uv_axes'];axis=patch['axis'];lo,hi=np.asarray(patch['uv_bounds']);nu,nv=patch['sample_grid'];coef=np.asarray(patch['plane_depth_from_uv1'])
            uv=np.array(np.meshgrid(np.linspace(lo[0],hi[0],nu),np.linspace(lo[1],hi[1],nv))).reshape(2,-1).T
            points=np.empty((len(uv),3));points[:,uv_axes]=uv;points[:,axis]=np.einsum('ij,j->i',uv,coef[:2])+coef[2]
            q=np.einsum('ij,nj->ni',V,points-eye);pixels=np.c_[500+f*q[:,0]/q[:,2],375+f*q[:,1]/q[:,2]]
            visible=(q[:,2]>.02)&(pixels[:,0]>=0)&(pixels[:,0]<1000)&(pixels[:,1]>=0)&(pixels[:,1]<750);pixels=pixels[visible]
            if not len(pixels):continue
            low,high=pixels.min(0),pixels.max(0)
            overlap=(depth>.02)&(screen[:,0]+radius>low[0])&(screen[:,0]-radius<high[0])&(screen[:,1]+radius>low[1])&(screen[:,1]-radius<high[1])
            ids=np.flatnonzero(overlap);ids=ids[np.argsort(depth[ids])]
            Q=Rotation.from_quat(columns(v[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',V@WORLD_FROM_RAW,Q)
            J=np.zeros((len(ids),2,3));J[:,0,0]=f/depth[ids];J[:,1,1]=f/depth[ids]
            J[:,0,2]=-f*np.clip(view[ids,0]/depth[ids],-1.3*500/f,1.3*500/f)/depth[ids]
            J[:,1,2]=-f*np.clip(view[ids,1]/depth[ids],-1.3*375/f,1.3*375/f)/depth[ids]
            B=np.einsum('nij,njk->nik',J,Q)*scales[ids,None,:];C=np.einsum('nik,njk->nij',B,B);C[:,0,0]+=.3;C[:,1,1]+=.3;inverse=np.linalg.inv(C)
            det=C[:,0,0]*C[:,1,1]-C[:,0,1]**2;mid=(C[:,0,0]+C[:,1,1])/2
            radius_exact=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)))
            fade=np.clip((2048/1080*750-radius_exact)/(1024/1080*750),0,1)
            residual=p[ids,axis]-np.einsum('ij,j->i',p[ids][:,uv_axes],coef[:2])-coef[2]
            dlo,dhi=patch['deep_bounds']
            eligible=(patch['behind_sign']*residual>.025)&(p[ids,axis]>dlo)&(p[ids,axis]<dhi)&(np.abs(p[ids])<7).all(1)
            if primary_rows is not None:eligible&=ids<primary_rows
            if 'center_bounds' in patch:
                blo,bhi=np.asarray(patch['center_bounds']);eligible&=((p[ids]>=blo)&(p[ids]<=bhi)).all(1)
            selected=np.zeros(len(ids),bool);coverage=[]
            for pixel in pixels:
                d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d)
                a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)))
                T=np.r_[1,np.cumprod(1-a[:-1])];visible_weight=a*T
                selected|=(visible_weight>.003)&eligible;coverage.append(float(visible_weight.sum()))
            picked_all[ids[selected]]=True
            record={'camera':camera['id'],'surface':patch['id'],'rays':len(pixels),'contributors':int(selected.sum()),'scene_alpha_median':float(np.median(coverage))};stats.append(record);print(record,flush=True)
    weights=base.copy();weights[picked_all]=1
    report={'method':'Full perspective Gaussian opacity contribution >.003 at visible seat interior rays in three views; keep actual deeper support',
        'contributor_count':int(picked_all.sum()),'additional_or_strengthened_count':int((picked_all&(base<1)).sum()),
        'world_bounds':[p[picked_all].min(0).tolist(),p[picked_all].max(0).tolist()] if picked_all.any() else None,'views':stats}
    return weights,report,np.flatnonzero(picked_all)


def attenuate(v,m):
    out=v.copy();a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));a=np.clip(a*m,1e-8,1-1e-8);out['opacity']=np.log(a/(1-a));return out


def refine_crown(base,out):
    """Suppress only attributed original background exposed through the crown."""
    if base.resolve()==out.resolve():raise ValueError('Keep the reviewed base trial immutable')
    prior=json.loads((base/'report.json').read_text());phone,_,_=read_ply(base/'iphone.ply');phone=phone.copy();ref,_,_=read_ply(base/'reference-seat.ply')
    data=np.load(base/'iphone-opacity-selection.npz');pw=np.zeros(len(phone),np.float32);pw[data['indices']]=1-data['multipliers']
    extra=np.empty(len(ref),dtype=phone.dtype)
    for name in phone.dtype.names:extra[name]=ref[name]
    traces=[];new=[]
    for iteration in range(3):
        scene=np.concatenate([phone,extra]);p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(scene,['x','y','z']).astype(float))
        weights,trace,selected=backing(scene,p,np.r_[pw,np.zeros(len(extra),np.float32)],CROWN,len(phone));weights=weights[:len(phone)]
        ids=np.flatnonzero(weights>pw+1e-6);trace.update({'iteration':iteration,'new_original_rows':len(ids)});traces.append(trace)
        print('Crown attributed pass',iteration,'new original rows',len(ids),flush=True)
        if not len(ids):break
        phone[ids]=attenuate(phone[ids],np.zeros(len(ids)));pw=weights;new.extend(ids.tolist())
    out.mkdir(parents=True,exist_ok=True);ids=np.flatnonzero(pw>0)
    np.savez_compressed(out/'iphone-opacity-selection.npz',indices=ids,multipliers=1-pw[ids]);np.save(out/'crown-original-indices.npy',np.asarray(new,dtype=np.int64))
    write_ply(out/'iphone.ply',phone);shutil.copyfile(base/'reference-seat.ply',out/'reference-seat.ply');shutil.copyfile(base/'reference-selection.npz',out/'reference-selection.npz')
    report={**prior,'status':'Unreviewed narrowly attributed crown correction','base_candidate':str(base),'iphone_changed_rows':len(ids),'crown_trace_surfaces':CROWN,'crown_new_original_rows':len(new),'crown_passes':traces,
        'crown_attribution':'New pale crown crease is caused by original white support rows such as 392443/2793224/2885811 at x≈1.73 behind the translucent captured seat; confirmed by exact hybrid pixel trace and independent visual review.',
        'reference_patch_sha256':sha256_file(out/'reference-seat.ply'),'reference_patch_unchanged_from_base':sha256_file(out/'reference-seat.ply')==sha256_file(base/'reference-seat.ply')}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--iphone',type=Path,default=ROOT/'raw/ambulance_exp11_boot_sharp.ply');ap.add_argument('--reference',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply');ap.add_argument('--out',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/trials/seat');ap.add_argument('--write-trial-phone',action='store_true');ap.add_argument('--crown-from',type=Path);args=ap.parse_args()
    if args.crown_from:refine_crown(args.crown_from,args.out);return
    if any((args.out/n).resolve() in [args.iphone.resolve(),args.reference.resolve()] for n in ['iphone.ply','reference-seat.ply']):raise ValueError('Source overwrite forbidden')
    phone,_,_=read_ply(args.iphone);ref,_,_=read_ply(args.reference)
    if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):raise ValueError('Reference SH3 required')
    args.out.mkdir(parents=True,exist_ok=True);results=[]
    for label,v in [('iphone',phone),('reference',ref)]:
        p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float));w,regions=base_weights(p);w,support,deep=backing(v,p,w);results.append((w,regions,support,deep));print(label,'selected',int((w>0).sum()),flush=True)
    (pw,pr,pbr,pdeep),(rw,rr,rbr,rdeep)=results;pi=np.flatnonzero(pw>0);ri=np.flatnonzero(rw>0)
    np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rw[ri]);np.savez_compressed(args.out/'deep-backing-indices.npz',iphone_indices=pdeep,reference_indices=rdeep)
    patch=attenuate(ref[ri],rw[ri]);write_ply(args.out/'reference-seat.ply',patch)
    if args.write_trial_phone:
        out=phone.copy();out[pi]=attenuate(phone[pi],1-pw[pi]);write_ply(args.out/'iphone.ply',out)
        assert all(np.array_equal(out[n],phone[n]) for n in phone.dtype.names if n!='opacity')
    assert all(np.array_equal(patch[n],ref[n][ri]) for n in ref.dtype.names if n!='opacity')
    report={'status':'Unreviewed single front-seat contributor transfer trial','iphone':str(args.iphone),'iphone_sha256':sha256_file(args.iphone),'reference':str(args.reference),'reference_sha256':sha256_file(args.reference),'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'iphone_regions':pr,'reference_regions':rr,'trace_surfaces':PATCHES,'backing_support':{'iphone':pbr,'reference':rbr},'iphone_only_changed_field':'opacity','reference_unchanged_fields':'all except feathered opacity','exclusions':['Phone and glass cabinet contents outside fixed seat','Screen and medical gear outside seat','Movable stretcher/bedding outside seat'],'review_requirements':['Logo and harness alignment, no duplicated straps','Frontal and grazing improvement over original','No changes to neighboring cabinet, floor, roof or mattress appearance'],'review_cameras':['front-seat','front-wall','left-grazing','mattress-top']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
