#!/usr/bin/env python3
"""One bounded trial of actual registered exposed bench appearance/support.

Selects the exposed cushion's observed records and measured deeper contributors.
No position, covariance, color or SH coefficient is generated or changed.
Original neighboring equipment is retained; the exposed cushion must pass same-view review without changing mobile bags
or nearby floor before integration.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import ROOT,columns,read_ply,sha256_file,write_ply
from repair_surfaces import WORLD_FROM_RAW

BOXES=[{'id':'exposed-central-bench','bounds':[[-.40,-.36,.43],[.18,-.255,.70]],'feather':.025}]
PATCHES=[{'id':'exposed-central-bench','axis':1,'uv_axes':[0,2],
 'uv_bounds':[[-.37,.45],[.15,.68]],'plane_depth_from_uv1':[-.001115,.008680,-.305487],
 'behind_sign':-1,'deep_bounds':[-4.5,-.255],'sample_grid':[19,9]}]


def smoothstep(x):
    x=np.clip(x,0,1);return x*x*(3-2*x)


def base_weights(p):
    weights=np.zeros(len(p),np.float32);regions=[]
    for box in BOXES:
        lo,hi=np.asarray(box['bounds']);edge=np.minimum(p-lo,hi-p).min(1)
        w=smoothstep(edge/box['feather']);weights=np.maximum(weights,w.astype(np.float32))
        regions.append({**box,'selected_count':int((w>0).sum())})
    return weights,regions


def backing(v,p,base,primary_rows=None):
    scales=np.exp(columns(v,['scale_0','scale_1','scale_2']).astype(float))
    alpha=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
    cameras=[c for c in json.loads(Path(__file__).with_name('pass2-cameras.json').read_text()) if c['id'] in ['bench-grazing','equipment-bag','right-grazing']]
    picked_all=np.zeros(len(v),bool);stats=[]
    for camera in cameras:
        eye=np.asarray(camera['position']);forward=np.asarray(camera['target'])-eye;forward/=np.linalg.norm(forward)
        right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=-np.cross(right,forward);V=np.array([right,down,forward])
        view=np.einsum('ij,nj->ni',V,p-eye);depth=view[:,2];safe=np.maximum(depth,.02)
        f=375/np.tan(np.radians(camera['fov'])/2)
        screen=np.c_[500+f*view[:,0]/safe,375+f*view[:,1]/safe]
        radius=4*f*scales.max(1)/safe+2
        for patch in PATCHES:
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
            if primary_rows is not None: eligible &= ids < primary_rows
            selected=np.zeros(len(ids),bool);coverage=[]
            for pixel in pixels:
                d=screen[ids]-pixel-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d)
                a=np.minimum(.99,alpha[ids]*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)))
                T=np.r_[1,np.cumprod(1-a[:-1])];visible_weight=a*T
                selected|=(visible_weight>.003)&eligible;coverage.append(float(visible_weight.sum()))
            picked_all[ids[selected]]=True
            record={'camera':camera['id'],'surface':patch['id'],'rays':len(pixels),'contributors':int(selected.sum()),'scene_alpha_median':float(np.median(coverage))};stats.append(record);print(record,flush=True)
    weights=base.copy();weights[picked_all]=1
    report={'method':'Full perspective Gaussian opacity contribution >.003 at visible exposed bench rays in three views; keep actual deeper support',
        'contributor_count':int(picked_all.sum()),'additional_or_strengthened_count':int((picked_all&(base<1)).sum()),
        'world_bounds':[p[picked_all].min(0).tolist(),p[picked_all].max(0).tolist()] if picked_all.any() else None,'views':stats}
    return weights,report,np.flatnonzero(picked_all)


def attenuate(v,m):
    out=v.copy();a=1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)));a=np.clip(a*m,1e-8,1-1e-8);out['opacity']=np.log(a/(1-a));return out


def guarded(base,out):
    prior=json.loads((base/'report.json').read_text());phone,_,_=read_ply(Path(prior['iphone']));ref,_,_=read_ply(Path(prior['reference']))
    ps=np.load(base/'iphone-opacity-selection.npz');rs=np.load(base/'reference-selection.npz');pi=ps['indices'];ri=rs['indices'];pm=ps['multipliers'].astype(float);rm=rs['multipliers'].astype(float)
    def guard(v):
        p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float));g=smoothstep((p[:,2]-.65)/.04)
        g*=((p[:,0]>-.42)&(p[:,0]<.20)&(p[:,1]>-.37)&(p[:,1]<-.245)&(p[:,2]<.78))
        return g
    pg=guard(phone[pi]);rg=guard(ref[ri]);pm=pm+(1-pm)*pg;rm=rm*(1-rg)
    pk=pm<1;rk=rm>0;pi=pi[pk];pm=pm[pk];ri=ri[rk];rm=rm[rk]
    np.savez_compressed(out/'iphone-opacity-selection.npz',indices=pi,multipliers=pm.astype(np.float32));np.savez_compressed(out/'reference-selection.npz',indices=ri,multipliers=rm.astype(np.float32))
    candidate=phone.copy();candidate[pi]=attenuate(phone[pi],pm);write_ply(out/'iphone.ply',candidate);write_ply(out/'reference-bench.ply',attenuate(ref[ri],rm))
    report={**prior,'status':'Unreviewed narrow rear-edge preservation variant','base_candidate':str(base),'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'rear_edge_guard':{'world_x':[-.42,.20],'world_y':[-.37,-.245],'world_z':[.65,.78],'z_restore_fade':[.65,.69],'phone_rows_at_least_partly_restored':int((pg>0).sum()),'reference_rows_at_least_partly_faded':int((rg>0).sum()),'reason':'Original physical rim opacity protects legitimate white wall behind the cushion; no wall or edge splats are deleted.'}}
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['rear_edge_guard'],indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--iphone',type=Path,default=ROOT/'raw/ambulance_exp11_boot_sharp.ply');ap.add_argument('--reference',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply');ap.add_argument('--out',type=Path,default=ROOT/'raw/ambulance-cleanup/pass3/trials/bench');ap.add_argument('--write-trial-phone',action='store_true');ap.add_argument('--guard-from',type=Path);args=ap.parse_args()
    if args.guard_from:
        args.out.mkdir(parents=True,exist_ok=True);guarded(args.guard_from,args.out);return
    if any((args.out/n).resolve() in [args.iphone.resolve(),args.reference.resolve()] for n in ['iphone.ply','reference-bench.ply']):raise ValueError('Source overwrite forbidden')
    phone,_,_=read_ply(args.iphone);ref,_,_=read_ply(args.reference)
    if not all(f'f_rest_{i}' in ref.dtype.names for i in range(45)):raise ValueError('Reference SH3 required')
    args.out.mkdir(parents=True,exist_ok=True);results=[]
    for label,v in [('iphone',phone),('reference',ref)]:
        p=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(v,['x','y','z']).astype(float));w,regions=base_weights(p);w,support,deep=backing(v,p,w);results.append((w,regions,support,deep));print(label,'selected',int((w>0).sum()),flush=True)
    (pw,pr,pbr,pdeep),(rw,rr,rbr,rdeep)=results
    pi=np.flatnonzero(pw>0);ri=np.flatnonzero(rw>0)
    patch=attenuate(ref[ri],rw[ri]);current=phone.copy();current[pi]=attenuate(phone[pi],1-pw[pi])
    extra=np.empty(len(patch),dtype=phone.dtype)
    for name in phone.dtype.names:extra[name]=patch[name]
    hybrid_passes=[];new_cloud=[]
    for iteration in range(3):
        scene=np.concatenate([current,extra]);positions=np.einsum('ij,nj->ni',WORLD_FROM_RAW,columns(scene,['x','y','z']).astype(float))
        candidate,trace,_=backing(scene,positions,np.r_[pw,np.zeros(len(extra),np.float32)],len(phone));candidate=candidate[:len(phone)]
        changed=np.flatnonzero(candidate>pw+1e-6);trace['iteration']=iteration;trace['additional_original_rows']=len(changed);hybrid_passes.append(trace)
        print('Hybrid bench trace',iteration,'new original rows',len(changed),flush=True)
        if not len(changed):break
        current[changed]=attenuate(current[changed],np.zeros(len(changed)));pw=candidate;new_cloud.extend(changed.tolist())
    pi=np.flatnonzero(pw>0)
    np.save(args.out/'additional-cloud-indices.npy',np.asarray(new_cloud,dtype=np.int64))
    np.savez_compressed(args.out/'iphone-opacity-selection.npz',indices=pi,multipliers=1-pw[pi]);np.savez_compressed(args.out/'reference-selection.npz',indices=ri,multipliers=rw[ri]);np.savez_compressed(args.out/'deep-backing-indices.npz',iphone_indices=pdeep,reference_indices=rdeep)
    patch=attenuate(ref[ri],rw[ri]);write_ply(args.out/'reference-bench.ply',patch)
    if args.write_trial_phone:
        out=phone.copy();out[pi]=attenuate(phone[pi],1-pw[pi]);write_ply(args.out/'iphone.ply',out)
        assert all(np.array_equal(out[n],phone[n]) for n in phone.dtype.names if n!='opacity')
    assert all(np.array_equal(patch[n],ref[n][ri]) for n in ref.dtype.names if n!='opacity')
    report={'status':'Unreviewed single exposed-bench contributor transfer trial','iphone':str(args.iphone),'iphone_sha256':sha256_file(args.iphone),'reference':str(args.reference),'reference_sha256':sha256_file(args.reference),'iphone_changed_rows':len(pi),'reference_added_rows':len(ri),'iphone_regions':pr,'reference_regions':rr,'trace_surfaces':PATCHES,'hybrid_trace_passes':hybrid_passes,'backing_support':{'iphone':pbr,'reference':rbr},'iphone_only_changed_field':'opacity','reference_unchanged_fields':'all except feathered opacity','exclusions':['All bags and equipment outside exposed central cushion','Bench rounded edge outside x/z bounds','Original stretcher and nearby floor'],'review_requirements':['Original bags and restraints preserved','Clear exposed cushion improvement from grazing view','No changes to neighboring floor, mattress or equipment appearance'],'review_cameras':['bench-grazing','equipment-bag','mattress-grazing']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
