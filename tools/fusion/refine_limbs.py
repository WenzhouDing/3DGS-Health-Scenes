#!/usr/bin/env python3
"""Reproducible limb feature alignment experiments; writes separate review files."""
import json, sys
import os
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/fusion-limb-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
sys.path.insert(0,str(Path(__file__).resolve().parent))
import pipeline as p
OUT=p.ROOT/'raw/fusion-work/limb-refinement'
OUT.mkdir(parents=True,exist_ok=True)

def requested_scale(default):
    return float(sys.argv[sys.argv.index('--scale')+1]) if '--scale' in sys.argv else default

HARDWARE={
 'left':{'front':[[.36730,-.00608,1.12993],[.25566,-.02722,1.16861],[.49110,-.01618,1.52720],[.42122,-.02473,1.54908]],
         'back':[[-.38231,.07271,.81691],[-.26339,.09415,.84852],[-.44766,.02528,1.25752],[-.39183,.00082,1.26616]]},
 'right':{'front':[[-.37810,-.02475,1.12056],[-.26734,-.01137,1.16593],[-.51993,-.01722,1.51169],[-.45099,-.00519,1.53905]],
          'back':[[.29750,.09634,.84261],[.17174,.11224,.86904],[.35360,.04248,1.28466],[.27670,.05238,1.29719]]}}
FRONT_ELBOW_UPDATES={'left_elbow':[.3839064154997863,.0011946936093699852,.2685460886187963],
                     'right_elbow':[-.3558778211475503,-.047657546478842475,.255655440325286]}

def hardware_fits():
    config,lf,lb,f,b,fl,bl,report=load();results={}
    for side in ['left','right']:
        fs=np.array(HARDWARE[side]['front']);bs=np.array(HARDWARE[side]['back'])
        scale=requested_scale(config['alignment']['scale'])
        for suffix in ['shin','foot','thigh']:
            name=side+'_'+suffix;i=[x[0] for x in p.BODY_PARTS].index(name);part=p.BODY_PARTS[i]
            tf=report['parts'][i]['sourceToFrontRaw'];oldR=np.array(tf['rotation']);oldt=np.array(tf['translation'])
            ff,fn,fc,_=proxy(f[fl==i],lf,part);bb,bn,bc,_=proxy(b[bl==i],lb,part)
            _,_,_,depth,_=p.part_frame(lf,part)
            fm=np.abs(p.dot(fn,depth))<.90
            ff,fn,fc=ff[fm],fn[fm],fc[fm]
            bm=np.abs(p.dot(p.rotate(bn,oldR),depth))<.90
            bb,bn,bc=bb[bm],bn[bm],bc[bm]
            oldbb=p.apply_transform(bb,oldR,oldt,tf['scale']);oldbn=p.rotate(bn,oldR)
            before=evaluate(ff,fn,fc,oldbb,oldbn,bc)
            if suffix=='shin':
                # Four metallic screw centers, two at each rigid end, determine
                # translation, bend direction and axial twist without skin sliding.
                R,t=p.fit_rigid(bs*scale,fs)
                method='rigid fit of paired lateral/medial knee and ankle screw centers'
                used=np.arange(4)
            else:
                used=np.array([2,3]) if suffix=='foot' else np.array([0,1])
                sa,sb=bs[used];ta,tb=fs[used];pivot=(ta+tb)/2
                axis=(tb-ta)/np.linalg.norm(tb-ta)
                R0=p.align_vectors(oldR@(sb-sa),tb-ta)@oldR
                t0=pivot-scale*R0@((sa+sb)/2)
                base=p.apply_transform(bb,R0,t0,scale);base_n=p.rotate(bn,R0)
                def objective(deg):
                    dR=Rotation.from_rotvec(axis*np.deg2rad(deg)).as_matrix()
                    metric=evaluate(ff,fn,fc,p.rotate(base-pivot,dR)+pivot,p.rotate(base_n,dR),bc)
                    return sum(x['cost'] for x in metric)
                angles=np.linspace(-35,35,71)
                scores=np.array([objective(a) for a in angles]);best=int(scores.argmin())
                lo,hi=angles[max(0,best-2)],angles[min(len(angles)-1,best+2)]
                optimum=minimize_scalar(objective,bounds=(lo,hi),method='bounded',options={'xatol':.025})
                dR=Rotation.from_rotvec(axis*np.deg2rad(optimum.x)).as_matrix();R=dR@R0;t=dR@(t0-pivot)+pivot
                method='screw-pair fixed hinge axis + bounded same-normal color overlap rotation'
                if suffix=='thigh':
                    # Cloth is not a rigid registration feature. Keep initial
                    # rotation about the knee axis until visible knee shell evidence.
                    R,t=R0,t0;method='knee screw-pair axis and center; initial unobservable knee-axis twist retained'
            after=evaluate(ff,fn,fc,p.apply_transform(bb,R,t,scale),p.rotate(bn,R),bc)
            residual_before=np.linalg.norm(p.apply_transform(bs[used],oldR,oldt,tf['scale'])-fs[used],axis=1)
            residual_after=np.linalg.norm(p.apply_transform(bs[used],R,t,scale)-fs[used],axis=1)
            result={'method':method,'before':before,'after':after,
                    'hardware':{'front':fs[used].tolist(),'back':bs[used].tolist(),'features':['knee_outer','knee_inner','ankle_outer','ankle_inner'] if suffix=='shin' else ['ankle_outer','ankle_inner'] if suffix=='foot' else ['knee_outer','knee_inner'],
                      'beforeResiduals':residual_before.tolist(),'afterResiduals':residual_after.tolist()},
                    'sourceToFrontRaw':{'scale':scale,'rotation':R.tolist(),'translation':t.tolist()}}
            oldpath=OUT/(name+'.json')
            if oldpath.exists() and not (OUT/(name+'-unconstrained.json')).exists():oldpath.rename(OUT/(name+'-unconstrained.json'))
            oldpath.write_text(json.dumps(result,indent=2)+'\n');results[name]=result
            print(name,'hardware residual',residual_before,'->',residual_after,'overlap',sum(x['cost'] for x in before),'->',sum(x['cost'] for x in after),flush=True)
    (OUT/'hardware-results.json').write_text(json.dumps(results,indent=2)+'\n')

def proxy(data,landmarks,part):
    data=data[(data[:,10]>-1)&(np.exp(data[:,7:10].max(axis=1))<.018)]
    data=data[p.voxel_indices(data[:,:3],.003)]
    xyz=data[:,:3].astype(float)
    _,idx=cKDTree(xyz).query(xyz,k=min(24,len(xyz)),workers=-1)
    near=xyz[idx];near-=near.mean(axis=1)[:,None,:]
    eig,v=np.linalg.eigh(np.einsum('nki,nkj->nij',near,near))
    normal=v[:,:,0]
    a,b,_,_,z=p.part_frame(landmarks,part)
    t=np.clip(p.dot(xyz-a,z)/np.linalg.norm(b-a),0,1)
    radial=xyz-a-t[:,None]*(b-a)
    normal[np.sum(radial*normal,axis=1)<0]*=-1
    valid=eig[:,0]/np.maximum(eig.sum(axis=1),1e-12)<.09
    rgb=np.clip(.5+.2820947918*data[:,11:14],0,1)
    # Chroma is stable under the two different source exposures; retain low-value
    # contrast at reduced strength to locate ring/panel edges.
    color=np.c_[rgb/(rgb.sum(axis=1)[:,None]+.08),rgb.mean(axis=1)*.3]
    return xyz[valid],normal[valid],color[valid],rgb[valid]

def correspond(b,bn,bc,f,fn,fc,maximum=.04):
    dist,idx=cKDTree(f).query(b,k=16,workers=-1)
    nd=np.sum(bn[:,None,:]*fn[idx],axis=2)
    cd=np.linalg.norm(bc[:,None,:]-fc[idx],axis=2)
    cost=dist+cd*.035+(1-nd)*.008
    cost[(nd<.72)|(dist>maximum)]=1e3
    choice=cost.argmin(axis=1);row=np.arange(len(b))
    fi=idx[row,choice];d=dist[row,choice];valid=cost[row,choice]<1
    return row[valid],fi[valid],d[valid],cd[row,choice][valid]

def evaluate(f,fn,fc,b,bn,bc):
    metrics=[]
    for a,an,ac,q,qn,qc in [(b,bn,bc,f,fn,fc),(f,fn,fc,b,bn,bc)]:
        si,ti,d,cd=correspond(a,an,ac,q,qn,qc,.035)
        metrics.append({'pairs':len(si),'median':float(np.median(d)),'mean':float(np.mean(d)),
          'coverage':float(len(si)/len(a)),'p90':float(np.quantile(d,.9)),
          'cost':float((np.minimum(d,.018).sum()+.018*(len(a)-len(d)))/len(a)),
          'color':float(np.median(cd))})
    return metrics

def fit(f,fn,fc,b,bn,bc,pivot,axis,angle):
    dR=Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix();dt=np.zeros(3)
    for it in range(28):
        moved=p.rotate(b-pivot,dR)+pivot+dt;mn=p.rotate(bn,dR)
        si,ti,d,cd=correspond(moved,mn,bc,f,fn,fc,.05 if it<8 else .035)
        s2,t2,d2,c2=correspond(f,fn,fc,moved,mn,bc,.05 if it<8 else .035)
        si,ti,d,cd=np.r_[si,t2],np.r_[ti,s2],np.r_[d,d2],np.r_[cd,c2]
        if len(si)<60:break
        q=moved[si];target=f[ti];normal=fn[ti]
        residual=q-target
        robust=1/np.sqrt(1+(d/.012)**2+(cd/.15)**2)
        # Symmetric normal-compatible correspondences protect the two shells.
        # Point-to-point term retains distinctive panel/ring positions along surfaces.
        J=np.c_[np.cross(q-pivot,normal)/.15,normal]
        r=np.sum(residual*normal,axis=1)
        JJ=[];rr=[]
        for k in range(3):
            n=np.eye(3)[k]
            JJ.append(np.c_[np.cross(q-pivot,n)/.15,np.tile(n,(len(q),1))]*.40)
            rr.append(residual[:,k]*.40)
        J=np.vstack([J]+JJ);r=np.concatenate([r]+rr);weights=np.tile(robust,4)
        prior=np.r_[Rotation.from_matrix(dR).as_rotvec()*.15,dt]
        reg=np.sqrt(len(si))*.035
        step=np.linalg.lstsq(np.vstack([J*weights[:,None],np.eye(6)*reg]),np.r_[-r*weights,-prior*reg],rcond=None)[0]
        omega=step[:3]/.15;shift=step[3:]
        rr=Rotation.from_rotvec(omega).as_matrix();dR=rr@dR;dt=rr@dt+shift
        if np.linalg.norm(step)<2e-6:break
    return dR,dt,evaluate(f,fn,fc,p.rotate(b-pivot,dR)+pivot+dt,p.rotate(bn,dR),bc)

def load():
    config=json.loads((p.ROOT/'tools/fusion/fusion-config.json').read_text())
    lf=json.loads((p.ROOT/config['frontLandmarks']).read_text())['landmarks'];lb=json.loads((p.ROOT/config['backLandmarks']).read_text())['landmarks']
    cache=OUT/'clean.npz'
    if cache.exists():
        c=np.load(cache);f,b,fl,bl=c['f'],c['b'],c['fl'],c['bl']
    else:
        f,_=p.read_scan(p.ROOT/config['front']);b,_=p.read_scan(p.ROOT/config['back'])
        f,fl,_,_=p.clean_scan(f,lf,config['filter'],p.BODY_PARTS,config.get('exclusions',{}).get('front',[]))
        b,bl,_,_=p.clean_scan(b,lb,config['filter'],p.BODY_PARTS,config.get('exclusions',{}).get('back',[]))
        np.savez(cache,f=f,b=b,fl=fl,bl=bl)
    if '--elbow-labels' in sys.argv:
        lf.update(FRONT_ELBOW_UPDATES)
        rules=json.loads((p.ROOT/'tools/fusion/shoulder-feature-evidence.json').read_text())['segmentationOverrides']
        fl,_=p.segment_points(f[:,:3],lf,p.BODY_PARTS,overrides=rules['front'])
        bl,_=p.segment_points(b[:,:3],lb,p.BODY_PARTS,overrides=rules['back'])
    baseline=OUT/'baseline-report.json';portable=p.ROOT/'tools/fusion/limb-feature-evidence.json'
    if baseline.exists():report=json.loads(baseline.read_text())
    elif portable.exists():
        e=json.loads(portable.read_text());report={'sources':[{'file':name,'sha256':value} for name,value in e['sourceHashes'].items()],
          'parts':[{'id':part[0],'sourceToFrontRaw':e['baselineTransforms'][part[0]]} for part in p.BODY_PARTS]}
        baseline.write_text(json.dumps(report,indent=2)+'\n')
    else:
        report=json.loads((p.ROOT/'raw/mannequin-fused/report.json').read_text());baseline.write_text(json.dumps(report,indent=2)+'\n')
    for source in report['sources']:
        if p.sha256_file(p.ROOT/source['file'])!=source['sha256']:
            raise ValueError('Registration evidence source hash mismatch: '+source['file'])
    return config,lf,lb,f,b,fl,bl,report

def evidence():
    _,_,_,_,_,_,_,report=load()
    names=[side+'_'+part for side in ['left','right'] for part in ['thigh','shin','foot','forearm','hand']]
    fits={name:json.loads((OUT/(name+'.json')).read_text()) for name in names if (OUT/(name+'.json')).exists()}
    for side in ['left','right']:
        variant=OUT/(side+'_forearm-elbow.json')
        if variant.exists():fits[side+'_forearm']=json.loads(variant.read_text())
    result={'version':1,'sourceHashes':{source['file']:source['sha256'] for source in report['sources']},
            'globalScale':.928,'frame':'native input PLY XYZ to front native XYZ; wxyz Gaussian rotation, log scales',
            'sourceNativeHardwareCenters':HARDWARE,
            'hardwareCenterMethod':'Within 0.10 source units of joint: opaque high-quality centers with RGB minimum >0.35 and max-minus-min <0.14; 0.0015 voxel deduplication; 0.004 connected components. Compact 0.013–0.023-wide metal screw-head components selected and checked against native anisotropic Gaussian renders from both sides, native and oblique views. Centers are coordinate-wise component medians.',
            'hardwareCorrespondenceOrder':['knee_outer','knee_inner','ankle_outer','ankle_inner'],
            'finalFrontLandmarkUpdates':FRONT_ELBOW_UPDATES,
            'elbowEvidence':'Original front elbow estimates fall on the forearm panel. Exact shoulder-pad transform maps the old native back elbow onto the actual visible front circumferential elbow seam. Revised front elbow boundary verified in five native source views and six fused full-arm context views.',
            'scaleEvidence':{'rightFourScrewSimilarityScale':.931421135801891,'rightSameSideKneeAnkleRatios':[.9271014013593862,.9349033120895441],'leftSameSideKneeAnkleRatios':[.9292047408140192,.9286883792789343],'chosenGlobalScale':.928,'note':'One common scale selected with independent shoulder-pad evidence; left ankle source has duplicated ghosts and is not suitable for local scaling.'},
            'fits':fits,
            'rejectedVariants':[{'method':'unconstrained clothing ICP for thighs','reason':'Changed shorts are not rigid features; excessive axial twist and 0.045-unit depth drift.'},{'method':'hard left ankle screw-pair fit for foot','reason':'Single back capture itself has duplicated and distorted ankle/foot; matching one ghost pair worsens heel/toe rendering.'},{'method':'large-Gaussian clipping over left shin','reason':'It removes legitimate smooth calf surface and opens holes in posterior views.'}],
            'qualityPolicy':{'left_foot':{'depthBias':.04,'backMaxScaleNative':.008/.928,'note':'Retain unique posterior columns; prefer crisp front contribution in overlapping ankle/toe regions.'},'left_shin':{'depthBias':.03}},
            'baselineTransforms':{part['id']:part['sourceToFrontRaw'] for part in report['parts']},
            'reproduceHardwareCommand':'.venv-fusion/bin/python -B tools/fusion/refine_limbs.py --hardware --scale .928',
            'reproduceForearmCommand':'.venv-fusion/bin/python -B tools/fusion/refine_limbs.py left_forearm right_forearm --scale .928 --elbow-labels',
            'reproduceFinalTransforms':'Use fits[part].sourceToFrontRaw directly after checking both source hashes. These are absolute source-native transforms, never incremental corrections.'}
    (p.ROOT/'tools/fusion/limb-feature-evidence.json').write_text(json.dumps(result,indent=2)+'\n')

def main():
    config,lf,lb,f,b,fl,bl,report=load();results={}
    selected=[x for x in sys.argv[1:] if x in [p0[0] for p0 in p.BODY_PARTS]] or [x[0] for x in p.BODY_PARTS if any(t in x[0] for t in ['upper_arm','thigh','shin','foot'])]
    for name in selected:
        i=[x[0] for x in p.BODY_PARTS].index(name);part=p.BODY_PARTS[i]
        print(name,flush=True)
        ff,fn,fc,frgb=proxy(f[fl==i],lf,part);bb,bn,bc,brgb=proxy(b[bl==i],lb,part)
        tf=report['parts'][i]['sourceToFrontRaw'];R=np.array(tf['rotation']);t=np.array(tf['translation']);scale=requested_scale(tf['scale'])
        sourcecenter=(np.array(lb[part[2]])+lb[part[3]])/2
        t=t+(tf['scale']-scale)*(R@sourcecenter)
        bb=p.apply_transform(bb,R,t,scale);bn=p.rotate(bn,R)
        a,z,x,y,axis=p.part_frame(lf,part);pivot=(a+z)/2
        # Evaluate actual side-facing overlap only, not anterior vs posterior shells.
        fm=np.abs(p.dot(fn,y))<.90;bm=np.abs(p.dot(bn,y))<.90
        ff0,fn0,fc0=ff[fm],fn[fm],fc[fm];bb0,bn0,bc0=bb[bm],bn[bm],bc[bm]
        before=evaluate(ff0,fn0,fc0,bb0,bn0,bc0)
        runs=[]
        for angle in [0,-12,12,-25,25]:
            dR,dt,metric=fit(ff0,fn0,fc0,bb0,bn0,bc0,pivot,axis,angle)
            score=sum(m['cost'] for m in metric)
            runs.append((score,dR,dt,metric,angle));print(' ',angle,score,metric,flush=True)
        score,dR,dt,after,angle=min(runs,key=lambda x:x[0])
        outR=dR@R;outt=dR@(t-pivot)+pivot+dt
        result={'before':before,'after':after,'startTwistDegrees':angle,
          'adjustment':{'rotationDegrees':Rotation.from_matrix(dR).as_euler('xyz',degrees=True).tolist(),'translation':dt.tolist()},
          'sourceToFrontRaw':{'scale':scale,'rotation':outR.tolist(),'translation':outt.tolist()}}
        results[name]=result
        np.savez_compressed(OUT/(name+'-proxies.npz'),front=ff,back=bb,after=p.rotate(bb-pivot,dR)+pivot+dt,front_rgb=frgb,back_rgb=brgb)
        (OUT/(name+('-elbow' if '--elbow-labels' in sys.argv else '')+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        fig,axs=plt.subplots(2,3,figsize=(15,10),facecolor='#202630')
        for row,bpoints in enumerate([bb,p.rotate(bb-pivot,dR)+pivot+dt]):
            for col,(u,v,title) in enumerate([(0,2,'XZ'),(1,2,'YZ'),(0,1,'XY')]):
                ax=axs[row,col];ax.set_facecolor('#202630')
                ax.scatter(ff[:,u],ff[:,v],s=.8,c='#30b7fa',alpha=.5)
                ax.scatter(bpoints[:,u],bpoints[:,v],s=.8,c='#fbb754',alpha=.5)
                ax.set_aspect('equal');ax.invert_yaxis() if v==2 else None
                ax.set_title(('Before ' if row==0 else 'After ')+title,color='white');ax.tick_params(colors='white')
        fig.suptitle(name+' : front cyan, back orange',color='white');fig.tight_layout();fig.savefig(OUT/(name+'-comparison.png'),dpi=160);plt.close(fig)
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')

def render_results(names):
    import render_gaussians as rg
    config,lf,lb,f,b,fl,bl,report=load()
    for name in names:
        i=[x[0] for x in p.BODY_PARTS].index(name);part=p.BODY_PARTS[i]
        candidate=json.loads((OUT/(name+'.json')).read_text())
        ff=f[fl==i];bb=b[bl==i]
        captures={}
        for label,tf in [('Before',report['parts'][i]['sourceToFrontRaw']),('After',candidate['sourceToFrontRaw'])]:
            aligned=p.transform_gaussians(bb,np.array(tf['rotation']),np.array(tf['translation']),tf['scale'])
            fw,bw=p.coverage_weights(ff,aligned,lf,part,config['fusion'])
            front,_=p.attenuate(ff,fw,config['fusion']['minWeight']);back,_=p.attenuate(aligned,bw,config['fusion']['minWeight'])
            captures[label]=np.r_[front,back]
        center=(np.array(lf[part[2]])+lf[part[3]])/2
        span=np.linalg.norm(np.array(lf[part[2]])-lf[part[3]])+max(part[4:])*2+.05
        rg.atlas(captures,OUT/(name+'-gaussians.png'),center,span,size=400,title=name+' actual anisotropic Gaussian before/after; identical framing')

def render_contexts():
    import render_gaussians as rg
    config,lf,lb,f,b,fl,bl,report=load()
    for side in ['left','right']:
        selected=['torso',side+'_upper_arm',side+'_forearm']
        if '--whole-arm' in sys.argv:selected=[side+'_upper_arm',side+'_forearm',side+'_hand']
        if '--legs' in sys.argv:selected=[side+'_thigh',side+'_shin',side+'_foot']
        captures={}
        for label in ['Before','After']:
            items=[]
            fll,bll=fl,bl
            if label=='After' and '--pad-labels' in sys.argv:
                rules=json.loads((p.ROOT/'raw/fusion-work/feature-audit/pad-segmentation-overrides.json').read_text())
                fll,_=p.segment_points(f[:,:3],lf,p.BODY_PARTS,overrides=rules['front'])
                bll,_=p.segment_points(b[:,:3],lb,p.BODY_PARTS,overrides=rules['back'])
            for i,part in enumerate(p.BODY_PARTS):
                if part[0] not in selected:continue
                tf=report['parts'][i]['sourceToFrontRaw']
                resultfile=OUT/(part[0]+'.json')
                if '--elbow-labels' in sys.argv and (OUT/(part[0]+'-elbow.json')).exists():resultfile=OUT/(part[0]+'-elbow.json')
                if label=='After' and resultfile.exists() and ('thigh' not in part[0] or '--thighs' in sys.argv):
                    tf=json.loads(resultfile.read_text())['sourceToFrontRaw']
                ff=f[fll==i];bb=b[bll==i]
                aligned=p.transform_gaussians(bb,np.array(tf['rotation']),np.array(tf['translation']),tf['scale'])
                fw,bw=p.coverage_weights(ff,aligned,lf,part,config['fusion'])
                front,_=p.attenuate(ff,fw,config['fusion']['minWeight']);back,_=p.attenuate(aligned,bw,config['fusion']['minWeight'])
                items.extend([front,back])
            captures[label]=np.concatenate(items)
        center=(np.array(lf[side+'_shoulder'])+lf[side+'_elbow'])/2;span=.58;name=side+'_shoulder-context'
        if '--whole-arm' in sys.argv:center=(np.array(lf[side+'_shoulder'])+lf[side+'_hand_tip'])/2;span=1.;name=side+'_whole-arm'
        if '--legs' in sys.argv:center=(np.array(lf[side+'_knee'])+lf[side+'_ankle'])/2;span=.90;name=side+'_leg-context'
        if '--elbow-labels' in sys.argv:name+='-resegmented'
        rg.atlas(captures,OUT/(name+'.png'),center,span,size=550,title=name+' actual Gaussians: before/after with unchanged neighboring segments')

if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='--render':render_results(sys.argv[2:])
    elif len(sys.argv)>1 and sys.argv[1]=='--context':render_contexts()
    elif len(sys.argv)>1 and sys.argv[1]=='--hardware':hardware_fits()
    elif len(sys.argv)>1 and sys.argv[1]=='--evidence':evidence()
    else:main()
