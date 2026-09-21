#!/usr/bin/env python3
"""Compare global scales on fixed same-facing lateral surface support.

Sources/configs are read only. Cached proxies and the numerical report go to
raw/fusion-work. The score includes capped penalties for missing matches;
reducing the inlier set cannot by itself improve this score.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import pipeline as p

SELECTED = ['torso', 'head', 'left_forearm', 'right_forearm', 'left_shin', 'right_shin']
SCALES = [.88, .90, .906196, .92, .94, .96]
REFERENCE = .906196


def cache_proxies(config, lf, lb, parts, output):
    inputs = {key: config[key] for key in ('front', 'back', 'filter', 'partRadii')}
    inputs['exclusions']=config.get('exclusions',{})
    inputs['segmentationOverrides']=config.get('segmentationOverrides',{})
    inputs.update({'frontLandmarks': lf, 'backLandmarks': lb,
                   'voxel': config['alignment']['voxel'], 'reference': REFERENCE,
                   'pipelineHash': hashlib.sha256(Path(p.__file__).read_bytes()).hexdigest()})
    for source in ('front','back'):
        stat=(p.ROOT/config[source]).stat()
        inputs[source+'Stat']=[stat.st_size,stat.st_mtime_ns]
    key=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()[:16]
    path=output/f'scale-proxies-{key}.npz'
    if path.exists():
        print('Using cached proxies',path.name,flush=True)
        with np.load(path) as saved:
            return {name: saved[name].copy() for name in saved.files}
    values={}
    for source,lm in [('front',lf),('back',lb)]:
        print('Preparing fixed proxies:',source,flush=True)
        data,_=p.read_scan(p.ROOT/config[source])
        data,labels,stats,_=p.clean_scan(data,lm,config['filter'],parts,config.get('exclusions',{}).get(source,[]),config.get('segmentationOverrides',{}).get(source,[]))
        print(' ',stats,flush=True)
        for i,part in enumerate(parts):
            if part[0] not in SELECTED:continue
            result=p.normal_proxy(data[labels==i],lm,part,config['alignment']['voxel']/(REFERENCE if source=='back' else 1))
            if result is None:raise ValueError(f'Insufficient proxy points: {source} {part[0]}')
            values[f'{source}_{part[0]}_xyz'],values[f'{source}_{part[0]}_normal']=result
        del data,labels
    np.savez_compressed(path,**values)
    return values


def nearest_signed(source,sn,target,tn,cap):
    distances,indices=cKDTree(target).query(source,workers=-1)
    compatible=np.sum(sn*tn[indices],axis=1)>.80
    return np.where(compatible,np.minimum(distances,cap),cap), compatible&(distances<cap)


def rotate_normals(normals,rotation):
    # Explicit contraction avoids spurious Accelerate/BLAS floating-point
    # status flags observed with finite unit-normal matrix products on macOS.
    result=np.einsum('ni,ji->nj',normals,rotation)
    if not np.isfinite(result).all():raise ValueError('Nonfinite transformed normal')
    return result


def score(f,fn,b,bn,cap):
    fd,fvalid=nearest_signed(f,fn,b,bn,cap)
    bd,bvalid=nearest_signed(b,bn,f,fn,cap)
    si,ti,d=p.match_overlap(b,bn,f,fn,cap)
    capped=np.r_[fd,bd]
    return {'frontSupport':len(f),'backSupport':len(b),'mutualCount':len(d),
            'mutualCoverage':float(2*len(d)/(len(f)+len(b))),
            'directionalCoverage':float((fvalid.sum()+bvalid.sum())/(len(f)+len(b))),
            'mutualMean':float(np.mean(d)) if len(d) else None,
            'mutualMedian':float(np.median(d)) if len(d) else None,
            'cappedMean':float(capped.mean()),'cappedMedian':float(np.median(capped))}


def refine_fixed(f,fn,b,bn,pivot,length,settings):
    """Pipeline's bounded point-plane prior, using the same frozen support for all scales."""
    dR=np.eye(3);dt=np.zeros(3);lever=max(.07,length/2)
    initial=score(f,fn,b,bn,settings['maxDistance'])
    for _ in range(settings['iterations']):
        moved=p.apply_transform(b-pivot,dR,dt)+pivot;mn=rotate_normals(bn,dR)
        si,ti,dist=p.match_overlap(moved,mn,f,fn,settings['maxDistance'])
        if len(si)<settings['minPairs']:break
        trim=dist<=np.quantile(dist,.8);si,ti=si[trim],ti[trim]
        points=moved[si];normal=fn[ti]
        residual=np.sum((points-f[ti])*normal,axis=1)
        robust=1/np.sqrt(1+(residual/.003)**2)
        jacobian=np.c_[np.cross(points-pivot,normal)/lever,normal]
        A=jacobian*robust[:,None];rhs=-residual*robust
        regularization=np.sqrt(len(si))*.2
        prior=np.r_[Rotation.from_matrix(dR).as_rotvec()*lever,dt]
        step=np.linalg.lstsq(np.vstack([A,np.eye(6)*regularization]),np.r_[rhs,-prior*regularization],rcond=None)[0]
        inc=Rotation.from_rotvec(step[:3]/lever).as_matrix()
        nextR=inc@dR;nextT=inc@dt+step[3:]
        if np.linalg.norm(Rotation.from_matrix(nextR).as_rotvec())>np.deg2rad(settings['maxRotationDegrees']) or np.linalg.norm(nextT)>settings['maxTranslation']:break
        dR,dt=nextR,nextT
        if np.linalg.norm(step)<1e-6:break
    moved=p.apply_transform(b-pivot,dR,dt)+pivot;mn=rotate_normals(bn,dR)
    final=score(f,fn,moved,mn,settings['maxDistance'])
    accepted=(final['mutualCount']>=settings['minPairs'] and final['cappedMean']<initial['cappedMean'])
    return (final if accepted else initial), {'accepted':bool(accepted),
        'rotationDegrees':float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))),
        'translation':float(np.linalg.norm(dt))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=p.ROOT/'tools/fusion/fusion-config.json')
    args=parser.parse_args()
    config=json.loads(args.config.read_text());lf=json.loads((p.ROOT/config['frontLandmarks']).read_text())['landmarks'];lb=json.loads((p.ROOT/config['backLandmarks']).read_text())['landmarks']
    parts=[tuple(list(part[:4])+config.get('partRadii',{}).get(part[0],list(part[4:]))) for part in p.BODY_PARTS]
    output=p.ROOT/'raw/fusion-work';output.mkdir(parents=True,exist_ok=True)
    proxy=cache_proxies(config,lf,lb,parts,output)
    settings=config['alignment'];cap=settings['maxDistance'];support={}
    _,reference,_=p.initialize_transforms(lf,lb,parts,{**settings,'scale':REFERENCE})
    for i,part in enumerate(parts):
        name=part[0]
        if name not in SELECTED:continue
        f,fn=proxy[f'front_{name}_xyz'],proxy[f'front_{name}_normal']
        b,bn=proxy[f'back_{name}_xyz'],proxy[f'back_{name}_normal']
        R,t=reference[i];placed=p.apply_transform(b,R,t,REFERENCE);turned=rotate_normals(bn,R)
        depth=p.part_frame(lf,part)[3]
        fside=abs(np.einsum('ni,i->n',fn,depth))<settings['maxAnteriorNormal'];bside=abs(np.einsum('ni,i->n',turned,depth))<settings['maxAnteriorNormal']
        f,fn=f[fside],fn[fside];b,bn,placed,turned=b[bside],bn[bside],placed[bside],turned[bside]
        # Fixed once, at an explicitly recorded reference. A generous gate
        # avoids selecting only the reference's already-perfect matches.
        _,fm=nearest_signed(f,fn,placed,turned,.040)
        _,bm=nearest_signed(placed,turned,f,fn,.040)
        if not fm.any() or not bm.any():raise ValueError(f'No common lateral support for {name}')
        fm=cKDTree(f[fm]).query(f)[0]<settings['voxel']*2
        bm=cKDTree(b[bm]).query(b)[0]<(settings['voxel']/REFERENCE)*2
        support[name]=(f[fm],fn[fm],b[bm],bn[bm])
        print('Fixed support',name,int(fm.sum()),int(bm.sum()),flush=True)
    report={'scales':SCALES,'referenceScale':REFERENCE,'settings':settings,
            'method':'Fixed proxy sampling; same-facing signed normals >.80; lateral normals below configured threshold. Freeze support once at reference scale with .040-source-unit generous normal-compatible neighborhood expanded 2 voxels. At every scale score all frozen samples symmetrically with .020-source-unit capped missing/incompatible-match penalty. Mutual match count and coverage reported separately. Same bounded rigid point-plane refinement and prior per scale; accept only capped-score improvement. No opposing-surface correspondence or anisotropic scaling. Distances are scan units, not independently calibrated metric units.',
            'results':[]}
    for scale in SCALES:
        _,transforms,_=p.initialize_transforms(lf,lb,parts,{**settings,'scale':scale})
        entry={'scale':scale,'parts':{}}
        for i,part in enumerate(parts):
            name=part[0]
            if name not in support:continue
            f,fn,b,bn=support[name];R,t=transforms[i]
            placed=p.apply_transform(b,R,t,scale);turned=rotate_normals(bn,R)
            initial=score(f,fn,placed,turned,cap)
            a=np.array(lf[part[2]]);end=np.array(lf[part[3]])
            refined,correction=refine_fixed(f,fn,placed,turned,(a+end)/2,np.linalg.norm(end-a),settings)
            entry['parts'][name]={'initial':initial,'refined':refined,'correction':correction}
        for stage in ['initial','refined']:
            rows=[row[stage] for row in entry['parts'].values()]
            weights=np.array([row['frontSupport']+row['backSupport'] for row in rows])
            entry[stage]={'partBalancedCappedMean':float(np.mean([row['cappedMean'] for row in rows])),
                'pointWeightedCappedMean':float(np.average([row['cappedMean'] for row in rows],weights=weights)),
                'partBalancedDirectionalCoverage':float(np.mean([row['directionalCoverage'] for row in rows])),
                'mutualCount':sum(row['mutualCount'] for row in rows),
                'support':int(weights.sum())}
        report['results'].append(entry)
        print(scale,entry['initial'],entry['refined'],flush=True)
    destination=output/'scale-check.json';destination.write_text(json.dumps(report,indent=2)+'\n')
    print('Wrote',destination,flush=True)


if __name__=='__main__':main()
