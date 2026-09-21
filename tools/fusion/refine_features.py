#!/usr/bin/env python3
"""Reproduce same-surface shoulder-pad registration and 3D segmentation patches.

The dark pad boundary is an ellipse in the native Y/Z projection and follows
arm curvature in X. We fit its image outline, recover the observed curved 3D
rim, and rigidly align that physical feature. A planar-circle fit is unsuitable.
All results are suggestions/audit artifacts; this script never alters a source
scan, pipeline configuration, or production fused output.
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mannequin-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from pipeline import ROOT, BODY_PARTS, read_scan, segment_points, rotate, dot, apply_transform

OUT=ROOT/'raw/fusion-work/feature-audit'


def ellipse_error(points,parameters):
    center=parameters[:2]
    axes=np.exp(parameters[2:4])
    c,s=np.cos(parameters[4]),np.sin(parameters[4])
    coordinates=np.einsum('ni,ji->nj',points-center,np.array([[c,s],[-s,c]]))
    return (np.sqrt(np.sum((coordinates/axes)**2,axis=1))-1)*np.sqrt(np.prod(axes))


def fit_ellipse(points,hint):
    parameters=np.r_[hint,np.log([.040,.030]),0.]
    for iteration in range(5):
        selected=abs(ellipse_error(points,parameters))<(.010 if iteration==0 else .004)
        if selected.sum()<30:raise ValueError('Insufficient physical rim support')
        parameters=least_squares(lambda p:ellipse_error(points[selected],p),parameters,
                                loss='soft_l1',f_scale=.0008,max_nfev=300).x
    return parameters,ellipse_error(points,parameters)


def fit_curved_patch(data,region,landmarks,settings):
    points=data[:,:3]
    mask=np.all((points>=region['boxMin'])&(points<=region['boxMax']),axis=1)
    colors=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
    luminance=dot(colors,np.array([.2126,.7152,.0722]))
    candidates=points[mask&(luminance<settings['darkLuminance'])]
    ellipse,error=fit_ellipse(candidates[:,1:3],region['ellipseCenterHint'])
    rim=candidates[abs(error)<.0025]
    y,z=(rim[:,1:3]-ellipse[:2]).T
    design=np.c_[np.ones(len(rim)),y,z,y*y,y*z,z*z]
    surface=np.linalg.lstsq(design,rim[:,0],rcond=None)[0]
    phi=np.arange(360)*np.pi/180
    c,s=np.cos(ellipse[4]),np.sin(ellipse[4])
    yz=np.einsum('ni,ji->nj',np.c_[np.exp(ellipse[2])*np.cos(phi),np.exp(ellipse[3])*np.sin(phi)],
                 np.array([[c,-s],[s,c]]))
    y,z=yz.T
    x=dot(np.c_[np.ones(len(phi)),y,z,y*y,y*z,z*z],surface)
    outline=np.c_[x,yz+ellipse[:2]]
    center=outline.mean(axis=0)
    labels,_=segment_points(rim,landmarks)
    metadata={'center':center.tolist(),'yzAxes':np.exp(ellipse[2:4]).tolist(),
              'yzRotation':float(ellipse[4]),'ringPoints':len(rim),
              'ringSegmentation':{BODY_PARTS[j][0]:int(sum(labels==j)) for j in np.unique(labels)},
              'medianEllipseResidual':float(np.median(abs(error[abs(error)<.0025]))),
              'medianSurfaceResidual':float(np.median(abs(dot(design,surface)-rim[:,0])))}
    return {'rim':rim,'outline':outline,'ellipse':ellipse,'surfaceCoefficients':surface},metadata,points[mask],colors[mask]


def patch_override(pack,part,data,landmarks):
    points=pack['outline'];center=points.mean(axis=0)
    _,_,axes=np.linalg.svd(points-center,full_matrices=False)
    local=rotate(points-center,axes)
    radii=np.max(abs(local[:,:2]),axis=0)*1.12
    halfdepth=max(.015,float(np.max(abs(local[:,2]))+.007))
    def contains(p):
        q=rotate(p-center,axes)
        return (np.sum((q[:,:2]/radii)**2,axis=1)<=1)&(abs(q[:,2])<halfdepth)
    mask=contains(data[:,:3]);labels,_=segment_points(data[mask,:3],landmarks)
    target=[p[0] for p in BODY_PARTS].index(part)
    return {'id':part.replace('_upper_arm','-shoulder-pad'),'part':part,'type':'orientedEllipseSlab',
            'center':center.tolist(),'axes':axes.tolist(),'radii':radii.tolist(),'halfDepth':halfdepth,
            'patchPoints':int(mask.sum()),'reassignedPoints':int(np.sum(labels!=target)),
            'coveredRimFraction':float(contains(pack['rim']).mean())}


def transform_metrics(front,back,transform):
    moved=apply_transform(back,np.array(transform['rotation']),np.array(transform['translation']),transform['scale'])
    distances=cKDTree(front).query(moved)[0]
    return {'centerError':float(np.linalg.norm(moved.mean(0)-front.mean(0))),
            'medianRimResidual':float(np.median(distances)),'p90RimResidual':float(np.quantile(distances,.9)),
            'centerDisplacement':(moved.mean(0)-front.mean(0)).tolist()}


def align_outlines(front,back,initial,scale,free_scale=False):
    r0=np.array(initial['rotation']);pivot=front.mean(0)
    source=rotate(back-back.mean(0),r0);target=cKDTree(front)
    def residual(v):
        moved=apply_transform(source,Rotation.from_rotvec(v[:3]).as_matrix(),pivot+v[3:6],v[6] if free_scale else scale)
        match=target.query(moved)[1]
        return (moved-front[match]).ravel()
    start=np.r_[np.zeros(6),scale] if free_scale else np.zeros(6)
    lower=[-.4]*3+[-.02]*3+([.85] if free_scale else [])
    upper=[.4]*3+[.02]*3+([1.05] if free_scale else [])
    solution=least_squares(residual,start,bounds=(lower,upper),loss='soft_l1',f_scale=.001,max_nfev=500).x
    scale=solution[6] if free_scale else scale
    correction=Rotation.from_rotvec(solution[:3]);r=correction.as_matrix()@r0
    t=pivot+solution[3:6]-scale*r@back.mean(0)
    transform={'scale':float(scale),'rotation':r.tolist(),'translation':t.tolist()}
    return {'sourceToFrontRaw':transform,'metrics':transform_metrics(front,back,transform),
            'correctionFromSeedDegrees':correction.as_euler('xyz',degrees=True).tolist()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec',type=Path,default=ROOT/'tools/fusion/shoulder-features.json')
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--torso',action='store_true',help='fit physical torso side-panel ports using torso-features.json')
    args=parser.parse_args()
    if args.torso:return torso_ports(args.output)
    settings=json.loads(args.spec.read_text());out=args.output
    out.mkdir(parents=True,exist_ok=True)
    evidence={'method':'paired curved shoulder-pad rims; fixed scale rigid registration',
              'sourceHashes':{},'patches':{},'alignment':{},'segmentationOverrides':{'front':[],'back':[]},
              'limitations':['Rim geometry alone has weak rotational observability around a nearly circular pad normal.',
                             'Check full upper arm, elbow and torso attachment after applying any fitted transform.',
                             'Free scale estimates are an independent diagnostic; do not rescale individual parts.']}
    fitted={}
    for capture in ('front','back'):
        data,metadata=read_scan(ROOT/settings['sources'][capture]);evidence['sourceHashes'][capture]=metadata['sha256']
        alpha=1/(1+np.exp(-np.clip(data[:,10],-40,40)))
        good=(alpha>settings['minOpacity'])&(np.exp(data[:,7:10].max(1))<settings['maxScale'])
        data=data[good];landmarks=json.loads((ROOT/f'tools/fusion/{capture}-landmarks.json').read_text())['landmarks']
        for side in ('left','right'):
            center=np.array(landmarks[side+'_shoulder']);crop=data[np.linalg.norm(data[:,:3]-center,axis=1)<.215]
            pack,meta,points,colors=fit_curved_patch(crop,settings['regions'][capture][side],landmarks,settings)
            fitted[(capture,side)]=pack
            np.savez_compressed(out/f'{capture}-{side}-pad-ellipse.npz',**pack)
            evidence['patches'][capture+'_'+side]=meta
            evidence['segmentationOverrides'][capture].append(patch_override(pack,side+'_upper_arm',crop,landmarks))
            fig,axes=plt.subplots(1,3,figsize=(12,4))
            for axis,(x,y) in zip(axes,[(0,1),(0,2),(1,2)]):
                axis.scatter(points[:,x],points[:,y],c=colors,s=2)
                axis.scatter(pack['rim'][:,x],pack['rim'][:,y],c='#fa2390',s=3)
                axis.plot(pack['outline'][:,x],pack['outline'][:,y],c='cyan',lw=.7)
                axis.set_aspect('equal');axis.grid();axis.set_title(capture+' '+side+' '+ 'XYZ'[x]+'/'+ 'XYZ'[y])
            fig.tight_layout();fig.savefig(out/f'{capture}-{side}-pad-fit.png',dpi=160);plt.close(fig)
    for side in ('left','right'):
        front=fitted[('front',side)]['outline'];back=fitted[('back',side)]['outline']
        fixed=align_outlines(front,back,settings['registrationSeeds'][side],settings['scale'])
        variable=align_outlines(front,back,fixed['sourceToFrontRaw'],settings['scale'],True)
        fixed['comparison']={'baseline':transform_metrics(front,back,settings['baselineTransforms'][side]),
                             'seed':transform_metrics(front,back,settings['registrationSeeds'][side])}
        fixed['independentScaleFit']=variable
        evidence['alignment'][side]=fixed
        (out/f'{side}-pad-transform.json').write_text(json.dumps(fixed,indent=2)+'\n')
        print(side,json.dumps(fixed['metrics']),'independent scale',variable['sourceToFrontRaw']['scale'],flush=True)
    (out/'pad-segmentation-overrides.json').write_text(json.dumps(evidence['segmentationOverrides'],indent=2)+'\n')
    (out/'shoulder-feature-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')




def torso_ports(out=OUT):
    """Refit the identifiable bilateral port details; preserve shell separation."""
    settings=json.loads((ROOT/'tools/fusion/torso-features.json').read_text())
    initial=settings['initialTransform'];r0=np.array(initial['rotation']);t0=np.array(initial['translation']);scale=settings['scale']
    observed={};hashes={}
    for capture in ('front','back'):
        data,meta=read_scan(ROOT/settings['sources'][capture]);hashes[capture]=meta['sha256']
        if capture=='back':data[:,:3]=apply_transform(data[:,:3],r0,t0,scale)
        observed[capture]=data
    features=[];source=[];target=[];weights=[]
    for feature in settings['features']:
        entry={'name':feature['name'],'fitWeightsXYZ':feature['fitWeightsXYZ']}
        for capture in ('front','back'):
            data=observed[capture];p=data[:,:3];rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
            luminance=dot(rgb,np.array([.2126,.7152,.0722]));seed=feature[capture+'SeedYZ']
            mask=(p[:,0]*feature['sideSign']>.1)&(abs(p[:,0])<.205)&(np.linalg.norm(p[:,1:3]-seed,axis=1)<feature['radius'])
            mask&=(data[:,10]>-.8)&(np.exp(data[:,7:10].max(1))<.012/(scale if capture=='back' else 1))
            mask&=(luminance<.13) if feature['colorKind']=='dark' else ((luminance>.3)&(rgb.max(1)-rgb.min(1)<.2))
            q=p[mask];q=q[np.unique(np.floor(q/.001),axis=0,return_index=True)[1]]
            if len(q)<12:raise ValueError('Insufficient feature support: '+capture+' '+feature['name'])
            center=np.median(q,axis=0).astype(float)
            native=rotate(((center-t0)/scale)[None,:],r0.T)[0] if capture=='back' else center
            entry[capture]={'nativeCenter':native.tolist(),'initialAlignedCenter':center.tolist(),'support':len(q),
                            'robustSpread':np.quantile(abs(q-center),.8,axis=0).tolist()}
            (target if capture=='front' else source).append(center)
        weights.append(feature['fitWeightsXYZ']);features.append(entry)
    target=np.array(target);source=np.array(source);weights=np.array(weights);pivot=np.array([0,.02,.35])
    def residual(v):
        moved=apply_transform(source-pivot,Rotation.from_rotvec(v[:3]).as_matrix(),pivot+v[3:])
        return ((moved-target)*weights).ravel()
    solution=least_squares(residual,np.zeros(6),loss='soft_l1',f_scale=.002,max_nfev=1000).x
    delta=Rotation.from_rotvec(solution[:3]).as_matrix();shift=pivot+solution[3:]-delta@pivot
    r=delta@r0;t=delta@t0+shift
    error=apply_transform(source,delta,shift)-target
    result={'method':'paired physical torso side-panel port centers with reduced uncertain recess-depth weight',
            'sourceHashes':hashes,'sourcePaths':settings['sources'],'initialTransform':initial,'selectors':settings['features'],'scale':scale,'features':features,
            'sourceToFrontRaw':{'scale':scale,'rotation':r.tolist(),'translation':t.tolist()},
            'baselineYZResiduals':np.linalg.norm(source[:,1:]-target[:,1:],axis=1).tolist(),
            'finalYZResiduals':np.linalg.norm(error[:,1:],axis=1).tolist(),
            'residualXYZ':error.tolist(),'medianYZResidual':float(np.median(np.linalg.norm(error[:,1:],axis=1))),
            'correctionDegrees':Rotation.from_matrix(delta).as_euler('xyz',degrees=True).tolist(),
            'pivot':pivot.tolist(),'translationAtPivot':solution[3:].tolist(),
            'limitations':['The left exterior dark recess is inconsistently reconstructed in depth; its X coordinate receives weight 0.05.',
                           'Feature-seed coordinates identify physical hardware in both actual Gaussian source renders; they are not opposite-shell surface correspondences.'],
            'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_features.py --torso'}
    out.mkdir(parents=True,exist_ok=True);(out/'torso-port-candidate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
