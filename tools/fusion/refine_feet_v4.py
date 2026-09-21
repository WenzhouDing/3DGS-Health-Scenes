#!/usr/bin/env python3
"""Reproducible source-only foot cleanup, measured on frozen v3 fusion."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from render_gaussians import ROOT,load_fused,atlas
from pipeline import BODY_PARTS,part_frame,dot

BASE=ROOT/'raw/fusion-work/refinement-v4/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v4/feet'

def load():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'baseline.npz'
    report=json.loads((BASE/'report.json').read_text())
    if cache.exists():
        d=np.load(cache)
        return d['data'],d['labels'],d['sources'],d['vertex'],report
    data,labels,sources,_=load_fused(BASE)
    vertex=np.load(BASE/'source-vertex-indices.npy')
    chosen=[i for i,p in enumerate(report['parts']) if 'shin' in p['id'] or 'foot' in p['id']]
    keep=np.isin(labels,chosen)
    np.savez(cache,data=data[keep],labels=labels[keep],sources=sources[keep],vertex=vertex[keep])
    return data[keep],labels[keep],sources[keep],vertex[keep],report

def masks(data,labels,sources,report,side,mode):
    ids={p['id']:i for i,p in enumerate(report['parts'])}
    foot=(labels==ids[side+'_foot'])
    sigma=np.exp(data[:,7:10]);largest=sigma.max(1)
    opacity=1/(1+np.exp(-data[:,10]))
    rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
    if mode=='size':
        return {f'Max sigma {s}':~(foot&(largest>s)) for s in [.015,.01,.0075,.005]}
    if mode=='confidence':
        return {f'Low confidence alpha<{a}, sigma>{s}':~(foot&(largest>s)&(opacity<a)) for a,s in [(.5,.01),(.75,.01),(.5,.007),(.75,.007)]}
    if mode=='color':
        lf=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
        part=next(p for p in BODY_PARTS if p[0]==side+'_foot')
        a,b,x,y,z=part_frame(lf,part)
        depth=dot(data[:,:3]-(a+b)/2,y)
        return {f'Posterior anomalous color {threshold}':~(foot&(depth>0)&((rgb[:,2]-rgb[:,1])>threshold)) for threshold in [.04,0,-.03,-.06]}
    raise ValueError(mode)

def repair_color(data,labels,report,side,threshold,amount=.8):
    """Appearance-only trial: robust neighboring skin color at posterior outliers."""
    ids={p['id']:i for i,p in enumerate(report['parts'])}
    foot=labels==ids[side+'_foot']
    rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
    lf=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
    part=next(p for p in BODY_PARTS if p[0]==side+'_foot')
    a,b,x,y,z=part_frame(lf,part)
    depth=dot(data[:,:3]-(a+b)/2,y)
    hardware=np.array(report['parts'][ids[side+'_shin']]['featureEvidence']['hardware']['front'])[2:]
    away_hardware=cKDTree(hardware).query(data[:,:3])[0]>.025
    region=foot&(depth>0)&(data[:,2]>1.555)&(data[:,2]<1.665)&away_hardware
    lum=dot(rgb,np.array([.2126,.7152,.0722]))
    healthy=foot&(rgb[:,0]-rgb[:,1]>.025)&(rgb[:,1]-rgb[:,2]>.025)&(lum>.18)&(lum<.58)
    ref=np.where(healthy)[0]
    _,unique=np.unique(np.floor(data[ref,:3]/.008).astype(int),axis=0,return_index=True)
    ref=ref[unique]
    indices=np.where(region)[0]
    distance,nearest=cKDTree(data[ref,:3]).query(data[indices,:3],k=12)
    target=np.median(rgb[ref[nearest]],axis=1)
    delta=np.linalg.norm(rgb[indices]-target,axis=1)
    changed=indices[(delta>threshold)&(distance[:,0]<.025)]
    colors=target[(delta>threshold)&(distance[:,0]<.025)]
    out=data.copy()
    out[changed,11:14]=((rgb[changed]*(1-amount)+colors*amount)-.5)/.28209479177387814
    return out,changed

def final_selection(data,labels,sources,report):
    ids={p['id']:i for i,p in enumerate(report['parts'])}
    hardware=np.array(report['parts'][ids['right_shin']]['featureEvidence']['hardware']['front'])[2:]
    remove=np.isin(labels,[ids['right_foot'],ids['right_shin']])&(sources==1)&(data[:,1]>.03)&(data[:,2]>1.495)&(data[:,2]<1.55)
    rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
    exposed_green=np.isin(labels,[ids['right_foot'],ids['right_shin']])&(sources==1)&(data[:,1]>.005)&(data[:,2]>1.48)&(data[:,2]<1.56)&(rgb[:,1]>rgb[:,0]+.02)
    remove|=exposed_green
    remove&=cKDTree(hardware).query(data[:,:3])[0]>.025
    repaired=data.copy();changed=[]
    for side in ['left','right']:
        candidate,indices=repair_color(data,labels,report,side,.12,.8)
        indices=indices[~remove[indices]]
        repaired[indices,11:14]=candidate[indices,11:14]
        changed.extend(indices.tolist())
    return ~remove,repaired,np.array(changed,dtype=int)

def export(data,labels,sources,vertex,report,render=True):
    keep,repaired,changed=final_selection(data,labels,sources,report)
    folder=ROOT/'tools/fusion/cleanup-masks';folder.mkdir(exist_ok=True)
    metadata=[]
    for source,name in enumerate(['front','back']):
        base={'version':1,'sourceCapture':name,'sourceFile':report['sources'][source]['file'],
              'sourceSha256':report['sources'][source]['sha256'],
              'sourceHashes':{s['file']:s['sha256'] for s in report['sources']},
              'baseline':str(BASE.relative_to(ROOT)),
              'reproduce':'.venv-fusion/bin/python -B tools/fusion/refine_feet_v4.py --mode final'}
        remove=np.where((sources==source)&~keep)[0]
        if len(remove):
            values=np.unique(vertex[remove]).astype(np.uint32)
            path=folder/f'feet-v4-{name}.npy';np.save(path,values)
            doc={**base,'purpose':'Remove residual intrusive posterior right ankle splats beyond the previous V3 cut.',
                 'maskFile':str(path.relative_to(ROOT)),'uniqueSourceIndices':len(values),
                 'maskDtype':'uint32','rejectedFusedCount':len(remove),
                 'applyStage':'after fusion attenuation, before export; never recalculate donor coverage',
                 'rule':{'parts':['right_shin','right_foot'],'source':'back','frontRawYGreaterThan':.03,'frontRawZBetween':[1.495,1.55],'protectedScrewRadius':.025,
                         'additionalExposedGreenSplats':{'frontRawYGreaterThan':.005,'frontRawZBetween':[1.48,1.56],'greenMinusRedGreaterThan':.02}},
                 'proof':str((OUT/'right-final.png').relative_to(ROOT))}
            (folder/f'feet-v4-{name}.json').write_text(json.dumps(doc,indent=2)+'\n');metadata.append(doc)
        selected=changed[sources[changed]==source]
        selected=selected[np.argsort(vertex[selected])]
        if len(selected):
            path=folder/f'feet-v4-color-{name}.npz'
            np.savez(path,indices=vertex[selected].astype(np.uint32),
                     f_dc=repaired[selected,11:14].astype(np.float32),original_f_dc=data[selected,11:14].astype(np.float32))
            assert len(np.unique(vertex[selected]))==len(selected)
            doc={**base,'purpose':'Localized appearance correction of dark/purple/gray sole contamination; this is repaired color, not recovered scan detail.',
                 'overrideFile':str(path.relative_to(ROOT)),'uniqueSourceIndices':len(selected),
                 'changedFields':['f_dc_0','f_dc_1','f_dc_2'],
                 'recipe':{'thresholdRGBEuclidean':.12,'neighborBlend':.8,'neighborCount':12,'referenceVoxel':.008,
                           'referenceHealthyRGB':{'redMinusGreenGreaterThan':.025,'greenMinusBlueGreaterThan':.025,'luminanceBetween':[.18,.58]},
                           'parts':['left_foot','right_foot'],'frontRawZBetween':[1.555,1.665],
                           'posteriorDepthGreaterThan':0,'protectedAnkleScrewRadius':.025,'nearestHealthyReferenceMaxDistance':.025,
                           'target':'channel median of 12 nearest healthy reference samples; blend 80% target +20% original RGB'},
                 'geometry':'All centers, rotations, scales, opacities and labels are unchanged.',
                 'proof':[str((OUT/(side+'-final.png')).relative_to(ROOT)) for side in ['left','right']],
                 'limits':'No new fine geometry or source texture is invented. Existing toe creases and hardware are outside this posterior sole repair.'}
            (folder/f'feet-v4-color-{name}.json').write_text(json.dumps(doc,indent=2)+'\n');metadata.append(doc)
    (OUT/'evidence.json').write_text(json.dumps({'documents':metadata,
        'rejected':['Global size caps uncover streaks and holes; broad splats carry real sole coverage.',
                    'Posterior color-pruning beyond blue>green removes hardware and valid skin.',
                    'Gaussian footprint shrinkage adds local roughness without recovering detail.'],
        'pruned':int((~keep).sum()),'recolored':len(changed)},indent=2)+'\n')
    print(json.dumps({'pruned':int((~keep).sum()),'recolored':len(changed),
                      'documents':[{k:d[k] for k in ['purpose','sourceCapture','uniqueSourceIndices']} for d in metadata]},indent=2),flush=True)
    if not render:return
    for side in ['left','right']:
        subset=np.isin(labels,[i for i,p in enumerate(report['parts']) if p['id'] in [side+'_foot',side+'_shin']])
        captures={'Current V3':data[subset],'V4 cleanup and local color repair':repaired[subset&keep]}
        center=[(.49 if side=='left' else -.49),-.04,1.61]
        atlas(captures,OUT/f'{side}-final.png',center,.32,800,
              title='Same Gaussian geometry and views: localized sole appearance repair and residual ankle artifact pruning',
              views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Right side',[-1,0,0]),('Left side',[1,0,0]),
                     ('Front 45 degrees',[1,-1,0]),('Back 45 degrees',[-1,1,0])])
        atlas(captures,OUT/f'{side}-sole-final.png',center,.32,800,
              title='High-resolution underside and heel verification: original above, appearance repair below',
              views=[('Sole',[0,.7,1]),('Sole 45 degrees',[.5,1,.6]),('Heel',[0,1,-.6])])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--side',choices=['left','right'],default='left')
    p.add_argument('--mode',choices=['size','confidence','color','repair','footprint','final'],default='size')
    a=p.parse_args()
    data,labels,sources,vertex,report=load()
    if a.mode=='final':
        export(data,labels,sources,vertex,report)
        return
    indices=[i for i,p in enumerate(report['parts']) if p['id'] in [a.side+'_shin',a.side+'_foot']]
    subset=np.isin(labels,indices)
    captures={'Current V3':data[subset]}
    if a.mode=='footprint':
        i=next(i for i,p in enumerate(report['parts']) if p['id']==a.side+'_foot')
        region=labels==i
        for factor in [.95,.9,.85,.75]:
            candidate=data.copy();candidate[region,7:10]+=np.log(factor)
            captures[f'Appearance-only sigma factor {factor}']=candidate[subset]
    elif a.mode=='repair':
        for threshold in [.3,.2,.12,.06]:
            candidate,changed=repair_color(data,labels,report,a.side,threshold)
            print('color threshold',threshold,'changed',len(changed),flush=True)
            captures[f'Appearance-only repair threshold {threshold}']=candidate[subset]
    else:
        for name,keep in masks(data,labels,sources,report,a.side,a.mode).items():
            print(name,'removed',int((~keep).sum()),flush=True)
            captures[name]=data[subset&keep]
    atlas(captures,OUT/f'{a.side}-{a.mode}.png',[(.49 if a.side=='left' else -.49),-.04,1.61],.29,480,
          title='Foot detail source pruning / same full Gaussian parameters for all retained points',
          views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Side',[1,0,0]),('Opposite side',[-1,0,0]),('Sole',[0,.7,1]),('Heel',[0,1,-.6])])

if __name__=='__main__':main()
