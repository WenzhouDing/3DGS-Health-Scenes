#!/usr/bin/env python3
"""Inspect conservative v3 lower-leg cleanup against an immutable v2 snapshot."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from render_gaussians import ROOT, load_fused, atlas

BASE = ROOT/'raw/fusion-work/cleanup-v3/baseline'
OUT = ROOT/'raw/fusion-work/cleanup-v3/legs'


def load():
    cache=OUT/'baseline-legs.npz'
    if cache.exists():
        d=np.load(cache)
        return d['data'],d['labels'],d['sources'],d['vertex'],json.loads((BASE/'report.json').read_text())
    data,labels,sources,report=load_fused(BASE)
    vertex=np.load(BASE/'source-vertex-indices.npy')
    ids=[i for i,p in enumerate(report['parts']) if 'shin' in p['id'] or 'foot' in p['id']]
    keep=np.isin(labels,ids)
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez(cache,data=data[keep],labels=labels[keep],sources=sources[keep],vertex=vertex[keep])
    return data[keep],labels[keep],sources[keep],vertex[keep],report


def foot_envelope(data,labels,sources,report,side,margin):
    """Back-only outer-envelope cut, learned from reliable front foot samples."""
    part=next(i for i,p in enumerate(report['parts']) if p['id']==side+'_foot')
    front=data[(labels==part)&(sources==0)]
    opacity=1/(1+np.exp(-front[:,10]))
    scale=np.exp(front[:,7:10]).max(1)
    points=front[(opacity>.2)&(scale<.012),:3]
    d=cKDTree(points).query(points,k=7)[0][:,-1]
    points=points[d<.006]
    hull=ConvexHull(points)
    planes=hull.equations
    candidate=(labels==part)&(sources==1)
    distances=np.max(np.einsum('ni,ji->nj',data[candidate,:3],planes[:,:3])+planes[:,3],axis=1)
    keep=np.ones(len(data),bool)
    keep[np.where(candidate)[0][distances>margin]]=False
    return keep,planes


def chosen_cleanup(data,labels,sources,report):
    """Boolean keep mask with individually inspected, front-frame predicates."""
    ids={p['id']:i for i,p in enumerate(report['parts'])}
    rgb=.5+.28209479177387814*data[:,11:14]
    largest=np.exp(data[:,7:10]).max(1)
    keep,planes=foot_envelope(data,labels,sources,report,'left',.006)
    rules={'left_back_foot_outside_front_envelope':~keep}
    rules['left_back_shin_duplicated_heel']=(labels==ids['left_shin'])&(sources==1)&(data[:,2]>1.56)
    rules['right_back_toe_white_wisps']=(labels==ids['right_foot'])&(sources==1)&(data[:,2]>1.685)&(rgb.min(1)>.3)
    rules['right_back_posterior_ankle_spur']=np.isin(labels,[ids['right_shin'],ids['right_foot']])&(sources==1)&(data[:,1]>.04)&(data[:,2]>1.495)&(data[:,2]<1.55)
    rules['left_front_posterior_shin_bright_streaks']=(labels==ids['left_shin'])&(sources==0)&(data[:,1]>.02)&(data[:,2]>1.18)&(data[:,2]<1.45)&(rgb.min(1)>.35)&(largest>.008)
    for side in ['left','right']:
        i=ids[side+'_shin'];indices=np.where(labels==i)[0]
        xyz=data[indices,:3]
        spacing=cKDTree(xyz).query(xyz,k=7)[0][:,-1]
        hardware=np.array(report['parts'][i]['featureEvidence']['hardware']['front'])
        near_joint=cKDTree(hardware).query(xyz)[0]<.025
        remove=np.zeros(len(data),bool)
        remove[indices[(spacing>.01)&~near_joint]]=True
        rules[side+'_shin_sparse_centers_outside_hardware']=remove
    remove=np.logical_or.reduce(list(rules.values()))
    return ~remove,rules,planes


def export_final(data,labels,sources,vertex,report):
    keep,rules,planes=chosen_cleanup(data,labels,sources,report)
    maskdir=ROOT/'tools/fusion/cleanup-masks';maskdir.mkdir(exist_ok=True)
    paths=[]
    for source,name in enumerate(['front','back']):
        values=np.unique(vertex[(sources==source)&~keep]).astype(np.uint32)
        path=maskdir/f'legs-{name}.npy';np.save(path,values)
        paths.append({'source':name,'sourceFile':report['sources'][source]['file'],
                      'sourceSha256':report['sources'][source]['sha256'],
                      'file':str(path.relative_to(ROOT)),'count':len(values)})
        selected=(sources==source)&~keep
        metadata={
            'version':1,'purpose':'Remove verified foot ghosts, toe wisps and sparse shin remnants without changing poses.',
            'sourceCapture':name,'sourceFile':report['sources'][source]['file'],
            'sourceSha256':report['sources'][source]['sha256'],
            'sourceHashes':{s['file']:s['sha256'] for s in report['sources']},
            'baseline':str(BASE.relative_to(ROOT)),
            'maskFile':str(path.relative_to(ROOT)),'maskDtype':'uint32',
            'maskMeaning':f'sorted unique ORIGINAL {name} PLY vertex indices; hash guarded',
            'applyStage':'after coverage confidence weighting and opacity attenuation, before final export; do not trigger donor rescue again',
            'rejectedFusedCount':int(selected.sum()),'uniqueSourceIndices':len(values),
            'rejectedByPart':{p['id']:int((selected&(labels==i)).sum()) for i,p in enumerate(report['parts']) if (selected&(labels==i)).any()},
            'exactRecipe':'tools/fusion/cleanup-masks/legs-recipe.json',
            'ruleCounts':{n:int((v&(sources==source)).sum()) for n,v in rules.items()},
            'review':'Five variant families inspected from six actual Gaussian cameras, then final six-view foot and shin plus three underside/heel cameras.',
            'limitations':'Left posterior sole remains softer/discolored because the back capture is intrinsically distorted. A small right posterior ankle remnant is preserved to avoid hollowing valid material.',
            'proof':['raw/fusion-work/cleanup-v3/legs/'+n+'-final.png' for n in ['left-foot','right-foot','left-shin','right-shin','left-sole','right-sole']],
            'reproduce':'.venv-fusion/bin/python -B tools/fusion/cleanup_legs.py --mode final'}
        (maskdir/f'legs-{name}.json').write_text(json.dumps(metadata,indent=2)+'\n')
    evidence={'version':1,'baseline':str(BASE.relative_to(ROOT)),
        'baselinePlySha256':hashlib.sha256((BASE/'mannequin_fused.ply').read_bytes()).hexdigest(),
        'coordinateFrame':'front source native (raw front), head -Z, posterior +Y',
        'apply':'Exclude source-native vertex IDs after baseline fusion attenuation; do not recompute donor coverage.',
        'masks':paths,'removedTotal':int((~keep).sum()),
        'rules':{name:{'count':int(mask.sum()),'front':int((mask&(sources==0)).sum()),'back':int((mask&(sources==1)).sum())} for name,mask in rules.items()},
        'recipe':{
            'left_back_foot_outside_front_envelope':{'reference':'front left_foot fused opacity>.2, largest sigma<.012, seventh nearest sample including self<.006; scipy ConvexHull','signedPlaneMargin':.006,'planes':planes.tolist()},
            'left_back_shin_duplicated_heel':{'part':'left_shin','source':'back','zGreaterThan':1.56},
            'right_back_toe_white_wisps':{'part':'right_foot','source':'back','zGreaterThan':1.685,'minDCrgbGreaterThan':.3},
            'right_back_posterior_ankle_spur':{'parts':['right_shin','right_foot'],'source':'back','yGreaterThan':.04,'zBetween':[1.495,1.55]},
            'left_front_posterior_shin_bright_streaks':{'part':'left_shin','source':'front','yGreaterThan':.02,'zBetween':[1.18,1.45],'minDCrgbGreaterThan':.35,'largestSigmaGreaterThan':.008},
            'shin_sparse_centers_outside_hardware':{'parts':['left_shin','right_shin'],'sources':['front','back'],'sixthOtherNeighborDistanceGreaterThan':.01,'protectedRadiusAroundFourVerifiedScrewCenters':.025}},
        'rejected':['Whole right-foot front envelope removes legitimate heel and sole.','Global shin max-scale removes broad valid surface splats.','Removing all left back foot does not fix shin-labeled heel ghost and loses useful sole coverage.','Right ankle Y>.02 cut removes too much valid rear ankle.'],
        'rerun':'.venv-fusion/bin/python -B tools/fusion/cleanup_legs.py --mode final'}
    (maskdir/'legs-recipe.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (OUT/'recipe.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'masks':paths,'rules':evidence['rules'],'removedTotal':evidence['removedTotal']},indent=2),flush=True)
    for side in ['left','right']:
        ids=[i for i,p in enumerate(report['parts']) if p['id'] in [side+'_shin',side+'_foot']]
        subset=np.isin(labels,ids)
        variants={'V2 baseline':data[subset],'Cleaned candidate':data[subset&keep]}
        atlas(variants,OUT/f'{side}-foot-final.png',[(.477 if side=='left' else -.49),-.035,1.57],.34,520,
              title='Exact Gaussian render, same alignment and cameras; inspected source-only artifact removal')
        atlas(variants,OUT/f'{side}-shin-final.png',[(.39 if side=='left' else -.39),-.015,1.34],.5,460,
              title='Knee/ankle hardware protected; sparse and stray source splats removed')
        atlas(variants,OUT/f'{side}-sole-final.png',[(.49 if side=='left' else -.49),-.04,1.62],.30,520,
              views=[('Toe underside',[0,.7,1]),('Sole oblique',[.5,1,.6]),('Heel underside',[0,1,-.6])],
              title='Additional sole and heel checks with matching cameras')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mode',default='envelope',choices=['envelope','local','ankle','shin','spur','final'])
    ap.add_argument('--side',default='left',choices=['left','right'])
    a=ap.parse_args()
    data,labels,sources,vertex,report=load()
    if a.mode=='final':
        export_final(data,labels,sources,vertex,report)
        return
    side=a.side
    partids=[i for i,p in enumerate(report['parts']) if p['id'] in [side+'_shin',side+'_foot']]
    subset=np.isin(labels,partids)
    variants={'V2 baseline':data[subset]}
    if a.mode=='envelope':
        for margin in [.012,.006,.003,0.]:
            keep,planes=foot_envelope(data,labels,sources,report,side,margin)
            print(side,margin,'removed',int((~keep).sum()),flush=True)
            variants[f'Front envelope +{margin:.3f}']=data[subset&keep]
    elif a.mode=='local':
        foot=next(i for i,p in enumerate(report['parts']) if p['id']==side+'_foot')
        back=(labels==foot)&(sources==1)
        rgb=.5+.28209479177387814*data[:,11:14]
        if side=='left':
            keep,_=foot_envelope(data,labels,sources,report,side,.006)
            masks={
                'Envelope + remove bright heel': keep&~(back&(data[:,1]>-.02)&(rgb.min(1)>.3)),
                'Envelope + front heel Y>0': keep&~(back&(data[:,1]>0)),
                'Envelope + front heel Y>-.02': keep&~(back&(data[:,1]>-.02)),
                'Front foot only': ~back,
            }
        else:
            envelope,_=foot_envelope(data,labels,sources,report,side,.003)
            masks={
                'Toe envelope Z>1.665': ~(back&(data[:,2]>1.665)&~envelope),
                'Bright toe only': ~(back&(data[:,2]>1.685)&(rgb.min(1)>.3)),
                'Front toe Z>1.68': ~(back&(data[:,2]>1.68)),
                'Front toe Z>1.66': ~(back&(data[:,2]>1.66)),
            }
        for label,keep in masks.items():
            print(label,'removed',int((~keep).sum()),flush=True)
            variants[label]=data[subset&keep]
    elif a.mode=='ankle':
        shin=next(i for i,p in enumerate(report['parts']) if p['id']==side+'_shin')
        foot=next(i for i,p in enumerate(report['parts']) if p['id']==side+'_foot')
        backshin=(labels==shin)&(sources==1)
        if side=='left':
            base,_=foot_envelope(data,labels,sources,report,side,.006)
        else:
            rgb=.5+.28209479177387814*data[:,11:14]
            base=~((labels==foot)&(sources==1)&(data[:,2]>1.685)&(rgb.min(1)>.3))
        for z in [1.56,1.55,1.54,1.53]:
            keep=base&~(backshin&(data[:,2]>z))
            print(side,'back shin z>',z,'removed',int((~keep).sum()),flush=True)
            variants[f'Foot cleanup + back shin Z>{z}']=data[subset&keep]
    elif a.mode=='shin':
        shin=next(i for i,p in enumerate(report['parts']) if p['id']==side+'_shin')
        indices=np.where(labels==shin)[0]
        dd=cKDTree(data[indices,:3]).query(data[indices,:3],k=7)[0][:,-1]
        for maxdist in [.012,.01,.008]:
            keep=np.ones(len(data),bool)
            keep[indices[dd>maxdist]]=False
            print(side,'shin sixthneighbor',maxdist,'removed',int((~keep).sum()),flush=True)
            variants[f'Shin sixth neighbor >{maxdist}']=data[subset&keep]
    else:
        rgb=.5+.28209479177387814*data[:,11:14]
        scale=np.exp(data[:,7:10]).max(1)
        if side=='right':
            for y in [.05,.04,.03,.02]:
                remove=subset&(sources==1)&(data[:,1]>y)&(data[:,2]>1.495)&(data[:,2]<1.55)
                print('right ankle y>',y,'removed',int(remove.sum()),flush=True)
                variants[f'Posterior ankle y>{y}']=data[subset&~remove]
        else:
            shin=next(i for i,p in enumerate(report['parts']) if p['id']==side+'_shin')
            for y in [.02,.0,-.01]:
                remove=(labels==shin)&(sources==0)&(data[:,1]>y)&(data[:,2]>1.18)&(data[:,2]<1.45)&(rgb.min(1)>.35)&(scale>.008)
                print('left shin bright large y>',y,'removed',int(remove.sum()),flush=True)
                variants[f'Bright large posterior y>{y}']=data[subset&~remove]
    center=[(.477 if side=='left' else -.49),-.035,1.57]
    span=.34
    if a.mode=='shin' or (a.mode=='spur' and side=='left'):center=[(.39 if side=='left' else -.39),-.015,1.34];span=.5
    atlas(variants,OUT/f'{side}-foot-{a.mode}-iterations.png',center,span,420,
          title='Back foot clipping against reliable front surface envelope; aligned source poses unchanged')


if __name__=='__main__':main()
