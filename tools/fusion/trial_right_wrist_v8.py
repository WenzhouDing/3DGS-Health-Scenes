"""Recompute all source weights for bounded wrist/finger transition candidates."""
import json
import numpy as np
from pipeline import ROOT, BODY_PARTS, transform_gaussians, coverage_weights, attenuate, load_cleanup_masks
from render_gaussians import atlas, render
from right_wrist_transition import preserve_pad_align_fingers

BASE=ROOT/'raw/fusion-work/refinement-v8/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v8/transition-trials'
OUT.mkdir(parents=True, exist_ok=True)
cfg=json.loads((BASE/'fusion-config.json').read_text())
report=json.loads((BASE/'report.json').read_text())
lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
parent=report['parts'][11]['sourceToFrontRaw']; prior=report['parts'][12]['sourceToFrontRaw']
parameters={'referenceForearm':parent,'referenceHand':prior,'origin':[-.697,.03,.597],
            'longitudinalAxis':[-.626,0,.7798],'longitudinalFade':[.065,.11]}
native={s:dict(np.load(ROOT/f'raw/fusion-work/refinement-v8/pad-inspection/{s}-full-native.npz')) for s in ['front','back']}
masks,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],report['sources'])),ROOT)
base=np.load(ROOT/'raw/fusion-work/refinement-v8/root-audit/right-wrist-context.npz')
captures={'Baseline':base['data']}
for end,thumb in [(.125,False),(.125,True)]:
    spec={**parameters,'longitudinalFade':[.065,end]}
    if thumb:spec.update({'lateralAxis':[.7798,0,.626], 'thumbLongitudinalFade':[.05,.085], 'thumbLateralFade':[-.044,-.058]})
    f=native['front']['data'][native['front']['labels']==12]
    b=native['back']['data'][native['back']['labels']==12]
    corrected,audit=preserve_pad_align_fingers(b,spec)
    corrected=transform_gaussians(corrected,np.array(parent['rotation']),np.array(parent['translation']),parent['scale'])
    fw,bw=coverage_weights(f,corrected,lm,BODY_PARTS[12],cfg['fusion'])
    rows=[];indices=[];sources=[]
    for si,(source,data,weights) in enumerate([('front',f,fw),('back',corrected,bw)]):
        ids=native[source]['indices'][native[source]['labels']==12]
        dd,kept=attenuate(data,weights,cfg['fusion']['minWeight'])
        keep=~np.isin(ids[kept],masks[source]);rows.append(dd[keep]);indices.append(ids[kept][keep]);sources.append(np.full(keep.sum(),si,np.uint8))
    context=base['labels']==11
    d=np.concatenate([base['data'][context],*rows]);l=np.r_[base['labels'][context],np.full(sum(map(len,rows)),12,np.uint8)]
    s=np.concatenate([base['sources'][context],*sources]);ix=np.concatenate([base['indices'][context],*indices])
    stem=f'transition-{end:g}'+('-thumb' if thumb else '');np.savez_compressed(OUT/(stem+'.npz'),data=d,labels=l,sources=s,indices=ix)
    (OUT/(stem+'.json')).write_text(json.dumps({'parameters':spec,'audit':audit},indent=2)+'\n')
    captures[f'Knuckle fade .065 to {end:g}'+(' + thumb' if thumb else '')]=d
    print(end,audit,flush=True)
    render(d,[0,0,1],[-.761,.045,.654],.24,width=1150,height=750).save(OUT/(stem+'-below.png'))
atlas(captures,OUT/'thumb-transition-comparison.png',[-.72,.045,.605],.45,800,
      views=[('Pad',[0,1,0]),('Palm',[0,-1,0]),('Outer',[-1,.2,0]),('Below',[0,0,1]),
             ('Distal',[-.7,0,.7]),('Pad oblique',[-.5,1,.15])],
      title='Rigid pad continuity + preserved distal fingers; fresh coverage and cleanup')
