"""Stage ablation for the rigid right-hand gap; scratch output only."""
import json
import numpy as np
from pipeline import ROOT, BODY_PARTS, transform_gaussians, coverage_weights, attenuate, load_cleanup_masks
from render_gaussians import load_fused, atlas, render

BASE = ROOT/'raw/fusion-work/refinement-v10/baseline'
OUT = ROOT/'raw/fusion-work/refinement-v10/coverage-audit'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((BASE/'fusion-config.json').read_text());report=json.loads((BASE/'report.json').read_text())
    lm=json.loads((BASE/'front-landmarks.json').read_text())['landmarks']
    for rule in cfg.get('jointBoundaries',{}).get('front',[]):lm[rule['landmark']]=rule['pivot']
    native={s:dict(np.load(ROOT/f'raw/fusion-work/refinement-v8/pad-inspection/{s}-full-native.npz')) for s in ['front','back']}
    masks,_=load_cleanup_masks(cfg['cleanupMasks'],dict(zip(['front','back'],report['sources'])),ROOT)
    aligned={};ids={};caps={};statistics={}
    tr=report['parts'][11]['sourceToFrontRaw'];u=np.array([-.626,0,.7798]);u/=np.linalg.norm(u)
    for s in native:
        hand=native[s]['labels']==12;d=native[s]['data'][hand];ids[s]=native[s]['indices'][hand]
        aligned[s]=d if s=='front' else transform_gaussians(d,np.array(tr['rotation']),np.array(tr['translation']),tr['scale'])
        visibility=np.load(ROOT/f'raw/fusion-work/refinement-v6/right-hand/{s}-full-visibility.npz')
        assert np.array_equal(ids[s],visibility['indices'])
        station=np.einsum('ni,i->n',aligned[s][:,:3]-[-.697,.03,.597],u)
        assert np.isfinite(station).all()
        caps[s]=(station>.09)&(visibility['seen']>visibility['opposite_seen'])&(visibility['observed_ratio']>.03)&(visibility['seen']>.05)
        statistics[s]={'qualityClean':len(d),'observedCaps':int(caps[s].sum()),'cleanupRemoved':int(np.isin(ids[s],masks[s]).sum())}
    weights=dict(zip(['front','back'],coverage_weights(aligned['front'],aligned['back'],lm,BODY_PARTS[12],cfg['fusion'])))
    d,l,s,_=load_fused(BASE);ix=np.load(BASE/'source-vertex-indices.npy');context=(l==11)&(d[:,0]<-.54)
    captures={'V9 current':d[np.isin(l,[11,12])]}
    for name,mode in [('full-confidence-cleaned','all-alpha'),('observed-caps-full-alpha','caps'),('no-v6-mask','no-v6'),('full-native-diagnostic','all-native')]:
        rows=[d[context]];labels=[l[context]];sources=[s[context]];indices=[ix[context]]
        counts={}
        for si,source in enumerate(['front','back']):
            w=weights[source].copy()
            if mode in ['all-alpha','all-native']:w[:]=1
            if mode=='caps':w[caps[source]]=1
            dd,retained=attenuate(aligned[source],w,cfg['fusion']['minWeight']);ii=ids[source][retained]
            keep=~np.isin(ii,masks[source])
            if mode=='no-v6':
                # Only expose the effect of the single directional visibility mask.
                old=np.load(ROOT/f'tools/fusion/cleanup-masks/hand-v6-right-{source}.npy')
                keep=~np.isin(ii,np.setdiff1d(masks[source],old))
            if mode=='all-native':keep[:]=True
            if mode=='caps':keep|=caps[source][retained]
            rows.append(dd[keep]);labels.append(np.full(keep.sum(),12,np.uint8));sources.append(np.full(keep.sum(),si,np.uint8));indices.append(ii[keep])
            counts[source]=int(keep.sum())
        data=np.concatenate(rows);np.savez_compressed(OUT/(name+'.npz'),data=data,labels=np.concatenate(labels),sources=np.concatenate(sources),indices=np.concatenate(indices))
        captures[name]=data;statistics[name]=counts
        render(data,[0,0,1],[-.769,.046,.665],.22,width=1200,height=800).save(OUT/(name+'-below.png'))
    atlas(captures,OUT/'coverage-ablation.png',[-.753,.048,.644],.32,size=700,
          views=[('Below',[0,0,1]),('Distal',[-.7,0,.7]),('Outer',[-1,0,0]),('Palm',[0,-1,0]),('Back',[0,1,0])],
          title='Diagnostic source/opacity ablation. Original geometry; one rigid forearm-hand pose.')
    (OUT/'counts.json').write_text(json.dumps(statistics,indent=2)+'\n')
    print(json.dumps(statistics,indent=2),flush=True)


if __name__=='__main__':main()
