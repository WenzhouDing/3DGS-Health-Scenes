#!/usr/bin/env python3
"""Read-only native left-arm collar measurement and mechanical registration audit.

Writes only a separate v5 experiment directory. Reuses the v4 native ring-plane
measurement method, with explicit left-arm landmarks and parent transform.
"""
import json
import numpy as np
import refine_arm_v4 as collar
from pipeline import ROOT
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pipeline import dot
OUT=ROOT/'raw/fusion-work/refinement-v5/arm-audit'
BASE=ROOT/'raw/fusion-work/refinement-v5/baseline'
def ORIGINAL_CONTEXT():
    return (json.loads((BASE/'front-landmarks.json').read_text())['landmarks'],
            json.loads((BASE/'back-landmarks.json').read_text())['landmarks'],
            json.loads((BASE/'feature-transforms.json').read_text())['parts'])

def left_context():
    lf,lb,fits=ORIGINAL_CONTEXT()
    for key in ['shoulder','elbow','wrist','handtip']:
        if 'left_'+key in lf:lf['right_'+key]=lf['left_'+key]
        if 'left_'+key in lb:lb['right_'+key]=lb['left_'+key]
    fits['right_upper_arm']=fits['left_upper_arm']
    return lf,lb,fits

def angle(a,b):
    return float(np.degrees(np.arccos(np.clip(a@b/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))

def measure_left_collar():
    """Left-specific seam search; its collar is proximal to the old landmark."""
    rng=np.random.default_rng(82);results={};fig,axes=plt.subplots(2,3,figsize=(14,8))
    for row,capture in enumerate(['front','back']):
        c=np.load(OUT/f'{capture}-collar-native.npz');p=c['local'];lum=c['lum'];angle0=c['angle'];good=(p[:,0]>-.035)&(p[:,0]<.020)&(lum<.28);q=p[good];bins=np.floor((angle0[good]+np.pi)/(2*np.pi)*24).astype(int);best=None
        design=np.c_[q[:,1:],np.ones(len(q))]
        for _ in range(8000):
            idx=rng.choice(len(q),3,replace=False)
            try:v=np.linalg.solve(design[idx],q[idx,0])
            except np.linalg.LinAlgError:continue
            if np.linalg.norm(v[:2])>.25 or not -.023<v[2]<.008:continue
            residual=abs(q[:,0]-dot(design,v));inside=residual<.0015;counts=np.bincount(bins[inside],minlength=24);score=np.minimum(counts,8).sum()
            if best is None or score>best[0]:best=(score,v,inside)
        _,v,inside=best
        for _ in range(5):
            v=least_squares(lambda x:q[inside,0]-dot(design[inside],x),v,loss='soft_l1',f_scale=.0006).x
            inside=abs(q[:,0]-dot(design,v))<.0018
        normal=np.r_[1.,-v[:2]];normal/=np.linalg.norm(normal);axis=c['frame'].T@normal;radial=q[inside,1:]
        def radius_res(x):return (np.sqrt(np.sum(((radial-x[:2])/np.exp(x[2:]))**2,axis=1))-1)*.055
        e=least_squares(radius_res,[0,0,np.log(.060),np.log(.057)],bounds=([-.025,-.025,np.log(.04),np.log(.04)],[.025,.025,np.log(.085),np.log(.085)]),loss='soft_l1',f_scale=.002).x
        local_center=np.array([v[2]+v[0]*e[0]+v[1]*e[1],e[0],e[1]]);pivot=c['center']+c['frame'].T@local_center;points=c['data'][good][inside,:3]
        results[capture]={'pivot':pivot.tolist(),'axis':axis.tolist(),'localPlane':v.tolist(),'radialCenter':e[:2].tolist(),'radialRadii':np.exp(e[2:]).tolist(),'inlierCount':int(sum(inside)),'occupiedAngularBins':int(sum(np.bincount(bins[inside],minlength=24)>0)),'planeMedianResidual':float(np.median(abs(q[inside,0]-dot(design[inside],v))))}
        np.savez_compressed(OUT/f'{capture}-collar-rim.npz',points=points,pivot=pivot,axis=axis)
        for col,(xx,yy) in enumerate([(0,1),(0,2),(1,2)]):
            a=axes[row,col];a.scatter(c['data'][:,xx],c['data'][:,yy],c=c['rgb'],s=.5);a.scatter(points[:,xx],points[:,yy],c='magenta',s=3);a.scatter(pivot[xx],pivot[yy],c='lime',s=20);a.set_aspect('equal');a.grid();a.set_title(capture+' '+'XYZ'[xx]+'/'+ 'XYZ'[yy])
        print(capture,results[capture],flush=True)
    fig.tight_layout();fig.savefig(OUT/'left-collar-fit.png',dpi=180);(OUT/'collar-measurements.json').write_text(json.dumps(results,indent=2)+'\n')

def audit():
    measure=json.loads((OUT/'collar-measurements.json').read_text());lf,lb,fits=ORIGINAL_CONTEXT();f=measure['front'];b=measure['back'];fa=np.array(f['axis']);ba=np.array(b['axis']);fp=np.array(f['pivot']);bp=np.array(b['pivot']);fv=np.array(lf['left_wrist'])-fp;bv=np.array(lb['left_wrist'])-bp
    rows={}
    for part in ['left_upper_arm','left_forearm','left_hand']:
        t=fits[part]['sourceToFrontRaw'];r=np.array(t['rotation']);tr=np.array(t['translation']);s=t['scale'];rows[part]={'sourceToFrontRaw':t,'collarNormalResidualDegrees':angle(fa,r@ba),'collarCenterResidualVector':(s*r@bp+tr-fp).tolist(),'collarCenterResidual':float(np.linalg.norm(s*r@bp+tr-fp)),'wristDirectionResidualDegrees':angle(fv,r@bv),'wristLandmarkResidualVector':(s*r@np.array(lb['left_wrist'])+tr-lf['left_wrist']).tolist()}
    pr=np.array(rows['left_upper_arm']['sourceToFrontRaw']['rotation']);cr=np.array(rows['left_forearm']['sourceToFrontRaw']['rotation']);rv=Rotation.from_matrix(cr@pr.T).as_rotvec();rva=rv/np.linalg.norm(rv)
    result={'anatomicalSide':'left','screenIdentification':'black circular socket and long gray cable; anatomical right carries yellow tubes','nativeCollars':measure,'frontCollarAxisToWristDegrees':angle(fa,fv),'backCollarAxisToWristDegrees':angle(ba,bv),'currentTransforms':rows,'relativeRotation':{'degrees':float(np.degrees(np.linalg.norm(rv))),'axisFrontRaw':rva.tolist(),'angleToFrontCollarAxisDegrees':min(angle(rva,fa),angle(-rva,fa))}}
    (OUT/'left-arm-mechanical-audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def parent_candidates():
    """Fit one rigid upper arm to pad outline, collar center and collar normal."""
    from pipeline import apply_transform,rotate
    measure=json.loads((OUT/'collar-measurements.json').read_text());lf,lb,fits=ORIGINAL_CONTEXT();base=fits['left_upper_arm']['sourceToFrontRaw'];r0=np.array(base['rotation']);t0=np.array(base['translation']);scale=base['scale'];cf=np.array(measure['front']['pivot']);cb=np.array(measure['back']['pivot']);af=np.array(measure['front']['axis']);ab=np.array(measure['back']['axis']);ff=np.load(ROOT/'raw/fusion-work/feature-audit/front-left-pad-ellipse.npz')['outline'];bb=np.load(ROOT/'raw/fusion-work/feature-audit/back-left-pad-ellipse.npz')['outline'];pivot=ff.mean(0);tree=cKDTree(ff)
    def unpack(v):
        q=Rotation.from_rotvec(v[:3]).as_matrix();return q@r0,q@(t0-pivot)+pivot+v[3:]
    def stats(r,t):
        points=apply_transform(bb,r,t,scale);d=tree.query(points)[0]
        return {'padRimMedian':float(np.median(d)),'padRimP90':float(np.quantile(d,.9)),'padCenterResidual':float(np.linalg.norm(points.mean(0)-ff.mean(0))),'collarCenterResidual':float(np.linalg.norm(scale*r@cb+t-cf)),'collarNormalResidualDegrees':angle(r@ab,af),'rotationChangeDegrees':float(np.degrees(Rotation.from_matrix(r@r0.T).magnitude()))}
    variants={'baseline':{'sourceToFrontRaw':base,'metrics':stats(r0,t0)}}
    for normal_weight in [0,.2,.5,1,2]:
        def residual(v):
            r,t=unpack(v);points=apply_transform(bb,r,t,scale);idx=tree.query(points)[1]
            # Balance regions by RMS, not point count. The pad fixes translation
            # and its visible contour; a separate distant collar fixes ambiguity.
            pad=(points-ff[idx]).ravel()/(np.sqrt(len(points))*.0015)
            center=(scale*r@cb+t-cf)/.002
            normal=(r@ab-af)/.03*normal_weight
            return np.r_[pad,center,normal]
        sol=least_squares(residual,np.zeros(6),bounds=([-.2]*3+[-.015]*3,[.2]*3+[.015]*3),loss='linear',max_nfev=500,xtol=1e-12,ftol=1e-12,gtol=1e-12);r,t=unpack(sol.x)
        key='normal-weight-'+str(normal_weight);variants[key]={'sourceToFrontRaw':{'scale':scale,'rotation':r.tolist(),'translation':t.tolist()},'metrics':stats(r,t),'normalWeight':normal_weight,'objective':'curved pad outline RMS/.0015 + collar center/.002 + normal difference/.03*normalWeight'}
        print(key,variants[key]['metrics'],flush=True)
    (OUT/'left-parent-candidates.json').write_text(json.dumps({'nativeCollars':measure,'candidates':variants},indent=2)+'\n')

def left_data():
    from pipeline import clean_scan,read_scan,BODY_PARTS
    cache=OUT/'left-clean.npz';cfg=json.loads((BASE/'fusion-config.json').read_text());lf,lb,fits=ORIGINAL_CONTEXT()
    if cache.exists():return dict(np.load(cache)),lf,lb,fits,cfg
    records={}
    for capture,lm in [('front',lf),('back',lb)]:
        data,_=read_scan(ROOT/cfg[capture]);data,labels,_,indices=clean_scan(data,lm,cfg['filter'],BODY_PARTS,cfg.get('exclusions',{}).get(capture,[]),cfg.get('segmentationOverrides',{}).get(capture,[]),cfg['filter'].get('partOverrides',{}).get(capture,{}));m=np.isin(labels,[4,5,6]);records[capture]=data[m];records[capture+'Labels']=labels[m];records[capture+'Indices']=indices[m]
    np.savez_compressed(cache,**records);return records,lf,lb,fits,cfg

def render_candidates():
    from pipeline import BODY_PARTS,transform_gaussians,coverage_weights,attenuate
    from render_gaussians import load_fused,atlas
    from cleanup_masks import keep_source_rows
    records,lf,lb,fits,cfg=left_data();measured=json.loads((OUT/'collar-measurements.json').read_text());parents=json.loads((OUT/'left-parent-candidates.json').read_text())['candidates'];masks={cap:[] for cap in ['front','back']}
    for filename in cfg.get('cleanupMasks',[]):
        meta=json.loads((ROOT/filename).read_text());masks[meta['sourceCapture']].append(np.load(ROOT/meta['maskFile']))
    masks={cap:np.unique(np.concatenate(parts)) for cap,parts in masks.items()}
    for cap in ['front','back']:
        data=records[cap];labels=records[cap+'Labels'];p=np.array(measured[cap]['pivot']);ax=np.array(measured[cap]['axis']);station=dot(data[:,:3]-p,ax);rad=np.linalg.norm(data[:,:3]-p-station[:,None]*ax,axis=1);selected=np.isin(labels,[4,5])&(abs(station)<.11)&(rad<.11);labels[selected]=np.where(station[selected]<=0,4,5)
    lf['left_elbow']=measured['front']['pivot'];lb['left_elbow']=measured['back']['pivot'];oldf=fits['left_forearm']['sourceToFrontRaw'];oldh=fits['left_hand']['sourceToFrontRaw'];rfo=np.array(oldf['rotation']);tfo=np.array(oldf['translation']);rho=np.array(oldh['rotation']);tho=np.array(oldh['translation']);base,lab,src,_=load_fused(BASE);rows={'V4 baseline':base[np.isin(lab,[4,5,6])]}
    for key in ['baseline','normal-weight-0.5','normal-weight-1']:
        parent=parents[key]['sourceToFrontRaw'];rp=np.array(parent['rotation']);tp=np.array(parent['translation']);delta=rp@rfo.T;hand={'scale':parent['scale'],'rotation':(delta@rho).tolist(),'translation':(delta@(tho-tfo)+tp).tolist()};out=[];labs=[];sources=[];ids=[]
        for i in [4,5,6]:
            tf=hand if i==6 else parent;r=np.array(tf['rotation']);t=np.array(tf['translation']);ff=records['front'][records['frontLabels']==i];bb=transform_gaussians(records['back'][records['backLabels']==i],r,t,tf['scale']);fi=records['frontIndices'][records['frontLabels']==i];bi=records['backIndices'][records['backLabels']==i];fw,bw=coverage_weights(ff,bb,lf,BODY_PARTS[i],cfg['fusion']);ff,fk=attenuate(ff,fw,cfg['fusion']['minWeight']);bb,bk=attenuate(bb,bw,cfg['fusion']['minWeight']);fi=fi[fk];bi=bi[bk];fm=keep_source_rows(fi,masks['front']);bm=keep_source_rows(bi,masks['back']);ff=ff[fm];bb=bb[bm];fi=fi[fm];bi=bi[bm];out.extend([ff,bb]);labs.extend([np.full(len(ff),i),np.full(len(bb),i)]);sources.extend([np.zeros(len(ff)),np.ones(len(bb))]);ids.extend([fi,bi])
        data=np.concatenate(out);rows[key]=data;np.savez_compressed(OUT/(key+'-fused.npz'),data=data,labels=np.concatenate(labs),sources=np.concatenate(sources),sourceIndices=np.concatenate(ids));(OUT/(key+'-transforms.json')).write_text(json.dumps({'left_upper_arm':parent,'left_forearm':parent,'left_hand':hand},indent=2)+'\n')
    views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Outer',[1,0,0]),('Inner',[-1,0,0]),('Cable oblique',[.7,1,-.3]),('Opposite oblique',[-.7,-1,-.3])]
    atlas(rows,OUT/'zero-twist-collar-candidates.png',[.37,.02,.27],.38,550,views=views,title='Anatomical LEFT arm: v4 vs one-parent zero-twist alternatives')
    atlas(rows,OUT/'zero-twist-fullarm-candidates.png',[.54,.05,.41],.9,500,views=views,title='Full left arm, including carried hand and existing source cleanup')

def accepted_evidence():
    from pipeline import sha256_file
    candidates=json.loads((OUT/'left-parent-candidates.json').read_text());chosen='normal-weight-0.5';transforms=json.loads((OUT/(chosen+'-transforms.json')).read_text());native=candidates['nativeCollars'];p=np.array(native['front']['pivot']);axis=np.array(native['front']['axis']);_,_,old=ORIGINAL_CONTEXT();parent=transforms['left_upper_arm'];child=transforms['left_forearm'];hand=transforms['left_hand'];rp=np.array(parent['rotation']);tp=np.array(parent['translation']);rc=np.array(child['rotation']);tc=np.array(child['translation']);rh=np.array(hand['rotation']);th=np.array(hand['translation']);rco=np.array(old['left_forearm']['sourceToFrontRaw']['rotation']);tco=np.array(old['left_forearm']['sourceToFrontRaw']['translation']);rho=np.array(old['left_hand']['sourceToFrontRaw']['rotation']);tho=np.array(old['left_hand']['sourceToFrontRaw']['translation'])
    trial_checks={'handRelativeRotationMaxError':float(np.max(abs(rc.T@rh-rco.T@rho))),'handRelativeTranslationMaxError':float(np.max(abs(rc.T@(th-tc)-rco.T@(tho-tco))))}
    # The close-up wrist gate rejected carrying the previous hand through the
    # new arm delta: it reintroduced doubled finger edges. Restore the accepted
    # absolute hand fit as the new wrist rest pose; future arm edits carry it.
    transforms['left_hand']=old['left_hand']['sourceToFrontRaw'];rh=rho.copy();th=tho.copy()
    checks={'childParentRotationMaxError':float(np.max(abs(rc-rp))),'childParentTranslationMaxError':float(np.max(abs(tc-tp))),'handBaselineAbsoluteRotationMaxError':float(np.max(abs(rh-rho))),'handBaselineAbsoluteTranslationMaxError':float(np.max(abs(th-tho)))}
    assert max(checks.values())<1e-12
    source_hashes=json.loads((BASE/'feature-transforms.json').read_text())['sourceHashes'];cfg=json.loads((BASE/'fusion-config.json').read_text())
    for source,digest in source_hashes.items():
        if sha256_file(ROOT/cfg[source])!=digest:raise ValueError('Source changed: '+source)
    old_parent=old['left_upper_arm']['sourceToFrontRaw'];rop=np.array(old_parent['rotation']);top=np.array(old_parent['translation']);old_relative_r=rco@rop.T;old_relative_t=tco-old_relative_r@top;old_pivot=np.array(ORIGINAL_CONTEXT()[0]['left_elbow']);old_rotvec=Rotation.from_matrix(old_relative_r).as_rotvec();old_axis=old_rotvec/np.linalg.norm(old_rotvec)
    old_relative={'rotationDegrees':float(np.degrees(np.linalg.norm(old_rotvec))),'axisFrontRaw':old_axis.tolist(),'angleToMeasuredCollarAxisDegrees':min(angle(old_axis,axis),angle(-old_axis,axis)),'offsetAtOldLandmark':(old_relative_r@old_pivot+old_relative_t-old_pivot).tolist(),'offsetAtOldLandmarkNorm':float(np.linalg.norm(old_relative_r@old_pivot+old_relative_t-old_pivot)),'offsetAtMeasuredCollar':(old_relative_r@p+old_relative_t-p).tolist(),'offsetAtMeasuredCollarNorm':float(np.linalg.norm(old_relative_r@p+old_relative_t-p))}
    result={'version':1,'anatomicalSide':'left','visualIdentification':'screen-right arm in user screenshot, with black circular socket and long gray cable; not yellow-tube arm',
      'sourceHashes':source_hashes,'globalScale':parent['scale'],'frame':'original source PLY XYZ to front source raw XYZ; source units uncalibrated',
      'baseline':str(BASE.relative_to(ROOT)),'baselineHashes':{name:sha256_file(BASE/name) for name in ['feature-transforms.json','front-landmarks.json','back-landmarks.json','fusion-config.json']},
      'method':'one rigid parent fit to curved shoulder pad + distant measured collar center + soft collar plane normal; zero child twist with no bend or translation; previous clean absolute hand fit retained as a reviewed wrist rest pose',
      'mechanicalConstraint':{'parentPart':'left_upper_arm','childPart':'left_forearm','pivotFrontRaw':p.tolist(),'axisFrontRaw':axis.tolist(),'pivotPrepared':(p*[1,-1,-1]).tolist(),'axisPrepared':(axis*[1,-1,-1]).tolist(),'twistDegrees':0.,'independentTranslation':[0,0,0],'independentSwingDegrees':0,'formula':'Rc=Q Rp; tc=Q(tp-pivot)+pivot; Q=axisAngle(axis,twistDegrees)'},
      'nativeCollars':native,'transforms':transforms,'parentMetrics':candidates['candidates'][chosen]['metrics'],'parentBaselineMetrics':candidates['candidates']['baseline']['metrics'],'validation':checks,
      'handRestPose':{'sourceToFrontRaw':transforms['left_hand'],'carriedByPart':'left_forearm','referenceForearmSourceToFrontRaw':transforms['left_forearm'],'relativeRotation':(rc.T@rh).tolist(),'relativeTranslation':(rc.T@(th-tc)).tolist(),'policy':'Restore the v4 clean hand absolute registration as the wrist rest pose, then carry this newly reviewed rest pose through future changes to the accepted forearm transform. This does not preserve the old forearm-relative hand transform.','reason':'Six hand and six cuff views show renewed doubled/purple finger edges under the carried trial; retaining the old absolute hand cleans these edges without opening the wrist or separating the cuff.'},
      'rejectedCarriedHandTrial':{'sourceToFrontRaw':hand,'validation':trial_checks,'meaning':'These near-zero errors describe the rejected trial, not the final hand rest pose.','reason':'Carrying the previous hand through the arm correction displaced the wrist landmark by approximately0.01416 source units and reintroduced hand overlap ghosts.'},
      'previousRelativeTransform':old_relative,'jointParentObjective':{'residualTerms':['curved pad outline nearest-surface XYZ / (sqrt(N)*0.0015)','collar center XYZ /0.002','collar unit-normal XYZ difference /0.03 *0.5'],'normalWeight':.5,'loss':'linear least-squares on normalized feature regions','selection':'normal-weight-0.5 visually accepted across six collar and six full-arm cameras','limits':'The native scans are not perfectly rigidly congruent. Native plane equality is not a hard constraint; prioritizing it further harms both pad contour and collar-center fit.'},
      'segmentationRule':{'parts':['left_upper_arm','left_forearm'],'maximumAbsStation':.11,'maximumRadius':.11,'proximalPart':'left_upper_arm','distalPart':'left_forearm','landmark':'left_elbow','order':'apply after baseline clean_scan; update landmarks for coverage after relabeling; use only existing neighboring part labels','nativePlanes':native},
      'cablePolicy':'A connected gray cable crossing the collar belongs to the parent/socket, not the skin capsules. This zero-twist fit keeps every native cable point under one transform already; use the separate reviewed source-ID ownership selection before allowing twist tuning.',
      'interpretation':['V4 corrected the anatomical right/yellow-tube arm; the screenshot identifies the anatomical left/gray-cable arm.','The old left child included roughly 0.025 units of free relative translation, creating a false step through the cable and cuff.','Native collar centers and planes are measured from350/360 ring samples over23/21 angular sectors; median plane errors are about0.0007.','Fitting pad plus collar corrects parent ambiguity before deriving the child, rather than inventing a child swing.','The two scans are not perfectly congruent: forcing exact collar-normal equality worsens pad and center residuals. The plane normal is therefore a soft parent feature. Remaining normal mismatch is reported, not hidden by an independent bend.','Zero twist is selected because source cable continuity and multi-angle shell/cuff appearance agree. No matching front-side gray cable is available to recover an independent inter-capture cable angle.','The unchanged-parent zero-twist variant restored cable continuity but produced wrist ghosts. The balanced parent improves the cuff, but final detailed hand review still finds a carried-hand regression. Restoring the prior absolute hand registration retains the earlier cleanup while the revised cuff remains continuous.'],
      'reproduceCommand':'.venv-fusion/bin/python -B tools/fusion/audit_arm_v5.py --measure --audit --parents --render --hand-rest --evidence',
      'reviewImages':[str((OUT/n).relative_to(ROOT)) for n in ['left-collar-fit.png','zero-twist-collar-candidates.png','zero-twist-fullarm-candidates.png','hand-absolute-comparison.png','wrist-absolute-comparison.png']]}
    (OUT/'final-left-arm-transforms.json').write_text(json.dumps(transforms,indent=2)+'\n')
    (OUT/'left-arm-axial-evidence.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(checks,indent=2))

def render_hand_rest():
    """Evaluate the earlier clean hand against the revised wrist in actual GS."""
    from render_gaussians import load_fused,atlas
    base,lab,src,_=load_fused(BASE);trial=np.load(OUT/'normal-weight-0.5-fused.npz');d=trial['data'];labels=trial['labels'];final=np.concatenate([d[labels!=6],base[lab==6]]);np.savez_compressed(OUT/'final-left-arm-rest.npz',data=final,labels=np.r_[labels[labels!=6],np.full(sum(lab==6),6)])
    rows={'V4 baseline':base[np.isin(lab,[5,6])],'V5 carried hand':d[np.isin(labels,[5,6])],'V5 arm + previous hand':np.concatenate([d[labels==5],base[lab==6]])}
    views=[('Front',[0,-1,0]),('Back',[0,1,0]),('Right',[-1,0,0]),('Left',[1,0,0]),('Front oblique',[-.7,-1,-.1]),('Back oblique',[.7,1,-.1])]
    atlas(rows,OUT/'hand-absolute-comparison.png',[.76,.09,.64],.30,700,views=views,title='Final hand gate: prior clean hand, carried trial, previous absolute hand with corrected arm')
    atlas(rows,OUT/'wrist-absolute-comparison.png',[.675,.074,.547],.30,650,views=views,title='Same alternatives: wrist/cuff continuity from six angles')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--measure',action='store_true');p.add_argument('--audit',action='store_true');p.add_argument('--parents',action='store_true');p.add_argument('--render',action='store_true');p.add_argument('--hand-rest',action='store_true');p.add_argument('--evidence',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);collar.OUT=OUT;collar.context=left_context
    if a.measure:collar.unwrap();measure_left_collar()
    if a.audit:audit()
    if a.parents:parent_candidates()
    if a.render:render_candidates()
    if a.hand_rest:render_hand_rest()
    if a.evidence:accepted_evidence()
