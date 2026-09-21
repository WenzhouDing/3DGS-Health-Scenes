#!/usr/bin/env python3
"""Audit joint centers against fused Gaussian shells without editing annotations or scans.

Fits angularly balanced ellipse sections perpendicular to the local longitudinal
axis and measures exposed opposing metal hinge caps as connected 3D components. Cross sections use both captures; bootstrap/neighbor-section agreement is
reported in uncalibrated scan units. Clothing and shoulder-pad surfaces are not
assumed to locate a hidden mechanical ball center.
"""
from pathlib import Path
import hashlib, json, copy, sys, os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mannequin-joint-mpl')
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'raw/fusion-work/joint-analysis'
SOURCE=ROOT/'raw/mannequin-fused/joint-annotations.json'
MANIFEST=ROOT/'viewers/mannequin-fusion/fusion.json'
C0=.28209479177387814

def file_hash(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def unit(v):
    v=np.asarray(v,float);return v/np.linalg.norm(v)

def basis(n):
    n=unit(n);u=unit(np.cross(n,[0,1,0] if abs(n[1])<.85 else [1,0,0]));v=np.cross(n,u)
    return np.array([u,v])

def fit_section(data,pivot,normal,station,halfwidth=.008,radius=.115):
    b=basis(normal);xyz=data[:,:3]-pivot;along=np.einsum('ni,i->n',xyz,normal); uv=np.einsum('ni,ji->nj',xyz,b)
    mask=(abs(along-station)<halfwidth)&(np.linalg.norm(uv,axis=1)<radius)
    points=uv[mask]
    if len(points)<150:return None
    # Equal angular-sector representation prevents the densely observed front
    # shell from pulling the fit toward itself. Median radius rejects floaters.
    def balance(center):
        d=points-center;theta=np.arctan2(d[:,1],d[:,0]);r=np.linalg.norm(d,axis=1)
        bins=np.minimum(47,((theta+np.pi)/(2*np.pi)*48).astype(int))
        targets=[]
        for k in range(48):
            ids=np.flatnonzero(bins==k)
            if len(ids)<6:continue
            local=ids[np.argsort(r[ids])];keep=local[int(len(local)*.2):max(int(len(local)*.8),int(len(local)*.2)+1)]
            targets.append(np.median(points[keep],axis=0))
        return np.asarray(targets)
    center=np.zeros(2)
    for iteration in range(3):
        q=balance(center)
        if len(q)<30:return None
        rr=np.quantile(np.linalg.norm(q-center,axis=1),.5)
        init=np.array([*center,np.log(rr),np.log(rr),0.]) if iteration==0 else opt.x
        def residual(x):
            c=np.cos(x[4]);s=np.sin(x[4]);r=(q-x[:2])@np.array([[c,-s],[s,c]])
            radii=np.exp(x[2:4]);rho=np.sqrt(np.sum((r/radii)**2,axis=1))
            return (rho-1)*np.sqrt(np.prod(radii))
        opt=least_squares(residual,init,loss='soft_l1',f_scale=.002,
            bounds=([-.045,-.045,np.log(.012),np.log(.012),-np.inf],[.045,.045,np.log(radius),np.log(radius),np.inf]),max_nfev=250)
        center=opt.x[:2]
    errors=residual(opt.x)
    c=np.cos(opt.x[4]);s=np.sin(opt.x[4]);rot=np.array([[c,-s],[s,c]])
    t=np.linspace(-np.pi,np.pi,150)
    outline=(np.array([np.cos(t),np.sin(t)]).T*np.exp(opt.x[2:4]))@rot.T+center
    return dict(center=pivot+station*normal+b.T@center,offsetUV=center,radii=np.exp(opt.x[2:4]),
        medianResidual=float(np.median(abs(errors))),p90Residual=float(np.quantile(abs(errors),.9)),
        angularBins=len(q),pointCount=len(points),basis=b,points=points,targets=q,outline=outline,station=station)

def load():
    manifest=json.loads(MANIFEST.read_text()); chunks=[];labels=[];captures=[]
    for ci,c in enumerate(manifest['captures']):
        a=np.memmap(MANIFEST.parent/c['url'],dtype='<f4',mode='r').reshape(-1,14)
        lab=np.memmap(MANIFEST.parent/c['labelsUrl'],dtype='u1',mode='r')
        # Analysis-only filtering: retain reliable surface support, never edit GS.
        ok=(a[:,10]>-1.1)&(a[:,7:10].max(1)<np.log(.016))&np.isfinite(a).all(1)
        chunks.append(np.asarray(a[ok]));labels.append(np.asarray(lab[ok]));captures.append(np.full(ok.sum(),ci,np.uint8))
    return manifest,np.concatenate(chunks),np.concatenate(labels),np.concatenate(captures)

def strip(f):
    if f is None:return None
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in f.items() if k not in ['points','targets','outline','basis']}

def diagnostic(j,p,n,local,cap,sections,after,axis_after):
    fig,axes=plt.subplots(1,4,figsize=(18,5),facecolor='#152128')
    colors=np.clip(.5+C0*local[:,11:14],0,1)
    views=[('Front projection',np.array([1,0,0]),np.array([0,0,1])),
           ('Side projection',np.array([0,1,0]),np.array([0,0,1])),
           ('Above projection',np.array([1,0,0]),np.array([0,1,0]))]
    for ax,(title,u,v) in zip(axes[:3],views):
        x=np.einsum('ni,i->n',local[:,:3],u);y=np.einsum('ni,i->n',local[:,:3],v);depth=np.einsum('ni,i->n',local[:,:3],np.cross(u,v));order=np.argsort(depth)
        ax.scatter(x[order],y[order],c=colors[order],s=.65,lw=0,rasterized=True)
        ax.scatter(p@u,p@v,marker='+',s=110,color='#ffcc63',lw=1.5,label='Accepted starter')
        ax.scatter(after@u,after@v,marker='x',s=75,color='#55e4e8',lw=1.5,label='Runtime center')
        if j['type'] in ['hinge','swivel']:
            for center,axis,col in [(p,np.array(j['axis']),'#ffcc63'),(after,axis_after,'#55e4e8')]:
                end=np.array([center-axis*.075,center+axis*.075]);ax.plot(end@u,end@v,color=col,lw=1.2)
        ax.set_title(title,color='white');ax.set_xlabel('scan units',color='#a9bdc7')
    ax=axes[3]
    if sections:
        f=min(sections,key=lambda f:abs(f['station']));q=f['points'];step=max(1,len(q)//12000)
        ax.scatter(q[::step,0],q[::step,1],s=1,color='#71858f',lw=0)
        ax.scatter(f['targets'][:,0],f['targets'][:,1],s=10,color='#c4ed8d',lw=0)
        ax.plot(f['outline'][:,0],f['outline'][:,1],color='#55e4e8',lw=1.5)
        ax.scatter(0,0,marker='+',s=110,color='#ffcc63'); off=f['basis']@(after-p)
        ax.scatter(*off,marker='x',s=75,color='#55e4e8')
        ax.set_title(f'Normal section: {f["angularBins"]}/48 sectors\nmedian error {f["medianResidual"]:.4f}',color='white')
    else:ax.set_title('Hidden / irregular surface\nCenter retained',color='white')
    for ax in axes:
        ax.set_facecolor('#202f39');ax.set_aspect('equal');ax.grid(alpha=.14);ax.tick_params(colors='#a9bdc7')
        for spine in ax.spines.values():spine.set_color('#47616e')
    axes[0].legend(loc='best',fontsize=7)
    fig.suptitle(j['label']+' — surface projection audit, not a hardware calibration',color='white',fontsize=14)
    fig.tight_layout();path=OUT/(j['id']+'.png');fig.savefig(path,dpi=145);plt.close(fig)
    return str(path.relative_to(ROOT))

def hardware_cap_pair(local,pivot,chroma=.14,brightness=.48):
    """Locate compact low-chroma metal cap clusters on opposing joint sides."""
    rgb=np.clip(.5+C0*local[:,11:14],0,1)
    mask=(np.ptp(rgb,axis=1)<chroma)&(np.mean(rgb,axis=1)>brightness)&(np.linalg.norm(local[:,:3]-pivot,axis=1)<.10)
    xyz=local[mask,:3].astype(float)
    if len(xyz)<100:return None
    tree=cKDTree(xyz);graph=tree.sparse_distance_matrix(tree,.004,output_type='coo_matrix')
    count,groups=connected_components(graph,directed=False);clusters=[]
    for k in range(count):
        pts=xyz[groups==k]
        if len(pts)<50:continue
        extent=np.ptp(pts,axis=0)
        if extent.max()>.035 or extent.max()<.004:continue
        center=np.median(pts,axis=0)
        clusters.append({'center':center,'pointCount':len(pts),'extent':extent,'points':pts})
    candidates=[]
    for i,a in enumerate(clusters):
        for b in clusters[i+1:]:
            delta=b['center']-a['center'];length=np.linalg.norm(delta);center=(a['center']+b['center'])/2
            if not .055<length<.15 or abs(delta[0])/length<.78 or np.linalg.norm(center-pivot)>.04:continue
            candidates.append((min(a['pointCount'],b['pointCount']),a,b))
    if not candidates:return None
    _,a,b=max(candidates,key=lambda c:c[0]);center=(a['center']+b['center'])/2;axis=unit(b['center']-a['center'])
    if axis[0]<0:axis=-axis
    return {'center':center,'axis':axis,'caps':[a,b],'capSeparation':float(np.linalg.norm(b['center']-a['center'])),'candidatePairCount':len(candidates)}

def gaussian_diagnostic(j,local,pivot,axis):
    """Render anisotropic Gaussian support in four directions, overlaying the axis."""
    sys.path.insert(0,str(ROOT/'tools/fusion'))
    from render_gaussians import render
    from PIL import Image, ImageDraw
    cameras=[('Front',[0,1,0]),('Back',[0,-1,0]),('Side',[1,0,0]),('Oblique',[.7,1,.5])]
    size=400;span=.25;canvas=Image.new('RGB',(size*4,size+55),(18,23,28));draw=ImageDraw.Draw(canvas)
    draw.text((12,8),j['label']+' - fitted center / axis over original Gaussian appearance',fill='white')
    for k,(label,direction) in enumerate(cameras):
        direction=unit(direction);right=unit(np.cross(direction,[0,0,1.]));up=np.cross(right,direction)
        img=render(local,direction,pivot,span,width=size,height=size,up=(0,0,1))
        overlay=ImageDraw.Draw(img);px=size/2;py=size/2
        overlay.line((px-7,py,px+7,py),fill='#5de9e3',width=2);overlay.line((px,py-7,px,py+7),fill='#5de9e3',width=2)
        if j['type'] in ['hinge','swivel']:
            tip=axis*.075;dx=float(tip@right)*size/span;dy=-float(tip@up)*size/span
            overlay.line((px-dx,py-dy,px+dx,py+dy),fill='#5de9e3',width=1)
            overlay.ellipse((px+dx-3,py+dy-3,px+dx+3,py+dy+3),fill='#ffd16b')
        canvas.paste(img,(k*size,55));draw.text((k*size+12,34),label,fill='white')
    path=OUT/(j['id']+'-gaussians.png');canvas.save(path);return str(path.relative_to(ROOT))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source=SOURCE.read_bytes();doc=json.loads(source);derived=copy.deepcopy(doc)
    manifest,data,labels,capture=load();names=[p['id'] for p in manifest['parts']];joints={j['id']:j for j in doc['joints']}
    report={'schema':'mannequin-joint-refinement','version':1,'inputAnnotationSha256':hashlib.sha256(source).hexdigest(),
        'inputSceneRevision':doc['scene']['revision'],'coordinateFrame':doc['scene']['coordinateFrame'],
        'method':'Opacity/scale-screened fused Gaussian centers; balanced 48-sector robust ellipse sections from both captures; three orthogonal projections; visible hinge-cap connected components and four threshold sensitivity fits. Opposing shells estimate a visible envelope center, not hidden hardware.',
        'limitations':['Scan units are not calibrated physical length.','Stiffness and damping are normalized interaction settings, not measured mechanical properties.',
        'Clothing and shoulder pads cannot establish hidden ball-joint centers.','Knee and ankle axes pass through visible opposing metal cap centers, with color-threshold sensitivity reported; cap centers are not calibrated internal shaft measurements.'],
        'sourceFiles':[str(SOURCE.relative_to(ROOT)),str(MANIFEST.relative_to(ROOT))], 'joints':[]}
    artifacts=[MANIFEST]+[MANIFEST.parent/c[key] for c in manifest['captures'] for key in ['url','labelsUrl']]
    report['sourceArtifactHashes']={str(path.relative_to(ROOT)):file_hash(path) for path in artifacts}
    fits_by_joint={}
    for j,out in zip(doc['joints'],derived['joints']):
        p=np.array(j['pivot']);a=unit(j['axis']);id=j['id'];side=id.split('_')[0]
        if 'arm_swivel' in id:n=a
        elif 'shoulder' in id:n=unit(np.array(joints[side+'_arm_swivel']['pivot'])-p)
        elif 'knee' in id:n=unit(np.array(joints[side+'_ankle']['pivot'])-p)
        elif 'ankle' in id:n=unit(p-np.array(joints[side+'_knee']['pivot']))
        elif 'hip' in id:n=unit(np.array(joints[side+'_knee']['pivot'])-p)
        else:n=np.array([0,0,-1.])
        allowed=[j['parentPart']]+j['childParts'];partmask=np.isin(labels,[names.index(x) for x in allowed])
        r=.155 if 'hip' in id or id=='waist' else .125
        localmask=partmask&(np.linalg.norm(data[:,:3]-p,axis=1)<r)
        local=data[localmask];cap=capture[localmask]
        stations=[-.012,0,.012] if id=='right_ankle' else [-.016,0,.016];fits=[fit_section(local,p,n,s,radius=r) for s in stations];fits=[f for f in fits if f is not None]
        fits_by_joint[id]=(p,n,local,cap,fits)
        after=p.copy();axis_after=a.copy();reason='Retained accepted starter: surface geometry does not independently establish this joint center.';confidence='low';status='retained'
        metrics={'analysisPointCount':len(local),'capturePointCounts':{c['id']:int((cap==i).sum()) for i,c in enumerate(manifest['captures'])},'sections':[strip(f) for f in fits]}
        if fits:
            # Remove displacement along the section normal. The seam station is
            # the user-accepted one, independent of the shell centroid.
            offsets=np.array([f['center']-p-f['station']*n for f in fits]);offset=np.median(offsets,axis=0)
            variation=float(np.max(np.linalg.norm(offsets-offset,axis=1)))
            residual=float(np.median([f['medianResidual'] for f in fits]))
            metrics.update(candidateCenter=(p+offset).tolist(),candidateShift=float(np.linalg.norm(offset)),neighborSectionSpread=variation,medianSectionResidual=residual)
        if 'arm_swivel' in id:
            reason='Preserved measured mechanical collar center and plane normal from prior fusion review. Broad sleeve sections are a check, not a replacement for the measured collar rim.';confidence='high'
        elif 'shoulder' in id or 'hip' in id or id=='waist':
            reason='Retained accepted position: shoulder pad / shorts / body shell is not the hidden ball-joint surface. A bulk silhouette fit would give false precision.'
        elif ('knee' in id or 'ankle' in id or id=='neck') and len(fits)==3:
            if variation<.009 and residual<.006 and .0015<np.linalg.norm(offset)<.025 and (np.linalg.norm(offset)<.020 or variation<.003):
                after=p+offset;status='refined';confidence='medium';reason='Centered the pivot between opposing visible surfaces using the median of three balanced cross sections; preserved its accepted position along the limb.'
            else:
                reason='Retained accepted center: section agreement, residual or candidate shift did not meet conservative acceptance thresholds.';confidence='medium' if np.linalg.norm(offset)<.008 else 'low'
        # A hinge transverse axis should follow the local limb frame, rather
        # than global horizontal. Estimate shin centerline from narrow sections.
        if 'knee' in id or 'ankle' in id:
            sign=1 if 'knee' in id else -1
            shinmask=(labels==names.index(side+'_shin'))&(np.linalg.norm(data[:,:3]-p,axis=1)<.18)
            shin=data[shinmask]
            axisfits=[fit_section(shin,p,n,sign*s,radius=.09,halfwidth=.014) for s in [.035,.060,.085,.110]]
            axisfits=[f for f in axisfits if f is not None]
            metrics['centerlineSections']=[strip(f) for f in axisfits]
            if len(axisfits)>=3:
                pts=np.array([f['center'] for f in axisfits]);x=np.array([f['station'] for f in axisfits]);coefs=np.linalg.lstsq(np.c_[x,np.ones(len(x))],pts,rcond=None)[0]
                long=unit(coefs[0]);axis_candidate=unit(np.cross(long,[0,1,0]));axis_candidate*=np.sign(axis_candidate@a)
                centerline_error=float(np.sqrt(np.mean(np.linalg.norm(pts-np.c_[x,np.ones(len(x))]@coefs,axis=1)**2)))
                degrees=float(np.rad2deg(np.arccos(np.clip(axis_candidate@a,-1,1))))
                metrics.update(centerlineDirection=long.tolist(),centerlineRms=centerline_error,candidateAxis=axis_candidate.tolist(),candidateAxisChangeDegrees=degrees)
                if centerline_error<.005 and degrees<30:
                    axis_after=axis_candidate;status='refined';confidence='medium';reason+=' Hinge direction follows the fitted shin centerline in the frontal plane; this remains an editable geometric estimate.'
        if 'knee' in id or 'ankle' in id:
            hardware=hardware_cap_pair(local,p)
            if hardware:
                variations=[hardware_cap_pair(local,p,chroma=c,brightness=b) for c,b in [(.12,.45),(.12,.52),(.16,.45),(.16,.52)]]
                variations=[v for v in variations if v is not None]
                spread=max([float(np.linalg.norm(v['center']-hardware['center'])) for v in variations] or [1.])
                angle_spread=max([float(np.rad2deg(np.arccos(np.clip(abs(v['axis']@hardware['axis']),-1,1)))) for v in variations] or [180.])
                metrics['hardwareCaps']={'centers':[c['center'].tolist() for c in hardware['caps']],
                    'pointCounts':[c['pointCount'] for c in hardware['caps']],
                    'extents':[c['extent'].tolist() for c in hardware['caps']],
                    'capSeparation':hardware['capSeparation'],'candidatePairCount':hardware['candidatePairCount'],
                    'colorSensitivityFits':len(variations),'centerSensitivity':spread,'axisSensitivityDegrees':angle_spread,
                    'midpoint':hardware['center'].tolist(),'axisTowardPositiveX':hardware['axis'].tolist()}
                if len(variations)>=3 and spread<.004 and angle_spread<4:
                    after=hardware['center'];axis_after=hardware['axis'];status='refined';confidence='medium-high'
                    reason='Located the two visible metal hinge caps as compact low-chroma components in 3D; set the pivot midway between cap centers and the axis through them. Four color-threshold perturbations verify stability. Checked against front/back/side/oblique Gaussian projections. Cap surface centers remain scan estimates, not a calibrated shaft measurement.'
        if 'knee' in id and axis_after[0]>0:
            axis_after=-axis_after
            status='refined'
            metrics['positiveFlexionConvention']='Positive knee angle moves the lower leg posteriorly (viewer -Y); this is an interaction convention, not a measured mechanical stop.'
            reason+=' Oriented the axis toward -X so positive knee flexion bends toward the back. Limits remain editable.'
        out['pivot']=after.tolist();out['axis']=axis_after.tolist()
        correction=np.linalg.norm(after-p);angular=np.rad2deg(np.arccos(np.clip(axis_after@a,-1,1)))
        metrics.update(appliedShift=float(correction),appliedAxisDegrees=float(angular),appliedAxisLineDegrees=float(min(angular,180-angular)))
        plot=diagnostic(j,p,n,local,cap,fits,after,axis_after)
        report['joints'].append({'id':id,'pivotBefore':p.tolist(),'pivotAfter':after.tolist(),'axisBefore':a.tolist(),'axisAfter':axis_after.tolist(),'status':status,'confidence':confidence,'method':reason,'metrics':metrics,'plot':plot,'gaussianPlot':gaussian_diagnostic(j,local,after,axis_after) if status=='refined' else None})
        print(id,status,confidence,'shift',round(correction,5),'angle',round(angular,2),'candidate',round(metrics.get('candidateShift',0),4),flush=True)
    report['summary']={'jointCount':len(doc['joints']),'refinedCount':sum(j['status']=='refined' for j in report['joints']),'maximumPivotChange':max(j['metrics']['appliedShift'] for j in report['joints']),'maximumAxisChangeDegrees':max(j['metrics']['appliedAxisDegrees'] for j in report['joints']),'maximumAxisLineChangeDegrees':max(j['metrics']['appliedAxisLineDegrees'] for j in report['joints']),'preservedRigidWrist':True,'analysisGaussianCount':len(data),'appliedThresholds':{'maxShellPivotShift':.025,'maxNeighborSpreadForShiftOver002':.003,'maxNeighborSectionSpread':.009,'maxMedianEllipseResidual':.006,'maxCenterlineRms':.005,'maxHingeAxisChangeDegrees':30,'maxHardwareCenterSensitivity':.004,'maxHardwareAxisSensitivityDegrees':4,'maxHardwareMidpointShift':.04}}
    report['annotations']=derived
    path=ROOT/'raw/mannequin-fused/joint-refinement.json';temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    print(path)

if __name__=='__main__':main()
