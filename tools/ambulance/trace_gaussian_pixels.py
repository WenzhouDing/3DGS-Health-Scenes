"""Trace SH0 iPhone contributors with the review renderer's Gaussian math."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'tools'),str(ROOT/'tools/ambulance')]
from cleanup import read_ply,columns,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W

OUT=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--input',type=Path,default=ROOT/'raw/ambulance_exp11_boot_sharp.ply');ap.add_argument('--extra',type=Path);ap.add_argument('--tag',default='pad');ap.add_argument('--camera',default='pads-close');ap.add_argument('--targets',type=Path);ap.add_argument('--camera-file',type=Path);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--observed',type=Path);args=ap.parse_args();OUT=args.out;OUT.mkdir(parents=True,exist_ok=True)
v,_,_=read_ply(args.input)
primary_count=len(v)
if args.extra:
    extra,_,_=read_ply(args.extra);common=np.empty(len(extra),dtype=v.dtype)
    for name in v.dtype.names:common[name]=extra[name]
    v=np.concatenate([v,common])
camera_list=json.loads((ROOT/'tools/ambulance/pass2-cameras.json').read_text())
if args.camera_file:camera_list+=json.loads(args.camera_file.read_text())
camera=next(c for c in camera_list if c['id']==args.camera)
pos=np.array(camera['position']);forward=np.array(camera['target'])-pos;forward/=np.linalg.norm(forward)
right=np.cross(forward,[0,1,0]);right/=np.linalg.norm(right);down=-np.cross(right,forward);basis=np.array([right,down,forward])
f=375/np.tan(np.radians(camera['fov'])/2)
targets=[('upper-cloud',[400,180]),('upper-dark',[530,245]),('upper-bottom-fringe',[460,292]),('lower-cloud',[450,450]),('lower-dark',[530,520]),('wall',[500,340]),('upper-trim',[500,70])]
if args.targets:targets=json.loads(args.targets.read_text())
collected=[[] for _ in targets]
for start in range(0,len(v),150000):
    b=v[start:start+150000];p=np.einsum('ij,nj->ni',W,columns(b,['x','y','z']).astype(float));cam=np.einsum('ij,nj->ni',basis,p-pos)
    ids=np.flatnonzero(cam[:,2]>.02)
    if not len(ids):continue
    cc=cam[ids];depth=cc[:,2];s=np.exp(columns(b[ids],['scale_0','scale_1','scale_2']).astype(float));alpha=1/(1+np.exp(-np.clip(b['opacity'][ids].astype(float),-40,40)))
    R=Rotation.from_quat(columns(b[ids],['rot_1','rot_2','rot_3','rot_0']).astype(float)).as_matrix()
    R=np.einsum('ij,njk->nik',basis@W,R);C=np.einsum('nik,njk->nij',R*s[:,None,:],R*s[:,None,:])
    J=np.zeros((len(ids),2,3));J[:,0,0]=f/depth;J[:,1,1]=f/depth
    J[:,0,2]=-f*np.clip(cc[:,0]/depth,-1.3*500/f,1.3*500/f)/depth
    J[:,1,2]=-f*np.clip(cc[:,1]/depth,-1.3*375/f,1.3*375/f)/depth
    cov=np.einsum('nij,njk,nlk->nil',J,C,J);cov[:,0,0]+=.3;cov[:,1,1]+=.3
    det=cov[:,0,0]*cov[:,1,1]-cov[:,0,1]**2;mid=(cov[:,0,0]+cov[:,1,1])/2
    radius=3*np.sqrt(mid+np.sqrt(np.maximum(.1,mid*mid-det)))
    fade=np.clip((2048/1080*750-radius)/(1024/1080*750),0,1)
    inverse=np.linalg.inv(cov);pixels=np.c_[500+f*cc[:,0]/depth,375+f*cc[:,1]/depth]
    for i,(label,xy) in enumerate(targets):
        d=pixels-np.array(xy)-.5;power=-.5*np.einsum('ni,nij,nj->n',d,inverse,d)
        a=np.minimum(.99,alpha*fade*np.maximum(0,np.exp(power)-np.exp(-4.5)))
        contributing=np.flatnonzero(a>1e-7)
        for j in contributing:
            index=int(start+ids[j]);rgb=.5+.28209479177387814*np.array([v[f'f_dc_{k}'][index] for k in range(3)])
            collected[i].append({'index':index,'source':'extra' if index>=primary_count else 'primary','depth':float(depth[j]),'pixel_alpha':float(a[j]),'center_alpha':float(alpha[j]),'rgb':rgb.tolist(),'world':p[ids[j]].tolist(),'sigma':s[j].tolist()})
has_sh=any(n.startswith('f_rest_') for n in v.dtype.names)
image=np.asarray(Image.open(args.observed).convert('RGB'))/255 if args.observed else None
reports=[]
for (label,xy),records in zip(targets,collected):
    records.sort(key=lambda r:r['depth']);T=1.;color=np.zeros(3);used=[]
    for r in records:
        weight=T*r['pixel_alpha'];color+=weight*np.array(r['rgb']);T*=1-r['pixel_alpha'];r['visible_weight']=weight;used.append(r)
        if T<1e-4:break
    used.sort(key=lambda r:-r['visible_weight'])
    report={'label':label,'pixel':xy,'predicted_rgb':color.tolist(),'observed_rgb':(image[xy[1],xy[0]].tolist() if image is not None else None),'remaining_transmittance':T,
        'visible_mass_center_alpha_below_05':sum(r['visible_weight'] for r in used if r['center_alpha']<.5),
        'visible_mass_center_alpha_above_09':sum(r['visible_weight'] for r in used if r['center_alpha']>.9),
        'visible_mass_maxsigma_above_002':sum(r['visible_weight'] for r in used if max(r['sigma'])>.02),
        'visible_mass_minmaxratio_below_001':sum(r['visible_weight'] for r in used if min(r['sigma'])/max(r['sigma'])<.01),
        'visible_mass_center_z_above_104':sum(r['visible_weight'] for r in used if r['world'][2]>1.04),
        'visible_mass_center_z_above_11':sum(r['visible_weight'] for r in used if r['world'][2]>1.1),
        'visible_mass_center_z_above_20':sum(r['visible_weight'] for r in used if r['world'][2]>2.),
        'opacity_if_only_center_z_below_104':1-np.prod([1-r['pixel_alpha'] for r in used if r['world'][2]<=1.04]),
        'contributors':used[:25]}
    if label.startswith(('upper-','lower-')) and label!='upper-trim':
        coef=np.array([-.08291,.06183,.78042] if label.startswith('upper-') else [-.08171,-.06387,.79002])
        distances=[(r['world'][2]-np.dot(coef,np.r_[r['world'][:2],1]))/np.sqrt(1+np.sum(coef[:2]**2)) for r in used]
        report['physical_plane_z_from_xy1']=coef.tolist()
        report['visible_mass_within_physical_plane_002']=sum(r['visible_weight'] for r,d in zip(used,distances) if abs(d)<.02)
        report['visible_mass_inward_of_physical_plane_002']=sum(r['visible_weight'] for r,d in zip(used,distances) if d<-.02)
        report['visible_mass_behind_physical_plane_002']=sum(r['visible_weight'] for r,d in zip(used,distances) if d>.02)
    reports.append(report);print(label,'RGB',color,'observed',(image[xy[1],xy[0]] if image is not None else None),'alpha mass .5/.9',report['visible_mass_center_alpha_below_05'],report['visible_mass_center_alpha_above_09'],flush=True)
(OUT/f'{args.tag}-pixel-contributors.json').write_text(json.dumps({'camera':camera,'source':str(args.input),'resolution':[1000,750],
    'source_sha256':sha256_file(args.input),'extra_source':str(args.extra) if args.extra else None,'extra_sha256':sha256_file(args.extra) if args.extra else None,
    'color_model':'DC-only: for SH3 or mixed inputs predicted_rgb omits higher coefficients; opacity/depth/visible weights remain valid' if has_sh or args.extra else 'full SH0 color',
    'reports':reports},indent=2)+'\n')
