#!/usr/bin/env python3
"""Local orthographic Gaussian renderer for reproducible alignment close-ups.

Renders the actual anisotropic covariance, DC color and sigmoid opacity with
depth-sorted alpha compositing. This is not a point-center scatter plot.
Inputs to render() use the original front scan frame (head toward -Z).
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from scipy.spatial.transform import Rotation
from numba import njit
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_mannequin import read_ply, FIELDS

VIEWS = [('Front', [0,-1,0]), ('Back', [0,1,0]),
         ('Right', [-1,0,0]), ('Left', [1,0,0]),
         ('Front oblique', [.8,-1,-.15]), ('Back oblique', [-.8,1,-.15])]
BG = np.array([.07,.09,.11], dtype=np.float32)

@njit
def _raster(xy, inverse, radius, colors, alpha, width, height, background):
    pixels = np.empty((height,width,3), dtype=np.float32)
    pixels[:,:,:] = background
    for i in range(len(xy)):
        cx,cy = xy[i]
        rx,ry = radius[i]
        x0=max(0,int(np.floor(cx-rx))); x1=min(width-1,int(np.ceil(cx+rx)))
        y0=max(0,int(np.floor(cy-ry))); y1=min(height-1,int(np.ceil(cy+ry)))
        aa,ab,bb = inverse[i]
        for y in range(y0,y1+1):
            dy=y+.5-cy
            for x in range(x0,x1+1):
                dx=x+.5-cx
                power=-.5*(aa*dx*dx+2*ab*dx*dy+bb*dy*dy)
                if power < -4.5: continue
                opacity=min(.99,alpha[i]*np.exp(power))
                if opacity < 1/255: continue
                for c in range(3):
                    pixels[y,x,c]=pixels[y,x,c]*(1-opacity)+colors[i,c]*opacity
    return pixels

def render(data, direction, center, span, width=640, height=640, tint=None,
           up=(0,0,-1), background=BG):
    """Return PIL RGB image; span is the image's vertical size in scan units."""
    direction=np.asarray(direction,float);direction/=np.linalg.norm(direction)
    right=np.cross(direction,up)
    if np.linalg.norm(right)<1e-8: right=np.cross(direction,[0,1,0])
    right/=np.linalg.norm(right); vertical=np.cross(right,direction)
    basis=np.array([right,-vertical])
    xyz=np.asarray(data[:,:3],float)-center
    projected=np.einsum('ni,ji->nj',xyz,basis)
    sigma=np.exp(np.clip(data[:,7:10],-30,3))
    margin=sigma.max(1)*3
    keep=(abs(projected[:,0])<span*width/height/2+margin)&(abs(projected[:,1])<span/2+margin)
    data=data[keep];xyz=xyz[keep];projected=projected[keep];sigma=sigma[keep]
    if len(data)==0:return Image.new('RGB',(width,height),tuple((np.asarray(background)*255).astype(int)))
    q=data[:,3:7].astype(float);norm=np.linalg.norm(q,axis=1)
    q[norm<1e-12]=[1,0,0,0]
    rotations=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix()
    axes=np.einsum('ij,njk->nik',basis,rotations)*sigma[:,None,:]
    pixel_scale=height/span
    cov=np.einsum('nik,njk->nij',axes,axes)*pixel_scale**2
    cov[:,0,0]+=.3;cov[:,1,1]+=.3
    determinant=cov[:,0,0]*cov[:,1,1]-cov[:,0,1]**2
    inverse=np.c_[cov[:,1,1],-cov[:,0,1],cov[:,0,0]]/determinant[:,None]
    radii=3*np.sqrt(np.c_[cov[:,0,0],cov[:,1,1]])
    xy=projected*pixel_scale+np.array([width,height])/2
    alpha=1/(1+np.exp(-np.clip(data[:,10].astype(float),-40,40)))
    colors=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
    if tint is not None:
        tint=np.asarray(tint)
        colors=np.broadcast_to(tint,colors.shape) if tint.ndim==1 else tint[keep]
    order=np.argsort(np.einsum('ni,i->n',xyz,direction),kind='stable')
    image=_raster(np.ascontiguousarray(xy[order]),np.ascontiguousarray(inverse[order]),
                  np.ascontiguousarray(radii[order]),np.ascontiguousarray(colors[order]),
                  np.ascontiguousarray(alpha[order]),width,height,np.asarray(background,dtype=np.float32))
    return Image.fromarray(np.rint(np.clip(image,0,1)*255).astype(np.uint8))

def atlas(captures, path, center, span, size=480, views=VIEWS, title='', source_colors=False):
    """One row per named Nx14 capture, one column per camera; consistent framing."""
    canvas=Image.new('RGB',(len(views)*size,len(captures)*(size+28)+52),(18,23,28))
    draw=ImageDraw.Draw(canvas);draw.text((12,12),title,fill='white')
    for row,(name,data) in enumerate(captures.items()):
        for col,(label,direction) in enumerate(views):
            print('render',name,label,len(data),flush=True)
            img=render(data,direction,center,span,width=size,height=size)
            x=col*size;y=52+row*(size+28)
            canvas.paste(img,(x,y+28));draw.text((x+10,y+7),name+' / '+label,fill='white')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
    return path

def load_fused(directory):
    directory=Path(directory);cols,_,_=read_ply(directory/'mannequin_fused.ply')
    data=np.column_stack([cols[k] for k in FIELDS])
    # Convert the exported [x,-y,-z] pose back to raw front for common diagnostics.
    sys.path.insert(0,str(Path(__file__).parent));from pipeline import transform_gaussians
    data=transform_gaussians(data,np.diag([1.,-1.,-1.]),np.zeros(3))
    return data,np.load(directory/'part-labels.npy'),np.load(directory/'capture-labels.npy'),json.loads((directory/'report.json').read_text())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=ROOT/'raw/mannequin-fused')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parts',default='head,neck')
    p.add_argument('--center',type=float,nargs=3,default=[0,-.04,-.14])
    p.add_argument('--span',type=float,default=.52);p.add_argument('--size',type=int,default=480)
    p.add_argument('--both-only',action='store_true');args=p.parse_args()
    data,labels,sources,report=load_fused(args.input)
    chosen=[i for i,part in enumerate(report['parts']) if part['id'] in args.parts.split(',')]
    if args.parts!='all':
        keep=np.isin(labels,chosen);data=data[keep];sources=sources[keep]
    captures={'Both':data} if args.both_only else {'Front contribution':data[sources==0],'Back contribution':data[sources==1],'Both':data}
    atlas(captures,args.out,args.center,args.span,args.size,title='Anisotropic Gaussian render / '+args.parts+' / '+str(args.input))

if __name__=='__main__': main()
