#!/usr/bin/env python3
"""Fixed-camera sensitivity of the left-hand back contribution; no production edits."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
from pipeline import ROOT, transform_gaussians
from render_gaussians import load_fused, render

BASE=ROOT/'raw/fusion-work/refinement-v7/baseline'
OUT=ROOT/'raw/fusion-work/refinement-v7/root-audit'
OUT.mkdir(parents=True,exist_ok=True)

def load():
    cache=OUT/'baseline-context.npz'
    if cache.exists():
        c=np.load(cache);return c['data'],c['labels'],c['sources']
    d,l,s,r=load_fused(BASE)
    selected=np.isin(l,[5,6])&(d[:,0]>.625)
    d,l,s=d[selected],l[selected],s[selected]
    np.savez_compressed(cache,data=d,labels=l,sources=s)
    return d,l,s

def main():
    d,l,s=load();selected=(l==6)&(s==1)
    pivot=np.array([.704,.065,.584]);long=np.array([.68,0,.7332]);long/=np.linalg.norm(long)
    lateral=np.array([long[2],0,-long[0]])
    variants=[('Current',np.eye(3),np.zeros(3))]
    variants += [(f'Depth {v:+.3f}',np.eye(3),np.array([0,v,0])) for v in [-.012,-.006,.006,.012]]
    variants += [(f'Roll {v:+} deg',Rotation.from_rotvec(long*np.deg2rad(v)).as_matrix(),np.zeros(3)) for v in [-8,-4,4,8]]
    variants += [(f'Pitch {v:+} deg',Rotation.from_rotvec(lateral*np.deg2rad(v)).as_matrix(),np.zeros(3)) for v in [-4,4]]
    views=[('From below',[0,0,1]),('Along fingers',[.7,0,.7]),('Low oblique',[.5,-.6,1])]
    for label,direction in views:
        size=580;canvas=Image.new('RGB',(4*size,3*(size+26)),(18,23,28));draw=ImageDraw.Draw(canvas)
        for i,(name,q,shift) in enumerate(variants):
            dd=d.copy();dd[selected]=transform_gaussians(dd[selected],q,pivot-q@pivot+shift)
            x=(i%4)*size;y=(i//4)*(size+26)
            canvas.paste(render(dd,direction,[.786,.105,.667],.235,width=size,height=size),(x,y+26))
            draw.text((x+8,y+8),name+' / '+label,fill='white')
        canvas.save(OUT/('pose-trials-'+label.lower().replace(' ','-')+'.png'))
    (OUT/'pose-trials.json').write_text(json.dumps({'baseline':str(BASE.relative_to(ROOT)),
        'pivot':pivot.tolist(),'trials':[{'name':n,'rotation':q.tolist(),'shift':v.tolist()} for n,q,v in variants],
        'note':'Fixed previous fusion weights for diagnostic sensitivity only. Candidate must recompute coverage before acceptance.'},indent=2)+'\n')

if __name__=='__main__':main()
