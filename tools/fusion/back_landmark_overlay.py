#!/usr/bin/env python3
"""Plot current manually initialized back landmarks in three raw-coordinate views."""
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mannequin-back-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from back_diagnostics import ROOT, load

data = json.loads((ROOT / 'raw/fusion-work/back-landmarks.json').read_text())
landmarks = data['landmarks']
points, colors, _ = load()
take = np.linspace(0, len(points)-1, min(350000, len(points)), dtype=int)
points, colors = points[take], colors[take]
bones = [('head','neck'), ('neck','chest'), ('chest','waist'), ('waist','pelvis')]
for side in ('left','right'):
    bones += [('neck',f'{side}_shoulder'), (f'{side}_shoulder',f'{side}_elbow'),
              (f'{side}_elbow',f'{side}_wrist'), (f'{side}_wrist',f'{side}_hand_tip'),
              ('pelvis',f'{side}_hip'), (f'{side}_hip',f'{side}_knee'),
              (f'{side}_knee',f'{side}_ankle'), (f'{side}_ankle',f'{side}_toe')]
fig, axes = plt.subplots(1,3,figsize=(20,12),constrained_layout=True)
for ax,(a,b,title) in zip(axes,[(0,2,'Raw X/Z: posterior view'),(2,1,'Raw Z/Y: side'),(0,1,'Raw X/Y: end')]):
    depth = ({0,1,2}-{a,b}).pop()
    order = np.argsort(points[:,depth])[::-1]
    ax.scatter(points[order,a],points[order,b],c=colors[order],s=.45,linewidths=0)
    for start,end in bones:
        p,q=np.asarray(landmarks[start]),np.asarray(landmarks[end])
        ax.plot([p[a],q[a]],[p[b],q[b]],color='#e32f95',linewidth=1.6)
    for name, point in landmarks.items():
        ax.scatter(point[a],point[b],s=16,c='#16c4d8',edgecolor='black',linewidth=.4,zorder=5)
        if a == 0 and b == 2:
            ax.annotate(name.replace('left_','L ').replace('right_','R '),(point[a],point[b]),
                        xytext=(4,4),textcoords='offset points',fontsize=7,color='#a51662')
    ax.set(aspect='equal',xlabel='xyz'[a],ylabel='xyz'[b],title=title)
    ax.grid(alpha=.2)
fig.savefig(ROOT/'raw/fusion-work/back-landmark-overlay.png',dpi=150)
print('Wrote raw/fusion-work/back-landmark-overlay.png')
