#!/usr/bin/env python3
"""Repeatable, source-preserving inspection of the back Gaussian scan."""
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mannequin-back-matplotlib')
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_mannequin import read_ply

def load():
    vertices, count, _ = read_ply(ROOT / 'raw/mannequin_back_119999.ply')
    xyz = np.column_stack([vertices[a] for a in 'xyz'])
    color = np.clip(np.column_stack([vertices[f'f_dc_{a}'] for a in range(3)]) * .28209479177387814 + .5, 0, 1)
    opacity = 1 / (1 + np.exp(-vertices['opacity']))
    scale = np.exp(np.column_stack([vertices[f'scale_{a}'] for a in range(3)])).max(axis=1)
    valid = np.isfinite(xyz).all(axis=1) & (opacity > .08) & (scale < .03)
    return xyz[valid], color[valid], opacity[valid]

def main():
    output = ROOT / 'raw/fusion-work'
    output.mkdir(parents=True, exist_ok=True)
    xyz, rgb, opacity = load()
    # Evenly spaced original indices preserve coverage without random changes.
    take = np.linspace(0, len(xyz) - 1, min(260000, len(xyz)), dtype=int)
    xyz, rgb, opacity = xyz[take], rgb[take], opacity[take]
    fig, axes = plt.subplots(2, 3, figsize=(20, 12), constrained_layout=True)
    for ax, (a, b, title) in zip(axes.flat[:3], [(0, 2, 'Raw X/Z top projection'), (0, 1, 'Raw X/Y end projection'), (2, 1, 'Raw Z/Y side projection')]):
        depth = ({0,1,2} - {a,b}).pop()
        order = np.argsort(xyz[:, depth])
        ax.scatter(xyz[order, a], xyz[order, b], c=rgb[order], s=.6, linewidths=0)
        ax.set(xlabel='xyz'[a], ylabel='xyz'[b], title=title, aspect='equal')
        ax.grid(alpha=.2)
    axes[1,0].scatter(xyz[:,0],xyz[:,2],c=xyz[:,1],s=.6,linewidths=0,cmap='turbo',vmin=-.13,vmax=.25)
    axes[1,0].set(xlabel='x',ylabel='z',title='Raw Y depth (blue low, red high)',aspect='equal')
    axes[1,1].hist(xyz[:,1], bins=250)
    axes[1,1].set(xlabel='raw y',title='Height density')
    axes[1,2].scatter(xyz[:,0], xyz[:,2],c=np.where(xyz[:,1,None] > -.055,rgb,[.15,.15,.15]),s=.6,linewidths=0)
    axes[1,2].set(xlabel='x',ylabel='z',title='Color above raw Y=-0.055; lower gray',aspect='equal')
    fig.savefig(output / 'back-orthographic.png', dpi=160)
    plt.close(fig)
    fig=plt.figure(figsize=(20,14), constrained_layout=True)
    for number,(elev,azim) in enumerate([(20,-60),(20,60),(20,135),(65,-90)],1):
        ax=fig.add_subplot(2,2,number,projection='3d')
        # Plot (x,z,y) to make raw y vertical.
        ax.scatter(xyz[::3,0],xyz[::3,2],xyz[::3,1],c=rgb[::3],s=.4,depthshade=False)
        ax.set(xlabel='raw x',ylabel='raw z',zlabel='raw y',title=f'elev={elev}, azim={azim}')
        ax.set_box_aspect([2,2.3,.5]);ax.view_init(elev=elev,azim=azim)
    fig.savefig(output / 'back-oblique.png',dpi=160)
    plt.close(fig)
    print(f'Wrote {output}/back-orthographic.png and back-oblique.png; sampled {len(xyz):,} points.')

if __name__ == '__main__':
    main()
