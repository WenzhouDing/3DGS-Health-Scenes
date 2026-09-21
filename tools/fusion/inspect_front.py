#!/usr/bin/env python3
"""Generate six color/depth views of the cleaned front scan without changing it."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mannequin-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_mannequin import FIELDS, read_ply


def main():
    out = ROOT / 'raw/fusion-work'
    out.mkdir(exist_ok=True)
    values, count, _ = read_ply(ROOT / 'raw/mannequin_front_119999.ply')
    data = np.column_stack([values[field] for field in FIELDS])
    opacity = 1 / (1 + np.exp(-data[:, 10]))
    scales = np.exp(data[:, 7:10])
    mask = np.isfinite(data).all(1) & (opacity > .12) & (scales.max(1) < .045)
    # Remove numerically degenerate splats only for the visible diagnostic.
    mask &= scales.max(1) > 1e-5
    sample = data[np.flatnonzero(mask)[::2]]
    xyz = sample[:, :3]
    colors = np.clip(.5 + .28209479177387814 * sample[:, 11:14], 0, 1)
    views = [('Anterior / raw -Y', [0, -1, 0]),
             ('Posterior contact side / raw +Y', [0, 1, 0]),
             ('Side / raw -X', [-1, 0, 0]),
             ('Side / raw +X', [1, 0, 0]),
             ('Anterior oblique', [.85, -1, .20]),
             ('Posterior oblique', [-.85, 1, .20])]
    fig, axes = plt.subplots(2, 3, figsize=(20, 16), facecolor='#20252c')
    for ax, (label, camera) in zip(axes.flat, views):
        camera = np.asarray(camera, float)
        camera /= np.linalg.norm(camera)
        right = np.cross(camera, [0, 0, -1.])
        right /= np.linalg.norm(right)
        up = np.cross(right, camera)
        projected = np.einsum('ij,kj->ik', xyz, np.array([right, up, camera]))
        order = np.argsort(projected[:, 2])
        ax.set_facecolor('#343b44')
        ax.scatter(projected[order, 0], projected[order, 1], c=colors[order],
                   s=.55, linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.grid(alpha=.1)
        ax.set_title(label, color='white', fontsize=13)
        ax.tick_params(colors='#c5cbd3')
        for spine in ax.spines.values():
            spine.set_color('#707780')
    fig.suptitle('Cleaned front scan: six depth-ordered point views', color='white', fontsize=19)
    fig.tight_layout()
    fig.savefig(out / 'front-six-views.png', dpi=170)
    np.save(out / 'front.npy', data)
    stats = {'sourceCount': count, 'diagnosticCount': len(sample),
             'rawBounds': [data[:, :3].min(0).tolist(), data[:, :3].max(0).tolist()],
             'quantileLevels': [0, .01, .1, .5, .9, .99, 1],
             'opacityQuantiles': np.quantile(opacity, [0, .01, .1, .5, .9, .99, 1]).tolist(),
             'maxScaleQuantiles': np.quantile(scales.max(1), [0, .01, .1, .5, .9, .99, 1]).tolist(),
             'scalesAllBelow1eMinus5': int((scales.max(1) < 1e-5).sum())}
    (out / 'front-statistics.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))


if __name__ == '__main__':
    main()
