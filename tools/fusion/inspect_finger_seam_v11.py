#!/usr/bin/env python3
"""Independent source-native, covariance-aware right finger seam audit.

Only writes diagnostic artifacts. No registration, source mutation or masking.
"""
import argparse
import json
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR', '/tmp/finger-seam-v11-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, sha256_file, transform_gaussians
from render_gaussians import atlas, load_fused, _raster, render
from scipy.ndimage import distance_transform_edt
from PIL import Image, ImageDraw

BASE = ROOT / 'raw/fusion-work/refinement-v11/baseline'
OUT = ROOT / 'raw/fusion-work/refinement-v11/gap-audit'
NATIVE = ROOT / 'raw/fusion-work/refinement-v8/pad-inspection'
ORIGIN = np.array([-.697, .03, .597])
U = np.array([-.626, 0., .7798]); U /= np.linalg.norm(U)
V = np.array([U[2], 0., -U[0]])
BASIS = np.array([V, [0, 1., 0], U])
BANDS = {'index': [-.053, -.028], 'middle': [-.028, .002],
         'ring': [.002, .029], 'little': [.029, .050]}
# Wider partitions at the empty space between shafts, not the conservative
# center-fit bands above. A support audit must not truncate lateral edges.
SUPPORT_BANDS = {'index': [-.060, -.0235], 'middle': [-.0235, .0055],
                 'ring': [.0055, .0325]}


def local(data):
    return transform_gaussians(data, BASIS, -np.einsum('ij,j->i', BASIS, ORIGIN))


def load():
    OUT.mkdir(parents=True, exist_ok=True)
    data, labels, captures, report = load_fused(BASE)
    tr = report['parts'][12]['sourceToFrontRaw']
    assert tr == report['parts'][11]['sourceToFrontRaw']
    original_ids = np.load(BASE / 'source-vertex-indices.npy')
    sources = {}
    for si, name in enumerate(['front', 'back']):
        meta = report['sources'][si]
        if sha256_file(ROOT / meta['file']) != meta['sha256']:
            raise ValueError('Source capture hash mismatch')
        native = np.load(NATIVE / f'{name}-full-native.npz')
        hand = native['labels'] == 12
        raw = native['data'][hand]; ids = native['indices'][hand]
        aligned = raw if si == 0 else transform_gaussians(raw, np.array(tr['rotation']), np.array(tr['translation']), tr['scale'])
        vis = np.load(ROOT / f'raw/fusion-work/refinement-v6/right-hand/{name}-full-visibility.npz')
        if not np.array_equal(ids, vis['indices']):
            raise ValueError('Native visibility index mismatch')
        retained = (captures == si) & (labels == 12)
        current = data[retained]; current_ids = original_ids[retained]
        strong = (vis['seen'] > vis['opposite_seen']) & (vis['seen'] > .05) & (vis['observed_ratio'] > .03)
        sources[name] = {'data': aligned, 'local': local(aligned), 'ids': ids,
                         'fused': current, 'fusedLocal': local(current), 'fusedIds': current_ids,
                         'strong': strong, 'seen': vis['seen'], 'opposite': vis['opposite_seen'],
                         'ratio': vis['observed_ratio']}
    np.savez_compressed(OUT / 'source-geometry.npz', **{name + '_' + key: value for name, d in sources.items() for key, value in d.items()})
    return sources, report


def covariance(data):
    q = data[:, 3:7].astype(float)
    R = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_matrix()
    a = R * np.exp(data[:, None, 7:10])
    return np.einsum('nik,njk->nij', a, a)


def conditional_alpha(data, cov, station, extent, size=500):
    """Cross-section of source Gaussian ellipsoids, including native opacity.

    This is a 3D density support diagnostic, not a replacement for the actual
    perspective render. Conditional covariance preserves radial thinness.
    """
    variance = np.maximum(cov[:, 2, 2], 1e-14)
    du = station - data[:, 2]
    alpha = 1 / (1 + np.exp(-np.clip(data[:, 10], -40, 40)))
    alpha = alpha * np.exp(-.5 * du * du / variance)
    keep = alpha > 1 / 255
    c = cov[keep]; var = variance[keep]; du = du[keep]
    mean = data[keep, :2] + c[:, :2, 2] * (du / var)[:, None]
    conditional = c[:, :2, :2] - np.einsum('ni,nj->nij', c[:, :2, 2], c[:, :2, 2]) / var[:, None, None]
    lo_v, hi_v, lo_y, hi_y = extent
    px = size / (hi_v - lo_v)
    assert abs((hi_y - lo_y) - (hi_v - lo_v)) < 1e-9
    C = conditional * px * px
    C[:, 0, 0] += .1; C[:, 1, 1] += .1
    det = C[:, 0, 0] * C[:, 1, 1] - C[:, 0, 1] ** 2
    inverse = np.c_[C[:, 1, 1], -C[:, 0, 1], C[:, 0, 0]] / det[:, None]
    radii = 3 * np.sqrt(np.c_[C[:, 0, 0], C[:, 1, 1]])
    xy = (mean - [lo_v, lo_y]) * px
    image = _raster(np.ascontiguousarray(xy), np.ascontiguousarray(inverse), np.ascontiguousarray(radii),
                    np.ones((len(mean), 3)), np.ascontiguousarray(alpha[keep]), size, size, np.zeros(3))
    return image[:, :, 0], 1 / px


def side_support(sources, prefix='retained'):
    """Measure real Gaussian support at the two lateral sides of each shaft."""
    stations = [.12, .135, .15, .165]
    rows = []; evidence_ids = {}; fig, axes = plt.subplots(4, 3, figsize=(12, 14))
    for col, (finger, (lo, hi)) in enumerate(SUPPORT_BANDS.items()):
        center_v = (lo + hi) / 2
        for row, station in enumerate(stations):
            # Frame contains the whole shaft and both sides, with equal axes.
            extent = (center_v - .025, center_v + .025, .005, .055)
            fields = {}; selected_data = {}
            for source, d in sources.items():
                p = d['fusedLocal']; take = (p[:, 0] > lo) & (p[:, 0] < hi)
                selected_data[source] = p[take]
                evidence_ids[f'{finger}_{source}'] = d['fusedIds'][take]
                fields[source], pixel = conditional_alpha(p[take], covariance(p[take]), station, extent)
            # White is joint support, source colors are independent occupancy.
            rgb = np.zeros((*fields['front'].shape, 3))
            rgb[:, :, 2] = fields['front']; rgb[:, :, 0] = fields['back']
            rgb[:, :, 1] = .6 * np.minimum(fields['front'], fields['back'])
            ax = axes[row, col]; ax.imshow(rgb, extent=extent, origin='lower')
            ax.set_title(f'{finger}, U={station:g}'); ax.set_aspect('equal')
            ax.set_xlabel('V'); ax.set_ylabel('Y')
            Vgrid = extent[0] + (np.arange(500) + .5) * pixel
            # Use measured common lateral center at this section, not midpoint
            # of the segmentation band, which differs for curved fingers.
            side_centers = []
            for source, p in selected_data.items():
                near = abs(p[:, 2] - station) < .002
                if near.sum(): side_centers.append(np.median(p[near, 0]))
            vc = np.mean(side_centers)
            for side, sign in [('minusV', -1), ('plusV', 1)]:
                lateral = sign * (Vgrid - vc) > .0035
                front_mask = (fields['front'] > .1) & lateral[None, :]
                back_mask = (fields['back'] > .1) & lateral[None, :]
                if np.any(front_mask) and np.any(back_mask):
                    distances, indices = distance_transform_edt(~front_mask, return_indices=True)
                    where = np.argwhere(back_mask)
                    best = where[np.argmin(distances[back_mask])]
                    nearest = indices[:, best[0], best[1]]
                    gap = float(distances[tuple(best)] * pixel)
                    pback = [extent[0] + (best[1]+.5)*pixel, extent[2]+(best[0]+.5)*pixel]
                    pfront = [extent[0] + (nearest[1]+.5)*pixel, extent[2]+(nearest[0]+.5)*pixel]
                    ax.plot([pfront[0], pback[0]], [pfront[1], pback[1]], color='yellow', lw=1.2)
                    rows.append({'finger': finger, 'station': station, 'side': side,
                                 'conditionalSupportGapAtAlpha0_1': gap, 'frontBoundaryVY': pfront, 'backBoundaryVY': pback})
    fig.suptitle(f'{prefix}: retained Gaussian ellipsoid sections: blue=front, red=back; yellow=nearest same-side support\nIncludes full covariance and actual opacity. Occupancy threshold .1; this is a 3D support diagnostic, not a photograph.')
    fig.tight_layout(); fig.savefig(OUT / f'{prefix}-side-covariance-support.png', dpi=160); plt.close(fig)
    document = {'method': side_support.__doc__, 'threshold': .1, 'sections': rows,
                'limitations': 'Gaussian density support is not a calibrated solid surface. Narrow plane intersections are sensitive to source splat thickness. Compare actual grazing render and retained/full observed source ablations.'}
    (OUT / f'{prefix}-side-covariance-support.json').write_text(json.dumps(document, indent=2) + '\n')
    np.savez_compressed(OUT / f'{prefix}-shaft-side-support-source-indices.npz', **evidence_ids)
    print(json.dumps(rows, indent=2))


def grazing_views(sources):
    data = np.concatenate([d['fused'] for d in sources.values()])
    white = data.copy(); white[:, 11:14] = .5 / .28209479177387814
    center = ORIGIN + np.einsum('i,ij->j', np.array([-.006, .028, .14]), BASIS)
    views = [('Plus side palm grazing', [1, -.1, .15]), ('Minus side palm grazing', [-1, -.1, .15]),
             ('Plus side palm lean', [1, -.3, .15]), ('Minus side palm lean', [-1, -.3, .15]),
             ('Plus side back grazing', [1, .1, .15]), ('Minus side back grazing', [-1, .1, .15]),
             ('Plus distal grazing', [.8, -.1, .6]), ('Minus distal grazing', [-.8, -.1, .6])]
    atlas({'V10 native colors': data, 'V10 identical alpha / white': white}, OUT / 'grazing-color-and-white.png',
          center, .23, size=850, views=views,
          title='True Gaussian grazing renders: white preserves identical means, covariance and alpha; distinguishes color crease from coverage slit')


def isolated_depth(sources):
    data = np.concatenate([d['fusedLocal'] for d in sources.values()])
    canvas = Image.new('RGB', (3 * 750, 5 * 778 + 52), (18, 23, 28)); draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), 'Isolated native finger rays: depth shades near-side V bright and far-side V dark; exact original Gaussian covariance/alpha', fill='white')
    rows = [('Native color +V', [1, -.05, 0], 'color'), ('White alpha +V', [1, -.05, 0], 'white'),
            ('Depth +V', [1, -.05, 0], 'depth'), ('Depth -V', [-1, -.05, 0], 'depth'),
            ('Native color -V', [-1, -.05, 0], 'color')]
    for col, (finger, (lo, hi)) in enumerate(SUPPORT_BANDS.items()):
        take = (data[:, 0] > lo) & (data[:, 0] < hi) & (data[:, 2] > .105) & (data[:, 2] < .198)
        original = data[take]; center = [(lo + hi) / 2, .03, .15]
        for row, (name, direction, kind) in enumerate(rows):
            candidate = original.copy()
            if kind == 'white': candidate[:, 11:14] = .5 / .28209479177387814
            if kind == 'depth':
                level = .12 + .78 * np.clip((original[:, 0] - lo) / (hi - lo), 0, 1)
                if direction[0] < 0: level = 1 - level
                candidate[:, 11:14] = (level[:, None] - .5) / .28209479177387814
            img = render(candidate, direction, center, .11, width=750, height=750)
            y = 52 + row * 778; canvas.paste(img, (col * 750, y + 28))
            draw.text((col * 750 + 12, y + 8), f'{finger}: {name}', fill='white')
    canvas.save(OUT / 'isolated-finger-depth.png')


def compare_isolated(sources, candidate_path):
    baseline = np.concatenate([d['fusedLocal'] for d in sources.values()])
    n = np.load(candidate_path); candidate = local(n['data'][n['labels'] == 12])
    canvas = Image.new('RGB', (3 * 750, 4 * 778 + 52), (18, 23, 28)); draw = ImageDraw.Draw(canvas)
    draw.text((12, 12), 'Isolated +V near-side seam: exact Gaussian color/alpha/covariance; depth exposes far-wall visibility', fill='white')
    rows = [('Baseline color', baseline, False), (candidate_path.stem + ' color', candidate, False),
            ('Baseline depth', baseline, True), (candidate_path.stem + ' depth', candidate, True)]
    for col, (finger, (lo, hi)) in enumerate(SUPPORT_BANDS.items()):
        for row, (name, data, depth) in enumerate(rows):
            take = (data[:, 0] > lo) & (data[:, 0] < hi) & (data[:, 2] > .105) & (data[:, 2] < .198)
            selected = data[take].copy()
            if depth:
                level = .12 + .78 * np.clip((selected[:, 0] - lo) / (hi - lo), 0, 1)
                selected[:, 11:14] = (level[:, None] - .5) / .28209479177387814
            img = render(selected, [1, -.05, 0], [(lo + hi)/2, .03, .15], .11, width=750, height=750)
            y = 52 + row * 778; canvas.paste(img, (col * 750, y + 28))
            draw.text((col * 750 + 12, y + 8), f'{finger}: {name}', fill='white')
    canvas.save(OUT / f'{candidate_path.stem}-isolated-comparison.png')


def inspect(sources, report, render_images):
    fig, axes = plt.subplots(1, 2, figsize=(12, 10), sharex=True, sharey=True)
    for ax, (name, d) in zip(axes, sources.items()):
        q = d['local']; strong = d['strong']
        lost = strong & ~np.isin(d['ids'], d['fusedIds'])
        ax.scatter(q[strong, 0], q[strong, 2], c=q[strong, 1], s=2, cmap='coolwarm', vmin=.005, vmax=.055)
        ax.scatter(q[lost, 0], q[lost, 2], facecolors='none', edgecolors='lime', s=8, linewidths=.4, label='Observed but missing')
        for finger, band in BANDS.items():
            ax.text(np.mean(band), .215, finger, ha='center')
        for u in [.12, .135, .15, .165, .18, .19, .20]: ax.axhline(u, color='gray', lw=.4)
        ax.set_title(name); ax.set_aspect('equal'); ax.grid(); ax.legend(); ax.set_xlabel('Lateral V'); ax.set_ylabel('Longitudinal U')
    axes[0].set_xlim(-.075, .060); axes[0].set_ylim(.22, .06)
    fig.suptitle('V10 source observations in common rigid pose: color=depth, green=excluded original rows')
    fig.tight_layout(); fig.savefig(OUT / 'native-finger-plan.png', dpi=170); plt.close(fig)
    stations = [.12, .135, .15, .165, .18, .19, .20]
    fig, axes = plt.subplots(len(stations), 2, figsize=(12, 3 * len(stations)), sharex=True, sharey=True)
    for row, station in enumerate(stations):
        for col in range(2):
            ax = axes[row, col]
            for name, color in [('front', 'tab:blue'), ('back', 'tab:orange')]:
                d = sources[name]; points = d['local'] if col == 0 else d['fusedLocal']
                mask = abs(points[:, 2] - station) < .002
                if col == 0: mask &= d['strong']
                ax.scatter(points[mask, 0], points[mask, 1], s=3, alpha=.45, color=color, label=name)
            ax.set_title(f'U={station:g} +/- .002 / ' + ['Full strong observed', 'Final retained'][col])
            ax.set_aspect('equal'); ax.grid(); ax.legend(); ax.set_ylabel('Depth Y')
    for ax in axes[-1]: ax.set_xlabel('Lateral V')
    axes[0, 0].set_xlim(-.065, .055); axes[0, 0].set_ylim(-.005, .065)
    fig.suptitle('Actual native means at fresh common-frame slices: identify same-side edge joins, not opposing palm/back planes')
    fig.tight_layout(); fig.savefig(OUT / 'finger-sections.png', dpi=150); plt.close(fig)
    cap_support = {}; caps = {}
    for finger, (lo, hi) in BANDS.items():
        caps[finger] = {}
        for name, d in sources.items():
            q = d['local']; mask = d['strong'] & (q[:, 0] > lo) & (q[:, 0] < hi) & (q[:, 2] > .11)
            cutoff = np.quantile(q[mask, 2], .95)
            selected = mask & (q[:, 2] >= cutoff)
            cap_support[f'{finger}_{name}'] = d['ids'][selected]
            caps[finger][name] = {'q95U': float(cutoff), 'capMedianVYU': np.median(q[selected, :3], 0).tolist(),
                                 'capCount': int(selected.sum()), 'capRetained': int(np.isin(d['ids'][selected], d['fusedIds']).sum())}
    np.savez_compressed(OUT / 'cap-support-source-indices.npz', **cap_support)
    evidence = {'sourceHashes': {m['file']: m['sha256'] for m in report['sources']},
                'sourceToFrontRaw': report['parts'][12]['sourceToFrontRaw'],
                'frame': {'origin': ORIGIN.tolist(), 'axesVYU': BASIS.tolist()},
                'caps': caps, 'warning': 'Cap medians combine complementary sides; depth difference is not a zero target. Native source IDs identify real observations.'}
    (OUT / 'seam-geometry.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if render_images:
        rows = {'Current fused': np.concatenate([d['fusedLocal'] for d in sources.values()]),
                'Front retained': sources['front']['fusedLocal'],
                'Back retained': sources['back']['fusedLocal'],
                'Both full strong observed': np.concatenate([d['local'][d['strong']] for d in sources.values()])}
        atlas(rows, OUT / 'canonical-finger-seam.png', [0, .025, .14], .21, size=800,
              views=[('Along fingers', [0, 0, 1]), ('Distal low front', [0, -.28, 1]),
                     ('Distal low back', [0, .28, 1]), ('Outer edge', [-1, 0, 0]),
                     ('Palm', [0, -1, 0]), ('Back', [0, 1, 0])],
              title='V10 exact anisotropic Gaussian contributions / finger seam audit')
    print(json.dumps(caps, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--side-support', action='store_true')
    parser.add_argument('--grazing', action='store_true')
    parser.add_argument('--isolated-depth', action='store_true')
    parser.add_argument('--side-candidate', type=Path)
    parser.add_argument('--isolated-compare', type=Path)
    args = parser.parse_args()
    if args.isolated_compare:
        source_data, _ = load()
        compare_isolated(source_data, args.isolated_compare)
    elif args.side_candidate:
        n = np.load(args.side_candidate)
        sources = {}
        for si, name in enumerate(['front', 'back']):
            take = (n['labels'] == 12) & (n['sources'] == si)
            sources[name] = {'fusedLocal': local(n['data'][take]), 'fusedIds': n['indices'][take]}
        side_support(sources, prefix=args.side_candidate.stem)
    elif args.isolated_depth:
        source_data, _ = load()
        isolated_depth(source_data)
    elif args.grazing:
        source_data, _ = load()
        grazing_views(source_data)
    elif args.side_support:
        source_data, _ = load()
        side_support(source_data)
    else:
        inspect(*load(), args.render)
