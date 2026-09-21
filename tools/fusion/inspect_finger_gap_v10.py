#!/usr/bin/env python3
"""Read-only per-finger source geometry and gap-location diagnostics.

All transforms are the frozen V9 single rigid forearm-hand pose. Original source
IDs are retained throughout. Cross-section centers are diagnostics, not a
proposal to collapse complementary palm and dorsal surfaces.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/finger-gap-v10-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, sha256_file, transform_gaussians
from render_gaussians import atlas, load_fused

BASE = ROOT / 'raw/fusion-work/refinement-v10/baseline'
OUT = ROOT / 'raw/fusion-work/refinement-v10/gap-audit'
NATIVE = ROOT / 'raw/fusion-work/refinement-v8/pad-inspection'
ORIGIN = np.array([-.697, .03, .597])
U = np.array([-.626, 0., .7798]); U /= np.linalg.norm(U)
V = np.array([U[2], 0., -U[0]])
BASIS = np.array([V, [0, 1., 0], U])


def local(data):
    return transform_gaussians(data, BASIS, -np.einsum('ij,j->i', BASIS, ORIGIN))


def load():
    OUT.mkdir(parents=True, exist_ok=True)
    report = json.loads((BASE / 'report.json').read_text())
    tr = report['parts'][12]['sourceToFrontRaw']
    if tr != report['parts'][11]['sourceToFrontRaw']:
        raise ValueError('Expected one frozen right forearm/hand rigid pose')
    sources = {}
    data, labels, captures, _ = load_fused(BASE)
    original_ids = np.load(BASE / 'source-vertex-indices.npy')
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
    return sources, report


def inspect(sources, report, render_images):
    fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True, sharey=True)
    for row, station in enumerate([.105, .135, .16, .18]):
        for col in range(2):
            for name, color in [('front', 'tab:blue'), ('back', 'tab:orange')]:
                d = sources[name]
                points = d['local'] if col == 0 else d['fusedLocal']
                mask = abs(points[:, 2] - station) < .003
                if col == 0:
                    mask &= d['strong']
                axes[row, col].scatter(points[mask, 0], points[mask, 1], s=3, alpha=.3, color=color, label=name)
            ax = axes[row, col]
            ax.set_title(f'U={station:g} +/- .003 / ' + ['Full strong observed', 'Final retained'][col])
            ax.set_aspect('equal'); ax.grid(); ax.legend(); ax.set_ylabel('Depth from reference wrist (scan units)')
    for ax in axes[-1]: ax.set_xlabel('Lateral V (scan units)')
    axes[0, 0].set_xlim(-.07, .06); axes[0, 0].set_ylim(-.04, .065)
    fig.suptitle('Finger cross-sections: compare matching lateral edges; palm/back separation alone is physical thickness')
    fig.tight_layout(); fig.savefig(OUT / 'finger-cross-sections.png', dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 9), sharex=True, sharey=True)
    for ax, (name, d) in zip(axes, sources.items()):
        q = d['local']; strong = d['strong']
        lost = strong & ~np.isin(d['ids'], d['fusedIds'])
        ax.scatter(q[strong, 0], q[strong, 2], c=q[strong, 1], s=2, cmap='coolwarm', vmin=-.02, vmax=.055)
        ax.scatter(q[lost, 0], q[lost, 2], facecolors='none', edgecolors='lime', s=8, linewidths=.4, label='Observed but missing')
        ax.set_title(name); ax.set_aspect('equal'); ax.grid(); ax.legend(); ax.set_xlabel('Lateral V'); ax.set_ylabel('Longitudinal U')
    axes[0].invert_yaxis(); fig.tight_layout(); fig.savefig(OUT / 'native-finger-plan.png', dpi=170); plt.close(fig)
    np.savez_compressed(OUT / 'source-geometry.npz', **{
        name + '_' + key: value for name, d in sources.items() for key, value in d.items()})
    evidence = {'sourceHashes': {m['file']: m['sha256'] for m in report['sources']},
                'sourceToFrontRaw': report['parts'][12]['sourceToFrontRaw'],
                'origin': ORIGIN.tolist(), 'axisU': U.tolist(), 'axisV': V.tolist(),
                'normalInterpretation': 'Cross-section blue/orange sheets are complementary; only shared same-side edges or matching cap features constrain registration.',
                'nativeObservedMissing': {name: int(np.sum(d['strong'] & ~np.isin(d['ids'], d['fusedIds']))) for name, d in sources.items()}}
    (OUT / 'gap-geometry.json').write_text(json.dumps(evidence, indent=2) + '\n')
    if render_images:
        rows = {'Current fused': np.concatenate([d['fusedLocal'] for d in sources.values()]),
                'Front full strong observed': sources['front']['local'][sources['front']['strong']],
                'Back full strong observed': sources['back']['local'][sources['back']['strong']],
                'Both full strong observed': np.concatenate([d['local'][d['strong']] for d in sources.values()])}
        atlas(rows, OUT / 'canonical-finger-surfaces.png', [0, .025, .12], .25, size=900,
              views=[('Palm', [0, -1, 0]), ('Back', [0, 1, 0]), ('Along fingers', [0, 0, 1]), ('Outer edge', [-1, 0, 0])],
              title='Frozen rigid forearm-hand pose / V horizontal, U down / original-source surfaces')
    print(json.dumps(evidence, indent=2))


def circle_audit():
    """Fit complementary curved finger arcs, compare inferred shaft centers.

    These are inferred centerline constraints with explicit fit residuals and
    supporting original IDs, not asserted one-to-one Gaussian landmarks.
    """
    cache = np.load(OUT / 'source-geometry.npz')
    report = json.loads((BASE / 'report.json').read_text())
    baseline = report['parts'][12]['sourceToFrontRaw']
    bands = {'index': [-.053, -.028], 'middle': [-.028, .002],
             'ring': [.002, .029], 'little': [.029, .050]}
    sections = {}; support_ids = {}
    for finger, (lo, hi) in bands.items():
        sections[finger] = {}
        for source in ['front', 'back']:
            points = cache[source + '_local']; visible = cache[source + '_strong']
            measurements = []
            stations = [.125, .135, .145, .155, .165] if finger != 'little' else [.110, .1175, .125]
            for station in stations:
                take = visible & (abs(points[:, 2] - station) < .003) & (points[:, 0] > lo) & (points[:, 0] < hi)
                selected = points[take, :2].astype(float)
                if len(selected) < 40:
                    continue
                def residual(parameters):
                    return np.linalg.norm(selected - parameters[:2], axis=1) - parameters[2]
                initial = [np.median(selected[:, 0]), np.median(selected[:, 1]) + (.008 if source == 'front' else -.008), .012]
                fit = least_squares(residual, initial, bounds=([lo, -.03, .004], [hi, .07, .03]), loss='soft_l1', f_scale=.0007)
                error = abs(residual(fit.x))
                accepted = (.0055 < fit.x[2] < .019 and np.median(error) < .0015
                            and np.quantile(selected[:, 0], .95) - np.quantile(selected[:, 0], .05) > .010)
                key = f'{finger}_{source}_{station:g}'
                support_ids[key] = cache[source + '_ids'][take]
                measurements.append({'station': station, 'centerLocalVYU': [*fit.x[:2], station],
                                     'radius': float(fit.x[2]), 'medianRadialResidual': float(np.median(error)),
                                     'p90RadialResidual': float(np.quantile(error, .9)),
                                     'supportCount': len(selected), 'supportIndicesKey': key, 'accepted': bool(accepted)})
            sections[finger][source] = measurements

    candidates = {'baseline': baseline}
    paths = list((ROOT / 'raw/fusion-work/refinement-v10/registration').glob('*.json'))
    paths += list((ROOT / 'raw/fusion-work/refinement-v10/root-pose').glob('*.json'))
    for path in sorted(paths):
        document = json.loads(path.read_text())
        if 'right_hand' in document:
            candidates[path.stem] = document['right_hand']
    results = {}
    R0 = np.array(baseline['rotation']); t0 = np.array(baseline['translation'])
    for name, transform in candidates.items():
        Q = np.array(transform['rotation']) @ R0.T
        translation = np.array(transform['translation']) - np.einsum('ij,j->i', Q, t0)
        rows = []
        for finger, measures in sections.items():
            front = [m for m in measures['front'] if m['accepted']]
            back = [m for m in measures['back'] if m['accepted']]
            if len(front) < 2 or len(back) < 2:
                continue
            back_centers = np.array([m['centerLocalVYU'] for m in back])
            world = np.einsum('ni,ij->nj', back_centers, BASIS) + ORIGIN
            moved = np.einsum('ni,ji->nj', world, Q) + translation
            moved_local = np.einsum('ni,ji->nj', moved - ORIGIN, BASIS)
            order = np.argsort(moved_local[:, 2]); moved_local = moved_local[order]
            back_radii = np.array([m['radius'] for m in back])[order]
            for m in front:
                u = m['station']
                if not moved_local[0, 2] <= u <= moved_local[-1, 2]:
                    continue
                target = np.array(m['centerLocalVYU'])
                center = np.array([np.interp(u, moved_local[:, 2], moved_local[:, 0]),
                                   np.interp(u, moved_local[:, 2], moved_local[:, 1]), u])
                delta = center - target
                combined_radius = m['radius'] + np.interp(u, moved_local[:, 2], back_radii)
                rows.append({'finger': finger, 'station': u, 'deltaV': float(delta[0]),
                             'deltaDepth': float(delta[1]), 'centerSeparation': float(np.linalg.norm(delta[:2])),
                             'inferredCombinedDepthOverMatchedDiameter': float(1 + delta[1] / combined_radius)})
        results[name] = {'sections': rows, 'medianCenterSeparation': float(np.median([p['centerSeparation'] for p in rows])) if rows else None,
                         'perFinger': {f: {'medianDeltaV': float(np.median([p['deltaV'] for p in rows if p['finger'] == f])),
                                          'medianDeltaDepth': float(np.median([p['deltaDepth'] for p in rows if p['finger'] == f])),
                                          'medianCenterSeparation': float(np.median([p['centerSeparation'] for p in rows if p['finger'] == f])),
                                          'inferredCombinedDepthOverMatchedDiameter': float(np.median([p['inferredCombinedDepthOverMatchedDiameter'] for p in rows if p['finger'] == f]))}
                                       for f in bands if any(p['finger'] == f for p in rows)}}
    document = {'method': 'robust circles fitted independently to complementary observed shaft arcs at narrow longitudinal sections; transformed center trajectories compared at common U',
                'limitations': 'Inferred centers assume a locally circular shaft section. Radius-bound, short-span or high-residual fits are rejected. Little-finger caps may not satisfy the shaft model. Do not treat arc-to-arc depth as a zero target.',
                'frame': {'origin': ORIGIN.tolist(), 'axesVYU': BASIS.tolist()}, 'sections': sections, 'candidates': results}
    (OUT / 'shaft-center-audit.json').write_text(json.dumps(document, indent=2) + '\n')
    np.savez_compressed(OUT / 'shaft-center-support-source-indices.npz', **support_ids)
    print(json.dumps({name: {'median': row['medianCenterSeparation'], 'perFinger': row['perFinger']} for name, row in results.items()}, indent=2))


def reslice_audit(candidate_paths):
    """Independently re-slice the moved source, rather than moving old fits.

    Original lateral bands identify each source finger. The narrow cutting
    plane is recomputed in the fixed front frame after each rigid transform.
    This exposes station/obliquity errors in the center-trajectory diagnostic.
    """
    cache = np.load(OUT / 'source-geometry.npz')
    report = json.loads((BASE / 'report.json').read_text())
    baseline = report['parts'][12]['sourceToFrontRaw']
    candidates = {'baseline': baseline}
    for path in candidate_paths:
        path = Path(path)
        candidates[path.stem] = json.loads(path.read_text())['right_hand']
    bands = {'index': [-.053, -.028], 'middle': [-.028, .002], 'ring': [.002, .029]}
    stations = [.135, .145, .155]
    R0 = np.array(baseline['rotation']); t0 = np.array(baseline['translation'])
    results = {}; fits = {}; supporting_ids = {}
    for name, transform in candidates.items():
        Q = np.array(transform['rotation']) @ R0.T
        translation = np.array(transform['translation']) - np.einsum('ij,j->i', Q, t0)
        back_world = np.einsum('ni,ji->nj', cache['back_data'][:, :3], Q) + translation
        back_local = np.einsum('ni,ji->nj', back_world - ORIGIN, BASIS)
        rows = []; fits[name] = {}
        for finger, (lo, hi) in bands.items():
            for station in stations:
                measures = {}
                for source in ['front', 'back']:
                    native_local = cache[source + '_local'][:, :3]
                    current_local = native_local if source == 'front' else back_local
                    take = (cache[source + '_strong'] & (native_local[:, 0] > lo)
                            & (native_local[:, 0] < hi) & (abs(current_local[:, 2] - station) < .003))
                    selected = current_local[take, :2].astype(float)
                    if len(selected) < 40:
                        continue
                    def residual(parameters):
                        return np.linalg.norm(selected - parameters[:2], axis=1) - parameters[2]
                    initial = [np.median(selected[:, 0]), np.median(selected[:, 1]) + (.008 if source == 'front' else -.008), .012]
                    fit = least_squares(residual, initial, bounds=([lo - .02, -.04, .004], [hi + .02, .08, .03]), loss='soft_l1', f_scale=.0007)
                    error = abs(residual(fit.x))
                    key = f'{name}_{finger}_{source}_{station:g}'
                    supporting_ids[key] = cache[source + '_ids'][take]
                    accepted = (.0055 < fit.x[2] < .019 and np.median(error) < .0015
                                and np.quantile(selected[:, 0], .95) - np.quantile(selected[:, 0], .05) > .010)
                    measures[source] = {'centerVY': fit.x[:2].tolist(), 'radius': float(fit.x[2]),
                                        'medianResidual': float(np.median(error)), 'supportCount': len(selected),
                                        'supportIndicesKey': key, 'accepted': bool(accepted)}
                    fits[name][f'{finger}_{source}_{station:g}'] = (selected, fit.x)
                if all(s in measures and measures[s]['accepted'] for s in ['front', 'back']):
                    delta = np.array(measures['back']['centerVY']) - measures['front']['centerVY']
                    rows.append({'finger': finger, 'station': station, 'deltaV': float(delta[0]),
                                 'deltaDepth': float(delta[1]), 'centerSeparation': float(np.linalg.norm(delta)),
                                 'inferredCombinedDepthOverMatchedDiameter': float(1 + delta[1] / (measures['front']['radius'] + measures['back']['radius'])),
                                 'fits': measures})
        results[name] = {'sections': rows, 'medianCenterSeparation': float(np.median([r['centerSeparation'] for r in rows])),
                         'perFinger': {finger: {field: float(np.median([r[field] for r in rows if r['finger'] == finger]))
                                               for field in ['deltaV', 'deltaDepth', 'centerSeparation', 'inferredCombinedDepthOverMatchedDiameter']}
                                       for finger in bands}}
    fig, axes = plt.subplots(len(candidates), 3, figsize=(15, 3.5 * len(candidates)), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 180)
    for row, name in enumerate(candidates):
        for col, station in enumerate(stations):
            ax = axes[row, col]
            for finger in bands:
                for source, color in [('front', 'tab:blue'), ('back', 'tab:orange')]:
                    key = f'{finger}_{source}_{station:g}'
                    if key not in fits[name]:
                        continue
                    points, fit = fits[name][key]
                    ax.scatter(points[:, 0], points[:, 1], s=2, color=color, alpha=.35)
                    ax.plot(fit[0] + fit[2] * np.cos(theta), fit[1] + fit[2] * np.sin(theta), color=color, lw=.7, alpha=.6)
                    ax.plot(fit[0], fit[1], '+', color=color, ms=5)
            display_name = {'baseline': 'Before: V9 rigid pose', 'center-fit': 'After: upstream rigid center fit'}.get(name, name)
            ax.set_title(f'{display_name}\nU={station:g}', fontsize=9); ax.grid(alpha=.2); ax.set_aspect('equal')
            ax.set_xlim(-.06, .04); ax.set_ylim(-.005, .055)
            ax.set_xlabel('Lateral position V (uncalibrated scan units)')
            if col == 0:
                ax.set_ylabel('Depth Y (uncalibrated scan units)')
    fig.suptitle('Observed finger arcs at identical front-frame sections: blue = front, orange = back\nLeft to right within each panel: index, middle, ring. Circles and + centers are independent robust fits.')
    fig.tight_layout(); fig.savefig(OUT / 'transformed-source-cross-sections.png', dpi=170); plt.close(fig)
    (OUT / 'resliced-shaft-center-audit.json').write_text(json.dumps({'method': reslice_audit.__doc__, 'candidates': results}, indent=2) + '\n')
    np.savez_compressed(OUT / 'resliced-shaft-support-source-indices.npz', **supporting_ids)
    print(json.dumps({name: {k: v for k, v in result.items() if k != 'sections'} for name, result in results.items()}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--circle-audit', action='store_true')
    parser.add_argument('--reslice-candidates', nargs='+', type=Path)
    args = parser.parse_args()
    if args.reslice_candidates:
        reslice_audit(args.reslice_candidates)
    elif args.circle_audit:
        circle_audit()
    else:
        inspect(*load(), args.render)
