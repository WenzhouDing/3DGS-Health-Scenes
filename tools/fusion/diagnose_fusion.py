#!/usr/bin/env python3
"""Inspect pre-feather, aligned source proxies without modifying a fusion result.

Requires NumPy and Matplotlib. Input arrays are <part>_front / <part>_back,
each N x 14, in the original front PLY coordinate frame. These diagnostics
show sampled centers, not Gaussian rasterization or calibrated uncertainty.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/mannequin-mpl')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
FRONT_COLOR = '#40d8ed'
BACK_COLOR = '#ffad40'
BG = '#202730'
PANEL = '#2d3641'
VIEWS = [('Anterior, −Y', [0, -1, 0]), ('Posterior, +Y', [0, 1, 0]),
         ('Right side, −X', [-1, 0, 0]), ('Left side, +X', [1, 0, 0]),
         ('Anterior oblique', [.85, -1, .2]), ('Posterior oblique', [-.85, 1, .2])]
SECTION_PARTS = [('head', 'neck', 'head', [.35, 1., 1.45]),
                 ('torso', 'neck', 'waist', [.20, .50, .80]),
                 ('left_forearm', 'left_elbow', 'left_wrist', [.20, .50, .80]),
                 ('right_forearm', 'right_elbow', 'right_wrist', [.20, .50, .80]),
                 ('left_shin', 'left_knee', 'left_ankle', [.20, .50, .80]),
                 ('right_shin', 'right_knee', 'right_ankle', [.20, .50, .80])]


def style(ax):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors='#bac4d0', labelsize=8)
    ax.grid(alpha=.12)
    for spine in ax.spines.values():
        spine.set_color('#637182')


def projected(xyz, camera):
    camera = np.asarray(camera, dtype=float)
    camera /= np.linalg.norm(camera)
    right = np.cross(camera, [0, 0, -1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, camera)
    return np.einsum('ij,kj->ik', xyz, np.array([right, up, camera]))


def source_legend(fig, at=.955):
    handles = [Line2D([], [], color=FRONT_COLOR, marker='o', ls='', label='Front capture'),
               Line2D([], [], color=BACK_COLOR, marker='o', ls='', label='Back capture, aligned')]
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, at),
                        ncol=2, frameon=False)
    for text in legend.get_texts():
        text.set_color('white')


def six_views(captures, output, source_colors=False):
    arrays = [np.concatenate([pair[source] for pair in captures.values()
                              if len(pair[source])], axis=0) for source in ('front', 'back')]
    data = np.concatenate(arrays)
    source = np.r_[np.zeros(len(arrays[0]), dtype=np.uint8), np.ones(len(arrays[1]), dtype=np.uint8)]
    colors = np.clip(.5 + .28209479177387814 * data[:, 11:14], 0, 1)
    fig, axes = plt.subplots(2, 3, figsize=(18, 15), facecolor=BG)
    for ax, (name, direction) in zip(axes.flat, VIEWS):
        p = projected(data[:, :3], direction)
        style(ax)
        if source_colors:
            # Interleave capture order to avoid putting one entire capture on top.
            order = np.random.default_rng(731).permutation(len(data))
            palette = np.asarray([matplotlib.colors.to_rgb(FRONT_COLOR),
                                  matplotlib.colors.to_rgb(BACK_COLOR)])
            ax.scatter(p[order, 0], p[order, 1], c=palette[source[order]],
                       s=.42, alpha=.34, linewidths=0, rasterized=True)
        else:
            order = np.argsort(p[:, 2], kind='stable')
            ax.scatter(p[order, 0], p[order, 1], c=colors[order],
                       s=.68, linewidths=0, rasterized=True)
        ax.set_aspect('equal')
        ax.set_title(name, color='white', fontsize=12)
        ax.set_xlabel('Projected horizontal / source units', color='#bac4d0', fontsize=8)
        ax.set_ylabel('Projected head-up / source units', color='#bac4d0', fontsize=8)
    if source_colors:
        title = 'Aligned sources: transparent superposition from six directions'
        subtitle = 'Pre-feather audit centers; hidden surfaces are also shown. Colors identify capture, not visibility or confidence.'
        source_legend(fig, at=.955)
        filename = 'source-superposition-six-views.png'
    else:
        title = 'Aligned geometry and captured color from six directions'
        subtitle = 'Depth-ordered, pre-feather audit points. This sparse point proxy is not a render of the final Gaussians.'
        filename = 'natural-color-six-views.png'
    fig.suptitle(title, color='white', fontsize=18, y=.994)
    fig.text(.5, .972, subtitle, color='#c4ceda', ha='center', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .927 if source_colors else .95])
    fig.savefig(output / filename, dpi=170)
    plt.close(fig)


def section_views(captures, landmarks, output, halfwidth):
    definitions = list(SECTION_PARTS)
    if 'skull_base' in landmarks and 'crown' in landmarks:
        definitions[0] = ('head', 'skull_base', 'crown', [.15, .50, .88])
        if 'neck' in captures:
            definitions.insert(1, ('neck', 'neck', 'skull_base', [.20, .50, .80]))
    fig, axes = plt.subplots(len(definitions), 3, figsize=(14, 3.5 * len(definitions) + 2), facecolor=BG)
    metrics = []
    for row, (part, start, end, fractions) in enumerate(definitions):
        a, b = np.asarray(landmarks[start]), np.asarray(landmarks[end])
        direction = b - a
        length = np.linalg.norm(direction)
        z = direction / length
        y = np.array([0., 1., 0.])
        y -= z * np.dot(y, z)
        y /= np.linalg.norm(y)
        x = np.cross(y, z)
        local = {source: np.einsum('ij,kj->ik', captures[part][source][:, :3] - a,
                                  np.array([x, y, z])) for source in ('front', 'back')}
        combined = np.concatenate(list(local.values()))
        limits = np.maximum(np.quantile(np.abs(combined[:, :2]), .995, axis=0) * 1.12, .045)
        for col, fraction in enumerate(fractions):
            ax = axes[row, col]
            style(ax)
            counts = {}
            for source, color in [('front', FRONT_COLOR), ('back', BACK_COLOR)]:
                points = local[source]
                slab = points[np.abs(points[:, 2] - fraction * length) <= halfwidth]
                counts[source] = len(slab)
                ax.scatter(slab[:, 0], slab[:, 1], c=color, s=2.3,
                           alpha=.53, linewidths=0, rasterized=True)
            ax.axhline(0, color='white', alpha=.20, lw=.7)
            ax.axvline(0, color='white', alpha=.20, lw=.7)
            ax.set_aspect('equal')
            ax.set_xlim(-limits[0], limits[0])
            ax.set_ylim(-limits[1], limits[1])
            center = a + fraction * direction
            ax.set_title(f"{part.replace('_', ' ')} · raw Z ≈ {center[2]:+.3f}\n"
                         f"front n={counts['front']:,} / back n={counts['back']:,}",
                         color='white', fontsize=10)
            ax.set_xlabel('Local lateral / source units', color='#bac4d0', fontsize=8)
            ax.set_ylabel('Local depth (+ posterior)', color='#bac4d0', fontsize=8)
            if not sum(counts.values()):
                ax.text(.5, .5, 'No audit points in this slab', transform=ax.transAxes,
                        color='#d0d8e2', ha='center', va='center', fontsize=9)
            metrics.append({'part': part, 'axisFraction': fraction,
                            'centerFrontRaw': center.tolist(), 'halfWidth': halfwidth,
                            'sampleCounts': counts})
    fig.suptitle('Same-part cross-sections: thickness and alignment checks', color='white', fontsize=17, y=.996)
    fig.text(.5, .979, f'Slabs ±{halfwidth:.3f} source units. Opposite anterior/posterior surfaces should remain separated.',
             color='#c4ceda', ha='center', fontsize=10)
    fig.text(.5, .966, 'Sparse audit counts are sampling support, not visibility, opacity, confidence, or calibrated uncertainty.',
             color='#c4ceda', ha='center', fontsize=9)
    source_legend(fig, at=.956)
    fig.tight_layout(rect=[0, 0, 1, .938])
    fig.savefig(output / 'same-part-cross-sections.png', dpi=165)
    plt.close(fig)
    return metrics


def constraint_plot(report, output):
    parts = report.get('parts', [])
    if not parts:
        return []
    fig, axes = plt.subplots(1, 3, figsize=(17, 9), facecolor=BG)
    y = np.arange(len(parts))
    labels = [part['id'].replace('_', ' ') for part in parts]
    residuals = [max(part.get('endpointResiduals', [float('nan')])) for part in parts]
    pairs = [part.get('finalPairs', part.get('initialPairs', 0)) for part in parts]
    normalized = []
    for part in parts:
        values = part.get('normalizedSingularValues')
        normalized.append(float(min(values)) if values else float('nan'))
    for ax in axes:
        style(ax)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
    axes[0].barh(y, residuals, color=FRONT_COLOR)
    axes[0].set_title('Largest endpoint residual', color='white')
    axes[0].set_xlabel('Source units; landmark estimates also have error', color='#bac4d0', fontsize=9)
    axes[1].barh(y, pairs, color=BACK_COLOR)
    axes[1].set_title('Eligible same-surface overlap pairs', color='white')
    axes[1].set_xlabel('Count; does not measure whole-body agreement', color='#bac4d0', fontsize=9)
    axes[2].barh(y, np.nan_to_num(normalized), color='#b48df5')
    axes[2].set_xlim(0, 1)
    axes[2].set_title('Normalized constraint strength σmin / σmax', color='white')
    axes[2].set_xlabel('Near 0 = poorly constrained direction; near 1 = balanced', color='#bac4d0', fontsize=8)
    for index, value in enumerate(normalized):
        if not np.isfinite(value):
            axes[2].text(.025, index, 'not estimated', va='center', color='#c4ceda', fontsize=8)
    fig.suptitle('Alignment support and normalized uncertainty indicators', color='white', fontsize=17, y=.98)
    fig.text(.5, .946, 'These are residual/support/conditioning diagnostics. No statistical uncertainty, confidence probability, or metric calibration is claimed.',
             color='#c4ceda', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, .927])
    fig.savefig(output / 'alignment-support.png', dpi=165)
    plt.close(fig)
    return [{'part': part['id'], 'maximumEndpointResidual': residual,
             'sameSurfacePairs': count,
             'normalizedConstraintStrength': strength if np.isfinite(strength) else None}
            for part, residual, count, strength in zip(parts, residuals, pairs, normalized)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=ROOT / 'raw/mannequin-fused/alignment-audit.npz')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--landmarks', type=Path, default=ROOT / 'tools/fusion/front-landmarks.json')
    parser.add_argument('--report', type=Path)
    parser.add_argument('--section-halfwidth', type=float, default=.018)
    args = parser.parse_args()
    if args.section_halfwidth <= 0:
        parser.error('--section-halfwidth must be positive')
    output = args.out or args.input.parent / 'diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    landmarks_path = args.landmarks
    if not landmarks_path.exists() and landmarks_path == ROOT / 'tools/fusion/front-landmarks.json':
        landmarks_path = ROOT / 'raw/fusion-work/front-landmarks.json'
    landmarks = json.loads(landmarks_path.read_text())['landmarks']
    report_path = args.report or args.input.parent / 'report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    with np.load(args.input, allow_pickle=False) as archive:
        parts = [key[:-6] for key in archive.files if key.endswith('_front')]
        captures = {}
        for part in parts:
            captures[part] = {}
            for source in ('front', 'back'):
                array = np.asarray(archive[f'{part}_{source}'])
                if array.ndim != 2 or array.shape[1] != 14 or not np.isfinite(array).all():
                    raise ValueError(f'{part}_{source} must contain finite N x 14 Gaussian records')
                captures[part][source] = array
    if not captures:
        raise ValueError('No <part>_front arrays found in alignment audit')
    missing = set(part[0] for part in SECTION_PARTS) - captures.keys()
    if missing:
        raise ValueError(f'Missing cross-section parts: {sorted(missing)}')
    six_views(captures, output, source_colors=False)
    six_views(captures, output, source_colors=True)
    sections = section_views(captures, landmarks, output, args.section_halfwidth)
    constraints = constraint_plot(report, output)
    summary = {'input': str(args.input), 'frame': 'original front scan raw coordinates',
               'dataStage': 'aligned audit before complementary-source feathering',
               'limitations': ['Sparse point-center projections are not a final Gaussian render.',
                               'Cyan/amber identify source capture, not visibility or statistical confidence.',
                               'Opposite body surfaces should not be collapsed onto one another.',
                               'Normalized singular values express conditioning, not calibrated uncertainty.'],
               'sampleCounts': {part: {source: len(data) for source, data in pair.items()}
                                for part, pair in captures.items()},
               'crossSections': sections, 'alignmentSupport': constraints}
    (output / 'diagnostic-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Wrote fusion diagnostics to {output}')


if __name__ == '__main__':
    main()
