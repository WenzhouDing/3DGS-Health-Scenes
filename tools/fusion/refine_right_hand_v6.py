#!/usr/bin/env python3
"""Frozen-baseline anatomical right hand underside/source registration audit."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pipeline import ROOT, read_scan, transform_gaussians, BODY_PARTS
from render_gaussians import atlas, load_fused

BASE = ROOT / 'raw/fusion-work/refinement-v6/baseline'
OUT = ROOT / 'raw/fusion-work/refinement-v6/right-hand'
CENTER = [-.761, .040, .664]
SPAN = .29
VIEWS = [('Palm', [0, -1, 0]), ('Back', [0, 1, 0]),
         ('Distal toward wrist', [-.63, .2, .77]),
         ('Distal palm', [-.63, -.65, .77]),
         ('Distal back', [-.63, .65, .77]),
         ('Finger edge', [-1, 0, 0])]


def load():
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / 'baseline.npz'
    report = json.loads((BASE / 'report.json').read_text())
    ids = {part['id']: i for i, part in enumerate(report['parts'])}
    if cache.exists():
        data = np.load(cache)
        return data['data'], data['labels'], data['sources'], data['indices'], report
    data, labels, sources, report = load_fused(BASE)
    indices = np.load(BASE / 'source-vertex-indices.npy')
    selected = np.isin(labels, [ids['right_hand'], ids['right_forearm']]) & (data[:, 0] < -.61)
    data, labels, sources, indices = [a[selected] for a in [data, labels, sources, indices]]
    np.savez_compressed(cache, data=data, labels=labels, sources=sources, indices=indices)
    return data, labels, sources, indices, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', action='store_true')
    parser.add_argument('--fit-clean', action='store_true')
    parser.add_argument('--clean', action='store_true')
    parser.add_argument('--full-clean', action='store_true')
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    data, labels, sources, indices, report = load()
    if args.fit:
        fit_candidate(data, labels, sources, indices, report)
        return
    if args.fit_clean:
        fit_candidate(data, labels, sources, indices, report,
                      np.load(OUT / 'cleanup-1.npz')['keep'])
        return
    if args.clean:
        cleanup_trials(data, labels, sources, indices, report)
        return
    if args.full_clean:
        full_cleanup(data, labels, sources, indices, report)
        return
    if args.refine:
        final_trials(data, labels, sources, indices, report)
        return
    if args.export:
        export_recommendation(data, labels, sources, indices, report)
        return
    atlas({'Current fused': data, 'Front contribution': data[sources == 0],
           'Back contribution': data[sources == 1]}, OUT / 'baseline-sources.png',
          CENTER, SPAN, size=850, views=VIEWS,
          title='Anatomical right hand / underside and distal-to-wrist / actual Gaussians')


def fit_candidate(data, labels, sources, indices, report, clean_mask=None):
    from refine_limbs import proxy, fit, evaluate
    from pipeline import part_frame
    ids = {part['id']: i for i, part in enumerate(report['parts'])}
    hand = ids['right_hand']
    lm = json.loads((BASE / 'front-landmarks.json').read_text())['landmarks']
    mask = np.ones(len(data), dtype=bool) if clean_mask is None else clean_mask
    ff = data[(labels == hand) & (sources == 0) & mask].copy()
    bb = data[(labels == hand) & (sources == 1) & mask].copy()
    for si, points in [(0, ff), (1, bb)]:
        original, metadata = read_scan(ROOT / report['sources'][si]['file'])
        assert metadata['sha256'] == report['sources'][si]['sha256']
        points[:, 10] = original[indices[(labels == hand) & (sources == si) & mask], 10]
    fp, fn, fc, _ = proxy(ff, lm, BODY_PARTS[hand])
    bp, bn, bc, _ = proxy(bb, lm, BODY_PARTS[hand])
    _, _, _, depth, axis = part_frame(lm, BODY_PARTS[hand])
    fm = abs(np.einsum('ni,i->n', fn, depth)) < .90
    bm = abs(np.einsum('ni,i->n', bn, depth)) < .90
    fp, fn, fc = fp[fm], fn[fm], fc[fm]
    bp, bn, bc = bp[bm], bn[bm], bc[bm]
    pivot = np.array(lm['right_wrist'])
    baseline = report['parts'][hand]['sourceToFrontRaw']
    trials = []
    before = evaluate(fp, fn, fc, bp, bn, bc)
    for angle in [0, -12, 12]:
        rotation, shift, metrics = fit(fp, fn, fc, bp, bn, bc, pivot, axis, angle)
        score = sum(row['cost'] for row in metrics)
        result = {'score': score, 'rotation': rotation.tolist(), 'shift': shift.tolist(),
                  'degrees': Rotation.from_matrix(rotation).as_euler('xyz', degrees=True).tolist(),
                  'metrics': metrics}
        trials.append(result)
        print('fit', angle, result, flush=True)
    best = min(trials, key=lambda row: row['score'])
    rotation = np.array(best['rotation']); shift = np.array(best['shift'])
    old_r = np.array(baseline['rotation']); old_t = np.array(baseline['translation'])
    final = {'scale': baseline['scale'], 'rotation': (rotation @ old_r).tolist(),
             'translation': (rotation @ (old_t - pivot) + pivot + shift).tolist()}
    result = {'baseline': baseline, 'sourceToFrontRaw': final, 'best': best,
              'before': before, 'trials': trials, 'method': 'symmetric compatible-normal hand-only overlap',
              'sourceHashes': {row['file']: row['sha256'] for row in report['sources']}}
    stem = 'geometry-clean' if clean_mask is not None else 'geometry-candidate'
    (OUT / f'{stem}.json').write_text(json.dumps(result, indent=2) + '\n')
    candidate = data.copy()
    selected = (labels == hand) & (sources == 1)
    candidate[selected] = transform_gaussians(candidate[selected], rotation,
                                              pivot - rotation @ pivot + shift)
    np.save(OUT / f'{stem}.npy', candidate)
    atlas({'Before': data[mask], 'Hand overlap candidate': candidate[mask]}, OUT / f'{stem}-comparison.png',
          CENTER, SPAN, size=850, views=VIEWS,
          title='Anatomical right hand / geometry-only change, unchanged source weighting')


def cleanup_trials(data, labels, sources, indices, report):
    from refine_hand_v4 import visibility
    ids = {part['id']: i for i, part in enumerate(report['parts'])}
    hand = ids['right_hand']
    seen = np.ones(len(data)); opposite_seen = np.ones(len(data))
    for si in [0, 1]:
        selected = (labels == hand) & (sources == si)
        original, meta = read_scan(ROOT / report['sources'][si]['file'])
        assert meta['sha256'] == report['sources'][si]['sha256']
        native = original[indices[selected]]
        _, seen[selected] = visibility(native, direction=[0, -1, 0])
        _, opposite_seen[selected] = visibility(native, direction=[0, 1, 0])
    np.savez_compressed(OUT / 'visibility.npz', seen=seen, opposite_seen=opposite_seen)
    captures = {'Before': data}
    geometry = np.load(OUT / 'geometry-candidate.npy')
    for factor in [3, 1]:
        keep = (labels != hand) | (opposite_seen <= factor * seen)
        np.savez_compressed(OUT / f'cleanup-{factor}.npz', keep=keep)
        captures[f'Visibility shell {factor} / unchanged pose'] = data[keep]
        if factor == 1:
            captures['Visibility shell 1 / overlap fit'] = geometry[keep]
        print('removed', factor, int((~keep).sum()), flush=True)
    atlas(captures, OUT / 'cleanup-trials.png', CENTER, SPAN, size=850,
          views=VIEWS, title='Right hand hidden-layer cleanup / all colors / geometry comparison')


def native_hand(report):
    from pipeline import clean_scan, apply_joint_boundaries, sha256_file
    cfg = json.loads((BASE / 'fusion-config.json').read_text())
    parts = [tuple(list(p[:4]) + cfg.get('partRadii', {}).get(p[0], list(p[4:]))) for p in BODY_PARTS]
    hid = [p[0] for p in parts].index('right_hand')
    cache = OUT / 'native-clean-hand.npz'
    records = {}
    for si, source in enumerate(['front', 'back']):
        lm = json.loads((BASE / f'{source}-landmarks.json').read_text())['landmarks']
        records[source + 'Landmarks'] = lm
    if cache.exists():
        for source in report['sources']:
            if sha256_file(ROOT / source['file']) != source['sha256']:
                raise ValueError('Source scan differs from frozen right-hand baseline')
        c = np.load(cache)
        for name in ['front', 'back', 'frontIndices', 'backIndices']: records[name] = c[name]
        return records, cfg, parts[hid]
    for si, source in enumerate(['front', 'back']):
        original, meta = read_scan(ROOT / cfg[source])
        assert meta['sha256'] == report['sources'][si]['sha256']
        lm = records[source + 'Landmarks']
        clean, lab, _, ids = clean_scan(original, lm, cfg['filter'], parts,
            cfg.get('exclusions', {}).get(source, []), cfg.get('segmentationOverrides', {}).get(source, []),
            cfg['filter'].get('partOverrides', {}).get(source, {}))
        lab, _ = apply_joint_boundaries(clean[:, :3], lab, lm, cfg.get('jointBoundaries', {}).get(source, []), parts)
        selected = lab == hid
        records[source], records[source + 'Indices'] = clean[selected], ids[selected]
        expected = report['parts'][hid][source + 'Before']
        assert int(selected.sum()) == expected, (source, selected.sum(), expected)
        print('full pre-fusion hand', source, int(selected.sum()), flush=True)
    np.savez_compressed(cache, **{name: records[name] for name in ['front', 'back', 'frontIndices', 'backIndices']})
    return records, cfg, parts[hid]


def full_cleanup(data, labels, sources, indices, report):
    from refine_hand_v4 import visibility
    from pipeline import coverage_weights, attenuate
    from cleanup_masks import load_cleanup_masks, keep_source_rows
    native, cfg, part = native_hand(report)
    hid = [p['id'] for p in report['parts']].index('right_hand')
    native_keep = {}
    for source in ['front', 'back']:
        _, seen = visibility(native[source], direction=[0, -1, 0])
        _, other = visibility(native[source], direction=[0, 1, 0])
        native_keep[source] = other <= seen
        np.savez_compressed(OUT / f'{source}-full-visibility.npz', seen=seen, opposite_seen=other,
                            keep=native_keep[source], indices=native[source + 'Indices'])
        print('full excluded', source, int((~native_keep[source]).sum()), flush=True)
    cleanup_masks, _ = load_cleanup_masks(cfg.get('cleanupMasks', []),
        dict(zip(['front', 'back'], report['sources'])), ROOT)
    captures = {'Before': data}
    for name, tr in [('Current rigid pose', report['parts'][hid]['sourceToFrontRaw']),
                     ('Original-overlap hand pose', json.loads((OUT / 'geometry-candidate.json').read_text())['sourceToFrontRaw'])]:
        f = native['front']; b = transform_gaussians(native['back'], np.array(tr['rotation']), np.array(tr['translation']), tr['scale'])
        fw, bw = coverage_weights(f, b, native['frontLandmarks'], part, cfg['fusion'])
        chosen = []
        for source, points, weights in [('front', f, fw), ('back', b, bw)]:
            contribution, retained = attenuate(points, weights, cfg['fusion']['minWeight'])
            keep = native_keep[source][retained] & keep_source_rows(native[source + 'Indices'][retained], cleanup_masks[source])
            chosen.append(contribution[keep])
        captures[name + ' / full visibility cleanup'] = np.concatenate([data[labels != hid], *chosen])
    atlas(captures, OUT / 'full-cleanup-comparison.png', CENTER, SPAN, size=850, views=VIEWS,
          title='Right hand actual pipeline coverage and cleanup / all pre-fusion source hand rows')


def final_trials(data, labels, sources, indices, report):
    from pipeline import coverage_weights, attenuate
    from cleanup_masks import load_cleanup_masks, keep_source_rows
    native, cfg, part = native_hand(report)
    hid = [p['id'] for p in report['parts']].index('right_hand')
    old = report['parts'][hid]['sourceToFrontRaw']
    old_r, old_t = np.array(old['rotation']), np.array(old['translation'])
    wrist_native = np.array(native['backLandmarks']['right_wrist'])
    pivot = old['scale'] * old_r @ wrist_native + old_t
    theta = -1.8151298907672602
    correction = Rotation.from_euler('y', theta, degrees=True).as_matrix()
    final = {'scale': old['scale'], 'rotation': (correction @ old_r).tolist(),
             'translation': (correction @ (old_t - pivot) + pivot).tolist()}
    evidence = {'version': 1, 'part': 'right_hand', 'sourceToFrontRaw': final,
        'baseline': old, 'sourceHashes': {row['file']: row['sha256'] for row in report['sources']},
        'method': 'one planar silhouette angle about the fixed current wrist',
        'correction': {'axisFrontRaw': [0, 1, 0], 'degrees': theta, 'pivotFrontRaw': pivot.tolist(),
                       'pivotNativeBack': wrist_native.tolist(), 'translationDegreesOfFreedom': 0},
        'outlineEvidence': str((OUT / 'fingertip-outline-measurements.json').relative_to(ROOT)),
        'limitation': 'Projected terminal silhouettes constrain this small planar correction; they do not establish a general 3D hand deformation. Large unconstrained overlap fits were rejected.'}
    (OUT / 'wrist-fixed-tip-candidate.json').write_text(json.dumps(evidence, indent=2) + '\n')
    masks, _ = load_cleanup_masks(cfg.get('cleanupMasks', []), dict(zip(['front','back'],report['sources'])), ROOT)
    vis = {s: np.load(OUT / f'{s}-full-visibility.npz') for s in ['front','back']}
    captures = {'Before': data}
    for factor, name, tr in [(1., 'Shell1 / current pose', old), (1., 'Shell1 / fixed-wrist -1.8deg', final),
                              (.5, 'Shell0.5 / fixed-wrist -1.8deg', final), (.25, 'Shell0.25 / fixed-wrist -1.8deg', final)]:
        f = native['front']; b = transform_gaussians(native['back'], np.array(tr['rotation']), np.array(tr['translation']), tr['scale'])
        weights = coverage_weights(f,b,native['frontLandmarks'],part,cfg['fusion'])
        chosen=[];chosen_ids=[];chosen_sources=[]
        for si, source, points, weight in [(0,'front',f,weights[0]),(1,'back',b,weights[1])]:
            contribution, retained = attenuate(points,weight,cfg['fusion']['minWeight'])
            clean = vis[source]['opposite_seen'] <= factor * vis[source]['seen']
            ids = native[source+'Indices'][retained]
            keep = clean[retained] & keep_source_rows(ids,masks[source])
            chosen.append(contribution[keep]);chosen_ids.append(ids[keep]);chosen_sources.append(np.full(keep.sum(),si,np.uint8))
        candidate = np.concatenate([data[labels!=hid],*chosen])
        captures[name] = candidate
        np.savez_compressed(OUT / ('trial-factor'+str(factor)+'-'+('tip' if tr is final else 'current')+'.npz'),
                            data=candidate,handData=np.concatenate(chosen),handIndices=np.concatenate(chosen_ids),
                            handSources=np.concatenate(chosen_sources))
    atlas(captures,OUT/'final-trials.png',CENTER,SPAN,size=850,views=VIEWS,
          title='Right hand / strict source visibility / bounded wrist-fixed silhouette correction')


def export_recommendation(data, labels, sources, indices, report):
    from refine_hand_v4 import visibility
    from pipeline import coverage_weights, attenuate
    from cleanup_masks import load_cleanup_masks, keep_source_rows
    native, cfg, part = native_hand(report)
    hid = [p['id'] for p in report['parts']].index('right_hand')
    hashes = {row['file']: row['sha256'] for row in report['sources']}
    keep_native = {}
    documents = []
    for si, source in enumerate(['front', 'back']):
        # Evaluate every hand row after source quality filtering, BEFORE fusion.
        ratio, seen = visibility(native[source], direction=[0, -1, 0])
        _, opposite = visibility(native[source], direction=[0, 1, 0])
        keep_native[source] = opposite <= seen
        rejected = np.unique(native[source + 'Indices'][~keep_native[source]]).astype(np.uint32)
        mask = OUT / f'right-hand-{source}.npy'
        np.save(mask, rejected)
        evidence = {
            'version': 1, 'purpose': 'remove unobserved interior hand splats that create duplicate fingers from underneath',
            'sourceCapture': source, 'sourceFile': report['sources'][si]['file'],
            'sourceSha256': report['sources'][si]['sha256'], 'sourceHashes': hashes,
            'baseline': str(BASE.relative_to(ROOT)), 'maskFile': str(mask.relative_to(ROOT)),
            'uniqueSourceIndices': len(rejected), 'maskMeaning': 'sorted original source PLY vertex rows',
            'applyStage': 'after coverage confidence weighting and opacity attenuation', 'part': 'right_hand',
            'algorithm': {'method': 'integrated visible alpha contribution from actual anisotropic Gaussian rasterization',
                'sourceRowsEvaluated': len(native[source]), 'population': 'ALL native hand rows after pre-fusion filtering, not exported baseline subset',
                'observedCameraNative': [0, -1, 0], 'oppositeCameraNative': [0, 1, 0],
                'cameraCenter': 'mean native positions of full pre-fusion hand', 'imageSize': 850, 'span': .30,
                'minPixelOpacity': 1 / 255, 'projectionVariancePixels': .3, 'sigmaCutoff': 3,
                'criterion': 'remove when opposite-camera visible alpha flux exceeds observed-camera flux',
                'oppositeToObservedMaximum': 1., 'colorThreshold': None, 'sourceOpacity': 'original sigmoid opacity'},
            'geometryChanged': False, 'colorChanged': False,
            'proof': str((OUT / 'recommended-eight-angles.png').relative_to(ROOT)),
            'reproduce': '.venv-fusion/bin/python -B tools/fusion/refine_right_hand_v6.py --export',
            'rejectedTrials': ['Unconstrained hand ICP can align complementary surfaces incorrectly; the cleaned fit drifted about .05 units.',
                'A wrist-fixed -1.815 degree planar silhouette correction modestly improved three tip projections but added little in the requested underside view; existing accepted wrist pose is retained.',
                'Palm-normal fits vary 14.6 to 34.9 degrees with patch location because palm and dorsum have different curvature. Bounded wrist-fixed plus/minus 3-degree trials moved the layered appearance without a consistent improvement across eight views, and are rejected.',
                'More aggressive source-dominance thresholds or minimum observed visibility removed extra surface opacity without a material additional underside improvement.'],
        }
        metadata_path = mask.with_suffix('.json')
        metadata_path.write_text(json.dumps(evidence, indent=2) + '\n')
        documents.append(str(metadata_path.relative_to(ROOT)))
        print('FINAL MASK', source, len(rejected), flush=True)
    all_masks, _ = load_cleanup_masks(cfg.get('cleanupMasks', []) + documents,
                                      dict(zip(['front', 'back'], report['sources'])), ROOT)
    tr = report['parts'][hid]['sourceToFrontRaw']
    f = native['front']; b = transform_gaussians(native['back'], np.array(tr['rotation']), np.array(tr['translation']), tr['scale'])
    weights = coverage_weights(f, b, native['frontLandmarks'], part, cfg['fusion'])
    chosen = []; retained_ids = []; retained_sources = []
    for si, source, points, weight in [(0, 'front', f, weights[0]), (1, 'back', b, weights[1])]:
        contribution, retained = attenuate(points, weight, cfg['fusion']['minWeight'])
        ids = native[source + 'Indices'][retained]
        keep = keep_source_rows(ids, all_masks[source])
        chosen.append(contribution[keep]); retained_ids.append(ids[keep]); retained_sources.append(np.full(keep.sum(), si, np.uint8))
    context = labels != hid
    candidate = np.concatenate([data[context], *chosen])
    output_labels = np.r_[labels[context], np.full(sum(map(len, chosen)), hid, np.uint8)]
    output_sources = np.concatenate([sources[context], *retained_sources])
    output_indices = np.concatenate([indices[context], *retained_ids])
    np.savez_compressed(OUT / 'candidate.npz', data=candidate, labels=output_labels,
                        sources=output_sources, indices=output_indices)
    result = {'version': 1, 'part': 'right_hand', 'sourceHashes': hashes, 'cleanupMasks': documents,
              'rigidTransformChanged': False, 'sourceToFrontRaw': tr,
              'colorChanged': False, 'coverageConfigurationChanged': False,
              'frontHandRetained': len(chosen[0]), 'backHandRetained': len(chosen[1]),
              'pipelineSimulation': 'exact frozen coverage_weights, attenuate, and existing cleanup masks plus new masks',
              'rejectedNormalTrial': {'accepted': False, 'measuredAngleRangeDegrees': [14.6, 34.9],
                  'boundedAnglesTestedDegrees': [-3, 3],
                  'reason': 'Opposite surfaces have different curvature; neither small normal step consistently improved all reviewed angles.',
                  'proof': str((OUT / 'normal-step-underside.png').relative_to(ROOT))},
              'limitation': 'Source underside/contact quality is incomplete. Some layered appearance and soft edges remain; no rigid correction is supported consistently enough across the reviewed angles.'}
    (OUT / 'recommendation.json').write_text(json.dumps(result, indent=2) + '\n')
    views = VIEWS + [('Below toward above', [0, 0, 1]), ('Below front oblique', [0, -.7, 1])]
    atlas({'Before': data, 'Observed source surfaces': candidate}, OUT / 'recommended-eight-angles.png',
          CENTER, SPAN, size=850, views=views,
          title='Anatomical right hand / accepted wrist pose preserved / full-source hidden-layer cleanup')
    atlas({'Before': data, 'Observed source surfaces': candidate}, OUT / 'recommended-underside-detail.png',
          [-.777, .047, .687], .18, size=950,
          views=[views[i] for i in [2, 3, 6, 7]],
          title='Right hand: distal and below-to-above closeups / original Gaussian rows only')


if __name__ == '__main__':
    main()
