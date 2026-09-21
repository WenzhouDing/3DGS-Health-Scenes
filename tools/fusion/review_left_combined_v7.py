"""Compare complete observed caps with and without a small digit alignment."""
from pathlib import Path
import numpy as np
from render_gaussians import atlas, render
from audit_left_depth_v7 import load, fused_trial, shift_digits, CENTER, ORIGIN, LONG
from pipeline import dot, ROOT

f, b, c, vis, report = load()
baseline = fused_trial(np.eye(3), np.zeros(3))
out = ROOT / 'raw/fusion-work/refinement-v7/root-audit'
cases = {'Baseline': baseline[0]}
for name, change in [('Caps restored', False), ('Caps and +U .0033', True)]:
    data, labels, sources, indices = fused_trial(np.eye(3), np.zeros(3), shift_digits if change else None)
    rows, labs, captures, ids = [], [], [], []
    for si, (source, original) in enumerate([('front', f), ('back', b)]):
        v = vis[source]
        station = dot(original[:, :3] - ORIGIN, LONG)
        restored = ((station > .105) & (v['seen'] > v['oppositeSeen'])
                    & (v['ratio'] > .05) & (v['seen'] > .05))
        moved = shift_digits(original) if change and si else original
        selected = (labels == 6) & (sources == si)
        current, current_ids = data[selected], indices[selected]
        keep = ~np.isin(current_ids, c[source + 'Indices'][restored])
        rows.extend([current[keep], moved[restored]])
        labs.append(np.full(keep.sum() + restored.sum(), 6, np.uint8))
        captures.append(np.full(keep.sum() + restored.sum(), si, np.uint8))
        ids.extend([current_ids[keep], c[source + 'Indices'][restored]])
    dd = np.concatenate([data[labels != 6], *rows])
    ll = np.concatenate([labels[labels != 6], *labs])
    ss = np.concatenate([sources[labels != 6], *captures])
    ii = np.concatenate([indices[labels != 6], *ids])
    cases[name] = dd
    prefix = 'combined-shift' if change else 'complete-caps'
    np.savez_compressed(out / (prefix + '.npz'), data=dd, labels=ll, sources=ss, indices=ii)
    render(dd, [0, 0, 1], CENTER, .225, width=1250, height=850).save(out / (prefix + '-detail.png'))
atlas(cases, out / 'caps-and-shift.png', CENTER, .33, 750,
      views=[('Under', [0, 0, 1]), ('Distal', [.7, 0, .7]), ('Palm', [0, -1, 0]),
             ('Back', [0, 1, 0]), ('Palm oblique', [.65, -.65, .8]), ('Back oblique', [.65, .65, .8])],
      title='Preserve finger thickness / complete observed caps + small lengthwise alignment')
