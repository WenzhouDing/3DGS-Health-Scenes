"""Compare the left-finger refinement with its frozen V6 export."""
import json
import numpy as np
from pipeline import ROOT, FIELDS, read_ply


def main():
    baseline = ROOT / 'raw/fusion-work/refinement-v7/baseline'
    output = ROOT / 'raw/mannequin-fused'
    records = []
    for directory in [baseline, output]:
        vertices, count, _ = read_ply(directory / 'mannequin_fused.ply')
        records.append((vertices, np.load(directory / 'part-labels.npy'),
                        np.load(directory / 'capture-labels.npy'),
                        np.load(directory / 'source-vertex-indices.npy'),
                        json.loads((directory / 'report.json').read_text())))
    before, after = records
    unchanged = []
    for part, entry in enumerate(before[4]['parts']):
        assert entry['sourceToFrontRaw'] == after[4]['parts'][part]['sourceToFrontRaw']
        if entry['id'] == 'left_hand':
            continue
        a, b = before[1] == part, after[1] == part
        np.testing.assert_array_equal(before[2][a], after[2][b])
        np.testing.assert_array_equal(before[3][a], after[3][b])
        for field in FIELDS:
            np.testing.assert_array_equal(before[0][field][a], after[0][field][b])
        unchanged.append({'part': entry['id'], 'gaussians': int(a.sum()),
                          'allFieldsAndSourceIdsBitExact': True})
    hand = next(i for i, p in enumerate(before[4]['parts']) if p['id'] == 'left_hand')
    hand_counts = {}
    for source in [0, 1]:
        a = (before[1] == hand) & (before[2] == source)
        b = (after[1] == hand) & (after[2] == source)
        common, ia, ib = np.intersect1d(before[3][a], after[3][b], return_indices=True)
        if source == 0:
            for field in FIELDS[:10] + FIELDS[11:]:
                np.testing.assert_array_equal(before[0][field][a][ia], after[0][field][b][ib])
        hand_counts[['front', 'back'][source]] = {
            'before': int(a.sum()), 'after': int(b.sum()),
            'addedSourceRows': int(b.sum() - len(common)),
            'removedSourceRows': int(a.sum() - len(common)),
        }
    result = {
        'passed': True, 'revision': after[4]['parameters']['revision'],
        'baseline': str(baseline.relative_to(ROOT)),
        'all16PartRigidTransformsUnchanged': True,
        'unchangedParts': unchanged,
        'leftHand': hand_counts,
        'frontHandExistingGeometryAndColorUnchanged': True,
        'backFingerLocalCorrection': after[4]['parts'][hand]['digitAlignment'],
        'fullOutputVerification': json.loads((output / 'verification.json').read_text()),
        'review': 'raw/mannequin-fused/left-hand-review/',
    }
    for name in ['left-hand-refinement-verification.json', 'hand-refinement-verification.json']:
        (output / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'unchangedParts': len(unchanged),
                      'leftHand': hand_counts, 'totalGaussians': after[4]['fusedCount']}, indent=2))


if __name__ == '__main__':
    main()
