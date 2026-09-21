"""Verify the complete right forearm and hand share one rigid source transform."""
import json
import numpy as np
from pipeline import ROOT, FIELDS, read_ply, transform_gaussians
from inspect_pad_v8 import verify_pad, OUT as PAD_AUDIT


def load(folder):
    vertices, _, _ = read_ply(folder / 'mannequin_fused.ply')
    return (vertices, np.load(folder / 'part-labels.npy'),
            np.load(folder / 'capture-labels.npy'),
            np.load(folder / 'source-vertex-indices.npy'),
            json.loads((folder / 'report.json').read_text()))


def main():
    baseline = ROOT / 'raw/fusion-work/refinement-v9/baseline'
    output = ROOT / 'raw/mannequin-fused'
    before, after = load(baseline), load(output)
    report = after[4]
    by_name = {p['id']: p for p in report['parts']}
    assert report['parameters']['rigidParts']['right_hand'] == 'right_forearm'
    assert not report['parameters'].get('rightWristRegistration')
    forearm = by_name['right_forearm']['sourceToFrontRaw']
    assert by_name['right_hand']['sourceToFrontRaw'] == forearm
    assert by_name['right_hand']['rigidWithPart'] == 'right_forearm'
    unchanged = []
    for label, part in enumerate(before[4]['parts']):
        if part['id'] in ['right_forearm', 'right_hand']:
            continue
        assert part['sourceToFrontRaw'] == report['parts'][label]['sourceToFrontRaw']
        a, b = before[1] == label, after[1] == label
        for i in [2, 3]:
            np.testing.assert_array_equal(before[i][a], after[i][b])
        for name in FIELDS:
            np.testing.assert_array_equal(before[0][name][a], after[0][name][b])
        unchanged.append({'part': part['id'], 'count': int(a.sum()),
                          'allFieldsSourceIdsAndRigidPoseBitExact': True})
    checks = []
    pairs = np.load(ROOT / 'raw/fusion-work/refinement-v8/pad-inspection/pad-cross-part-pairs.npz')
    for capture, meta in enumerate(report['sources']):
        raw, _, _ = read_ply(ROOT / meta['file'])
        rows = np.flatnonzero(np.isin(after[1], [11, 12]) & (after[2] == capture))
        original = np.column_stack([raw[k][after[3][rows]] for k in FIELDS])
        expected = original
        if capture:
            expected = transform_gaussians(original, np.array(forearm['rotation']),
                                           np.array(forearm['translation']), forearm['scale'])
        expected = transform_gaussians(expected, np.diag([1., -1., -1.]), np.zeros(3))
        actual = np.column_stack([after[0][k][rows] for k in FIELDS])
        np.testing.assert_allclose(actual[:, :10], expected[:, :10], atol=2e-7, rtol=0)
        np.testing.assert_array_equal(actual[:, 11:], expected[:, 11:])
        checks.append({'capture': meta['file'], 'allExportedArmAndHandRowsChecked': len(rows),
                       'maximumMeanAndCovarianceParameterError': float(np.abs(actual[:, :10] - expected[:, :10]).max()),
                       'noWristOrKnuckleDeformation': True})
        if capture:
            lookup = {int(after[3][i]): i for i in rows}
            retained = [(int(a), int(b)) for a, b in zip(pairs['forearmSourceIndices'], pairs['handSourceIndices'])
                        if int(a) in lookup and int(b) in lookup]
            ids_a, ids_b = np.array(retained).T
            native_a = np.column_stack([raw[k][ids_a] for k in FIELDS[:3]])
            native_b = np.column_stack([raw[k][ids_b] for k in FIELDS[:3]])
            ia, ib = [lookup[int(a)] for a in ids_a], [lookup[int(b)] for b in ids_b]
            actual_a = np.column_stack([after[0][k][ia] for k in FIELDS[:3]])
            actual_b = np.column_stack([after[0][k][ib] for k in FIELDS[:3]])
            expected_dist = np.linalg.norm(native_a - native_b, axis=1) * forearm['scale']
            actual_dist = np.linalg.norm(actual_a - actual_b, axis=1)
            error = np.abs(expected_dist - actual_dist)
            assert error.max() < 2e-7
            pair_check = {'retainedCrossBoundaryPairs': len(retained),
                          'maximumDistanceError': float(error.max()),
                          'meaning': 'Original distances across the hand/forearm label boundary are preserved at the shared scan scale.'}
    verify_pad(output)
    result = {'passed': True, 'revision': report['parameters']['revision'],
              'baseline': str(baseline.relative_to(ROOT)), 'gaussianCount': report['fusedCount'],
              'unchangedParts': unchanged, 'rigidAssembly': ['right_forearm', 'right_hand'],
              'sourceToFrontRaw': forearm, 'allRetainedAssemblyRows': checks,
              'padBoundaryDistanceCheck': pair_check,
              'rigidPad': json.loads((PAD_AUDIT / 'verification-mannequin-fused.json').read_text()),
              'fullOutputVerification': json.loads((output / 'verification.json').read_text()),
              'review': 'raw/mannequin-fused/rigid-wrist-review/'}
    for name in ['right-wrist-verification.json', 'hand-refinement-verification.json']:
        (output / name).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'unchangedParts': len(unchanged),
                      'assemblyChecks': checks, 'padBoundary': pair_check}, indent=2))


if __name__ == '__main__':
    main()
