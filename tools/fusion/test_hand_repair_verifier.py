"""Regression guards for source-only hand restoration verification."""
import copy
import unittest
import numpy as np
from pipeline import BODY_PARTS, FIELDS
from verify_arm_gap_v10 import verify_hand_only_changes


class HandRepairScopeTests(unittest.TestCase):
    def setUp(self):
        data = np.zeros(32, dtype=[(field, '<f4') for field in FIELDS])
        for i, field in enumerate(FIELDS):
            data[field] = np.arange(32, dtype=np.float32) * .001 + i * .01
        parts = [{'id': part[0], 'sourceToFrontRaw': {
            'rotation': np.eye(3).tolist(), 'translation': [0., 0., 0.], 'scale': 1.}}
            for part in BODY_PARTS]
        self.before = (data, np.tile(np.arange(16, dtype=np.uint8), 2),
                       np.repeat(np.arange(2, dtype=np.uint8), 16),
                       np.r_[np.arange(16), np.arange(100, 116)].astype(np.uint32),
                       {'parts': parts})
        self.after = copy.deepcopy(self.before)
        self.restored = {s: np.empty(0, np.uint32) for s in ['front', 'back']}
        self.colors = copy.deepcopy(self.restored)
        self.pad = np.array([111, 112], np.uint32)

    def check(self):
        return verify_hand_only_changes(self.before, self.after, self.restored, self.colors, self.pad)

    def append(self, source_id=40):
        arrays = list(self.after)
        arrays[0] = np.r_[arrays[0], arrays[0][[12]]]
        for i, value in [(1, 12), (2, 0), (3, source_id)]:
            arrays[i] = np.r_[arrays[i], np.array([value], dtype=arrays[i].dtype)]
        self.after = tuple(arrays)

    def test_accepts_declared_changes_and_reports_exact_scope(self):
        self.append()
        self.restored['front'] = np.array([12, 40], np.uint32)
        self.colors['front'] = np.array([12], np.uint32)
        self.after[0][FIELDS[10]][12] += .1
        self.after[0][FIELDS[11]][12] += .1
        proof = self.check()
        self.assertEqual(proof['allRigidTransformsBitExactToBaseline'], 16)
        self.assertEqual(proof['measuredBackPadRowsBitExactToBaseline'], 2)
        self.assertEqual(proof['handSourceChanges'][0]['newRowsInsideDeclaredRestoration'], 1)
        self.assertEqual(proof['handSourceChanges'][0]['retainedOpacityChangesInsideDeclaredRestoration'], 1)
        self.assertEqual(proof['handSourceChanges'][0]['retainedColorChangesInsideDeclaredRepair'], 1)

    def test_rejects_pose_drift_even_on_allowed_hand(self):
        self.after[4]['parts'][12]['sourceToFrontRaw']['translation'][0] = .0001
        with self.assertRaises(AssertionError): self.check()

    def test_rejects_undeclared_added_rows(self):
        self.append()
        with self.assertRaises(AssertionError): self.check()

    def test_rejects_undeclared_opacity_and_color_changes(self):
        for field in [FIELDS[10], FIELDS[11]]:
            with self.subTest(field=field):
                self.after = copy.deepcopy(self.before)
                self.after[0][field][12] += .01
                with self.assertRaises(AssertionError): self.check()

    def test_rejects_mean_and_covariance_edits(self):
        for field in [FIELDS[0], FIELDS[3], FIELDS[7]]:
            with self.subTest(field=field):
                self.after = copy.deepcopy(self.before)
                self.after[0][field][12] += .0001
                with self.assertRaises(AssertionError): self.check()

    def test_rejects_removed_or_duplicated_source_rows(self):
        keep = np.arange(32) != 12
        self.after = tuple(a[keep] if i < 4 else a for i, a in enumerate(self.after))
        with self.assertRaises(AssertionError): self.check()
        self.after = copy.deepcopy(self.before)
        self.append(source_id=13)
        self.restored['front'] = np.array([13], np.uint32)
        with self.assertRaises(AssertionError): self.check()

    def test_rejects_part_crossing_or_pad_changes(self):
        self.restored['front'] = np.array([13], np.uint32)
        with self.assertRaises(AssertionError): self.check()
        self.restored['front'] = np.empty(0, np.uint32)
        self.after[0][FIELDS[11]][27] += .01  # Back forearm row on measured pad.
        with self.assertRaises(AssertionError): self.check()


if __name__ == '__main__': unittest.main()
