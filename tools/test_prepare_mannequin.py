"""Run with: python3 -m unittest discover -s tools -p 'test_prepare_mannequin.py'."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from prepare_mannequin import FIELDS, main


class PrepareMannequinTests(unittest.TestCase):
    def write_ply(self, path, records, endian="<"):
        # Reversed properties and an unrelated normal exercise name-based parsing.
        names = list(reversed(FIELDS)) + ["nx"]
        dtype = np.dtype([(name, endian + "f4") for name in names])
        values = np.zeros(len(records), dtype=dtype)
        for column, name in enumerate(FIELDS):
            values[name] = records[:, column]
        fmt = "binary_little_endian" if endian == "<" else "binary_big_endian"
        header = "ply\nformat " + fmt + " 1.0\nelement vertex " + str(len(records)) + "\n"
        header += "".join("property float " + name + "\n" for name in names) + "end_header\n"
        path.write_bytes(header.encode() + values.tobytes())

    def run_prepare(self, source, output, *extra):
        with contextlib.redirect_stdout(io.StringIO()):
            return main(["--input", str(source), "--output", str(output), *extra])

    def test_field_order_endianness_rotation_and_source_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = np.zeros((3, 14), dtype=np.float32)
            records[:, :3] = [[.1, -.2, .3], [.4, .1, 1], [-.4, 0, .6]]
            records[:, 3:7] = [[1, 0, 0, 0], [2, 3, 4, 5], [0, 0, 0, 0]]
            records[:, 7:10] = -5
            records[:, 10] = 1
            for endian in ("<", ">"):
                with self.subTest(endian=endian):
                    source = root / "input.ply"
                    self.write_ply(source, records, endian)
                    original_bytes = source.read_bytes()
                    output = root / "output"
                    meta = self.run_prepare(source, output, "--max-splats", "0")
                    packed = np.fromfile(output / "scan.bin", dtype="<f4").reshape(-1, 14)
                    np.testing.assert_allclose(packed[:, :3], records[:, :3] * [1, -1, -1])
                    np.testing.assert_allclose(packed[0, 3:7], [0, 1, 0, 0])
                    np.testing.assert_allclose(packed[1, 3:7], np.array([-3, 2, -5, 4]) / np.sqrt(54))
                    np.testing.assert_allclose(packed[2, 3:7], [0, 1, 0, 0])
                    np.testing.assert_array_equal(packed[:, 7:], records[:, 7:])
                    self.assertEqual(meta["count"], 3)
                    self.assertEqual(meta["degenerateQuaternionsReplaced"], 1)
                    self.assertEqual(source.read_bytes(), original_bytes)
                    self.assertEqual(json.loads((output / "scan.json").read_text())["fields"], FIELDS)

    def test_filtering_deterministic_sampling_and_no_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records = np.zeros((8, 14), dtype=np.float32)
            records[:, 0] = np.arange(8) * .1
            records[:, 3] = 1
            records[:, 7:10] = -5
            records[:, 10] = 1
            records[1, 10] = -10  # Almost invisible.
            records[2, 7] = 0  # Excessive scale.
            records[3, 1] = 10  # Beyond the broad scene bounds.
            records[4, 12] = np.nan  # Always invalid, even with --no-filter.
            source = root / "input.ply"
            self.write_ply(source, records)
            output = root / "output"
            meta = self.run_prepare(source, output, "--max-splats", "2")
            self.assertEqual(meta["finiteCount"], 7)
            self.assertEqual(meta["filteredCount"], 4)
            packed = np.fromfile(output / "scan.bin", dtype="<f4").reshape(-1, 14)
            np.testing.assert_allclose(packed[:, 0], [0, .7])
            first = (output / "scan.bin").read_bytes()
            self.run_prepare(source, output, "--max-splats", "2")
            self.assertEqual((output / "scan.bin").read_bytes(), first)
            meta = self.run_prepare(source, output, "--max-splats", "0", "--no-filter")
            self.assertEqual(meta["count"], 7)


if __name__ == "__main__":
    unittest.main()
