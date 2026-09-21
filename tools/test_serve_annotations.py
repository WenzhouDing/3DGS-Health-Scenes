"""Exercise local annotation saves against temporary files, never the real scan."""

import copy
from functools import partial
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from serve import ANNOTATION_PATH, MAX_ANNOTATION_BYTES, NoCacheHandler, SCENE_PATH
from http.server import ThreadingHTTPServer


class QuietHandler(NoCacheHandler):
    def log_message(self, *args):
        pass


class AnnotationServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.scene = {
            "parts": [{"id": part} for part in
                      ("torso", "right_upper_arm", "right_forearm", "right_hand", "pelvis")],
            "report": {
                "parameters": {"revision": "test-scan-v1"},
                "sources": [{"file": "raw/front.ply", "sha256": "a" * 64},
                            {"file": "raw/back.ply", "sha256": "b" * 64}],
            },
        }
        manifest = cls.root / SCENE_PATH
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps(cls.scene), encoding="utf-8")
        (cls.root / "static.txt").write_text("local viewer fixture", encoding="utf-8")
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0),
                                        partial(QuietHandler, directory=cls.root))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.port = cls.server.server_port
        cls.origin = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()
        cls.temporary.cleanup()

    def setUp(self):
        self.path = self.root / ANNOTATION_PATH
        self.backup = self.path.with_name("joint-annotations.previous.json")
        self.path.unlink(missing_ok=True)
        self.backup.unlink(missing_ok=True)

    def document(self):
        return {
            "schema": "mannequin-joint-annotations", "version": 1,
            "scene": {
                "revision": "test-scan-v1",
                "sourceHashes": {"raw/front.ply": "a" * 64, "raw/back.ply": "b" * 64},
                "coordinateFrame": "fused-viewer-x-y-z", "units": "scan units",
            },
            "updatedAt": "2026-09-20T18:30:00.000Z",
            "joints": [
                {"id": "right_shoulder", "label": "Right shoulder", "type": "ball",
                 "parentPart": "torso", "childParts": ["right_upper_arm"],
                 "pivot": [-0.4, 0.1, 0.5], "axis": [1, 0, 0],
                 "limits": [[-90, 90], [-90, 90], [-90, 90]],
                 "stiffness": 30, "damping": 2, "notes": "Align to the pad.", "reviewed": True},
                {"id": "right_arm_swivel", "label": "Arm rotation", "type": "swivel",
                 "parentPart": "right_upper_arm", "childParts": ["right_forearm", "right_hand"],
                 "pivot": [-0.5, 0.1, 0.6], "axis": [0.2, 0, 0.8],
                 "limits": [[-180, 180], [0, 0], [0, 0]],
                 "stiffness": 20, "damping": 1, "notes": "No wrist joint.", "reviewed": False},
            ],
        }

    def request(self, method="POST", document=None, raw=None, path="/api/mannequin-joints", headers=None):
        headers = headers if headers is not None else {
            "Origin": self.origin, "Content-Type": "application/json"}
        if document is not None:
            raw = json.dumps(document).encode("utf-8")
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=3)
        try:
            connection.request(method, path, body=raw, headers=headers)
            response = connection.getresponse()
            content = response.read()
            result = json.loads(content) if response.getheader("Content-Type", "").startswith("application/json") else content
            return response.status, result, dict(response.getheaders())
        finally:
            connection.close()

    def test_missing_get_is_useful_and_static_no_cache_is_preserved(self):
        status, content, headers = self.request(method="GET")
        self.assertEqual(status, 404)
        self.assertIn("No saved joint", content["error"])
        self.assertIn("no-store", headers["Cache-Control"])
        status, content, headers = self.request(method="GET", path="/static.txt")
        self.assertEqual((status, content), (200, b"local viewer fixture"))
        self.assertIn("no-store", headers["Cache-Control"])

    def test_save_round_trip_and_previous_backup(self):
        first = self.document()
        status, content, headers = self.request(document=first)
        self.assertEqual(status, 200)
        self.assertEqual(content["path"], ANNOTATION_PATH.as_posix())
        self.assertTrue(content["ok"])
        self.assertTrue(content["savedAt"].endswith("Z"))
        self.assertEqual(self.request(method="GET")[1], first)
        self.assertFalse(self.backup.exists())
        second = copy.deepcopy(first)
        second["joints"][0]["notes"] = "The final pad position."
        self.assertEqual(self.request(document=second)[0], 200)
        self.assertEqual(json.loads(self.path.read_text()), second)
        self.assertEqual(json.loads(self.backup.read_text()), first)
        self.assertEqual(list(self.path.parent.glob(".joint-annotations-*.tmp")), [])

    def test_stale_scene_and_invalid_write_leave_existing_save_and_backup_untouched(self):
        document = self.document()
        self.assertEqual(self.request(document=document)[0], 200)
        self.assertEqual(self.request(document=document)[0], 200)
        previous, previous_backup = self.path.read_bytes(), self.backup.read_bytes()
        for key, value in (("revision", "other-scan"), ("sourceHashes", {}),
                           ("coordinateFrame", "raw-frame"), ("units", "mm")):
            with self.subTest(key=key):
                bad = copy.deepcopy(document)
                bad["scene"][key] = value
                self.assertEqual(self.request(document=bad)[0], 422)
                self.assertEqual(self.path.read_bytes(), previous)
                self.assertEqual(self.backup.read_bytes(), previous_backup)

    def test_cross_origin_missing_origin_and_non_loopback_host_are_rejected(self):
        for origin, host in (("http://example.com", f"127.0.0.1:{self.port}"),
                             ("null", f"127.0.0.1:{self.port}"),
                             (f"http://localhost:{self.port}", f"127.0.0.1:{self.port}"),
                             ("http://localhost:1", f"localhost:{self.port}"),
                             (f"http://evil.test:{self.port}", f"evil.test:{self.port}"),
                             (None, f"127.0.0.1:{self.port}")):
            with self.subTest(origin=origin, host=host):
                headers = {"Host": host, "Content-Type": "application/json"}
                if origin is not None:
                    headers["Origin"] = origin
                self.assertEqual(self.request(document=self.document(), headers=headers)[0], 403)
        self.assertFalse(self.path.exists())

    def test_localhost_same_origin_is_allowed(self):
        headers = {"Host": f"localhost:{self.port}", "Origin": f"http://localhost:{self.port}",
                   "Content-Type": "application/json; charset=utf-8"}
        self.assertEqual(self.request(document=self.document(), headers=headers)[0], 200)

    def test_body_and_content_type_guards(self):
        self.assertEqual(self.request(raw=b"", headers={"Origin": self.origin,
                         "Content-Type": "application/json", "Content-Length": str(MAX_ANNOTATION_BYTES + 1)})[0], 413)
        self.assertEqual(self.request(raw=b"not json")[0], 400)
        self.assertEqual(self.request(raw=b'{}\xff')[0], 400)
        self.assertEqual(self.request(raw=b'{"a":1,"a":2}')[0], 400)
        self.assertEqual(self.request(raw=b'{"pivot":NaN}')[0], 400)
        self.assertEqual(self.request(document=self.document(), headers={"Origin": self.origin,
                         "Content-Type": "text/plain"})[0], 415)
        self.assertFalse(self.path.exists())

    def test_invalid_joint_geometry_types_and_bounds(self):
        invalid = {
            "pivot": [False, 0, 0], "axis": [0, 0, 0], "limits": [[2, 1], [0, 0], [0, 0]],
            "stiffness": -1, "damping": "1", "reviewed": 1, "type": "wrist",
            "id": "../escape", "notes": None, "label": "", "parentPart": "missing",
        }
        for key, value in invalid.items():
            with self.subTest(key=key):
                document = self.document()
                document["joints"][0][key] = value
                self.assertEqual(self.request(document=document)[0], 422)
        document = self.document()
        document["updatedAt"] = "not a timestamp"
        self.assertEqual(self.request(document=document)[0], 422)
        self.assertFalse(self.path.exists())

    def test_invalid_child_ownership_cycles_and_wrist_are_rejected(self):
        cases = []
        duplicate = self.document()
        duplicate["joints"][0]["childParts"].append("right_forearm")
        cases.append(duplicate)
        cycle = self.document()
        cycle["joints"][0]["parentPart"] = "right_forearm"
        cases.append(cycle)
        wrist = self.document()
        wrist["joints"][1]["parentPart"] = "right_forearm"
        wrist["joints"][1]["childParts"] = ["right_hand"]
        cases.append(wrist)
        for document in cases:
            self.assertEqual(self.request(document=document)[0], 422)
        self.assertFalse(self.path.exists())

    def test_endpoint_cannot_choose_an_output_path(self):
        document = self.document()
        document["path"] = "../../escape.json"
        self.assertEqual(self.request(document=document)[0], 200)
        self.assertTrue(self.path.exists())
        self.assertFalse((self.root / "escape.json").exists())
        self.assertEqual(self.request(document=document, path="/api/anything-else")[0], 404)


if __name__ == "__main__":
    unittest.main()
