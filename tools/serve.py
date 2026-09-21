#!/usr/bin/env python3
"""Static file server and local mannequin annotation storage.

Localhost-only file server that sends `no-store`. Safari
in particular caches the viewer's JS and wasm hard enough to keep serving a stale copy
after you have already fixed something, which makes changes look like they had no
effect. Use this while iterating on `viewers/`; plain `http.server` is fine otherwise.

    python3 tools/serve.py            # http://localhost:8000
    python3 tools/serve.py 8765       # pick a port
"""

import ipaddress
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ANNOTATION_PATH = Path("raw/mannequin-fused/joint-annotations.json")
SCENE_PATH = Path("viewers/mannequin-fusion/fusion.json")
MAX_ANNOTATION_BYTES = 512 * 1024
SAVE_LOCK = threading.Lock()


def read_json(data):
    """Reject duplicate keys and JavaScript-incompatible nonfinite numbers."""
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"Invalid JSON number: {value}")

    return json.loads(data, object_pairs_hook=unique_keys, parse_constant=reject_constant)


def validate_annotations(document, scene):
    """Validate the saved contract against the exact currently displayed scan."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def finite(value, low, high):
        return (type(value) in (int, float) and math.isfinite(value)
                and low <= value <= high)

    require(isinstance(document, dict), "Annotations must be a JSON object.")
    require(document.get("schema") == "mannequin-joint-annotations",
            "Unrecognized annotation schema.")
    require(type(document.get("version")) is int and document["version"] == 1,
            "Only annotation version 1 is supported.")
    signature = document.get("scene")
    require(isinstance(signature, dict), "Missing scene identity.")
    require(signature.get("revision") == scene["report"]["parameters"]["revision"],
            "These annotations belong to a different scan revision. Reload the page first.")
    hashes = {source["file"]: source["sha256"] for source in scene["report"]["sources"]}
    require(signature.get("sourceHashes") == hashes,
            "Annotation source hashes do not match the current scan.")
    require(signature.get("coordinateFrame") == "fused-viewer-x-y-z",
            "Annotations must use fused-viewer-x-y-z coordinates.")
    require(signature.get("units") == "scan units", "Annotations must use scan units.")
    updated = document.get("updatedAt")
    require(isinstance(updated, str) and len(updated) <= 64
            and re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", updated),
            "updatedAt must be an ISO timestamp.")
    try:
        timestamp = datetime.fromisoformat(updated.replace("Z", "+00:00"))
        require(timestamp.tzinfo is not None, "updatedAt must include a timezone.")
    except (ValueError, TypeError):
        raise ValueError("updatedAt must be an ISO timestamp with a timezone.") from None

    joints = document.get("joints")
    require(isinstance(joints, list) and 1 <= len(joints) <= 128,
            "Annotations must contain 1 to 128 joints.")
    part_ids = {part["id"] for part in scene["parts"]}
    joint_ids, owned_parts = set(), set()
    graph = {part: [] for part in part_ids}
    for number, joint in enumerate(joints, 1):
        require(isinstance(joint, dict), f"Joint {number} must be an object.")
        identifier = joint.get("id")
        require(isinstance(identifier, str)
                and re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", identifier),
                f"Joint {number} has an invalid id.")
        require(identifier not in joint_ids, f"Duplicate joint id: {identifier}.")
        joint_ids.add(identifier)
        prefix = f"Joint {identifier}: "
        label, notes = joint.get("label"), joint.get("notes")
        require(isinstance(label, str) and len(label.strip()) > 0 and len(label) <= 120,
                prefix + "label must contain 1 to 120 characters.")
        require(isinstance(notes, str) and len(notes) <= 10000,
                prefix + "notes must be text of at most 10000 characters.")
        require(joint.get("type") in ("ball", "hinge", "swivel", "fixed"),
                prefix + "type must be ball, hinge, swivel, or fixed.")
        parent, children = joint.get("parentPart"), joint.get("childParts")
        require(isinstance(parent, str) and parent in part_ids,
                prefix + "parent part is not in the current scan.")
        require(isinstance(children, list) and len(children) > 0
                and all(isinstance(child, str) and child in part_ids for child in children),
                prefix + "choose at least one valid child part.")
        require(len(set(children)) == len(children), prefix + "child parts must be unique.")
        require(parent not in children, prefix + "parent cannot also be a child.")
        require(not owned_parts.intersection(children),
                prefix + "a child part already belongs to another joint.")
        for side in ("left", "right"):
            require((f"{side}_hand" in children) == (f"{side}_forearm" in children),
                    prefix + "the hand and forearm must remain one rigid child group.")
        owned_parts.update(children)
        graph[parent].extend(children)
        pivot, axis = joint.get("pivot"), joint.get("axis")
        require(isinstance(pivot, list) and len(pivot) == 3
                and all(finite(value, -100, 100) for value in pivot),
                prefix + "pivot must contain three finite coordinates within ±100 scan units.")
        require(isinstance(axis, list) and len(axis) == 3
                and all(finite(value, -sys.float_info.max, sys.float_info.max) for value in axis)
                and math.isfinite(math.hypot(*axis)) and math.hypot(*axis) > 1e-8,
                prefix + "axis must contain three finite values and be nonzero.")
        limits = joint.get("limits")
        require(isinstance(limits, list) and len(limits) == 3
                and all(isinstance(pair, list) and len(pair) == 2
                        and all(finite(value, -360, 360) for value in pair)
                        and pair[0] <= pair[1] for pair in limits),
                prefix + "limits must be three ordered degree pairs within ±360.")
        for property_name in ("stiffness", "damping"):
            require(finite(joint.get(property_name), 0, 100000),
                    prefix + property_name + " must be between 0 and 100000.")
        require(type(joint.get("reviewed")) is bool, prefix + "reviewed must be true or false.")

    visiting, visited = set(), set()

    def visit(part):
        require(part not in visiting, "Joint parent/child connections contain a cycle.")
        if part in visited:
            return
        visiting.add(part)
        for child in graph[part]:
            visit(child)
        visiting.remove(part)
        visited.add(part)

    for part in part_ids:
        visit(part)


def atomic_write(path, data):
    """Replace a fixed local file without exposing a partly written document."""
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".joint-annotations-",
                                         suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class NoCacheHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory=None, **kwargs):
        # Anchoring to the script also makes running it from another cwd reliable.
        super().__init__(*args, directory=str(directory or WORKSPACE_ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        super().end_headers()

    def json_response(self, status, payload):
        data = (json.dumps(payload, allow_nan=False) + "\n").encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if urlsplit(self.path).path != "/api/mannequin-joints":
            return super().do_GET()
        path = Path(self.directory) / ANNOTATION_PATH
        try:
            with SAVE_LOCK:
                document = read_json(path.read_text(encoding="utf-8"))
            self.json_response(200, document)
        except FileNotFoundError:
            self.json_response(404, {"ok": False, "error": "No saved joint annotations yet."})
        except (OSError, ValueError):
            self.json_response(500, {"ok": False, "error": "The saved annotations could not be read."})

    def same_origin(self):
        """Block cross-origin writes, including local DNS-rebinding hostnames."""
        hosts, origins = self.headers.get_all("Host", []), self.headers.get_all("Origin", [])
        if len(hosts) != 1 or len(origins) != 1:
            return False
        try:
            host = urlsplit("http://" + hosts[0])
            origin = urlsplit(origins[0])
            if host.username or host.password or host.path or host.query or host.fragment:
                return False
            hostname = host.hostname
            if hostname != "localhost":
                address = ipaddress.ip_address(hostname)
                if address.version != 4 or not address.is_loopback:
                    return False
            return (host.port or 80) == self.server.server_port and (
                origin.scheme == "http" and origin.hostname == hostname
                and (origin.port or 80) == self.server.server_port
                and not origin.username and not origin.password
                and not origin.path and not origin.query and not origin.fragment)
        except (ValueError, TypeError):
            return False

    def do_POST(self):
        if urlsplit(self.path).path != "/api/mannequin-joints":
            self.json_response(404, {"ok": False, "error": "Unknown local API endpoint."})
            return
        if not self.same_origin():
            self.json_response(403, {"ok": False, "error": "Save annotations from this local server's page."})
            return
        content_types = self.headers.get_all("Content-Type", [])
        if len(content_types) != 1 or self.headers.get_content_type() != "application/json":
            self.json_response(415, {"ok": False, "error": "Send annotations as application/json."})
            return
        lengths = self.headers.get_all("Content-Length", [])
        if self.headers.get("Transfer-Encoding") or len(lengths) != 1 or not lengths[0].isdigit():
            self.json_response(400, {"ok": False, "error": "A valid Content-Length is required."})
            return
        length = int(lengths[0])
        if length > MAX_ANNOTATION_BYTES:
            self.json_response(413, {"ok": False, "error": "Annotations must be smaller than 512 KB."})
            return
        try:
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete annotation request.")
            document = read_json(raw.decode("utf-8"))
        except (ValueError, UnicodeError, RecursionError):
            self.json_response(400, {"ok": False, "error": "Annotations must contain valid UTF-8 JSON."})
            return
        try:
            scene = read_json((Path(self.directory) / SCENE_PATH).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.json_response(503, {"ok": False, "error": "The current scan identity could not be loaded."})
            return
        try:
            validate_annotations(document, scene)
            serialized = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
        except (ValueError, TypeError, OverflowError, RecursionError) as error:
            self.json_response(422, {"ok": False, "error": str(error)})
            return
        path = Path(self.directory) / ANNOTATION_PATH
        try:
            with SAVE_LOCK:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    atomic_write(path.with_name("joint-annotations.previous.json"), path.read_bytes())
                atomic_write(path, serialized)
        except OSError:
            self.json_response(500, {"ok": False, "error": "Could not save annotations to the local workspace."})
            return
        self.json_response(200, {"ok": True, "path": ANNOTATION_PATH.as_posix(),
                                 "savedAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"serving on http://localhost:{port}  (no-store)")
    print("press ctrl-c to stop")
    try:
        ThreadingHTTPServer(("127.0.0.1", port), NoCacheHandler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
