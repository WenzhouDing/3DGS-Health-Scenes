#!/usr/bin/env python3
"""Bake measured ambulance contact proxies into the viewer's camera collider.

No Gaussian occupancy is used. MeshCollision.fromGlb in the bundled viewer
ignores node transforms, so all vertices are baked in ambulance world space.
The source config also drives the independent articulated body contact checker.
"""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import struct


def add(a, b): return [x + y for x, y in zip(a, b)]
def sub(a, b): return [x - y for x, y in zip(a, b)]
def mul(a, t): return [x * t for x in a]
def dot(a, b): return sum(x * y for x, y in zip(a, b))
def norm(a): return math.sqrt(dot(a, a))
def cross(a, b): return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]


def vec(a, size=3):
    assert len(a) == size and all(isinstance(v, (int, float)) and math.isfinite(v) for v in a)
    return [float(v) for v in a]


def plane_polygon(proxy):
    normal = vec(proxy['normal'])
    size = norm(normal)
    assert size > 1e-12
    normal = mul(normal, 1 / size)
    offset = float(proxy['offset']) / size
    assert math.isfinite(offset)
    lo, hi = [vec(p) for p in proxy['bounds']]
    assert all(a <= b for a, b in zip(lo, hi))
    corners = [[hi[i] if bits & (1 << i) else lo[i] for i in range(3)] for bits in range(8)]
    points = []

    def keep(p):
        if not any(norm(sub(q, p)) < 1e-8 for q in points):
            points.append(p)

    for a in range(8):
        for axis in range(3):
            b = a ^ (1 << axis)
            if b < a:
                continue
            da, db = dot(normal, corners[a]) - offset, dot(normal, corners[b]) - offset
            if abs(da) < 1e-10:
                keep(corners[a])
            if abs(db) < 1e-10:
                keep(corners[b])
            if da * db < 0:
                keep(add(corners[a], mul(sub(corners[b], corners[a]), da / (da-db))))
    assert len(points) >= 3, f"{proxy['id']}: plane does not intersect finite bounds"
    center = [sum(p[i] for p in points) / len(points) for i in range(3)]
    seed = [1, 0, 0] if abs(normal[0]) < .8 else [0, 1, 0]
    u = cross(normal, seed)
    u = mul(u, 1 / norm(u))
    v = cross(normal, u)
    points.sort(key=lambda p: math.atan2(dot(sub(p, center), v), dot(sub(p, center), u)))
    return points, [[0, i, i+1] for i in range(1, len(points)-1)]


def rotate(q, p):
    t = mul(cross(q[:3], p), 2)
    return add(p, add(mul(t, q[3]), cross(q[:3], t)))


def box_mesh(proxy):
    center, half = vec(proxy['center']), vec(proxy['halfExtents'])
    assert all(v > 0 for v in half)
    q = vec(proxy.get('rotation', [0, 0, 0, 1]), 4)
    assert norm(q) > 1e-12
    q = mul(q, 1 / norm(q))
    points = [add(center, rotate(q, [half[i] * sign[i] for i in range(3)]))
              for sign in itertools.product([-1, 1], repeat=3)]
    # Every quad is oriented outward; node transform remains identity.
    quads = [[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1],
             [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]]
    triangles = [t for a, b, c, d in quads for t in ([a, b, c], [a, c, d])]
    return points, triangles


def build(config):
    collision = config.get('collision', config)
    points, triangles, ranges = [], [], []
    ids = set()
    for kind, field, fn in [('surface', 'surfaces', plane_polygon), ('box', 'boxes', box_mesh)]:
        for proxy in collision.get(field, []):
            assert proxy['id'] and proxy['id'] not in ids, 'Duplicate/empty proxy ID'
            ids.add(proxy['id'])
            pp, tt = fn(proxy)
            start = len(points)
            ranges.append({'id': proxy['id'], 'type': kind, 'vertex_start': start,
                           'vertex_count': len(pp), 'triangle_start': len(triangles), 'triangle_count': len(tt)})
            points.extend(pp)
            triangles.extend([[v + start for v in t] for t in tt])
    assert points and triangles, 'No contact geometry'
    for a, b, c in triangles:
        assert norm(cross(sub(points[b], points[a]), sub(points[c], points[a]))) > 1e-12
    positions = struct.pack('<' + 'f' * (len(points) * 3), *(v for p in points for v in p))
    indices = struct.pack('<' + 'I' * (len(triangles) * 3), *(v for t in triangles for v in t))
    binary = positions + indices
    bounds = [[min(p[i] for p in points) for i in range(3)], [max(p[i] for p in points) for i in range(3)]]
    document = {
        'asset': {'version': '2.0', 'generator': 'build_contact_mesh.py; world-space measured surface proxies'},
        'scene': 0, 'scenes': [{'nodes': [0]}], 'nodes': [{'name': 'Ambulance measured contact surfaces', 'mesh': 0}],
        'meshes': [{'primitives': [{'attributes': {'POSITION': 0}, 'indices': 1, 'mode': 4, 'material': 0}]}],
        'materials': [{'doubleSided': True, 'pbrMetallicRoughness': {'baseColorFactor': [.6, .7, .8, 1], 'metallicFactor': 0, 'roughnessFactor': 1}}],
        'buffers': [{'byteLength': len(binary)}],
        'bufferViews': [{'buffer': 0, 'byteOffset': 0, 'byteLength': len(positions), 'target': 34962},
                        {'buffer': 0, 'byteOffset': len(positions), 'byteLength': len(indices), 'target': 34963}],
        'accessors': [{'bufferView': 0, 'componentType': 5126, 'count': len(points), 'type': 'VEC3', 'min': bounds[0], 'max': bounds[1]},
                      {'bufferView': 1, 'componentType': 5125, 'count': len(triangles) * 3, 'type': 'SCALAR'}],
        'extras': {'units': 'uncalibrated ambulance world scene units', 'proxyRanges': ranges,
                   'scope': 'Measured cabin boundaries and bounded furniture; no Gaussian haze occupancy or movable equipment reconstruction'}
    }
    encoded = json.dumps(document, separators=(',', ':')).encode()
    encoded += b' ' * (-len(encoded) % 4)
    binary += b'\x00' * (-len(binary) % 4)
    glb = struct.pack('<III', 0x46546c67, 2, 12 + 8 + len(encoded) + 8 + len(binary))
    glb += struct.pack('<II', len(encoded), 0x4e4f534a) + encoded
    glb += struct.pack('<II', len(binary), 0x004e4942) + binary
    return glb, {'vertices': len(points), 'triangles': len(triangles), 'bounds': bounds, 'proxies': ranges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('viewers/ambulance/bed-placement.json'))
    parser.add_argument('--out', type=Path, default=Path('viewers/ambulance/scene-collision.glb'))
    args = parser.parse_args()
    config_bytes = args.config.read_bytes()
    glb, report = build(json.loads(config_bytes))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(glb)
    report.update({'source_config': str(args.config), 'source_config_sha256': hashlib.sha256(config_bytes).hexdigest(),
                   'output_sha256': hashlib.sha256(glb).hexdigest(), 'bytes': len(glb),
                   'world_vertices_baked': True, 'node_transforms_identity': True,
                   'limits': ['Camera sphere/capsule queries use proxy surface triangles, not full captured equipment geometry.',
                              'Body checking additionally applies finite-plane half-spaces and shallow mattress support tolerance.',
                              'Scene units are not a verified metric calibration.']})
    args.out.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
