#!/usr/bin/env python3
"""Prepare an editable mannequin Gaussian scan; the source PLY is never changed.

Requires Python 3 and NumPy. The output keeps log scales, opacity logits and
degree-zero SH coefficients exactly as stored in the source. Only centers and
quaternions are rotated 180 degrees around X, making Y point above the mattress
and Z point toward the head. Run with --help for quality and filtering options.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["x", "y", "z", "rot_0", "rot_1", "rot_2", "rot_3",
          "scale_0", "scale_1", "scale_2", "opacity", "f_dc_0", "f_dc_1", "f_dc_2"]
PLY_TYPES = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


def read_ply(path):
    """Read named scalar vertex properties, independent of their order/types."""
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError("Input is not a PLY file")
        fmt = None
        elements = []
        current = None
        for _ in range(10000):
            line = stream.readline()
            if not line:
                raise ValueError("Truncated PLY header")
            tokens = line.decode("ascii").strip().split()
            if not tokens:
                continue
            if tokens[0] == "end_header":
                break
            if tokens[0] == "format":
                fmt = tokens[1]
            elif tokens[0] == "element":
                current = {"name": tokens[1], "count": int(tokens[2]), "properties": []}
                elements.append(current)
            elif tokens[0] == "property":
                if current is None:
                    raise ValueError("PLY property appears before its element")
                if tokens[1] == "list":
                    current["properties"].append((tokens[-1], None))
                else:
                    if tokens[1] not in PLY_TYPES:
                        raise ValueError(f"Unsupported PLY property type: {tokens[1]}")
                    current["properties"].append((tokens[2], PLY_TYPES[tokens[1]]))
        else:
            raise ValueError("PLY header is unexpectedly long")
        offset = stream.tell()

    vertex = next((item for item in elements if item["name"] == "vertex"), None)
    if not vertex or not vertex["count"]:
        raise ValueError("PLY must have a nonempty vertex element")
    names = [item[0] for item in vertex["properties"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate PLY vertex property names")
    missing = set(FIELDS) - set(names)
    if missing:
        raise ValueError(f"PLY is missing Gaussian properties: {', '.join(sorted(missing))}")
    if any(kind is None for _, kind in vertex["properties"]):
        raise ValueError("List-valued vertex properties are unsupported")

    preceding = elements[:elements.index(vertex)]
    if fmt in ("binary_little_endian", "binary_big_endian"):
        endian = "<" if fmt == "binary_little_endian" else ">"
        for element in preceding:
            if any(kind is None for _, kind in element["properties"]):
                raise ValueError("List-valued elements before vertices are unsupported")
            offset += element["count"] * np.dtype(element["properties"]).itemsize
        dtype = np.dtype([(name, endian + kind) for name, kind in vertex["properties"]])
        if path.stat().st_size < offset + dtype.itemsize * vertex["count"]:
            raise ValueError("PLY vertex data is truncated")
        vertices = np.memmap(path, mode="r", dtype=dtype, offset=offset, shape=(vertex["count"],))
    elif fmt == "ascii":
        with path.open("rb") as stream:
            stream.seek(offset)
            for element in preceding:
                for _ in range(element["count"]):
                    stream.readline()
            values = np.loadtxt(stream, max_rows=vertex["count"], ndmin=2)
        if values.shape != (vertex["count"], len(names)):
            raise ValueError("Unexpected ASCII PLY vertex data shape")
        vertices = {name: values[:, i] for i, name in enumerate(names)}
    else:
        raise ValueError(f"Unsupported PLY format: {fmt}")
    return vertices, vertex["count"], names


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "raw/mannequin_119999.ply")
    parser.add_argument("--output", type=Path, default=ROOT / "viewers/mannequin-rig")
    parser.add_argument("--max-splats", type=int, default=900000,
                        help="Deterministic sample size; 0 retains every filtered Gaussian (default: 900000)")
    parser.add_argument("--min-opacity", type=float, default=0.03,
                        help="Remove Gaussians with sigmoid(opacity) at or below this value (default: .03)")
    parser.add_argument("--max-scale", type=float, default=0.1,
                        help="Remove Gaussians with any exp(scale) at or above this value (default: .1)")
    parser.add_argument("--bounds", type=float, nargs=6, default=[-1.1, -.55, -.6, 1.1, .65, 2.25],
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        help="Broad scene bounds in original PLY coordinates")
    parser.add_argument("--no-filter", action="store_true",
                        help="Disable opacity, scale and scene-bound filters; invalid numeric records are always excluded")
    args = parser.parse_args(argv)
    if args.max_splats < 0 or not 0 <= args.min_opacity < 1 or args.max_scale <= 0:
        parser.error("Require --max-splats >= 0, 0 <= --min-opacity < 1 and --max-scale > 0")
    lower, upper = np.asarray(args.bounds[:3]), np.asarray(args.bounds[3:])
    if not np.all(lower < upper):
        parser.error("Each minimum bound must be below its maximum")
    source = args.input.resolve()
    vertices, count, source_fields = read_ply(source)
    valid = np.ones(count, dtype=bool)
    for field in FIELDS:
        valid &= np.isfinite(vertices[field])
    finite_count = int(valid.sum())
    if not args.no_filter:
        if args.min_opacity:
            valid &= vertices["opacity"] > math.log(args.min_opacity / (1 - args.min_opacity))
        for field in ("scale_0", "scale_1", "scale_2"):
            valid &= vertices[field] < math.log(args.max_scale)
        for i, field in enumerate(("x", "y", "z")):
            valid &= (vertices[field] >= lower[i]) & (vertices[field] <= upper[i])
    indices = np.flatnonzero(valid)
    filtered_count = len(indices)
    if not filtered_count:
        raise ValueError("No Gaussians survive the selected filters")
    if args.max_splats and len(indices) > args.max_splats:
        indices = indices[np.linspace(0, len(indices) - 1, args.max_splats, dtype=np.int64)]
    packed = np.column_stack([vertices[field][indices] for field in FIELDS]).astype("<f4")
    packed[:, 1:3] *= -1
    # Hamilton quaternion q' = (0, 1, 0, 0) * q, with source/storage order wxyz.
    q = packed[:, 3:7].copy()
    norms = np.linalg.norm(q.astype(np.float64), axis=1)
    degenerate = norms < 1e-12
    q[degenerate] = [1, 0, 0, 0]
    norms[degenerate] = 1
    q /= norms[:, None]
    packed[:, 3:7] = np.column_stack([-q[:, 1], q[:, 0], -q[:, 3], q[:, 2]])
    args.output.mkdir(parents=True, exist_ok=True)
    binary = args.output / "scan.bin"
    tmp_binary = binary.with_suffix(".bin.tmp")
    packed.tofile(tmp_binary)
    tmp_binary.replace(binary)
    try:
        source_name = source.relative_to(ROOT).as_posix()
    except ValueError:
        source_name = str(source)
    metadata = {
        "version": 1,
        "source": source_name,
        "sourceSha256": sha256_file(source),
        "sourceCount": count,
        "sourceFields": source_fields,
        "finiteCount": finite_count,
        "filteredCount": filtered_count,
        "count": len(indices),
        "file": "scan.bin",
        "format": "interleaved-float32-little-endian",
        "stride": len(FIELDS),
        "strideBytes": len(FIELDS) * 4,
        "fields": FIELDS,
        "quaternionOrder": "wxyz",
        "scaleEncoding": "natural-log",
        "opacityEncoding": "logit",
        "colorEncoding": "degree-zero-spherical-harmonics",
        "transform": {"position": "[x, -y, -z]", "rotationQuaternionWxyz": [0, 1, 0, 0],
                      "description": "180-degree rotation around X; Y above mattress, Z toward head; source units retained"},
        "bounds": {"min": packed[:, :3].min(axis=0).astype(float).tolist(),
                   "max": packed[:, :3].max(axis=0).astype(float).tolist()},
        "filter": {"enabled": not args.no_filter, "minOpacity": args.min_opacity,
                   "maxScale": args.max_scale, "sourceBounds": args.bounds,
                   "finiteRecordsRequired": True},
        "sampling": {"method": "evenly-spaced-source-indices-after-filtering", "maxSplats": args.max_splats},
        "degenerateQuaternionsReplaced": int(degenerate.sum()),
    }
    (args.output / "scan.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Prepared {len(indices):,} Gaussians from {count:,} source records "
          f"({filtered_count:,} after filtering).")
    print(f"Wrote {binary} ({binary.stat().st_size / 1e6:.1f} MB) and scan.json")
    return metadata


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        sys.exit(f"prepare_mannequin: {exc}")
