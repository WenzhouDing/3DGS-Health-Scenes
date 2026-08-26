#!/usr/bin/env python3
"""Re-log a COLMAP-style SfM .rrd into a legible Rerun visualization.

The raw dumps this reads come out of the usual `colmap -> rerun` loop, which logs
one static Pinhole+Transform3D entity per registered image. For a 2000-image
reconstruction that means 2000 overlapping frustums drawn on top of each other,
no timeline to scrub, and no ViewCoordinates -- so the viewer falls back to its
default up axis and orbiting fights you.

This script fixes both:

  * density -- camera centers collapse into ONE Points3D entity, view directions
    into ONE Arrows3D entity, and the path into one LineStrips3D per clip.
    Full frustums are drawn only every --frustum-stride images, plus a single
    frustum that walks the timeline so you can inspect poses one at a time.

  * orientation -- the reconstruction frame is levelled. The upright axis is
    recovered from the scene's own dominant planes (Manhattan-world RANSAC),
    disambiguated with the camera rig, and the scene is re-based so +Z is up,
    the floor sits at z=0 and the footprint is centred on the origin.

Usage:
    uv run --with rerun-sdk==0.36.3 --with numpy tools/sfm_vis.py raw/scene.rrd
    ... --save out.rrd      # write an .rrd instead of spawning the viewer
    ... --no-rectify        # keep the original frame (to see what changed)
    ... --up X,Y,Z          # force the up axis instead of estimating it

Note on SDK versions: rerun >=0.27 dropped the local dataframe reader, so nothing
that can *write* for a modern viewer can also *read* an .rrd off disk. This script
therefore shells out to a second interpreter that still has `rerun.dataframe`
(0.26.x) to extract the poses once, caches them as .npz, and does everything else
in the modern SDK. The reader is provisioned with uv (Python 3.12 + rerun-sdk 0.26.2)
rather than taken from PATH -- under `uv run` or any venv, PATH's `python3` IS the
modern SDK and cannot read, and the system one is often an EOL 3.9. Local interpreters
are only a fallback when uv is absent. Override with --reader-python.
"""

from __future__ import annotations

import argparse
import collections
import colorsys
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

try:  # only present on rerun-sdk <= 0.26
    from rerun import dataframe as rrd

    CAN_READ = True
except ImportError:  # pragma: no cover - depends on installed SDK
    rrd = None
    CAN_READ = False

FRAME_RE = re.compile(r"_(m?)(\d+)\.[A-Za-z]+$")


# --------------------------------------------------------------------------- io


def read_source(path):
    """Pull camera poses, intrinsics and the sparse cloud out of a source .rrd."""
    rec = rrd.load_recording(path)
    schema = rec.schema()

    cam_paths = sorted(
        {
            c.entity_path
            for c in schema.component_columns()
            if c.entity_path.startswith("/cameras/") and c.entity_path.count("/") >= 3
        }
    )
    if not cam_paths:
        sys.exit(f"{path}: no /cameras/<clip>/<image> entities found")

    view = rec.view(
        index="log_time",
        contents={
            "/cameras/**": [
                "Transform3D:translation",
                "Transform3D:mat3x3",
                "Pinhole:image_from_camera",
                "Pinhole:resolution",
            ]
        },
    )
    tbl = view.select_static().read_all()

    def col(name, default=None):
        if name not in tbl.column_names:
            return default
        v = tbl.column(name).to_pylist()[0]
        return None if v is None else v[0]

    cams = []
    for ent in cam_paths:
        t = col(f"{ent}:Transform3D:translation")
        m = col(f"{ent}:Transform3D:mat3x3")
        if t is None or m is None:
            continue
        stem = ent.rsplit("/", 1)[1]
        hit = FRAME_RE.search(stem)
        cams.append(
            {
                "path": ent,
                "clip": ent.split("/")[2],
                # the two numbering families ("000123" vs "m000123") are separate
                # samplings of the same clip in different index spaces -- keep them
                # apart so the trajectory does not zig-zag between them
                "kind": (hit.group(1) or "p") if hit else "p",
                "frame": int(hit.group(2)) if hit else 0,
                "C": np.asarray(t, float),
                "R": np.asarray(m, float).reshape(3, 3),
                "K": np.asarray(col(f"{ent}:Pinhole:image_from_camera"), float).reshape(3, 3),
                "res": np.asarray(col(f"{ent}:Pinhole:resolution"), float),
            }
        )

    pv = rec.view(
        index="log_time", contents={"/points": ["Points3D:positions", "Points3D:colors"]}
    ).select_static().read_all()
    P = np.asarray(pv.column("/points:Points3D:positions").to_pylist()[0], float)
    packed = np.asarray(pv.column("/points:Points3D:colors").to_pylist()[0]).astype(np.uint32)
    rgb = np.stack([(packed >> 24) & 255, (packed >> 16) & 255, (packed >> 8) & 255], 1).astype(
        np.uint8
    )

    # These dumps also carry a /camera_centers cloud holding EVERY registered image,
    # while only a strided subset gets a full Pinhole+Transform3D entity (4x fewer,
    # in the file this was written for). Positions alone still give the complete
    # trajectory, so pick them up and recover each one's clip.
    centers, center_clips = read_centers(rec, cams)
    return cams, P, rgb, centers, center_clips


def read_centers(rec, cams):
    """All registered camera centres, labelled by clip via the posed subset."""
    paths = {c.entity_path for c in rec.schema().component_columns()}
    if "/camera_centers" not in paths:
        return np.zeros((0, 3)), np.zeros(0, dtype=object)
    cv = rec.view(
        index="log_time",
        contents={"/camera_centers": ["Points3D:positions", "Points3D:colors"]},
    ).select_static().read_all()
    pos = np.asarray(cv.column("/camera_centers:Points3D:positions").to_pylist()[0], float)
    packed = np.asarray(cv.column("/camera_centers:Points3D:colors").to_pylist()[0]).astype(np.uint32)
    rgb = np.stack([(packed >> 24) & 255, (packed >> 16) & 255, (packed >> 8) & 255], 1)

    # the writer colours these per clip; posed cameras sit at exactly the same
    # coordinates, so an exact position match recovers colour -> clip
    lut = {tuple(np.round(p, 6)): i for i, p in enumerate(pos)}
    votes = collections.defaultdict(collections.Counter)
    for c in cams:
        i = lut.get(tuple(np.round(c["C"], 6)))
        if i is not None:
            votes[tuple(rgb[i])][c["clip"]] += 1
    colour_to_clip = {k: v.most_common(1)[0][0] for k, v in votes.items()}
    clips = np.array([colour_to_clip.get(tuple(c), "unassigned") for c in rgb], dtype=object)
    return pos, clips


def dump_cache(src, dst):
    cams, P, rgb, centers, center_clips = read_source(src)
    np.savez_compressed(
        dst,
        paths=np.array([c["path"] for c in cams]),
        C=np.array([c["C"] for c in cams]),
        R=np.array([c["R"] for c in cams]),
        K=np.array([c["K"] for c in cams]),
        res=np.array([c["res"] for c in cams]),
        P=P,
        rgb=rgb,
        centers=centers,
        center_clips=np.array([str(x) for x in center_clips]),
    )
    print(f"cached {len(cams)} poses, {len(centers)} centres + {len(P)} points -> {dst}")


def can_read_with(cmd):
    try:
        return subprocess.run(
            cmd + ["-c", "import rerun.dataframe"], capture_output=True, timeout=300
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def find_reader(explicit):
    """Find an interpreter holding rerun-sdk<=0.26, the last one that can read .rrd.

    Do NOT just trust `python3` off PATH: when this script runs under `uv run` or any
    venv, PATH's python3 is that ephemeral environment -- which has the modern SDK and
    therefore cannot read. The system interpreter is the likelier reader, and `uv` can
    always conjure one as a last resort.
    """
    if explicit:
        if can_read_with([explicit]):
            return [explicit]
        sys.exit(
            f"{explicit} has no `rerun.dataframe` module.\n"
            f"Install one with:  {explicit} -m pip install 'rerun-sdk==0.26.2'"
        )

    # Prefer a uv-provisioned reader over whatever system python happens to be around:
    # it pins Python 3.12 and rerun-sdk 0.26.2 explicitly, where the system interpreter
    # is often an EOL 3.9 that rerun warns about.
    uv = shutil.which("uv")
    if uv:
        cmd = [uv, "run", "--quiet", "--python", "3.12",
               "--with", "rerun-sdk==0.26.2", "--with", "numpy", "python"]
        if can_read_with(cmd):
            return cmd
        print("uv could not provision a reader; falling back to local interpreters")

    seen, cands = set(), []
    for p in ("/opt/homebrew/bin/python3", "/usr/local/bin/python3", "/usr/bin/python3",
              shutil.which("python3"), sys.executable):
        if p and p not in seen:
            seen.add(p)
            cands.append(p)
    for p in cands:
        if can_read_with([p]):
            return [p]

    sys.exit(
        "Could not find an interpreter with `rerun.dataframe` (rerun-sdk<=0.26), which is\n"
        "the last version able to read .rrd files off disk.\n"
        "Fix with either:\n"
        "  python3 -m pip install 'rerun-sdk==0.26.2'\n"
        "  ...or install uv, and this script will provision Python 3.12 + rerun-sdk 0.26.2 itself\n"
        "  ...or pass --reader-python /path/to/such/an/interpreter"
    )


def load(path, reader_python, cache=None):
    """Read the source recording, via a helper interpreter if this SDK cannot."""
    if CAN_READ:
        return read_source(path)

    if cache is None:
        stem = os.path.splitext(os.path.basename(path))[0]
        cache = os.path.join(tempfile.gettempdir(), f"{stem}.sfmcache.npz")
    if not os.path.exists(cache) or os.path.getmtime(cache) < os.path.getmtime(path):
        print(f"this SDK ({rr.__version__}) cannot read .rrd files; finding a reader")
        reader = find_reader(reader_python)
        print(f"   reading with: {' '.join(reader)}")
        subprocess.run(
            reader + [os.path.abspath(__file__), path, "--dump-cache", cache], check=True
        )
    d = np.load(cache, allow_pickle=False)
    if "centers" not in d:
        sys.exit(f"{cache} predates the /camera_centers support; delete it and re-run")
    cams = []
    for i, p in enumerate(d["paths"]):
        p = str(p)
        stem = p.rsplit("/", 1)[1]
        hit = FRAME_RE.search(stem)
        cams.append(
            {
                "path": p,
                "clip": p.split("/")[2],
                "kind": (hit.group(1) or "p") if hit else "p",
                "frame": int(hit.group(2)) if hit else 0,
                "C": d["C"][i],
                "R": d["R"][i],
                "K": d["K"][i],
                "res": d["res"][i],
            }
        )
    return cams, d["P"], d["rgb"], d["centers"], d["center_clips"]


# ------------------------------------------------------------------- geometry


def unit(v):
    return np.asarray(v, float) / np.linalg.norm(v)


def dominant_planes(pts, n_planes=8, thresh=0.05, seed=1):
    """Sequential RANSAC: peel off the largest planes and return their normals."""
    rng = np.random.default_rng(seed)
    work = pts.copy()
    out = []
    for _ in range(n_planes):
        if len(work) < 5000:
            break
        sub = work[rng.choice(len(work), min(120_000, len(work)), replace=False)]
        best = (0, None, 0.0)
        for _ in range(3000):
            p = sub[rng.choice(len(sub), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            norm = np.linalg.norm(n)
            if norm < 1e-6:
                continue
            n = n / norm
            hits = int((np.abs(sub @ n - n @ p[0]) < thresh).sum())
            if hits > best[0]:
                best = (hits, n, float(n @ p[0]))
        count, n, off = best
        if n is None:
            break
        for _ in range(10):  # least-squares refine on the full working set
            m = np.abs(work @ n - off) < thresh
            if m.sum() < 100:
                break
            A = work[m] - work[m].mean(0)
            n = unit(np.linalg.eigh(A.T @ A)[1][:, 0])
            off = float(np.median(work[m] @ n))
        m = np.abs(work @ n - off) < thresh * 1.2
        out.append((int(m.sum()), n.copy()))
        work = work[~m]
    return out


def upright_frame(pts, cams):
    """Recover the scene's upright axis from its dominant planes.

    Averaging the camera rig's own down-vectors is the obvious approach and it is
    biased: a handheld operator tilts down towards whatever they are filming, and
    with non-uniform headings that bias does not cancel. The building is the more
    reliable reference, so the planes decide the axis and the cameras only break
    the up/down sign and pick which plane family is the floor.
    """
    med = np.median(pts, 0)
    core = pts[np.linalg.norm(pts - med, axis=1) < 12]
    planes = dominant_planes(core)
    if not planes:
        return None, None

    # group parallel planes into families -- a room gives one family per wall pair
    fams = []
    for count, n in planes:
        match = next((f for f in fams if abs(n @ f["n"]) > 0.985), None)
        if match is None:
            fams.append({"n": n, "count": count})
        else:
            match["count"] += count
    fams.sort(key=lambda f: -f["count"])

    C = np.array([c["C"] for c in cams])
    R = np.array([c["R"] for c in cams])
    g_cam = unit(R[:, :, 1].mean(0))  # mean camera "down" -- rough, but signed

    # the floor family is the one whose normal is closest to the rig's gravity and
    # whose camera-height spread is tightest (cameras hold a roughly fixed height)
    best = None
    for f in fams[:4]:
        n = f["n"]
        ang = np.degrees(np.arccos(np.clip(abs(n @ g_cam), 0, 1)))
        if ang > 45:
            continue
        h = C @ n
        spread = float(np.percentile(h, 75) - np.percentile(h, 25))
        score = (ang, spread)
        if best is None or score < best[0]:
            best = (score, n)
    if best is None:
        return None, None
    n_floor = best[1]
    up = -n_floor if (n_floor @ g_cam) > 0 else n_floor  # gravity points down

    # complete the triad using the strongest family perpendicular to up
    horiz = None
    for f in fams:
        if abs(f["n"] @ up) < 0.25:
            horiz = f["n"]
            break
    Z = unit(up)
    X = unit(horiz - Z * (horiz @ Z)) if horiz is not None else unit(np.cross([0.0, 0.0, 1.0], Z))
    Y = np.cross(Z, X)
    return np.stack([X, Y, Z]), fams


def euler_xyz_deg(M):
    """Extrinsic X-then-Y-then-Z Euler angles, in degrees, for R = Rz@Ry@Rx."""
    sy = -M[2, 0]
    if abs(sy) < 0.999999:
        x = np.arctan2(M[2, 1], M[2, 2])
        y = np.arcsin(sy)
        z = np.arctan2(M[1, 0], M[0, 0])
    else:  # gimbal lock
        x = np.arctan2(-M[1, 2], M[1, 1])
        y = np.arcsin(np.clip(sy, -1, 1))
        z = 0.0
    return np.degrees([x, y, z])


# --------------------------------------------------------------------- coverage

VIRIDIS = np.array([[68,1,84],[72,40,120],[62,74,137],[49,104,142],[38,130,142],
                    [31,158,137],[53,183,121],[109,205,89],[253,231,37]], float)
MAGMA = np.array([[0,0,4],[40,11,84],[101,21,110],[159,42,99],[212,72,66],
                  [245,125,21],[250,193,39],[252,253,191]], float)


def ramp(v, lo, hi, anchors):
    t = np.clip((np.asarray(v, float) - lo) / max(hi - lo, 1e-9), 0, 1)
    idx = t * (len(anchors) - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.minimum(i0 + 1, len(anchors) - 1)
    f = (idx - i0)[:, None]
    return (anchors[i0] * (1 - f) + anchors[i1] * f).astype(np.uint8)


def plane_footprint(pts, res, pad=2):
    """Grid cells where a plane actually exists, dilated / hole-filled / eroded back."""
    xs = np.arange(pts[:, 0].min() - 0.5, pts[:, 0].max() + 0.5, res)
    ys = np.arange(pts[:, 1].min() - 0.5, pts[:, 1].max() + 0.5, res)
    hist, _, _ = np.histogram2d(
        pts[:, 0], pts[:, 1],
        bins=[np.append(xs, xs[-1] + res), np.append(ys, ys[-1] + res)],
    )
    m = hist >= 1
    for _ in range(pad):
        n = m.copy()
        n[1:] |= m[:-1]; n[:-1] |= m[1:]; n[:, 1:] |= m[:, :-1]; n[:, :-1] |= m[:, 1:]
        m = n
    free, out = ~m, np.zeros_like(m)          # flood the exterior so enclosed gaps fill
    if free[0, 0]:
        out[0, 0] = True
        stack = [(0, 0)]
        while stack:
            i, j = stack.pop()
            for a, b in ((i+1, j), (i-1, j), (i, j+1), (i, j-1)):
                if 0 <= a < m.shape[0] and 0 <= b < m.shape[1] and free[a, b] and not out[a, b]:
                    out[a, b] = True
                    stack.append((a, b))
    m = m | (~out & ~m)
    for _ in range(pad):
        n = m.copy()
        n[1:] &= m[:-1]; n[:-1] &= m[1:]; n[:, 1:] &= m[:, :-1]; n[:, :-1] &= m[:, 1:]
        m = n
    return xs, ys, m, hist


def plane_coverage(cells, normal, cams, K, res, max_incidence=75.0):
    """How many cameras actually see each cell, and from how wide a cone.

    Measured on a GRID over the plane, deliberately not on the sparse cloud: that
    cloud only contains points COLMAP managed to triangulate, so every point in it
    was seen by construction and it can never reveal a gap. A grid can.
    Occlusion is not modelled -- fine for a ceiling, optimistic for a floor with
    furniture on it.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    W, H = res
    n = np.zeros(len(cells), np.int32)
    vsum = np.zeros((len(cells), 3))
    best = np.zeros(len(cells))
    cos_max = np.cos(np.radians(max_incidence))
    for c in cams:
        v = c["Cr"] - cells
        dist = np.linalg.norm(v, axis=1)
        u = v / np.maximum(dist, 1e-9)[:, None]
        cosi = u @ normal
        idx = np.flatnonzero((cosi > cos_max) & (dist > 0.2))
        if not len(idx):
            continue
        Xc = (cells[idx] - c["Cr"]) @ c["Rr"]           # world -> camera (RDF)
        z = Xc[:, 2]
        f = z > 0.05
        idx, Xc, z = idx[f], Xc[f], z[f]
        if not len(idx):
            continue
        uu = fx * Xc[:, 0] / z + cx
        vv = fy * Xc[:, 1] / z + cy
        keep = (uu >= 0) & (uu < W) & (vv >= 0) & (vv < H)
        idx = idx[keep]
        n[idx] += 1
        vsum[idx] += u[idx]
        best[idx] = np.maximum(best[idx], cosi[idx])
    r = np.linalg.norm(vsum, axis=1) / np.maximum(n, 1)
    # spherical spread expressed as the half-angle of an equivalent uniform cone
    cone = np.degrees(np.arccos(np.clip(2 * r - 1, -1, 1)))
    cone[n < 2] = 0.0
    return n, cone, best


def log_coverage(cams, Pr, ceil_z, res=0.10):
    K = cams[0]["K"].T          # rerun stores Mat3x3 column-major; transpose for projection
    imres = cams[0]["res"]
    planes = [
        ("ceiling", ceil_z, np.array([0.0, 0.0, -1.0]), Pr[Pr[:, 2] > ceil_z - 0.35]),
        ("floor", 0.0, np.array([0.0, 0.0, 1.0]), Pr[Pr[:, 2] < 0.35]),
    ]
    for name, zp, normal, band in planes:
        if len(band) < 500:
            print(f"  {name}: too few points to define a footprint, skipped")
            continue
        xs, ys, mask, hist = plane_footprint(band, res)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        cells = np.stack([gx[mask], gy[mask], np.full(int(mask.sum()), zp)], 1)
        n, cone, best = plane_coverage(cells, normal, cams, K, imres)
        empty = 100 * ((hist < 1) & mask).sum() / max(mask.sum(), 1)
        print(f"\n  {name.upper()} — {len(cells)} cells over {mask.sum() * res * res:.1f} sq units")
        print(f"     never seen {100 * (n == 0).mean():.2f}%   seen 1-4 {100 * ((n >= 1) & (n < 5)).mean():.2f}%"
              f"   seen >=10 {100 * (n >= 10).mean():.1f}%")
        print(f"     views/cell median {np.median(n):.0f}  (p5 {np.percentile(n, 5):.0f}, p95 {np.percentile(n, 95):.0f})")
        if (n >= 2).any():
            print(f"     view-cone spread median {np.median(cone[n >= 2]):.1f} deg; "
                  f"{100 * ((cone < 15) & (n >= 2)).mean():.1f}% of cells under 15 deg")
        print(f"     best incidence median {np.degrees(np.arccos(np.clip(np.median(best), 0, 1))):.0f} deg from normal")
        print(f"     {empty:.0f}% of cells have no reconstructed points -> textureless, NOT unseen")

        # Stretch the ramp over the data's own p2..p98, not 1..max: real captures never
        # reach 1 view, so a fixed low end wastes most of the colours on empty range and
        # the map reads as flat yellow.
        lo, hi = np.percentile(n, 2), np.percentile(n, 98)
        lo, hi = float(max(lo, 1)), float(max(hi, lo + 1))
        rr.log(f"/world/coverage/{name}/views",
               rr.Points3D(cells,
                           colors=ramp(np.log10(np.maximum(n, 1)), np.log10(lo), np.log10(hi), VIRIDIS),
                           radii=res * 0.5), static=True)
        rr.log(f"/world/coverage/{name}/spread",
               rr.Points3D(cells, colors=ramp(cone, 0, 60, MAGMA), radii=res * 0.5), static=True)
        print(f"     colour key (views): dark purple <={lo:.0f}  ...  bright yellow >={hi:.0f}  (log scale)")
        print(f"     colour key (spread): black 0 deg -> red 34 deg -> pale yellow 60 deg (linear)")


# ---------------------------------------------------------------------- logging


def clip_colors(clips):
    out = {}
    for i, c in enumerate(sorted(clips)):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.72, 1.0)
        out[c] = [int(r * 255), int(g * 255), int(b * 255)]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rrd")
    ap.add_argument("--save", metavar="OUT.rrd", help="write an .rrd instead of spawning the viewer")
    ap.add_argument("--no-rectify", action="store_true", help="keep the original frame")
    ap.add_argument("--up", help="force the up axis, e.g. '0,-1,0'")
    ap.add_argument("--yaw", type=float, default=0.0,
                    help="extra rotation about up, in degrees. Levelling fixes which way is UP; "
                         "which way is FORWARD is a free choice, so use this to point the subject "
                         "down the axis you want (default aligns to the dominant wall)")
    ap.add_argument("--frustum-stride", type=int, default=48, help="draw one full frustum every N images (0=none)")
    ap.add_argument("--point-stride", type=int, default=1, help="subsample the cloud by N")
    ap.add_argument("--clip-radius", type=float, default=12.0, help="drop points farther than this from the median")
    ap.add_argument("--no-blueprint", action="store_true",
                    help="ship no blueprint, so the viewer picks its own default layout "
                         "(useful for isolating layout problems from data problems)")
    ap.add_argument("--coverage", action="store_true",
                    help="measure how well each surface was actually filmed (ceiling/floor grids)")
    ap.add_argument("--reader-python", default=None,
                    help="interpreter holding rerun-sdk<=0.26 (the last one that can read .rrd); auto-detected, falling back to a uv-provisioned one")
    ap.add_argument("--cache", help="path for the extracted-pose .npz cache")
    ap.add_argument("--dump-cache", metavar="OUT.npz", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.dump_cache:  # internal: run under the reader interpreter
        if not CAN_READ:
            sys.exit("--dump-cache needs an SDK with `rerun.dataframe` (rerun-sdk<=0.26)")
        dump_cache(args.rrd, args.dump_cache)
        return

    cams, P, rgb, centers, center_clips = load(args.rrd, args.reader_python, args.cache)
    print(f"loaded {len(cams)} posed cameras, {len(centers)} registered centres, "
          f"{len(P)} points from {args.rrd}")
    if len(centers) > len(cams):
        print(f"   note: only every ~{len(centers) / max(len(cams), 1):.0f}th registered image "
              f"carries a full pose; the rest contribute position only")

    # ---- levelling ---------------------------------------------------------
    M = np.eye(3)
    if not args.no_rectify:
        if args.up:
            Z = unit([float(x) for x in args.up.split(",")])
            X = unit(np.cross([0.0, 0.0, 1.0], Z)) if abs(Z[2]) < 0.9 else unit(np.cross([0.0, 1.0, 0.0], Z))
            M = np.stack([X, np.cross(Z, X), Z])
            print(f"up axis forced to {np.round(Z, 4).tolist()}")
        else:
            M, fams = upright_frame(P, cams)
            if M is None:
                print("!! could not estimate an upright frame; leaving the scene as-is")
                M = np.eye(3)
            else:
                up = M[2]
                print(f"estimated up axis (source frame): {np.round(up, 4).tolist()}")
                for nm, ax in [("+X", [1, 0, 0]), ("-X", [-1, 0, 0]), ("+Y", [0, 1, 0]),
                               ("-Y", [0, -1, 0]), ("+Z", [0, 0, 1]), ("-Z", [0, 0, -1])]:
                    a = np.degrees(np.arccos(np.clip(np.dot(up, ax), -1, 1)))
                    if a < 45:
                        print(f"   -> {a:.1f} deg away from {nm}")

    if args.yaw:
        a = np.radians(args.yaw)
        M = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]) @ M
        print(f"applied {args.yaw:g} deg extra yaw about up")

    Pr = P @ M.T
    Cen = centers @ M.T if len(centers) else centers
    for c in cams:
        c["Cr"] = M @ c["C"]
        c["Rr"] = M @ c["R"]

    # floor at z=0, footprint centred on the origin
    med = np.median(Pr, 0)
    keep = np.linalg.norm(Pr - med, axis=1) < args.clip_radius
    core = Pr[keep]
    hist, edges = np.histogram(core[:, 2], bins=400)
    centres = (edges[:-1] + edges[1:]) / 2
    strong = [i for i in np.argsort(hist)[::-1][:15]]
    floor_z = float(min(centres[i] for i in strong))
    ceil_z = float(max(centres[i] for i in strong))
    shift = np.array([core[:, 0].mean(), core[:, 1].mean(), floor_z])
    if args.no_rectify:
        shift = np.zeros(3)
    Pr = Pr - shift
    core = core - shift
    if len(Cen):
        Cen = Cen - shift
    for c in cams:
        c["Cr"] = c["Cr"] - shift
    print(f"floor z={floor_z:.2f}  ceiling z={ceil_z:.2f}  (height {ceil_z - floor_z:.2f} scene units)")

    if not args.no_rectify:
        print("\n--- reuse this rotation elsewhere ---")
        print(f"  matrix (row-major, p_new = R @ p_old): {np.round(M, 6).tolist()}")
        print(f"  euler XYZ deg, Z-up target (rerun):    {np.round(euler_xyz_deg(M), 3).tolist()}")
        swap = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])  # Z-up -> Y-up
        print(f"  euler XYZ deg, Y-up target (viewer):   {np.round(euler_xyz_deg(swap @ M), 3).tolist()}")
        print("  (verify the sign/order against your tool's convention before baking it in)\n")

    # ---- per-camera diagnostics -------------------------------------------
    cams.sort(key=lambda c: (c["clip"], c["kind"], c["frame"]))
    seqs = collections.defaultdict(list)
    for c in cams:
        seqs[(c["clip"], c["kind"])].append(c)

    up_axis = np.array([0.0, 0.0, 1.0])
    for seq in seqs.values():
        for i, c in enumerate(seq):
            fwd, right = c["Rr"][:, 2], c["Rr"][:, 0]
            c["pitch"] = float(np.degrees(np.arcsin(np.clip(fwd @ up_axis, -1, 1))))
            c["roll"] = float(np.degrees(np.arcsin(np.clip(right @ up_axis, -1, 1))))
            c["height"] = float(c["Cr"][2])
            if i == 0:
                c["step"] = c["turn"] = 0.0
            else:
                prev = seq[i - 1]
                # normalise by the frame gap: these clips are sampled unevenly, and
                # an un-normalised step makes every gap look like a tracking failure
                gap = max(1, c["frame"] - prev["frame"])
                c["step"] = float(np.linalg.norm(c["Cr"] - prev["Cr"]) / gap)
                rel = prev["Rr"].T @ c["Rr"]
                c["turn"] = float(
                    np.degrees(np.arccos(np.clip((np.trace(rel) - 1) / 2, -1, 1))) / gap
                )
        for key in ("step", "turn"):
            vals = np.array([c[key] for c in seq[1:]])
            med = np.median(vals) if len(vals) else 0.0
            for i, c in enumerate(seq):
                # the two frame-numbering families divide by very different gaps, so the
                # raw values are not comparable across sub-sequences. Express each as a
                # multiple of its own median: 1.0 is typical everywhere, spikes are real.
                c[f"{key}_ratio"] = (c[key] / med) if (med > 0 and i > 0) else (1.0 if i else 0.0)
                c[f"bad_{key}"] = i > 0 and c[key] > 6 * med and c[key] > 2 * np.percentile(vals, 95)
        for c in seq:
            c["suspect"] = c.get("bad_step", False) or c.get("bad_turn", False)

    print("\n--- per-sub-sequence continuity (a healthy track keeps max/med under ~5) ---")
    print(f"{'sub-sequence':32s} {'n':>4s} {'med step':>9s} {'max/med':>8s} {'max turn':>9s} {'height sd':>10s}")
    worst = 0.0
    for (clip, kind), seq in sorted(seqs.items()):
        if len(seq) < 3:
            continue
        steps = np.array([c["step"] for c in seq[1:]])
        turns = np.array([c["turn"] for c in seq[1:]])
        ratio = steps.max() / max(np.median(steps), 1e-9)
        worst = max(worst, ratio)
        print(f"{clip + '_' + kind:32s} {len(seq):4d} {np.median(steps):9.4f} {ratio:8.1f} "
              f"{turns.max():9.2f} {np.std([c['height'] for c in seq]):10.3f}")

    n_suspect = sum(c["suspect"] for c in cams)
    print(f"\n{len(seqs)} sub-sequences, {len(cams)} poses; {n_suspect} flagged as discontinuous "
          f"({100 * n_suspect / max(1, len(cams)):.1f}%); worst max/med step ratio {worst:.1f}")

    # ---- log ---------------------------------------------------------------
    bp = None if args.no_blueprint else blueprint(args.coverage)
    rr.init("sfm_review", spawn=not args.save, default_blueprint=bp)
    if args.save:
        rr.save(args.save, default_blueprint=bp)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    pts = Pr[keep][:: args.point_stride]
    cols = rgb[keep][:: args.point_stride]
    rr.log("/world/points", rr.Points3D(pts, colors=cols, radii=0.008), static=True)
    print(f"logged {len(pts)} points ({int((~keep).sum())} outliers dropped)")

    if args.coverage:
        print("\n--- surface coverage (grid over each plane, occlusion not modelled) ---")
        log_coverage(cams, Pr[keep], ceil_z - floor_z if not args.no_rectify else ceil_z)

    colors = clip_colors({c["clip"] for c in cams} | set(map(str, center_clips)))

    if len(Cen):
        # one entity per clip, not one for all 8075: keeps the entity count at 18 while
        # letting the viewer name and toggle each clip individually
        for clip in sorted(set(map(str, center_clips))):
            sel = np.array([str(c) == clip for c in center_clips])
            rr.log(
                f"/world/cameras/centers/{clip}",
                rr.Points3D(Cen[sel], colors=colors[clip], radii=0.02),
                static=True,
            )
    else:
        rr.log(
            "/world/cameras/centers",
            rr.Points3D(
                np.array([c["Cr"] for c in cams]),
                colors=np.array([colors[c["clip"]] for c in cams], dtype=np.uint8),
                radii=0.025,
            ),
            static=True,
        )
    centres_all = np.array([c["Cr"] for c in cams])
    cam_cols = np.array([colors[c["clip"]] for c in cams], dtype=np.uint8)
    for clip in sorted({c["clip"] for c in cams}):
        sub = [c for c in cams if c["clip"] == clip]
        rr.log(
            f"/world/cameras/directions/{clip}",
            rr.Arrows3D(
                origins=np.array([c["Cr"] for c in sub]),
                vectors=np.array([c["Rr"][:, 2] for c in sub]) * 0.25,
                colors=colors[clip],
            ),
            static=True,
        )
    bad = [c for c in cams if c["suspect"]]
    if bad:
        rr.log(
            "/world/cameras/discontinuous",
            rr.Points3D([c["Cr"] for c in bad], colors=[255, 40, 40], radii=0.07),
            static=True,
        )

    if len(Cen):
        # /camera_centers is stored grouped by clip and in capture order, so contiguous
        # runs are the walk itself -- at 4x the density of the posed subset
        strips = collections.defaultdict(list)
        start = 0
        for i in range(1, len(Cen) + 1):
            if i == len(Cen) or center_clips[i] != center_clips[start]:
                seg = Cen[start:i]
                if len(seg) >= 2:
                    step = np.linalg.norm(np.diff(seg, axis=0), axis=1)
                    # break the line where frames went unregistered, otherwise the strip
                    # draws a straight phantom leg across the scene
                    cut = np.flatnonzero(step > max(1.5, 8 * np.median(step))) + 1
                    for piece in np.split(seg, cut):
                        if len(piece) >= 2:
                            strips[str(center_clips[start])].append(piece)
                start = i
        for clip, segs in sorted(strips.items()):
            rr.log(
                f"/world/trajectory/{clip}",
                rr.LineStrips3D(segs, colors=[colors[clip]] * len(segs), radii=0.012),
                static=True,
            )
        print(f"logged {sum(len(v) for v in strips.values())} trajectory segments "
              f"across {len(strips)} clips")
    else:
        for (clip, kind), seq in sorted(seqs.items()):
            if len(seq) < 2:
                continue
            rr.log(
                f"/world/trajectory/{clip}_{kind}",
                rr.LineStrips3D([np.array([c["Cr"] for c in seq])], colors=[colors[clip]], radii=0.012),
                static=True,
            )

    if args.frustum_stride > 0:
        for c in cams[:: args.frustum_stride]:
            p = f"/world/keyframes/{c['clip']}_{c['kind']}/{c['frame']:06d}"
            rr.log(p, rr.Transform3D(translation=c["Cr"], mat3x3=c["Rr"]), static=True)
            rr.log(
                p,
                rr.Pinhole(
                    image_from_camera=c["K"],
                    resolution=c["res"],
                    image_plane_distance=0.28,
                    color=colors[c["clip"]],
                ),
                static=True,
            )

    for i, c in enumerate(cams):
        rr.set_time("capture", sequence=i)
        rr.log("/world/active", rr.Transform3D(translation=c["Cr"], mat3x3=c["Rr"]))
        rr.log(
            "/world/active",
            rr.Pinhole(
                image_from_camera=c["K"],
                resolution=c["res"],
                image_plane_distance=0.75,
                color=[255, 255, 255],
            ),
        )
        rr.log("/diagnostics/step_vs_typical", rr.Scalars(c["step_ratio"]))
        rr.log("/diagnostics/turn_vs_typical", rr.Scalars(c["turn_ratio"]))
        rr.log("/diagnostics/raw/step_per_frame", rr.Scalars(c["step"]))
        rr.log("/diagnostics/raw/turn_deg_per_frame", rr.Scalars(c["turn"]))
        rr.log("/diagnostics/height", rr.Scalars(c["height"]))
        rr.log("/diagnostics/roll_deg", rr.Scalars(c["roll"]))
        rr.log("/diagnostics/pitch_deg", rr.Scalars(c["pitch"]))

    if not args.save:
        print("\nviewer spawned -- scrub the 'capture' timeline to walk the poses one at a time")
    else:
        print(f"\nwrote {args.save}  ->  rerun {args.save}")


def blueprint(coverage=False):
    # The coverage grids sit ON the floor and ceiling planes and would otherwise paint
    # over the whole reconstruction, so they ship hidden -- toggle them in the tree.
    hidden = rrb.EntityBehavior(visible=False)
    main = rrb.Spatial3DView(
                origin="/world",
                name="reconstruction",
                background=[18, 20, 24],
                line_grid=rrb.LineGrid3D(visible=True, plane=rr.datatypes.Plane3D([0, 0, 1], 0.0)),
                overrides={
                    "/world/coverage": hidden,
                    "/world/cameras/directions": hidden,
                } if coverage else {"/world/cameras/directions": hidden},
            )
    return rrb.Blueprint(
        rrb.Horizontal(
            main,
            rrb.Vertical(
                rrb.TimeSeriesView(origin="/diagnostics/step_vs_typical", name="step vs typical  (1 = normal, >5 = suspect)"),
                rrb.TimeSeriesView(origin="/diagnostics/turn_vs_typical", name="turn vs typical  (1 = normal, >5 = suspect)"),
                rrb.TimeSeriesView(origin="/diagnostics/height", name="camera height above floor (scene units)"),
                rrb.TimeSeriesView(origin="/diagnostics/roll_deg", name="roll (deg, 0 = level)"),
            ),
            column_shares=[3, 1],
        ),
        rrb.BlueprintPanel(state="expanded"),
        rrb.SelectionPanel(state="collapsed"),
    )


if __name__ == "__main__":
    main()
