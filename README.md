# 3DGS Health Scenes

Interactive 3D Gaussian Splatting project page: a gallery of health-care and indoor
scenes, each viewable in the browser with real-time orbit/pan/zoom. Static files
only — no backend, no build step.

## Layout

- `index.html` — gallery landing page (one card per scene)
- `viewers/<scene>/` — standalone viewer per scene: `index.html` + JS/CSS +
  `index.sog` (the compressed scene data)

## How scenes are produced

Scenes are trained with [gsplat](https://github.com/nerfstudio-project/gsplat) and
exported as standard 3DGS PLY files. Each PLY is converted into a compressed
[SOG](https://developer.playcanvas.com/user-manual/gaussian-splatting/splat-formats/)
scene plus a standalone web viewer with
[@playcanvas/splat-transform](https://github.com/playcanvas/splat-transform) (MIT):

```sh
npx -y @playcanvas/splat-transform \
  <scene>.ply --filter-nan -r <x,y,z> \
  viewers/<scene>/index.html --unbundled -w
```

Two per-scene adjustments to expect:

- **Orientation** — gsplat scenes come out in the training frame, not viewer-space
  (+Y up). The ambulance scene needed `-r -90,0,0` (its up axis was +Z). Check a
  test render and bake whatever rotation levels the scene.
- **Initial camera** — the exported `viewers/<scene>/settings.json` defaults to
  position `[2,2,-2]` targeting the origin, which is usually wrong (for interior
  scenes it starts inside a wall). Edit `cameras[0].initial` (`position`, `target`,
  `fov`) after conversion. Candidate cameras can be tested without re-converting
  via the `?settings=<url>` viewer parameter. Note that re-running the converter
  with `-w` overwrites `settings.json` — re-apply the camera afterwards.
- **Camera coordinates are relative to the scene centroid.** The viewer re-centers
  the scene on the mean splat position at load (`calcFocalPoint`), so a camera at
  `[0,0,0]` sits at the centroid, not at the PLY origin. For a PLY whose content is
  far from the origin, convert `position`/`target` as `ply_coords - centroid`.
- **Floater haze (optional, not used so far)** — 360°/fisheye captures can contain
  a few very large, faint blobs that fog views from inside the scene. If a scene
  needs it, a size cap removes them: `-V scale_0,lt,0.3 -V scale_1,lt,0.3
  -V scale_2,lt,0.3` (decoded sizes in scene units; `--filter-floaters` does not
  catch these). It is lossy — the published scenes ship the full splat set.

The in-browser rendering is done by
[@playcanvas/supersplat-viewer](https://github.com/playcanvas/supersplat-viewer) (MIT).

Raw PLYs are not committed (see `.gitignore`) — they are hundreds of MB each and
fully redundant with the `.sog` files.

## Inspecting the SfM result

`tools/sfm_vis.py` re-logs a COLMAP-style `.rrd` (one static `Pinhole` + `Transform3D`
entity per registered image) into something you can actually read:

```sh
uv run --with rerun-sdk==0.36.3 --with numpy \
  tools/sfm_vis.py raw/<scene>.rrd --save raw/<scene>.review.rrd
```

It collapses every camera centre into one `Points3D` and every view direction into
one `Arrows3D`, draws one trajectory line per clip, and keeps full frustums only
every `--frustum-stride` images plus a single frustum that walks a `capture`
timeline. Side panels plot per-frame step, turn, height and roll — spikes there are
where a track broke.

Note that these dumps log *two* camera sets: `/camera_centers` holds every
registered image, while `/cameras/<clip>/<image>` carries a full pose for only a
strided subset (every 4th, in the file this was written for). Trajectories and
camera dots therefore come from `/camera_centers` — 4x denser, and the only
complete picture of coverage — while arrows, frustums and the orientation
diagnostics necessarily use the posed subset. Camera dots are coloured per source
clip, so a region touched by only one colour was seen from a single pass and will
be weak in the trained splat.

It also levels the scene. The upright axis is recovered from the reconstruction's
own dominant planes (Manhattan-world RANSAC), not from averaging the camera rig's
down-vectors — a handheld operator tilts toward whatever they are filming, and that
bias does not cancel. The cameras only break the up/down sign and pick which plane
family is the floor. Output is +Z up, floor at `z=0`, footprint centred, with
`ViewCoordinates.RIGHT_HAND_Z_UP` logged so orbiting behaves.

### Coverage

`--coverage` answers "was this surface actually filmed, and from enough angles?"
It grids the ceiling and floor planes and, for each cell, counts the cameras that
genuinely see it — in frustum, in front, incidence under 75° — and measures the
angular spread of those views as the half-angle of an equivalent cone. Spread is
what matters for splat quality: a cell seen 100 times from one spot is worse
constrained than one seen 10 times from all around.

Two traps this deliberately avoids. **Do not measure coverage on the sparse cloud** —
it only contains points COLMAP managed to triangulate, so every point in it was seen
by construction and it can never show you a gap. A grid over the plane can. And
**empty grid cells are not unseen cells**: the tool reports point density separately,
because a blank stretch of ceiling is textureless, not unfilmed. Occlusion is not
modelled, which is fine for a ceiling and optimistic for a floor with furniture on it.

Results log to `/world/coverage/<plane>/{views,spread}` as coloured grids you can
toggle in the 3D view.

The script prints the rotation it applied as a matrix and as Euler angles for both
a Z-up and a Y-up target, so the same levelling can be baked into the `-r` argument
of the splat conversion above. Verify the order/sign against the tool's convention
before trusting it. Levelling only fixes which way is *up*; use `--yaw` to choose
which way is *forward*. `--no-rectify` reproduces the original frame for comparison.

One wrinkle: rerun dropped its local dataframe reader after 0.26, so no single SDK
can both read an `.rrd` and write for a current viewer. The script provisions a
reader with uv (Python 3.12 + `rerun-sdk==0.26.2`) to extract poses once, then
caches them as `.npz`. It deliberately does *not* take `python3` from `PATH`: under
`uv run` that is the modern SDK, which cannot read, and the system interpreter is
often an EOL 3.9. Local interpreters are tried only if uv is missing; override with
`--reader-python`.

## Publishing the SfM review

`viewers/sfm-review/` embeds the [Rerun web viewer](https://github.com/rerun-io/rerun)
so the camera solve can be examined in a browser, no install required. It is built from
the `.rrd` the tool above produces:

```sh
uv run --with rerun-sdk==0.36.3 --with numpy tools/sfm_vis.py \
  raw/<scene>.rrd --save viewers/sfm-review/sfm-review.rrd --coverage --point-stride 2
```

`--point-stride 2` halves the cloud to ~9.5 MB, in line with the other scenes; the full
cloud is 17 MB and looks much the same at this zoom.

The viewer runtime is vendored from Rerun's own deployment, pinned to the **same version
that wrote the file** — viewer and `.rrd` are only compatible within one minor version:

```sh
mkdir -p viewers/sfm-review/app && cd viewers/sfm-review/app
for f in index.html re_viewer.js re_viewer_bg.wasm; do
  curl -sL --compressed "https://app.rerun.io/version/0.36.3/$f" -o "$f"
done
```

Those three files are Rerun's viewer verbatim, embedded in a same-origin iframe. No build
step, no patching, nothing to re-apply on a version bump.

### Safari needs `?renderer=webgl`

Under WebGPU the 3D viewport renders **black on Safari 26.4+** — the rest of the viewer UI
draws normally, the `.rrd` decodes, and nothing is logged. That is
[rerun-io/rerun#12788](https://github.com/rerun-io/rerun/issues/12788), not anything about
this repo. Rerun added a Safari→WebGL default in
[#12789](https://github.com/rerun-io/rerun/pull/12789) and it is present in 0.36.3, but
their `index.html` assigns `render_backend` from the query string unconditionally, so an
absent `?renderer` arrives as an explicit `null` and overrides it. The page passes
`renderer=webgl` for Safari explicitly. `?renderer=webgpu` forces the broken path if you
want to confirm the symptom.

Do not "simplify" that away: forcing WebGPU because `navigator.gpu` exists is exactly the
bug, since it does exist in Safari 26.

## Viewing locally

### Articulate the complete fused manikin

Open the [fused articulation page](http://127.0.0.1:8766/viewers/mannequin-articulation/)
to pose all 2,698,682 fused Gaussians using the saved joint map. It includes ball
shoulders and hips, measured axial arm swivels, knee/ankle hinges, adjustable
spring response, pose export/import and multiple inspection views. Both captures
move together, and each forearm, wrist pad and hand remains one rigid group.
The derived geometry audit uses opposing metal hinge caps for knee/ankle axes
and leaves hidden shoulder/hip centers at the accepted annotations. The original
joint map and scan files are preserved. See the
[articulation guide](viewers/mannequin-articulation/README.md).

### Local manikin articulation lab

`viewers/mannequin-rig/` adds editable ball and hinge joints, angular springs,
joint-placement controls and body-part segmentation to the static manikin scan.
The raw PLY remains unchanged. Start `python3 tools/serve.py 8766` from this
directory and open [the local lab](http://localhost:8766/viewers/mannequin-rig/).
See the [tuning guide](viewers/mannequin-rig/README.md) for joint stiffness,
pivot/axis placement, segmentation painting, rig export/import and scan quality.

### Local front/back manikin fusion

Open the [joint annotation tool](http://127.0.0.1:8766/viewers/mannequin-joints/)
to mark centers, rotation axes, movement limits and stiffness on the current
fused scan. Start in Front, check depth from a side view, then mark each joint
checked. **Save for articulation** writes
`raw/mannequin-fused/joint-annotations.json` locally; the preceding save is kept
as `joint-annotations.previous.json`. The 12 starter joints are editable and
unreviewed, and both forearm/hand pairs stay rigid. See the
[annotation guide](viewers/mannequin-joints/README.md). This page records the
joint map consumed by the fused articulation page; the annotation page itself
does not deform the scan.

`viewers/mannequin-fusion/` combines the two manikin captures using matched
shoulder pads, torso ports, molded ears, and knee/ankle hardware at one shared
scale. Version 11 (`feature-guided-v11-finger-side-coverage`) restores a narrow
strip of original-source Gaussians along the finger sides and corrects local
color outliers. It keeps the version 10 alignment: the anatomical right forearm,
wrist pad and hand are one rigid body, with no wrist joint. No finger positions,
Gaussian shapes or arm poses change in this refinement.

Version 7's restored left fingertips and version 5's joint shoulder-pad/collar
fit and continuous grey cable remain unchanged, along with earlier clothing,
leg and foot cleanup. Adjustable
posterior-hair fullness changes appearance and can be disabled with
`hairFullness.amount: 0`. `colorRepairStrength: 0` disables both sole and
finger-side color repairs; remove only their specific `colorOverrides` entries
to disable either separately. These corrections do not recover missing scan detail.

Open [Fusion Review](http://127.0.0.1:8766/viewers/mannequin-fusion/) using the same
local server and inspect the [finger-side comparison](http://127.0.0.1:8766/raw/mannequin-fused/finger-side-review/).
Select a hand, choose **Frame selection**, then **Feet**, and orbit toward the
palm and back. Earlier [left fingertip renders](http://127.0.0.1:8766/raw/mannequin-fused/left-hand-review/),
[both-hand underside renders](http://127.0.0.1:8766/raw/mannequin-fused/hand-review/),
[grey-cable arm renders](http://127.0.0.1:8766/raw/mannequin-fused/cable-review/)
and [feature-alignment renders](http://127.0.0.1:8766/raw/mannequin-fused/render-review/)
remain available. Regenerate matching before/after views of the finger sides,
wrist pad and hand underside:

```sh
.venv-fusion/bin/python -B tools/fusion/make_review.py --title "Finger sides: original scan coverage restored" \
  --before raw/fusion-work/refinement-v11/baseline --after raw/mannequin-fused \
  --out raw/mannequin-fused/finger-side-review --size 800 \
  --regions right-finger-sides,right-wrist-pad,right-hand-under
```

The original PLYs remain unchanged; the full export is
`raw/mannequin-fused/mannequin_fused.ply`. The right pad spans hand and forearm
labels; their earlier transforms differed by about 20.96°. The active recipe's
`rigidParts: {"right_hand": "right_forearm"}` enforces the exact same transform
for every right-hand and forearm Gaussian. Tune
`adjustments.right_forearm.twistDegrees` to rotate the complete assembly about
the existing collar axis. Independent hand rotation, translation and deformation
are rejected. The original pad's continuity supplies a constraint even though
the front scan has no reliable corresponding dorsal pad rim.

The `surfaceRestorations` rules restore reviewed source IDs for the left fingertips
and right finger sides
before attenuation and exempt those exact IDs from cleanup, never rescuing
quality-filter rejects. `leftHandDigitAlignment.amount: 0.0033` controls its
separate lengthwise shift; `0` disables only that shift. The retained left-hand shift
updates its Gaussian covariance. The right-side restoration preserves original
geometry and applies a separately switchable local color repair. Source softness and
some seams remain visible from grazing angles.

See the [fusion guide](viewers/mannequin-fusion/README.md) for regenerating the
export and tuning wrist continuity, segmentation, feature transforms, source
quality, cleanup masks, fingertip restoration and hair fullness. Editing landmarks alone does not replace a reviewed
feature transform. The fusion viewer reviews a static scan; the separate
fused articulation page moves it. Different cloth folds and incomplete source
surfaces can still leave seams.

### Scene viewers

The viewers load scene data with `fetch`, so they must be served over HTTP —
opening `index.html` via `file://` will not work:

```sh
python3 tools/serve.py
# then open http://localhost:8000/
```

`tools/serve.py` is `http.server` plus `no-store`. Use it while editing anything under
`viewers/`: Safari caches the viewer JS and wasm hard enough to keep serving a stale copy
after a fix, which makes the change look like it did nothing. Plain
`python3 -m http.server 8000` is fine for just looking at scenes.

Useful viewer URL parameters: `?noui` hides the UI chrome, `?poster=<img>` shows a
loading image, `?budget=<millions>` caps rendered splats for weak GPUs.
