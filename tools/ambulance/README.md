# iPhone ambulance reference repair

**Current scene: restored pass5 plus the bounded pass7 ceiling cleanup below.
The user rejected the entire pass6 floor and
lower cabinet-wall revision**, including `combined-v2`, `floor-v7`,
`opposite-floor-v4` and `wall-native-v3`. This supersedes all earlier pass6
component and visual acceptance. Do not reinstall those historical experiments.
See `raw/ambulance-cleanup/pass6/rejection.json` and the
[current ceiling comparison](../../raw/ambulance-cleanup/pass7/review/).
Further cleanup is limited to corner and ceiling haze using surface-plane priors.

## Pass7 ceiling cleanup after rollback

`raw/ambulance-cleanup/pass7/final-v1` starts from the restored pass5 pair and
physically removes only twelve exact iPhone rows attributed to exterior roof
haze. Every retained iPhone record and the entire reference file remain exact;
there are no additions, covariance edits or material changes. The installed scene
contains 5,525,878 Gaussians. Its removed centers lie beyond the measured cabin,
so no interior, floor or lower cabinet-wall records are removed.

Plane-guided ceiling and rear-bevel covariance trials were omitted because
matched views showed no meaningful visible gain. The remaining exterior support
is retained where removal would expose gaps. The independent numeric audit is
saved as `pass7/final-v1/independent-verification.json`; it does not establish
visual acceptance. The exact exported SOG passed 24 matched views, with explicit
floor/lower-wall restoration checks. `pass7/final-v1/installed-report.json`
records the installed SHA256 and review hashes; the numerical `report.json`
remains unchanged so its audit stays reproducible. The improvement is confined
to the front ceiling puffs; residual corner haze remains. The installed scene and
tools are included in the repository; raw review data remain local.

The source captures remain unchanged:

- `raw/ambulance_exp11_boot_sharp.ply`: detailed iPhone capture, DC color only.
- `raw/ambulance_exp_insta360.ply`: cleaner reference, degree-three spherical harmonics.

The second manual cleanup (`refined.ply`) is **rejected**. Its uniform roof/hatch
fills and aggressive layer projection made materials less believable. Passing
numeric checks did not establish visual quality. The local viewer was rolled
back before the new reference-transfer experiments.

## Capture-backed repair

`reference_alignment.py` registers the fixed cabin in the two captures. Roof
and pad appearance are evaluated separately from the geometry used for fitting.
The aligned reference keeps its actual Gaussian geometry, covariance and full
view-dependent color; it is not a texture-painted plane. The converter rotates
all SH coefficients when transforming the reference.

The registered reference lives at
`raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply`. Both it and
the original iPhone PLY use the native iPhone coordinate frame. The alignment,
held-out region checks, transform stability and SH export checks are recorded
beside it. One nonfinite source reference row is excluded.

The new experiments use explicit bounded masks:

- `transfer_reference_roof.py`: actual roof and optional inclined lamp fascias.
- `transfer_reference_pads.py`: right pad replacement, including measured
  background contributions and suppression of newly exposed original layers.
- `transfer_reference_walls.py`: cabinet/rear-door material experiments.
- `transfer_reference_seat.py`: attendant-seat capture transfer.
- `transfer_reference_bench.py`: exposed bench-center capture transfer.
- `pass3/diagnostics/reference_cleanup.py` (under the raw cleanup directory):
  conservative reference-guided opacity attenuation; this produced too little
  visible improvement to include.

A trial is not approved merely because it exists. Consult the inspection report
at `raw/ambulance-cleanup/pass3/review/findings.md` and the installed viewer's
`cleanup-report.json` for the precise accepted set. In particular, early pad
interior crops exposed colorful original layers and were rejected.

The accepted right-pad V5 uses full perspective Gaussian opacity tracing in
three views. Much of the reference's clean pad appearance is supplied by
Gaussians behind the apparent surface. Those contributing rows are retained,
including their original SH values. The combined model is traced again after
replacement: removing one original layer can expose another that was invisible
in the source scan. Only attributed original contributions behind the pad are
suppressed. This is an appearance repair; retained deep support is not evidence
of a physically accurate pad shell. Failed physical-only crops are archived.

`assemble_reference_repair.py` composes selected patch folders. It starts from
the original iPhone scan, changes only the selected opacity values, and appends
actual aligned reference rows. Overlapping masks are combined once and reference
row IDs are deduplicated. All geometry, covariance, DC color and SH values are
checked against their capture sources; unselected iPhone rows remain exact.
No new uniform backing sheets or colors are synthesized.

The reviewed component set is roof/fascia V2, left cabinet V3, right pads V5,
attendant seat V2, and exposed bench center V2. The bench retains the original
rear rim to avoid a coverage leak. `pass3/final-bench/report.json` records the
exact selection hashes and capture row counts: 5,429,055 combined Gaussians,
including 308,984 unique reference rows. To reproduce the composition from
the frozen selections:

```sh
.venv-fusion/bin/python -B tools/ambulance/assemble_reference_repair.py \
  --patch raw/ambulance-cleanup/pass3/trials/roof-v2 \
  --patch raw/ambulance-cleanup/pass3/trials/walls-v3 \
  --patch raw/ambulance-cleanup/pass3/trials/pads-v5 \
  --patch raw/ambulance-cleanup/pass3/trials/seat-v2 \
  --patch raw/ambulance-cleanup/pass3/trials/bench-v2 \
  --out raw/ambulance-cleanup/pass3/final-bench
```

Render the compressed output again after any recipe change; old image manifests
must not be reused for a changed model.

## Floor and wall follow-up

The pass4 composition adds seven bounded repairs to the accepted five-component
pass3 model: the visible aisle floor, vertical bench base, left backrests,
overhead cabinet recess, right bolted trim and white gap, small black cabinet
protector, and bright upper-pad fringe. The floor is distinct from the vertical
bench base. The original rear doors remain: both new rear-paint trials produced
an unwanted colored boundary and were rejected.

The source selections are frozen in these additional component folders under
`raw/ambulance-cleanup/pass4/`:

- `floor-v4`: full observed aisle footprint, source cable-curve protection,
  attributed raised gray haze removal and actual native appearance support.
- `plinth-v2`: captured vertical bench-base material, with four manually
  attributed overlay corrections. Blue source row 2231388 intentionally
  overrides the initial saturation guard; it is a cloud on the neutral face,
  not a yellow/red foreground restraint.
- `left-seats-v5`: two fixed backrests and anchors, retaining the original loose
  cushion straps. Oblique-view attribution removes a duplicate white-point
  cluster behind the lower pad.
- `walls-recess-v1`, `walls-right-trim-v1`, `walls-protector-v1`, and
  `walls-pad-edge-v1`: reviewed static wall details and a narrow source-only
  upper-pad fringe correction.

`pass4/combined-v1/report.json` records all twelve component paths, their source
hashes, selections and output hashes. Its uncompressed composition contains
5,478,961 records: 5,120,071 original iPhone rows and 358,890 unique reference
rows. Independent `verification.json` checks all fields and the min/max union;
4,179,865 unselected iPhone records remain byte-exact. Only selected opacity is
modified. The review includes the existing seventeen cameras plus two downward
floor views, with the same physical framing and near plane of .02.

`render_reference_trial.py` renders the two uncompressed PLYs together, avoiding
repeated SH compression during manual trials. Rotation is applied after merging.
The final SOG must still be independently rendered and checked before installation:

```sh
.venv-fusion/bin/python -B tools/ambulance/render_reference_trial.py \
  --input raw/ambulance-cleanup/pass4/combined-v1/iphone.ply \
  --input raw/ambulance-cleanup/pass4/combined-v1/reference-patches.ply \
  --camera-file raw/ambulance-cleanup/pass4/all-review-cameras.json \
  --label combined-v1 --out raw/ambulance-cleanup/pass4/review
```

The review preserves failed trials, original iPhone, the previously accepted
scene and native Insta360 as separate comparisons. Fine native floor wisps,
some pad-edge flecks, cabinet-glass blur and movable-object artifacts remain.
These appearance repairs do not assert a physically accurate recovered shell.

## Corner and wall-material follow-up

Pass5 uses the accepted pass4 model as its comparison baseline. The corner
covers have pale/blue reconstruction swirls absent from the native reference.
The coarse dirt around the monitor is different: pixel attribution places most
of it on the wall plane itself. Global opacity reduction cannot recover that
material. Fine laminate grain and crosshatch are present in the native capture
and are retained; bounded material regions use its actual measured splats.

- `transfer_reference_corners.py` handles fixed dark covers/header material,
  with separate protected clock, grille, net and equipment regions.
- `transfer_reference_white_walls.py` handles the wall column, backsplash,
  shelf underside and exposed right-wall material, preserving original fixtures.
- `refine_pad_residuals.py` traces the already mixed scene at denser pad-edge
  rays, then applies eight exact contributor corrections for five surviving
  white flecks. The accepted reference remains byte-exact for this component.
- `guard_header_controls.py`, `protect_corner_wall.py` and
  `protect_white_wall_label.py` preserve original roof, hatch, neighbouring-wall
  and printed-label contributions that are shared across a repair boundary.
  The earlier corner/header and shelf trials exposed hidden layers or
  mixed two versions of a label; those trials are not included in the final set.
- `trace_gaussian_pixels.py` records per-pixel opacity, depth and source IDs.
  Its reported color uses DC only; opacity attribution still includes the
  actual accepted reference coverage. Use full GPU renders for appearance.
- `verify_reference_composition.py` checks every output field, independently
  reconstructs the component min/max union and confirms that accepted source
  suppression and reference support are preserved. It does not approve visuals.

The installed `viewers/ambulance/cleanup-report.json` identifies the exact
accepted component set. Pass5's final comparison and QA manifest use newly
rendered exported-model images, with pass4 as the default before image. Trials
and numerical checks alone do not establish visual acceptance. Raw review artifacts stay local.

The pass5 composition has 21 components and 5,525,890 Gaussians, including
405,819 unique captured reference rows. Independent all-record checks preserve
3,681,877 unselected iPhone records byte-for-byte and every original geometry,
covariance and color field before SOG compression. The header's final V4 guard
uses full projected-ellipse overlap with the stepped roof silhouette: sparse
control rays missed thin strands. This deliberately retains some original
upper-header shading to preserve the accepted ceiling. Frozen selections live
in `pass5/accepted-components.json`; the complete numerical and component
provenance audits accompany `pass5/combined-v1/report.json`.

Example roof/fascia rebuild and composition:

```sh
.venv-fusion/bin/python -B tools/ambulance/reference_alignment.py --export
.venv-fusion/bin/python -B tools/ambulance/transfer_reference_roof.py \
  --reference raw/ambulance-cleanup/pass3/alignment/insta-aligned-iphone-raw.ply \
  --out raw/ambulance-cleanup/pass3/trials/roof-v2 --fascias
.venv-fusion/bin/python -B tools/ambulance/assemble_reference_repair.py \
  --patch raw/ambulance-cleanup/pass3/trials/roof-v2 \
  --out raw/ambulance-cleanup/pass3/roof-example
npx --yes @playcanvas/splat-transform@3.4.2 \
  raw/ambulance-cleanup/pass3/roof-example/iphone.ply \
  raw/ambulance-cleanup/pass3/roof-example/reference-patches.ply \
  raw/ambulance-cleanup/pass3/roof-example/hybrid.sog \
  -r -78.243,0.463,-4.499 -w
```

The rotation follows the output path so it applies to both inputs together.
Do not rotate only one input. Do not recenter the result. World position equals
`EulerXYZ(-78.243,0.463,-4.499) * diag(-1,-1,1) * rawPosition`; the converter
supplies the PLY format flip.

## Rejected pass6 floor and lower wall experiments

The user rejected **all latest pass6 floor and lower cabinet-wall changes**,
including the installed `combined-v2` and its `floor-v7`, `opposite-floor-v4`
and `wall-native-v3` components. The local scene has been restored exactly to
pass5 (`6e7aa0f7463a3e297531394aaba7054f0b7c7554099c18b168e0dcce5695da47`).
This latest rejection supersedes the earlier approval of floor V7 and all
component, numerical and 24-view acceptance statements saved in pass6 reports.
`pass6/rejection.json` records the controlling decision; frozen generation
reports remain unchanged as historical evidence.

The earlier generated wall-grain V4 and `combined-v1` were also rejected.
**Everything below describes rejected experimental recipes, not the current
viewer or approved changes. Do not install either pass6 composition.**

The rejected `wall-native-v3` experiment retained the actual registered Insta360 wall
positions, colors and SH, including natural spatial grain and shading. Bounded
tangent-shape cleanup reduces long needle splats without synthesizing a new
material. Masks follow the measured sloped metal trim and protect the original
printed label. Perspective contributor tracing removes attributed iPhone
overlay ghosts, including a final set of thirteen behind-wall colored rows.
Deep native contributors needed for coverage remain; this appearance repair
does not establish a physically accurate wall shell. The lower panel retains
some coarse captured reconstruction. It is not a translated texture patch.

The rejected floor V7 experiment translated a dense captured tread patch along the fitted
floor and retains a thin sampled-color base beneath it. The new tread's normal
sigma is .00015 scene units; its measured three-sigma support stays below the
floor plane and inside the aisle boundaries in the uncompressed PLY. The
12,859 marked reference film/web rows are physically omitted at composition.
The old rectangular floor mask is clipped against the measured bench face.
Captured curves and small dark marks remain, with bed, wheel and bench contacts
checked from low, downward and grazing views. Repeated captured tread is an
explicit reconstruction of missing coverage, not exact unobserved floor detail.

The rejected `opposite-floor-v4` experiment treated the narrow cabinet-side lane separately. It removes
470 attributed original films, extends the same captured tread with locally
sampled shadow color, and confines two raised original floor sheets normally
to the fitted floor. Bed, plinth and seat supports are protected. Both floor
components keep their new three-sigma footprints within measured safe strips.

`compose_surface_repairs.py` applies field-level differences against frozen
pass5 files, rejects conflicting edits, appends attributed additions, and
physically removes opacity-floor records. `row-provenance.npz` maps every
surviving row to its baseline or component source. The independent
`verify_surface_composition.py` checks the full retained/deleted partition,
all serialized fields, additions, source hashes and finite values. Original
capture files remain unchanged. The historical component list is saved in
`pass6/accepted-components.json`; despite its filename, its approval is now
superseded by the user rejection. Historical provenance accompanies
`pass6/combined-v2/report.json`.

Historical reproduction only — this command describes the rejected experiment
and is not an installation recipe:

```sh
.venv-fusion/bin/python -B tools/ambulance/compose_surface_repairs.py \
  --baseline raw/ambulance-cleanup/pass5/combined-v1 \
  --patch raw/ambulance-cleanup/pass6/floor-v7 \
  --patch raw/ambulance-cleanup/pass6/wall-native-v3 \
  --patch raw/ambulance-cleanup/pass6/opposite-floor-v4 \
  --out raw/ambulance-cleanup/pass6/combined-v2
.venv-fusion/bin/python -B tools/ambulance/verify_surface_composition.py \
  --directory raw/ambulance-cleanup/pass6/combined-v2
```

The prior installation used a separate floor-compaction audit and a fresh
review of the exact exported SOG. Those checks did not establish user acceptance
and must not override the subsequent rejection. The installer preserved its
reviewed composition report as historical evidence.

The existing separate `viewers/ambulance/index.voxel.json` collider is not
rebuilt by this repair. Before later collision checks, rebuild or validate the
collision representation against the repaired geometry. Thin-surface numerical
bounds describe the PLY before lossy SOG encoding and do not certify whole-scene
collision safety or calibrated metric dimensions.

## Visual acceptance

The archived, user-rejected pass6 review used all Gaussians and full perspective
anisotropic rendering from 24 fixed cameras: the earlier 19 close/frontal/grazing, opening and downward
views, two low bedside-floor views, two close/grazing wall-material views,
and a low view of the cabinet-side floor lane.
Near plane `.02`,
1000×750 output, FOV and background are matched. No image enhancement is used.
The native reference camera is inverse-mapped with scale-adjusted clipping to
show the same physical view. Each image manifest stores camera/source hashes.
These archived comparisons do not approve pass6. The current installed scene
and its review are pass5; the user rejection takes precedence over earlier QA.

```sh
python3 tools/ambulance/pass2_review.py \
  --input raw/ambulance-cleanup/pass3/roof-example/hybrid.sog --frame sog \
  --label roof-example --out raw/ambulance-cleanup/pass3/review
```

Judge material texture, reflections, seams, fixtures and unrelated control
regions, not simply fewer specks. Moving bedding, bag contents and some straps
differ between captures, so unrestricted fusion is inappropriate. Some haze on
unpatched iPhone objects remains. Numeric verification is separate from visual
acceptance.

The viewer has a minimum near clipping distance of `.08` scene units. Inspection
renders deliberately use `.02` to expose artifacts independently of that setting.
Do not regenerate the viewer HTML/JS and lose its coordinate/clipping fixes.

## Preserved historical versions

- `raw/ambulance-cleanup/original/`: original viewer assets.
- `raw/ambulance-cleanup/cleaned.ply`: initial conservative attenuation and bench
  backing; `cleanup.py`, `repair_surfaces.py`, `verify_cleanup.py` reproduce it.
- `raw/ambulance-cleanup/pass2/baseline/`: frozen first-pass viewer and PLY.
- `raw/ambulance-cleanup/refined.ply`: rejected second pass, reproduced by
  `refine_surfaces.py`, `architectural_cleanup.py`, `object_cleanup.py`.
  `verify_refinement.py` establishes numeric integrity only. Do not reinstall
  it on the strength of those checks or the superseded pass2 approval.
- `raw/ambulance-cleanup/pass3/rejected-pass2/`: rejected installed assets saved
  before rollback.

The user authorized publishing the installed scene and viewer. Rejected experiment
scripts remain historical tools; their outputs must not replace the accepted scene.
Raw captures and intermediate review artifacts are excluded by `.gitignore`.
