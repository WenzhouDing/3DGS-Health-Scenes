# Local front/back manikin fusion

This workflow combines the two Gaussian scans in the front capture's pose. The active revision, **`feature-guided-v11-finger-side-coverage`**, restores a narrow strip of original-source Gaussians along the finger sides and corrects localized color artifacts. The right forearm, wrist pad and hand remain one rigid body at the accepted version 10 pose. No wrist joint, finger displacement, Gaussian enlargement or new Gaussian centers are introduced. Processing and review stay local, and the original PLY files are untouched.

The review viewer displays a static fused scan. The separate [interactive manikin](../mannequin-articulation/) applies joints, poses, and springs to this same fused export. The published fusion viewer includes its prepared front/back captures and segmentation labels. Before/after review galleries remain local workspace artifacts, so their links are shown only when opening the viewer through a loopback HTTP address.

## Run and review

Run from the repository root. Reuse the existing `.venv-fusion`, or create one with Python 3.9–3.12. Package installation needs access to the package index.

```sh
python3 -m venv .venv-fusion
.venv-fusion/bin/python -m pip install numpy==2.0.2 scipy==1.13.1 matplotlib pillow numba==0.60.0
.venv-fusion/bin/python -B tools/fusion/pipeline.py
.venv-fusion/bin/python -B tools/fusion/verify_output.py
.venv-fusion/bin/python -B tools/fusion/verify_arm_gap_v10.py \
  --baseline raw/fusion-work/refinement-v11/baseline \
  --evidence tools/fusion/right-finger-seam-evidence.json \
  --review raw/mannequin-fused/finger-side-review/ \
  --changed-parts right_hand --allow-hand-color-repair
.venv-fusion/bin/python -B tools/fusion/make_review.py --title "Finger sides: original scan coverage restored" \
  --before raw/fusion-work/refinement-v11/baseline --after raw/mannequin-fused \
  --out raw/mannequin-fused/finger-side-review --size 800 \
  --regions right-finger-sides,right-wrist-pad,right-hand-under
python3 tools/serve.py 8766
```

Open [Fusion Review](http://127.0.0.1:8766/viewers/mannequin-fusion/) or the [finger-side before/after review](http://127.0.0.1:8766/raw/mannequin-fused/finger-side-review/). Reuse an existing server on port 8766. Opening the HTML directly with `file://` will not load the data.

- Drag to orbit, right-drag to pan, and use the wheel to zoom.
- Toggle **Front only**, **Back only**, and **Both** without moving the camera. These show each retained contribution after filtering and fusion weighting, rather than the untouched source.
- Use **Capture colors** to locate a source seam and **Body-part colors** to inspect segmentation. Select a part and choose **Frame selection** for close inspection.
- To inspect a hand from underneath, select that hand, choose **Frame selection**, then **Feet**, and orbit slightly toward the palm and back. Toggle captures without moving the camera.
- Inspect front, back, both sides, and oblique views. The viewer's report gives the actual recipe and registration evidence for that run.

The browser budget is **2,000,000 Gaussians per source**, large enough to display the complete current fused result. The PLY export always contains every retained Gaussian regardless of browser budget. Reduce `previewMaxPerSource` and regenerate if a smaller browser dataset is needed; consult `report.json` and the browser manifest for exact counts.

The version 11 gallery checks the finger sides, pad and hand underside from eight angles each: **48 before/after renders**. Every comparison uses identical cameras and renders the full anisotropic Gaussian shape, opacity, and color. It uses:

```text
Before: raw/fusion-work/refinement-v11/baseline
After:  raw/mannequin-fused
Review: raw/mannequin-fused/finger-side-review
```

Earlier evidence remains available in the [version 10 arm fit](http://127.0.0.1:8766/raw/mannequin-fused/hand-gap-review/), [version 9 rigid wrist review](http://127.0.0.1:8766/raw/mannequin-fused/rigid-wrist-review/), [version 7 left fingertip review](http://127.0.0.1:8766/raw/mannequin-fused/left-hand-review/), [version 6 both-hand underside review](http://127.0.0.1:8766/raw/mannequin-fused/hand-review/), [version 5 grey-cable review](http://127.0.0.1:8766/raw/mannequin-fused/cable-review/), [version 4 refinement review](http://127.0.0.1:8766/raw/mannequin-fused/refinement-review/), [version 3 hair/clothing cleanup review](http://127.0.0.1:8766/raw/mannequin-fused/cleanup-review/) and [feature-alignment review](http://127.0.0.1:8766/raw/mannequin-fused/render-review/). `make_review.py` still defaults to the earlier alignment baseline/output, so use the explicit version 11 command above for the current comparison.

The saved baseline must contain its PLY, part labels, capture labels, source-vertex indices, and report. For another comparison, pass `--before path/to/old-export --after path/to/new-export --out path/to/review`. Use `--regions right-shoulder,right-foot` for a focused rerender. The first render may take longer while Numba compiles the rasterizer.

Optional sampled-point diagnostics are still available with `tools/fusion/diagnose_fusion.py`. Their cross-sections and support plots help diagnose geometry, but the rendered review is the final appearance check.

## Files and coordinate conventions

| File | Purpose |
|---|---|
| `raw/mannequin_front_119999.ply`, `raw/mannequin_back_119999.ply` | Untouched source captures |
| [fusion-config.json](../../tools/fusion/fusion-config.json) | Active recipe, including paths, quality settings, segmentation, blending, and manual corrections |
| [front-landmarks.json](../../tools/fusion/front-landmarks.json), [back-landmarks.json](../../tools/fusion/back-landmarks.json) | Native-coordinate anatomical landmarks for each capture |
| [feature-transforms.json](../../tools/fusion/feature-transforms.json) | Reviewed absolute per-part transforms, guarded by both source hashes and one shared scale |
| [shoulder-feature-evidence.json](../../tools/fusion/shoulder-feature-evidence.json) | Paired curved pad rims, fits, and segmentation patches |
| [torso-features.json](../../tools/fusion/torso-features.json), [torso-feature-evidence.json](../../tools/fusion/torso-feature-evidence.json) | Six corresponding torso side-panel ports and their reviewed rigid fit |
| [head-feature-evidence.json](../../tools/fusion/head-feature-evidence.json) | Molded-ear alignment evidence and explicit skull/cervical-cuff segmentation |
| [limb-feature-evidence.json](../../tools/fusion/limb-feature-evidence.json) | Native knee/ankle screw centers, limb fits, elbow updates, and rejected experiments |
| [right-arm-axial-evidence.json](../../tools/fusion/right-arm-axial-evidence.json) | Measured right collar rings, native pivot/axis fits, and the accepted rotation-only relationship |
| [left-arm-axial-evidence.json](../../tools/fusion/left-arm-axial-evidence.json) | Joint shoulder-pad/collar fit, zero relative forearm twist, and reviewed clean hand rest alignment |
| [back-left-gray-cable.json](../../tools/fusion/part-masks/back-left-gray-cable.json), [refine_cable_v5.py](../../tools/fusion/refine_cable_v5.py) | Source-index cable ownership selection and its editable native-centerline recipe |
| [hair_shape.py](../../tools/fusion/hair_shape.py) | Posterior-hair appearance field, defaults, protected anchors and covariance update |
| [cleanup-masks/](../../tools/fusion/cleanup-masks/) | Reviewed original-source vertex selections and hash-guarded metadata |
| [cleanup_pants.py](../../tools/fusion/cleanup_pants.py), [cleanup_legs.py](../../tools/fusion/cleanup_legs.py), [cleanup_hand.py](../../tools/fusion/cleanup_hand.py) | Editable artifact-selection recipes and comparison renders |
| [refine_hand_v4.py](../../tools/fusion/refine_hand_v4.py), [refine_arm_v4.py](../../tools/fusion/refine_arm_v4.py), [refine_feet_v4.py](../../tools/fusion/refine_feet_v4.py) | Version 4 visibility cleanup, axial fitting, and localized foot appearance recipes |
| [refine_left_hand_v6.py](../../tools/fusion/refine_left_hand_v6.py), [refine_right_hand_v6.py](../../tools/fusion/refine_right_hand_v6.py) | Native-source hand inspection, underside comparisons and editable visibility-selection recipes |
| [left-hand-back-v6.json](../../tools/fusion/surface-restorations/left-hand-back-v6.json) | Hash-guarded original-source selection restoring reviewed back fingertips |
| [left-hand-front-caps-v7.json](../../tools/fusion/surface-restorations/left-hand-front-caps-v7.json), [left-hand-back-caps-v7.json](../../tools/fusion/surface-restorations/left-hand-back-caps-v7.json) | Additional observed fingertip surfaces, disjoint from earlier restoration selections |
| [hand_digits.py](../../tools/fusion/hand_digits.py), [left-hand-v7-evidence.json](../../tools/fusion/left-hand-v7-evidence.json) | Editable lengthwise finger correction, protected regions and reviewed evidence |
| [refine_left_caps_v7.py](../../tools/fusion/refine_left_caps_v7.py) | Regenerate the additional source-index fingertip restoration selections |
| [right-finger-seam-evidence.json](../../tools/fusion/right-finger-seam-evidence.json), [audit_hand_coverage_v11.py](../../tools/fusion/audit_hand_coverage_v11.py) | Current original-source finger-side restoration, color repair and reproducible selection |
| [right-arm-gap-evidence.json](../../tools/fusion/right-arm-gap-evidence.json), [refine_rigid_arm_v10.py](../../tools/fusion/refine_rigid_arm_v10.py) | Current rigid arm-chain fit, independent curved-shaft measurements, safeguards and visual review |
| [right-rigid-wrist-evidence.json](../../tools/fusion/right-rigid-wrist-evidence.json), [audit_rigid_wrist_v9.py](../../tools/fusion/audit_rigid_wrist_v9.py) | Historical version 9 evidence removing the wrist deformation |
| [back-right-wrist-pad.json](../../tools/fusion/part-masks/back-right-wrist-pad.json) | Hash-guarded original pad selection for verifying rigid continuity |
| [appearance_colors.py](../../tools/fusion/appearance_colors.py) | Hash-guarded sparse DC-color overrides with original-value validation |
| [mannequin_fused.ply](../../raw/mannequin-fused/mannequin_fused.ply) | Full retained fused Gaussian dataset |
| [report.json](../../raw/mannequin-fused/report.json) | Exact source hashes, counts, transforms, parameters, and evidence for the latest export |
| `raw/mannequin-fused/part-labels.npy` | One part label per exported Gaussian |
| `raw/mannequin-fused/capture-labels.npy` | Source per exported Gaussian: `0` front, `1` back |
| `raw/mannequin-fused/source-vertex-indices.npy` | Original source vertex index per exported Gaussian |
| `viewers/mannequin-fusion/{front,back}.bin` | Browser data, subject only to the configured preview budget |

The exported PLY and browser use `[front_raw_x, -front_raw_y, -front_raw_z]`: +Y faces the anterior surface and +Z points toward the head. Landmarks, feature transforms, and manual corrections use the original raw coordinates. These are scan units, not independently calibrated meters.

Anatomical left is front raw +X and back raw −X. Back-to-front initialization uses the proper rotation `diag(-1, -1, 1)`, not a reflection. Each reviewed transform then applies `front_position = scale × rotation × back_position + translation`.

Part label `i` identifies `report.json["parts"][i]["id"]`; do not assume alphabetical order. All three `.npy` arrays follow the exact full PLY row order. For row `j`, the capture label selects the source PLY and the source vertex index selects its original row. Browser label files follow the separately budgeted browser rows and cannot replace the full-export sidecars.

## Alignment and current cleanup

1. **Segment each source in 3D.** Elliptical anatomical capsules provide initial assignments. Source-specific shoulder-pad patches keep a complete pad on one rigid body; the earlier capsule boundary cut some pads between torso and upper arm. Explicit skull and cervical-cuff seam planes separate head, neck, and torso. Front elbow landmarks were moved from the forearm panel to the visible elbow seam after inspection from multiple directions.
2. **Fit corresponding physical features.** Curved shoulder-pad rims constrain the upper arms; six side-panel ports constrain the torso; knee and ankle screw heads constrain the legs; molded ear features constrain skull pose. Both forearms use a measured ring pivot and axis, with one twist angle relative to their upper-arm parent; each hand is carried with its forearm. The left upper-arm fit jointly uses its pad and collar, with zero relative forearm twist in the accepted pose. Changed clothing is not treated as a rigid registration target.
3. **Use one shared scale: 0.928.** Independent pad geometry and long knee-to-ankle screw baselines support this value. The registration transforms are rigid at that common scale. The separately configured hair field and retained left-finger correction are explicit local nonrigid steps, described below. There is no right-hand deformation.
4. **Preserve cable ownership and apply reviewed transforms.** After source quality cleaning and measured joint-boundary assignment, `sourcePartAssignments` keeps the selected back cable on `left_upper_arm`, including the section crossing the collar. This happens before part transforms and coverage. `featureTransforms` supplies the accepted fits; the pipeline checks input hashes, proper rotations, the shared scale, and both forearms' axial relationships. A part without a feature fit falls back to landmark initialization and bounded same-facing overlap refinement. Source means, quaternions, and all Gaussian axes transform consistently. Alignment itself does not change DC color; the optional foot and finger-side repairs are separate later steps.
5. **Apply the configured local corrections.** The entire right back forearm and hand use one rigid transform, with no local wrist or finger deformation. The back left fingers separately receive a small lengthwise shift with a smooth transition through the knuckles, preserving their wrist, thumb and depth. A hair field adds fullness only to posterior hair and the crown, equally in both aligned captures, leaving reviewed ear/face/anterior-neck anchors fixed. Each field updates the complete Gaussian covariance as `J Σ Jᵀ`, where `J` is its deformation Jacobian. Captured DC colors are unchanged.
6. **Filter and select complementary coverage.** Neighbor support removes isolated/contact remnants. Depth weighting favors the front capture on the front surface and the back capture on the back surface. Before attenuation, `surfaceRestorations` raises the weight of precisely reviewed source rows, restoring curved left fingertips and selected right finger-side coverage that the previous cut suppressed. It never bypasses source quality filtering. Optical-depth feathering uses `alpha_new = 1 - (1 - alpha_old)^weight`. The cut can vary by part, and unique coverage is preserved where enabled.
7. **Remove reviewed local artifacts after coverage.** `cleanupMasks` points to metadata for sorted `uint32` original-source vertex indices. Both original PLY hashes must match before a selection is accepted. These cuts remove inspected contact remnants and duplicate source surfaces after blending decisions, so deletion cannot reactivate donor rescue. Exact IDs in an approved surface restoration are exempted from those cuts. The cleanup adds no global Gaussian-size filter and performs no mesh filling.
8. **Apply optional localized foot and finger-side color repair.** `colorOverrides` identifies original source rows and their reviewed replacement DC colors. Each file includes the original values, which must match before application. `colorRepairStrength` blends between original and repaired color. Centers, quaternions, scales, opacity, and segmentation are unchanged by this step.

Version 5 changes the left upper-arm parent rotation by **2.799°**, reducing the measured collar-center discrepancy from about **0.009 to 0.00197 scan units**, while checking the shoulder pad jointly. The forearm has no independent translation or swing, and no additional relative twist in the accepted fit. The left hand preserves its earlier clean absolute alignment, with its wrist rest relationship reviewed from six angles; carrying the old relative hand pose into the new arm alignment exposed purple finger ghosts and was rejected. The back source contains the continuous cable; the contact-corrupted front source provides no reliable counterpart, so no front cable is invented.

The retained version 4 cleanup removes **12,742 additional dark left-hand splats** using their actual rendered visibility from the observed and unobserved sides of each native scan. It does not recolor the hand. It also removes **934 additional right-ankle artifact splats** and repairs DC color on **6,726 foot splats** to reduce local sole discoloration. These are refinement counts; use the latest `report.json` for total export counts and the exact applied recipe.

The back left ankle/foot already contains duplicate hardware and stretched reconstruction within that individual scan. Its final recipe therefore favors the clearer front contribution and filters unusually large back Gaussians. A rigid transform alone cannot repair an internally distorted source.

## Fine-tune position, segmentation, and seams

Select one part, toggle captures, inspect side and oblique views, change one cause, then regenerate both the export and render review. Use a separate recipe with `pipeline.py --config path/to/recipe.json` for experiments; paths inside the recipe resolve from the repository root. Pipeline runs replace their configured output files.

### Fine-tune the right finger-side repair

The remaining longitudinal slit came from weak near-side coverage. More shoulder or axial rotation did not remove it. Version 11 restores selected **original-source rows**, including rows previously removed by cleanup, and raises their coverage weights to the original opacity. The selection follows the measured finger shells on the positive lateral side of the index, middle and ring fingers. These lower-confidence source rows are selected by shell proximity, normal direction, size and connection to stronger observed surfaces; they are not newly captured detail.

The active source selections are `tools/fusion/surface-restorations/right-hand-{front,back}-sides-v11.json`. Their `maskFile` arrays contain sorted original PLY row IDs. Separate `tools/fusion/cleanup-masks/right-hand-{front,back}-side-color-v11.json` files guard the local replacement colors against the native values. The exact selectors, source hashes, counts, rejected trials and rendered review are recorded in `right-finger-seam-evidence.json`.

- To adjust which side rows return, edit `sector_review` in `audit_hand_coverage_v11.py`, regenerate a diagnostic candidate, and inspect it from below and both lateral directions. Copy a reviewed selection into the active JSON/NPY pair and update its count and metadata before rebuilding.
- `minimumWeight: 1` restores the selected rows' original opacity. A lower weight reduces their contribution; removing the two right-hand restoration rules disables their restoration entirely. Keep the older left-hand rules when changing only this hand.
- To disable only the finger-side color correction, remove its two paths from `colorOverrides`. The global `colorRepairStrength` affects the older foot repair too.
- Keep `rigidParts.right_hand: right_forearm` and the accepted feature transforms. Do not add wrist rotation or a local finger displacement to hide a coverage seam.

Regenerate the accepted diagnostic recipe in this order (these commands do not replace the active export or promoted selection files):

```sh
.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py --broad-sector
.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py --stabilize-all broad-sidewall
.venv-fusion/bin/python -B tools/fusion/audit_hand_coverage_v11.py --fix-green-patch
```

The last step traces the index-side discoloration to two source rows and uses nearby plausible color from the other capture. Its donor distances are scan units, without metric calibration. Review the `broad-sidewall-final` images before copying a revised recipe into the active source selections and color files.

The independent checks compare every right-arm Gaussian to its original source geometry, preserve all 16 poses, check the pad across the segmentation boundary, and require the other 15 body regions to be bit-for-bit unchanged. Exact conditional Gaussian sections and isolated finger renders distinguish near-side coverage from a farther surface visible through the slit. Source softness and molded creases can remain; overlapping projected alpha alone is not proof of a watertight surface.

### Keep the right forearm and hand rigid

The rounded rectangular dorsal pad crosses the `right_forearm`/`right_hand` segmentation labels, but there is no physical joint at that boundary. Version 8 kept the pad intact while deforming the knuckles toward an independently fitted hand pose. That model was mechanically incorrect and caused a duplicated hand from grazing angles. Version 9 removed that deformation. Version 10 reduced the separation between the finger shells by fitting the shoulder rotation about the matched pad center and the existing axial elbow swivel jointly. The total shoulder correction is 1.9994 degrees, and the accepted total axial twist is 0.75064 degrees. These are recorded fits, not extra viewer rotations.

The fit uses centers inferred independently from observed curved finger arcs, with shoulder-rim and collar-center safeguards. Nine independently re-sliced sections reduce median center mismatch from 0.00621 to 0.00122 scan units while preserving finger thickness. The shoulder pad center residual stays 0.000779, and its rim 90th-percentile residual is 0.001600. The collar-center mismatch is 0.003612 scan units, with 3.93 degrees of source ring-normal discrepancy; its actual seam was reviewed from several angles. Native texture defects remain, so these measurements do not claim a watertight reconstruction. Version 10 added no hand pruning, restoration, recoloring or local deformation. Version 11 keeps this fit and restores the remaining weak side coverage as described below.

The active recipe contains:

```json
"rigidParts": { "right_hand": "right_forearm" }
```

This is stronger than `carriedByPart`: the hand uses **exactly the same scale, rotation and translation** as the forearm, with no stored relative wrist pose. Every original source distance across the wrist boundary is preserved at the common scan scale. The front scan stays in its original frame. No trustworthy front dorsal-pad rim exists, so pad continuity is checked within the original back capture; no paired pad landmarks are invented.

To fine-tune, change `adjustments.right_forearm.twistDegrees`. This adds a rotation about the measured collar axis and carries the entire forearm, pad and hand. Check the pad, fingertips from underneath, forearm edges, collar and upper-arm shoulder pad together. Do not independently pitch, translate or warp the right hand. The pipeline rejects nonzero child adjustments or `rightWristRegistration` while the rigid relationship is active. The constraint also applies to `--skip-icp` diagnostic runs.

The full pre-correction version 9 export and recipe are saved in `raw/fusion-work/refinement-v10/baseline/`. The older `refinement-v9/baseline/` records the rejected version 8 deformation for historical comparison. Use a separate output recipe to reproduce historical versions. Current verification checks **all retained right upper-arm, forearm and hand Gaussians** against their rigid source transforms, including their covariance, and checks original neighbor distances across the pad's segmentation boundary.

### Move an aligned part

**Editing landmarks alone does not override an explicit feature transform.** Landmarks still affect segmentation, local coverage frames, and the pivot for manual corrections. To move a reviewed part, use `adjustments` after its absolute fit:

```json
"adjustments": {
  "left_foot": {
    "rotationDegrees": [0, 0, 1],
    "translation": [0.003, 0, 0]
  }
}
```

This modifies the aligned **back contribution**. Rotation uses SciPy's lowercase `xyz` Euler convention in raw front axes, around the midpoint of that part's front landmarks. Translation is added afterward in raw front coordinates. It is not expressed in the flipped browser frame.

Both constrained **forearms** instead accept only an extra axial twist. For the grey-cable arm:

```json
"adjustments": {
  "left_forearm": { "twistDegrees": 2 }
}
```

This adds two degrees around the measured left collar axis; `carriedByPart: "left_forearm"` makes the left hand follow future forearm adjustments from its reviewed clean rest alignment. Use `adjustments.right_forearm.twistDegrees` identically for the other arm. Free `rotationDegrees` or `translation` adjustments on either constrained forearm are rejected because they would bend or separate the collar. The upper arm supplies its parent pose. The grey cable stays with its upper-arm socket rather than acquiring a split at the elbow. This preserves the captured cable shape; flexible cable animation would require separate routing or physics.

To change the measured pivot or axis, update the reviewed `mechanicalConstraint` (`pivotFrontRaw`, `axisFrontRaw`, and base `twistDegrees`) and regenerate the matching child transform together; independently editing the forearm matrix fails validation. Inspect the connected arm, shoulder pad, collar, wrist and cable after any change.

For a new fitted pose, regenerate the feature evidence, inspect the candidate renders, and replace that part's `sourceToFrontRaw` entry under `feature-transforms.json["parts"]`. Keep the common scale and source hashes consistent. Removing a part's entry deliberately returns it to landmark/overlap fitting. The rigid right hand always follows its forearm even if its redundant feature entry is removed. `pipeline.py --skip-icp` bypasses both feature fits and ICP for a landmarks-only diagnostic export; use a separate output recipe for that comparison, with dependent local fields handled consistently.

### Change segmentation

- Edit each capture's landmark JSON in its own native coordinates. Use real joint seams and centers; wristbands and forearm panels are not joint centers.
- Adjust `partRadii` as `[lateralRadius, depthRadius]` when an anatomical capsule is too narrow or broad.
- Edit source-specific `segmentationOverrides` to change the oriented ellipse slabs surrounding a shoulder pad. Check the whole rim, its attachment, and the elbow afterward.
- Edit the `cervicalSegmentation` evidence referenced by the config to change the skull/cuff/torso planes. These rules override capsule competition around the neck.
- For either collar, edit its entry in `jointBoundaries.front` and `.back`. The grey-cable arm names `landmark: "left_elbow"`, `proximalPart: "left_upper_arm"`, `distalPart: "left_forearm"`, and its measured native `pivot` and `axis`; the opposite arm uses the corresponding `right_` names. `maximumAbsStation` and `maximumRadius` are both 0.11. These local boundary rules run **after source cleaning**, then override the elbow landmark for alignment and coverage frames. The original landmark files remain unchanged so their capsule filtering is preserved. Coordinate this seam edit with the forearm's mechanical constraint, whose pivot and axis use aligned front raw coordinates.

The general part definitions and rule evaluation are in [pipeline.py](../../tools/fusion/pipeline.py). Raw inspection tools `inspect_front.py` and `back_diagnostics.py` write diagnostic views to `raw/fusion-work/`; diagnostic landmark copies there do not override the paths in the active config.

### Keep the grey cable with its socket

This selection assigns part ownership; it does not delete cable rows. It uses **18,084 original back-source vertex indices**, guarded by both original scan hashes:

```json
"sourcePartAssignments": [
  {
    "selection": "tools/fusion/part-masks/back-left-gray-cable.json",
    "part": "left_upper_arm",
    "allowedParts": ["left_upper_arm", "left_forearm"],
    "requireRetained": true
  }
]
```

Assignments run after quality filtering and `jointBoundaries`, before per-part transforms and fusion. `requireRetained` compares the selected rows surviving quality cleaning with the final export; it does not force rejected source artifacts back into the model. Verification confirms those rows all survive and belong to the same upper-arm part. See `sourcePartAssignmentCounts` in the report for the actual applied counts.

To adjust membership, edit `BACK_CABLE_PATH` or `CABLE_RADIUS` in `refine_cable_v5.py`, then run:

```sh
.venv-fusion/bin/python -B tools/fusion/refine_cable_v5.py --extract-only
```

This writes the candidate JSON/NPY pair and native source review to `raw/fusion-work/refinement-v5/cable/`. Inspect posterior, side and oblique views, including the socket and free connector. Then either point the config's `selection` to the new work-directory JSON, or copy the reviewed pair into `tools/fusion/part-masks/` and update the JSON's `maskFile` to the copied NPY. Rerunning the generator alone does not replace the active selection. Regenerate the export and cable review after promoting it.

### Change source quality or the seam

| Setting | Meaning |
|---|---|
| `filter.minOpacity` | Minimum sigmoid opacity; default 0.025 |
| `filter.minScale`, `maxScale` | Bounds on the largest Gaussian axis in native source units; defaults 0.000015 and 0.04 |
| `filter.isolatedSpacing` | Sixth-neighbor support threshold; default 0.026 |
| `filter.contactDepth`, `contactSpacing` | Combined local contact-side depth and sparsity test; defaults 0.005 and 0.012 |
| `filter.partOverrides.<source>.<part>` | Source-specific opacity/scale limits after segmentation |
| `fusion.feather` | Width of the depth transition; default 0.009 |
| `fusion.donorColumnRadius` | Radius used to check complementary source coverage; default 0.02 |
| `fusion.partOverrides.<part>.depthBias` | Positive values prefer more front-capture coverage |
| `fusion.partOverrides.<part>.preserveUniqueBack` | Whether missing front donor columns may restore back contributions |
| `fusion.partOverrides.<part>.preserveUniqueFront` | Whether missing back donor columns may restore front contributions |
| `hairFullness.amount` | Optional aesthetic posterior-hair fullness; 0 disables it |
| `leftHandDigitAlignment.amount` | Lengthwise back-left-finger correction in raw scan units; 0 disables only this local deformation |
| `rigidParts.right_hand` | Must be `right_forearm`: one rigid body with no independent wrist pose or hand deformation |
| `sourcePartAssignments` | Hash-guarded source-index ownership overrides before transforms and coverage; `requireRetained` verifies post-cleaning selected rows survive export |
| `surfaceRestorations` | Reviewed source rows whose coverage weight is raised before attenuation and which are exempted from post-coverage cleanup |
| `cleanupMasks` | JSON metadata paths for reviewed post-coverage original-source selections |
| `colorOverrides` | JSON metadata paths for sparse foot DC-color repairs keyed by original source row |
| `colorRepairStrength` | Repair amount from 0 to 1; 0 uses original captured colors, 1 uses the reviewed repair |

Current local quality overrides are:

```json
"filter": {
  "partOverrides": {
    "back": { "left_foot": { "maxScale": 0.008620689655172414 } }
  }
},
"fusion": {
  "partOverrides": {
    "left_foot": { "depthBias": 0.04 },
    "left_shin": { "depthBias": 0.03 },
    "head": { "preserveUniqueBack": false },
    "neck": { "preserveUniqueBack": false }
  }
}
```

Merge these entries with the other settings rather than replacing the whole `filter` or `fusion` object. Disabling unique-back rescue for head/neck prevents known contact-side remnants from reappearing after seam reassignment; it does not disable the desired posterior contribution. The foot limit is native back scale: `0.008 / 0.928`. Do not apply it globally: tests on the shin removed legitimate broad Gaussians and opened holes.

The minimum-scale test also uses the **largest** axis, so thin but useful surface Gaussians are retained. `exclusions.front` and `.back` allow tightly inspected source-coordinate boxes for accessories or floaters. The final recipe includes targeted cleanup of observed neck contact-layer ghosts in both captures, checked against the intact neck surfaces from multiple directions. Feathering can soften a seam but cannot fix a geometric offset.

## Tune appearance and local cleanup

### Hair fullness

The active recipe has a `hairFullness` object. Its default `amount` is **0.014 scan units**; set `"amount": 0` to return to the aligned captured hair shape. This does not alter the original PLY or the rigid ear/skull registration.

```json
"hairFullness": {
  "amount": 0.014
}
```

Unspecified parameters use `DEFAULT_FULLNESS` in [hair_shape.py](../../tools/fusion/hair_shape.py): center `(X, Z) = (0.006, -0.225)`, radii `(0.15, 0.215)`, depth fade `[-0.012, 0.035]`, nape fade `[-0.095, -0.130]`, and crown lift ratio `0.5`. These values use aligned **front raw** coordinates. Change `amount` first and inspect both sides, high rear and crown. Changing the support or fade settings requires checking the protected ear/face anchors again; the output report records their maximum displacement and the covariance/Jacobian audit. The original posterior hair is flat in places, so this correction must not be interpreted as recovered physical shape.

### Left-finger alignment

The reviewed local correction moves the back capture's fingers **0.0033 scan units lengthwise**, with zero depth displacement. These coordinates have no metric calibration; the amount is not a measurement in millimeters. Three independently fitted finger-shaft center curves support the small lengthwise offset. Restoring complete observed fingertip coverage supplies most of the visible improvement; shifting the entire hand toward its opposite surface was rejected because it reduced finger thickness.

```json
"leftHandDigitAlignment": {
  "amount": 0.0033
}
```

Set `amount` to `0` to disable only the local shift. The source-index restorations below stay active. To tune, change the amount in small steps and regenerate the eight-angle review; compare palm/back outlines, fingertip thickness, the knuckle transition, thumb and wrist. Values are limited to ±0.006 scan units.

Unspecified parameters use [hand_digits.py](../../tools/fusion/hand_digits.py): origin `[0.704, 0.065, 0.584]`, normalized longitudinal axis `[0.68, 0, 0.7332]`, normalized lateral axis `[0.7332, 0, -0.68]`, `longitudinalFade: [0.08, 0.125]`, and `thumbFade: [0.043, 0.055]`. All use aligned front raw coordinates. The smooth fade leaves the wrist and thumb fixed; distal fingers receive a constant translation. Gaussian covariance changes only through the transition, using `J Σ Jᵀ`. This is an explicit local deformation after the unchanged rigid hand fit, not a new whole-hand transform. Colors and original opacity stay unchanged; normal coverage weighting is recomputed afterward.

If you later change the whole-hand or arm pose, update this field's origin and axes to match, or set `amount: 0` while refitting, then rerender. Restoration selections remain attached to original native-source IDs and do not need that coordinate adjustment.

### Hand undersides: restore coverage and remove hidden layers

The left fingers curve across the hand's simple coverage plane. This suppressed useful observed fingertip surfaces, while earlier cleanup also selected some of those rows. Version 6 restored a restricted back-source region; version 7 adds observed curved surfaces from both captures, including the index finger previously outside that region. The active recipe restores only inspected original-source IDs:

```json
"surfaceRestorations": [
  {
    "selection": "tools/fusion/surface-restorations/left-hand-back-v6.json",
    "part": "left_hand",
    "minimumWeight": 1
  },
  {
    "selection": "tools/fusion/surface-restorations/left-hand-front-caps-v7.json",
    "part": "left_hand",
    "minimumWeight": 1
  },
  {
    "selection": "tools/fusion/surface-restorations/left-hand-back-caps-v7.json",
    "part": "left_hand",
    "minimumWeight": 1
  }
]
```

`minimumWeight: 1` retains the original opacity of those reviewed rows. The pipeline raises their coverage weights before attenuation and exempts exactly those IDs from old cleanup selections. It never restores rows rejected by source quality filtering. Both source hashes, one-part membership, non-overlapping restoration selections, final retention and the minimum opacity are checked. Restoration itself changes no positions, Gaussian shapes, colors or transforms; the optional local finger shift is a separate step.

To reduce the restoration, lower `minimumWeight` within the interval from `fusion.minWeight` to 1, then check all eight views. The floor only raises weights; it never reduces a larger existing weight. Remove the rule entirely to disable both restoration and its cleanup exemption. Keep the old masks in place when comparing, and inspect fingertip coverage as well as finger separation.

Both hands' cleanup measures integrated visible alpha in each original capture from its observed and opposite directions. It evaluates the full native hand after quality filtering, before the fusion cut, and removes rows whose contribution is greater from the unobserved side, across all colors rather than only dark splats. The precise left fingertip restoration remains exempt. This targets the internal sheets that look like doubled fingers from below. Those version 6–7 changes preserved all 16 rigid part transforms; version 9 replaces the right wrist fit with a single rigid forearm-hand assembly as described above. Unconstrained refitting can incorrectly pull complementary palm/back surfaces together.

Regenerate the additional version 7 cap selections with:

```sh
.venv-fusion/bin/python -B tools/fusion/refine_left_caps_v7.py --raw-verify
```

The selector checks the original scans and selects observed native surfaces beyond longitudinal station 0.105, with observed visible-alpha flux greater than opposite-side flux, visibility ratio above 0.05 and observed flux above 0.05. It removes the old lateral restriction and excludes IDs already restored. Review the generated metadata and masks before rerunning the final export.

The earlier all-color cleanup recipes remain `refine_left_hand_v6.py` and `refine_right_hand_v6.py`. Regenerate those selections with:

```sh
.venv-fusion/bin/python -B tools/fusion/refine_left_hand_v6.py --export
.venv-fusion/bin/python -B tools/fusion/refine_right_hand_v6.py --export
```

They write candidate masks, metadata and actual Gaussian comparisons under `raw/fusion-work/refinement-v6/left-hand/` and `right-hand/`. After reviewing a revised selection, either update the relevant `cleanupMasks` paths to its work-directory JSONs, or promote the JSON/NPY pairs into the active mask directory and update each `maskFile`. Follow the same source-index and metadata discipline when changing the left restoration selection; inspect its recorded recipe and the left-hand script. Regenerating candidate files alone does not update active copies. Consult the final report for the applied restoration and cleanup counts. Source softness and creases can remain at grazing angles; these changes do not promise perfect alignment or invent missing surfaces.

### Local artifact selections

`cleanupMasks` contains JSON metadata paths, not the `.npy` masks directly:

```json
"cleanupMasks": [
  "tools/fusion/cleanup-masks/pants-back.json",
  "tools/fusion/cleanup-masks/legs-front.json",
  "tools/fusion/cleanup-masks/legs-back.json",
  "tools/fusion/cleanup-masks/hand-front.json",
  "tools/fusion/cleanup-masks/hand-back.json"
]
```

The example above shows the retained version 3 selections. Keep the additional hand and foot entries already in the active config when editing this list. Each document identifies a source capture, its sorted original vertex indices, and the hashes of **both** input PLYs. The masks are applied after source coverage and opacity weighting, with the exact restoration exceptions described above. Remove a metadata entry to disable that selection, then rerun the pipeline. Preserve the frozen comparison baselines: exported row indices belong to each baseline's own dataset, while mask values refer to the original PLY rows.

To change a selection, edit its recipe and rerun the selector before regenerating the final export:

```sh
# Lower-short contact sheets and neutral bright crotch/hem remnants.
.venv-fusion/bin/python -B tools/fusion/cleanup_pants.py

# Local foot envelope, toe/heel ghosts and sparse shin remnants.
.venv-fusion/bin/python -B tools/fusion/cleanup_legs.py --mode final

# Left-hand observed-surface and opposite-shell leakage selection.
.venv-fusion/bin/python -B tools/fusion/cleanup_hand.py

# Additional dark left-hand/finger layers measured by native Gaussian visibility.
.venv-fusion/bin/python -B tools/fusion/refine_hand_v4.py --mode final

# Residual right-ankle pruning and localized foot DC-color repair.
.venv-fusion/bin/python -B tools/fusion/refine_feet_v4.py --mode final
```

These commands regenerate selection files and review artifacts; they do not replace the source scans. The pants recipe leaves the front logo, primary rear fabric and knee hardware intact. Leg selections protect joint hardware and use local checks instead of a broad size cutoff over the shin. The hand recipe preserves supported palm/back surfaces and removes overlapping leakage.

The version 4 hand selector compares integrated visible alpha from the native observed and opposite cameras. It removes dark layers dominated by the unobserved side, with a separate distal-finger rule. Its source-addressed masks and six-angle palm/back and finger-detail reviews are written under `raw/fusion-work/refinement-v4/hand`. The active config uses reviewed copies under `tools/fusion/cleanup-masks/hand-v4-{front,back}.json`. After tuning, either point those two config entries to the newly generated work-directory JSONs, or copy the reviewed JSON/NPY pairs into the active directory and update each JSON's `maskFile`. Regenerating work-directory candidates alone does not replace the active copies.

The left hand also uses this narrower blending transition:

```json
"fusion": {
  "partOverrides": {
    "left_hand": {
      "depthBias": 0.008,
      "feather": 0.004,
      "preserveUniqueFront": false,
      "preserveUniqueBack": false
    }
  }
}
```

Merge this into the other fusion settings. To compare the hand with its earlier blending as well as disabling its masks, remove the `left_hand` override. Clothing folds still differ between captures; the cleanup trims artifacts without forcing those folds into a new shape.


### Foot color repair

The local foot repair is separately switchable from pruning. This excerpt shows the foot entries only; retain the existing finger-side entries when merging it into `colorOverrides`:

```json
"colorOverrides": [
  "tools/fusion/cleanup-masks/feet-v4-color-front.json",
  "tools/fusion/cleanup-masks/feet-v4-color-back.json"
],
"colorRepairStrength": 1
```

The global `colorRepairStrength` controls both foot and finger-side patches: use `0` for captured colors or `0.5` for half of each reviewed correction. Remove only the two foot paths to disable soles separately. The stored foot target already blends 80% neighboring reference color with 20% original color. This setting controls how far to move toward that target; it does not remove additional Gaussians.

The recipe in `refine_feet_v4.py` selects posterior sole color outliers above an RGB-distance threshold of 0.12, uses the median of 12 nearby plausible skin-color samples, and protects ankle screws and toe creases. Source IDs and both scan hashes guard every patch; each NPZ stores `indices`, `f_dc`, and `original_f_dc`. To change selection thresholds or the neighbor blend, edit the recipe, rerun `--mode final`, then regenerate the pipeline and review. Inspect the sole and oblique cameras as well as the front. Broad size pruning was rejected because it exposed streaks and holes; color repair improves appearance without claiming new geometric detail.

## Refit features and validate

These commands generate candidates/evidence; they do not automatically approve every candidate or replace the active `feature-transforms.json` entries:

```sh
# Shoulder pads; edit tools/fusion/shoulder-features.json first if needed.
.venv-fusion/bin/python -B tools/fusion/refine_features.py

# Torso: six side-panel ports configured in tools/fusion/torso-features.json.
.venv-fusion/bin/python -B tools/fusion/refine_features.py --torso

# Head/cervical cuff; --rerender recreates the native feature images.
.venv-fusion/bin/python -B tools/fusion/refine_head.py --scale .928 --rerender --render

# Knee/ankle hardware candidates.
.venv-fusion/bin/python -B tools/fusion/refine_limbs.py --hardware --scale .928

# Measured right collar, one-axis fitting, connected-arm renders and evidence.
.venv-fusion/bin/python -B tools/fusion/refine_arm_v4.py --all

# Current chain: fit shoulder rotation, shoulder yaw and axial swivel.
.venv-fusion/bin/python -B tools/fusion/refine_rigid_arm_v10.py --fit-centers --export

# Independently re-slice the moved original finger surfaces.
.venv-fusion/bin/python -B tools/fusion/inspect_finger_gap_v10.py --reslice-candidates raw/fusion-work/refinement-v10/registration/center-fit.json

# Grey-cable arm: joint pad/collar fit and final hand/wrist review.
.venv-fusion/bin/python -B tools/fusion/audit_arm_v5.py --measure --audit --parents --render --hand-rest --evidence
```

Edit the measured feature definitions rather than assuming approximate anatomical endpoints correspond perfectly. Shoulder and torso measurements are configured in `shoulder-features.json` and `torso-features.json`; the head workflow reads `head-feature-evidence.json`. Limb screw centers and front elbow updates are the `HARDWARE` and `FRONT_ELBOW_UPDATES` definitions in `refine_limbs.py`, also recorded in its evidence JSON. Update those definitions before refitting measurements; directly editing a recorded residual does not change the fit.

Limb fitting preserves a frozen baseline and checks source hashes, avoiding cumulative correction of an already edited result. Its accepted evidence explicitly rejects the left-foot hardware candidate because of duplicate source features. Rerunning `--hardware` produces that candidate for comparison; do not replace the accepted left-foot fit without reviewing it.

The earlier right-arm workflow uses the frozen version 3 state under `raw/fusion-work/refinement-v4/baseline`; its ring measurements and single-axis search are recorded in `right-arm-axial-evidence.json`. The historical version 9 sweep uses `raw/fusion-work/refinement-v9/baseline` and writes rigid forearm/hand candidates to `raw/fusion-work/refinement-v9/registration`, without a wrist deformation. Version 10 uses the frozen refinement-v10 baseline; its fit-centers recipe fits three bounded parameters to reliable curved-shaft centers plus pad/collar safeguards. Edit the bounds and residual weights in that function for experiments, then independently re-slice and render every candidate. These diagnostic commands do not replace active transforms. The grey-cable arm uses `raw/fusion-work/refinement-v5/baseline`, with the joint pad/collar parent fit, axial constraint and carried hand recorded in `left-arm-axial-evidence.json`. The cable selector above only regenerates ownership; it does not refit the arm. Review the measured seam, shoulder, connected arm and wrist before updating `jointBoundaries` and either active mechanical constraint. Do not replace a constrained forearm with an unconstrained overlap candidate merely because its nearest-surface residual is smaller.

Rebuild and verify the current version 11 source-coverage recipe with the accepted transforms unchanged:

```sh
.venv-fusion/bin/python -B -m unittest discover -s tools/fusion -p 'test_*.py'
.venv-fusion/bin/python -B tools/fusion/pipeline.py
.venv-fusion/bin/python -B tools/fusion/verify_output.py
.venv-fusion/bin/python -B tools/fusion/verify_arm_gap_v10.py \
  --baseline raw/fusion-work/refinement-v11/baseline \
  --evidence tools/fusion/right-finger-seam-evidence.json \
  --review raw/mannequin-fused/finger-side-review/ \
  --changed-parts right_hand --allow-hand-color-repair
.venv-fusion/bin/python -B tools/fusion/make_review.py --title "Finger sides: original scan coverage restored" \
  --before raw/fusion-work/refinement-v11/baseline --after raw/mannequin-fused \
  --out raw/mannequin-fused/finger-side-review --size 800 \
  --regions right-finger-sides,right-wrist-pad,right-hand-under
```

For a new pose or segmentation fit, freeze its own baseline and run `verify_output.py` plus a matching visual review. The hand-only check above deliberately rejects any change to the 16 accepted poses; it is for this coverage refinement.

The verifier checks exported values, quaternion normalization, source hashes, cleanup selections, and provenance samples against original Gaussian parameters and recorded transforms, including the configured hair field and its covariance update. It checks the selected foot and finger-side color replacements against their stored originals and expected repair strength, and cable ownership against the selected post-cleaning rows with retention counts. Surface restoration checks include the selected part, retained rows and the configured opacity floor. The report records both applied axial constraints and carried-hand relationships. Verification writes `raw/mannequin-fused/verification.json`. Numerical checks complement the rendered comparisons; low residuals on a small overlap patch do not prove an entire body part is correct.

The [arm-gap verification](../../raw/mannequin-fused/right-arm-gap-verification.json) checks every retained right upper-arm, forearm and hand Gaussian against its original source under the recorded rigid poses, including the unchanged native shape and transformed covariance. Right-hand source additions, opacity changes and colors must match the declared sparse repairs; all 16 rigid transforms match the frozen version 10 baseline. It also checks that the **15 other regions** retain the same source IDs, Gaussian values and rigid poses. The previous left-hand correction is included in that preservation check. Exact current counts and parameters are recorded in `report.json`; numerical checks complement the multi-angle visual review.

Version 7's earlier [left-hand preservation check](../../raw/mannequin-fused/left-hand-refinement-verification.json) records its own comparison against the version 7 baseline; its all-16-transforms-unchanged result describes that earlier revision, not the new right wrist fit.

## Remaining limits

The wig, clothing, lighting, and exposure differ between captures. Parts of the back left ankle/foot are intrinsically duplicated or distorted, and hidden/contact surfaces remain incomplete. Feature alignment, source selection and local cleanup improve the fusion but do not create missing capture observations, reconstruct a watertight mesh, or guarantee invisible seams. Hair fullness deliberately edits existing hair geometry; the foot and finger-side repairs deliberately edit selected colors. Both are separately switchable. Missing fine foot detail cannot be recovered by deleting more splats.

To articulate the fused result later, connect its exported part labels to a hierarchy and pivot transforms in the articulation lab. Preserve the source-index sidecars so subsequent segmentation edits remain traceable to the original captures.
