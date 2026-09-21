# Manikin articulation lab

This local editor makes `raw/mannequin_119999.ply` poseable by assigning its
Gaussians to rigid body parts and attaching those parts to a joint hierarchy.
It includes a head/neck ball joint, ball shoulders and hips, axial swivels at both arm collars,
wrists, knees and ankles, and a fixed torso: 13 movable joints in total.
The pivots, axes, limits and segmentation are starter estimates for you to refine.

The source PLY stays unchanged. The prepared editing copy currently contains
900,000 Gaussians from 3,406,096 source records. This is a local browser tool;
it does not upload the scan or require Blender, a server account or a build step.

## Open the editor

From the repository directory:

```sh
python3 tools/serve.py 8766
```

Open [the local articulation lab](http://localhost:8766/viewers/mannequin-rig/).
Keep the terminal running; press Ctrl-C when finished. Use HTTP rather than
opening `index.html` as a file. A WebGL 2 capable browser and GPU are required.
The rendering engine is included locally in `vendor/`. The server binds only to
`127.0.0.1`; the scan stays on this computer.

If `scan.bin` is missing, prepare it before opening the page. The preprocessing
script needs Python 3 and NumPy:

```sh
python3 -m venv .venv-mannequin
.venv-mannequin/bin/python -m pip install numpy
.venv-mannequin/bin/python tools/prepare_mannequin.py
```

If your Python already has NumPy, `python3 tools/prepare_mannequin.py` is enough.

## Try the motion

1. In **01 Pose**, click **Try a pose** to move the arms, head and knees.
2. Select a joint using **Active joint** or a joint marker. Shoulder and hip ball
   joints expose three rotation sliders; a hinge exposes one **Rotation** slider.
   Each arm swivel exposes **Twist**, rotating along the arm at its collar.
3. Drag a joint marker up/down to change its first target rotation. For ball
   joints, left/right also changes the second rotation. Use the third slider for
   the remaining axis. This changes joint angles; it is not an inverse-kinematics
   hand or foot handle.
4. Turn **Spring motion** off for direct posing, or leave it on to see the joint
   move toward its target. **Nudge joint** adds angular velocity to the first axis.
5. **Target → rest** sends the selected joint's target to zero. **Rest pose**
   immediately resets all joint angles, targets and velocities.

Drag empty space to orbit, right-drag or Shift-drag to pan, and scroll to zoom.
**Top**, **Side**, **End** and **Perspective** provide repeatable camera views.
Toggle **Environment / bed**, **Joint markers**, **Segment colors** and
**Isolate selected part** while inspecting the result.

## Tune stiffness and damping

Each active angular coordinate follows this independent spring equation:

```text
I θ̈ + c θ̇ + k (θ − θtarget) = 0
```

The controls display angles in degrees; the spring calculation uses radians and
seconds. `k` is angular stiffness, `c` is angular damping and `I` is the rotational
inertia parameter. If physically calibrated, their units would be N·m/rad,
N·m·s/rad and kg·m² respectively. Here they are tuning parameters: the scan's
distance scale is not guaranteed to be meters, and no body masses or anatomical
measurements have been fitted. Increasing `I` makes the response slower for the
same stiffness. A ball joint uses the same parameters independently on all three
Euler coordinates.

| Desired response | Change |
| --- | --- |
| Stronger pull toward the target | Increase `k` |
| Less ringing or overshoot | Increase `c` |
| Slower response | Increase `I` |
| No pull toward a target | Set `k = 0`; use **Nudge joint** to test |
| Free rotation until a hard limit | Set both `k = 0` and `c = 0`, then nudge |
| Immediate positioning | Turn **Spring motion** off |

A useful starting damping value is `c = 2√(kI)`, the critical damping value for
one unrestricted scalar spring. For example, `k = 25`, `I = 1`, `c = 10` settles
without oscillation in that model; `c = 3` rings more. Joint limits act as hard
stops and remove outward velocity. With `k = 0`, moving the target slider alone
does not move the joint.

The torso is fixed. There is no gravity, bed contact, self-collision, friction,
muscle model or transfer of forces between articulated bodies. A leg can pass
through the mattress. These controls model angular response, not a calibrated
medical simulator or a coupled rigid-body physics engine.

## Place the joints accurately

Entering **02 Joints** or **03 Segment** temporarily displays the rest pose for
editing and clears motion velocity. Your target angles are preserved; returning
to **01 Pose** resumes motion toward those targets. Export from **01 Pose** if
you also want to capture the currently displayed posed angles.

1. Open **02 Joints**, select a shoulder and choose **Top**.
2. Drag the selected marker to the anatomical pivot. Dragging moves it in the
   camera plane; it does not pick a point on the scanned surface.
3. Switch to **Side** or **End** to set its depth. Finish with the numeric
   **Pivot in scan coordinates** X/Y/Z fields. Pivots often belong inside the
   joint, rather than on the visible skin.
4. Return to **01 Pose** and test a small rotation first. Repeat until the limb
   turns around the intended center.
5. Work down each chain: shoulder → elbow → wrist, then hip → knee → ankle.

All pivots and selection shapes are stored in the prepared scan's **rest-world
coordinates**, including child joints. Do not enter an elbow location relative
to the shoulder. Moving a shoulder pivot does not automatically relocate the
rest elbow pivot or its segmentation volumes; adjust those separately if needed.
During posing, a shoulder carries the elbow, forearm and hand through the joint
hierarchy.

The prepared scan is the source rotated 180° around X:

```text
[x_editor, y_editor, z_editor] = [x_source, -y_source, -z_source]
```

Positive Y points above the mattress and positive Z points toward the head.
Source distance units are retained. Starter left/right labels use +X for the
manikin's left and −X for its right; verify that convention against your intended
anatomical labeling.

**Joint type** selects ball, hinge/swivel or fixed. For a hinge, **Rotation axis** is the
axis it rotates around, expressed in its local frame. The UI normalizes this
nonzero vector. **Local frame orientation** rotates that frame relative to the
scan's rest axes; parent motion subsequently carries it. Start with a zero frame
and tune the hinge axis directly. If bending goes backward, negate the axis or
adjust the angle limits to your desired convention.

The starter right arm now uses the circular collar measured in the front scan.
Its prepared-coordinate pivot is `[-0.359085, 0.038332, -0.278961]` and its axis
is `[-0.624909, -0.181747, -0.759248]`. This is one axial rotation, with no bending
at that seam. The shoulder remains a ball joint. The ±90° swivel limits are
editable starter values, not measured mechanical stops. The upper-arm and
forearm capsule endpoints also meet at the updated collar.

The grey-cable arm (anatomical left; screen-right in the reported closeup) also
uses its own measured collar: prepared pivot `[0.372255, -0.008710, -0.270393]`
and axis `[0.633916, -0.139061, -0.760797]`. Its motion is an axial swivel with
editable ±90° starter limits. In the separate fused export, all retained grey
cable splats belong to the socket's upper-arm part so a segmentation seam cannot
split the cable between two transforms. Flexible cable bending is not simulated.

Saved personal rigs are preserved. To load this corrected starter into an
existing editing session, first export your own rig if needed, then choose
**Restore starter rig**. This lab still uses its earlier 900,000-Gaussian editing
copy; the new fused scan is reviewed separately in the Fusion Review viewer.

Ball joints use three bounded Euler XYZ rotations: X, then Y, then Z. Their limits
are independent angle ranges, not an anatomical swing/twist cone. Large combined
rotations can be unintuitive near Euler singularities. Fit the frame to the joint
and use modest ranges while refining segmentation. Keep zero inside each limit
range if you want **Rest pose** to reproduce the scanned pose exactly.

## Refine the body-part segmentation

Every Gaussian belongs to exactly one part. This is a rigid assignment, with no
blend weights at the shoulder, elbow or other boundaries. Work in **03 Segment**
with **Segment colors** on, and inspect each part from several directions.

1. Select a **Body part**. **Attached to** chooses which joint moves it. For
   example, an upper arm attaches to its shoulder, a forearm to its elbow, and a
   hand to its wrist.
2. Select its **Selection volume**. A capsule is a line segment between **Start
   point** and **End point**, expanded by **Radius** with round end caps. A box is
   axis-aligned between its **Minimum corner** and **Maximum corner**. Multiple
   volumes can describe one part; use **+ Capsule**, **+ Box** or **Remove**.
3. Set the volume endpoints near the part's centerline, then increase its radius
   until it includes the surface without capturing the mattress or adjacent body
   part. Keep each box minimum at or below the matching maximum.
4. Click **Apply segmentation**. Changing volumes is pending until applied; the
   Gaussian count and rendered colors then update.
5. Use **Isolate selected part** to spot stray mattress splats and missing skin.
   Turn isolation off again to see how neighboring assignments meet.
6. Enable **Paint selected part**, set a small **3D brush radius**, and click the
   scan to correct individual areas. Each click records a sphere in rest space.
   Click **Apply segmentation** to display its effect.
7. To remove an area from a limb, select **Mattress / unassigned** and paint it.
   This assigns splats to the environment; it does not delete source data.
   **Undo paint** removes the most recent stroke; **Clear paint** removes every
   saved stroke across all parts. Apply again after either action.

Paint is a sequence of 3D sphere assignments, not a continuous screen-space
brush. The picker approximates the front splat center near the pointer; it is
not an exact opaque-surface depth test. A stroke affects every splat center
inside its sphere, including splats behind thin surfaces. Orbit, use a smaller
brush and apply frequently near joints. Temporarily disable painting to orbit
with a left drag.

Volume overlaps have a deterministic rule: a capsule scores the distance to its
centerline divided by its radius; a containing box scores `0.95`. The lowest
score wins; equal scores keep the earlier segment in the JSON array. Anything
outside all volumes becomes environment. Saved brush strokes are applied after
volumes and later strokes override earlier ones. If a volume edit seems to have
no effect, check whether a previous brush stroke is overriding it.

Masks and paint live in rest space, so they remain useful when preparing a denser
copy of the same source scan. They are not hardcoded Gaussian indices.

## Save and continue later

Changes to the rig autosave in this browser's local storage. Use the same browser,
host and port to find that autosave again; `localhost:8766` and
`127.0.0.1:8766` have separate storage. Browser data clearing removes the autosave.
Display toggles, the camera and instantaneous spring velocity are not saved.

**Export rig** downloads `mannequin-rig.json` containing joint pivots and hierarchy,
types, frames, axes, limits, spring settings, the current angles and targets,
selection volumes, and paint strokes. Export milestones to durable files.
**Import rig** loads that JSON and rebuilds segmentation. The source hash is
checked when present, so a rig from another source scan is rejected. Opening a
saved target with spring motion active lets the current angles settle toward it.

The export is an editor configuration. It does not contain the scan, bake a posed
PLY, generate a mesh, or create a Blender armature. Keep the source/prepared scan
alongside the rig if moving the project to another machine.

**Restore starter rig** replaces the current browser rig, including segmentation,
paint and pose, with `default-rig.json`. Export your refinements first. Editing
`default-rig.json` on disk does not override an existing browser autosave until
you restore the starter rig. To make an exported configuration your new starter,
copy it over that file, reload the page, and restore.

## Increase quality or reduce the workload

Preparation defaults to at most 900,000 splats (50.4 MB of binary data). It first
removes nonfinite records and, by default, excludes opacity ≤ 0.03, any Gaussian
scale ≥ 0.1 source units, and centers outside broad source-coordinate scene
bounds. It then takes evenly spaced indices through the remaining records.
The current source yields 2,323,827 records after these filters.

Run commands from the repository root using a Python with NumPy:

```sh
# Faster iteration on a weaker GPU:
.venv-mannequin/bin/python tools/prepare_mannequin.py --max-splats 350000

# All splats that pass the default filters:
.venv-mannequin/bin/python tools/prepare_mannequin.py --max-splats 0

# All finite source records, with neither sampling nor opacity/scale/bounds filters:
.venv-mannequin/bin/python tools/prepare_mannequin.py --max-splats 0 --no-filter

# Restore the standard editing copy:
.venv-mannequin/bin/python tools/prepare_mannequin.py

# See source bounds and other options:
.venv-mannequin/bin/python tools/prepare_mannequin.py --help
```

These commands replace `scan.bin` and `scan.json` in this viewer directory and
leave the raw PLY unchanged. Reload the page afterward. The full current source
is about 190.7 MB before GPU resources and sorting buffers; it needs substantially
more memory than that while loading and rebuilding. More splats and more paint
strokes make **Apply segmentation** slower. Finish coarse editing at a smaller
count, export, then test the denser copy.

The preparer preserves the 14 properties needed by this scan: center, quaternion,
log scale, opacity logit and degree-zero spherical-harmonic color. It rotates
centers and quaternions into editor coordinates. Higher-order SH properties in
another input file are not retained by this preparation format.

## How the articulation works

`rig-core.mjs` evaluates rotations about rest-space pivots and composes the parent
chain. Each body part becomes a PlayCanvas Gaussian resource, with its completed
joint transform applied to the entity. For a Gaussian center μ and covariance Σ,
the rigid transform is:

```text
μ′ = R μ + t
Σ′ = R Σ Rᵀ
```

Rotating the Gaussian's orientation as well as its center preserves its
anisotropic ellipsoid. Its scale, opacity and base color remain unchanged.
`stepSpring` solves the scalar angular spring for a constant target during each
frame and projects onto joint limits. The editor caps the simulated frame step
at 0.05 seconds after a stall.

Rigid segmentation can expose gaps, stretched-looking splats or seams when a
joint bends. A static scan also contains baked lighting, occlusion and incomplete
surfaces. Moving a limb can reveal an underside that was never reconstructed;
joint tuning cannot recover that missing appearance. Start with small motions,
clean up assignments near the joint and evaluate the actual views you need.

For a later deformable version, add per-Gaussian blend weights near joints and a
skinning transform that also updates covariance. For gravity, contact or forces,
add collision proxies and a coupled articulated physics solver, then drive these
same Gaussian part transforms from its body poses. Those are extensions to this
editor; the current version supplies rigid parts and independent angular springs.

## Files and dependencies

| File | Purpose |
| --- | --- |
| `default-rig.json` | Editable starter joints, segmentation volumes and settings |
| `app.mjs` | Browser editor, selection, painting, rendering and persistence |
| `rig-core.mjs` | Hierarchy, classification, constraint and spring math |
| `scan.bin`, `scan.json` | Generated editing data and source/filter metadata |
| `../../tools/prepare_mannequin.py` | PLY preprocessing; never writes the source |
| `../../tools/test_mannequin_rig.mjs` | Math and validation checks |
| `../../tools/test_prepare_mannequin.py` | Preprocessing checks |

The vendored renderer is **PlayCanvas Engine 2.22.1**, revision `73787b3`, under
the [included MIT license](vendor/LICENSE.playcanvas) and
[upstream license at the pinned version](https://github.com/playcanvas/engine/blob/v2.22.1/LICENSE).
See PlayCanvas's [Gaussian splatting documentation](https://developer.playcanvas.com/user-manual/gaussian-splatting/)
and [GSplatComponent API](https://api.playcanvas.com/engine/classes/GSplatComponent.html)
for the rendering APIs. All runtime assets used by this lab are served locally.
