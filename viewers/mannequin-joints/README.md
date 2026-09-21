# Annotate the manikin joints

Open [the local joint annotation page](http://127.0.0.1:8766/viewers/mannequin-joints/). It uses the complete fused Gaussian scan from the fusion review.

This page records where the joints are, how they should move, and which body regions move together. It does not articulate or modify the scan yet. Save your annotations, then ask for the articulation to be built from that joint map.

## Place a joint in three dimensions

1. Choose a joint and click **Focus joint**. **Isolate region** hides the other regions if they obstruct your view.
2. Select **Front**. Drag the numbered marker to the joint center, or choose **Place center** and click its location.
3. Select **Left** or **Right** and correct its depth. A front view only determines two coordinates; a perpendicular view supplies the third. Repeat the two views as needed.
4. Orbit around the connection and check it from behind or below. Aim for the center of the pin, collar or ball socket, inside the body, rather than a point on its visible surface.
5. Choose the motion type, check its axis and limits, and add any useful notes. Tick **I checked this joint from multiple views** when ready. **Next unchecked** moves through the remaining joints.

Markers remain visible through the scan so buried joint centers can be edited. They are annotations, not surface detections. Dragging a center moves it in a plane parallel to the current view and preserves its depth in that view; it never snaps to the nearest Gaussian surface. The X, Y and Z fields allow numerical adjustment.

Drag the background to orbit, right-drag to pan and scroll to zoom. **Full body** returns to the entire scan. The capture selector lets you inspect the front and back contributions separately without changing the joint map.

## Choose the motion and axis

| Type | Annotation meaning |
| --- | --- |
| **Ball** | Rotation in three directions around one center, such as a shoulder. The three angle ranges refer to the viewer's X, Y and Z axes at rest. |
| **Swivel** | Rotation around the displayed yellow axis, such as the arm's axial collar. |
| **Hinge** | Bending around the displayed yellow axis, such as a knee pin. |
| **Fixed** | A rigid connection that follows its parent without separate rotation. |

For a hinge or swivel, drag the yellow handle to align the axis with the actual pin or shaft. Check it from two perpendicular views; a line that looks correct from the front can still point too far forward or backward. **Set axis**, the X/Y/Z direction fields and the axis presets offer other ways to edit it. **Flip** reverses its direction and reverses the angle range so the requested travel remains the same.

Limits are degrees relative to the current scanned pose. Hinge and swivel limits apply around their yellow axis. The supplied limits, stiffness and damping are starting suggestions for the later articulation, not measurements of this manikin. Stiffness describes resistance and return toward rest; damping describes how quickly movement settles. No physical force units are assigned because the scan is not metrically calibrated.

## Keep rigid pieces together

The 12 starters cover the neck, waist, shoulders, arm swivels, hips, knees and ankles. Every starter is initially unchecked. Arm swivel centers and axes come from the measured collar alignment; other positions are estimates to review.

There are no wrist joints. Each forearm, wrist pad and hand stays one rigid body. In **Connected body parts**, selecting either a hand or its forearm selects the pair. The shoulder connects the upper arm to the torso, and the arm swivel connects the forearm/hand group to that upper arm. Parent connections form the chain for future articulation.

The neck starter carries both the neck and head; the waist starts fixed. Change their settings to describe your actual manikin. Body regions are the existing fusion segmentation. This page changes their joint assignments, not their Gaussian segmentation boundaries. Use notes for any boundaries or hardware that should be treated differently later.

## Save and continue later

On the published website, the editor starts from the accepted joint map, including a matching geometry refinement when available. Edits and compatible drafts stay in your browser. Use **Export JSON** to keep or share a copy; this does not change the published manikin. Import that file in the local editor and use **Save for articulation** to update the workspace. The local save button and API are available only through a loopback HTTP address (`localhost` or `127.0.0.1`).

- **Save for articulation** writes `raw/mannequin-fused/joint-annotations.json` inside this local project. Each successful replacement preserves the previous file as `raw/mannequin-fused/joint-annotations.previous.json`. Saving does not push or upload anything.
- A browser draft is also kept automatically after edits. A newer compatible draft is restored when the page reopens; otherwise it loads the saved project file. Browser drafts are specific to the browser and local address/port and do not replace the project save.
- **Export JSON** downloads an extra copy. **Import** opens a compatible annotation file as a draft; use **Save for articulation** to make it the project copy.
- **Undo** and **Redo** work across edits and imports in the current session. One marker drag is one undo step. Use Ctrl/Cmd+Z and Ctrl/Cmd+Shift+Z when focus is outside a text field. **Reset this joint** restores a known starter joint and can also be undone.

Incomplete review is allowed: save as often as useful. Changes to a center, axis, type or attachment clear that joint's checked flag. A note can be added without clearing it. When the map is ready, save once more and tell the assistant to use the saved joint annotations for articulation.

## Coordinate and file contract

Annotations use `schema: "mannequin-joint-annotations"`, version 1. Coordinates match the rendered fused export: `[front raw X, -front raw Y, -front raw Z]`. Positive X is the manikin's anatomical left, positive Y is toward its front, and positive Z points toward its head. Distances are **scan units**, not millimeters or meters.

Each file records the fused scan revision and both original source hashes. Import and project saving reject another scan revision or coordinate frame, nonfinite values, reversed angle ranges, duplicate ownership, cycles and separate wrist groups. This prevents a joint map from silently being attached to the wrong geometry. Nothing in an annotation file is used as a destination path; project saving always uses the fixed local path above.

Open the page through the running local server, rather than by double-clicking its HTML file. If the server is not running, start it from the repository root with `python3 tools/serve.py 8766` and reopen the page. Do not start a second copy while that port is already serving the project.
