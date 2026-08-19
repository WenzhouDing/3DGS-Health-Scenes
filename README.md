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
  `fov`) after conversion; a good target is the median of high-opacity splat
  positions. Candidate cameras can be tested without re-converting via the
  `?settings=<url>` viewer parameter. Note that re-running the converter with `-w`
  overwrites `settings.json` — re-apply the camera afterwards.

The in-browser rendering is done by
[@playcanvas/supersplat-viewer](https://github.com/playcanvas/supersplat-viewer) (MIT).

Raw PLYs are not committed (see `.gitignore`) — they are hundreds of MB each and
fully redundant with the `.sog` files.

## Viewing locally

The viewers load scene data with `fetch`, so they must be served over HTTP —
opening `index.html` via `file://` will not work:

```sh
python3 -m http.server 8000
# then open http://localhost:8000/
```

Useful viewer URL parameters: `?noui` hides the UI chrome, `?poster=<img>` shows a
loading image, `?budget=<millions>` caps rendered splats for weak GPUs.
