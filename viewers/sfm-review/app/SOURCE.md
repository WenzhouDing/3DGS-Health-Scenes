# Vendored Rerun viewer

Mirrored verbatim from `https://app.rerun.io/version/0.36.3/`:

```sh
for f in index.html re_viewer.js re_viewer_bg.wasm; do
  curl -sL --compressed "https://app.rerun.io/version/0.36.3/$f" -o "$f"
done
```

Three files, unmodified, loaded by their own `index.html` in a same-origin iframe. Nothing
to patch and nothing to re-apply on a version bump. Keep the version matching the SDK that
wrote the `.rrd` — viewer and recording are compatible only within one minor version.

`@rerun-io/web-viewer` on npm is an equally valid source; it is the same viewer. Its wasm is
8.7 MB larger purely because it retains a debug `name` section (code section 34,401,009 vs
34,411,857 bytes; identical `target_features`; same rustc hash). It ships as an ESM
`--target web` build and needs one patch — its `import("./re_viewer")` has no extension,
which browsers cannot resolve. This mirror avoids that, and is smaller.

## Safari must be given `?renderer=webgl`

Under WebGPU the 3D viewport renders black on Safari 26.4+ while the rest of the UI draws
normally and nothing is logged — [rerun-io/rerun#12788](https://github.com/rerun-io/rerun/issues/12788).
Rerun added a Safari→WebGL default in
[#12789](https://github.com/rerun-io/rerun/pull/12789), present in 0.36.3, but this
`index.html` assigns `render_backend` from the query string unconditionally, so an absent
`?renderer` arrives as an explicit `null` and overrides that default. The embedding page
therefore passes `renderer=webgl` for Safari explicitly. Verified on Safari 26.6.2.
