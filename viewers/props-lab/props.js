/**
 * Props Lab — browser-local prop editing in the shared SuperSplat viewer.
 *
 * Loads GLB props (raw/mesh) into the gsplat scene, with manual
 * position/rotation/scale controls, plus simple gravity + ground-plane
 * collision ("Drop"). Layouts persist to localStorage per scene and can be
 * exported as JSON.
 *
 * Uses the shared ambulance engine and accepted scene/collision assets.
 * Layout edits stay in this browser until exported as JSON.
 */
import { sceneTools } from '../ambulance/index.js?v=surface-frame-v3';

const GRAVITY = -9.81;      // m/s^2, world -Y
const RESTITUTION = 0.25;   // bounce energy retained on floor hit
const REST_SPEED = 0.25;    // m/s — below this after a bounce, come to rest
const MAX_DT = 0.05;        // clamp dt against loading hitches
const RAY_LIFT = 0.1;       // rays start this far above the prop bottom
const SNAP_UP = 0.15;       // max upward correction out of a penetrated surface
// Voxel solids mark the dense core of the splats. On sharp surfaces the voxel
// top matches what the renderer shows (bias ~0), but diffuse regions (the dark
// rubber floor) voxelize up to ~0.25 m below the visible surface. The bias is
// therefore calibrated per drop by comparing a picker sample of the visible
// surface with a voxel ray at the same spot; this is the clamp for it.
const MAX_LAND_BIAS = 0.35;
const PLANE_GRACE = 0.05;   // prop this far below the ground plane still lands on it
const MAX_FALL = 25;        // fall distance safety: stop + restore after this

export const initProps = async ({ viewer, config }) => {
    // ---- resolve viewer internals -----------------------------------------
    const app = viewer?.app ?? viewer?.global?.app;
    const cameraEntity = viewer?.camera ?? viewer?.global?.camera;
    if (!app) {
        console.error('props-lab: cannot reach the PlayCanvas app from the viewer instance', viewer);
        return;
    }

    const scene = config?.scene ?? 'ambulance';
    const storageKey = `propsLab:${scene}`;

    // Saved layouts remain in the capture's source frame. A common parent
    // rotates their full transforms with the leveled scene, without decomposing
    // the stored Euler angles. Picking, gravity and position controls use world
    // coordinates through Entity.getPosition/setPosition as before.
    const sourceFrame = new sceneTools.Entity('Props Lab source frame');
    const sceneRotation = viewer.global?.config?.sceneRotation ?? null;
    const hasSceneRotation = sceneRotation && sceneRotation.slice(0, 3).some(value => Math.abs(value) > 1e-12);
    if (sceneRotation) sourceFrame.setLocalRotation(...sceneRotation);
    app.root.addChild(sourceFrame);

    const requestRender = () => { app.renderNextFrame = true; };

    // ---- manifest ---------------------------------------------------------
    let manifest = { props: [] };
    try {
        const resp = await fetch('./manifest.json');
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        manifest = await resp.json();
    } catch (err) {
        console.error('props-lab: failed to load manifest.json', err);
    }
    if (!Array.isArray(manifest?.props)) manifest = { props: [] };

    // ---- state ------------------------------------------------------------
    const state = {
        groundY: null,      // manual ground plane; null = rely on scene collision only
        props: [],          // { uid, id, name, url, entity, euler, scale, falling, vy, dropStart }
        selected: null,     // uid
        nextUid: 1
    };

    // Voxel/mesh collision loaded by the viewer. The cleaned ambulance uses
    // its shared measured mesh; other scenes and URL overrides keep their own.
    const getCollision = () => viewer.inputController?.collision ?? null;

    // Keep the first measured mesh hit. Only voxel occupancy can contain the
    // thin phantom slabs for which the legacy skip heuristic was designed.
    const sceneSurfaceBelow = (col, x, y, z, maxDist) => {
        const hit = col.queryRay(x, y, z, 0, -1, 0, maxDist);
        if (!hit) return null;
        if (col.triangles) return hit;
        if (col.isFreeAt && col.isFreeAt(x, hit.y - 0.08, z)) {
            const below = col.queryRay(x, hit.y - 0.08, z, 0, -1, 0, 0.35);
            if (below) return below;
        }
        return hit;
    };

    // Per-drop landing-bias calibration. The voxel ray gives the predicted
    // landing point on the target surface; project THAT point to the screen
    // and ask the splat picker (which ignores meshes, so the prop is
    // transparent to it) where the visible surface sits at that pixel. The
    // difference is the local fuzz bias between rendered surface and voxel
    // core. Falls back to 0 when the landing spot is off-screen or occluded.
    const calibrateDropBias = async (p) => {
        try {
            const col = getCollision();
            // A measured collider is already the intended support surface.
            // Recalibrating it against Gaussian haze would move that surface.
            if (!col || col.triangles || !viewer.picker) return 0;
            const aabb = worldAabb(p.entity);
            if (!aabb) return 0;
            const cx = (aabb.min.x + aabb.max.x) / 2;
            const cz = (aabb.min.z + aabb.max.z) / 2;
            const vhit = sceneSurfaceBelow(col, cx, aabb.min.y + RAY_LIFT, cz, 5);
            if (!vhit) return 0;
            const cam = viewer.global?.camera?.camera;
            if (!cam) return 0;
            const s = cam.worldToScreen({ x: vhit.x, y: vhit.y, z: vhit.z });
            const cv = app.graphicsDevice.canvas;
            const nx = s.x / cv.clientWidth;
            const ny = s.y / cv.clientHeight;
            if (s.z < 0 || nx < 0.02 || nx > 0.98 || ny < 0.02 || ny > 0.98) return 0;
            // median of a few samples around the landing pixel; negative bias is
            // allowed — phantom fuzz voxels can sit ABOVE the visible surface,
            // and the prop should sink through them onto what the eye sees
            const samples = [];
            const rejects = [];
            for (const [ox, oy] of [[0, 0], [-0.02, 0], [0.02, 0], [0, -0.02], [0, 0.02]]) {
                const sx = nx + ox;
                const sy = ny + oy;
                if (sx < 0.02 || sx > 0.98 || sy < 0.02 || sy > 0.98) continue;
                const hit = await viewer.picker.pickSurface(sx, sy);
                if (!hit?.position) { rejects.push('miss'); continue; }
                // same surface only: a distant pick means the spot is occluded
                const d = Math.hypot(hit.position.x - vhit.x, hit.position.z - vhit.z);
                if (d > 0.4) { rejects.push(`dist ${d.toFixed(2)}`); continue; }
                const b = hit.position.y - vhit.y;
                if (b >= -0.3 && b <= MAX_LAND_BIAS) samples.push(b);
                else rejects.push(`bias ${b.toFixed(2)}`);
            }
            window.__plCalib = { vhit: [vhit.x, vhit.y, vhit.z], samples: [...samples], rejects };
            if (!samples.length) return 0;
            samples.sort((a, b) => a - b);
            return samples[Math.floor(samples.length / 2)];
        } catch {
            return 0;
        }
    };

    // saved-layout entries whose GLB failed to load; kept so a re-save never
    // silently drops them (cleared only by the Clear button)
    let unrestored = [];
    // suppress per-change saves while the restore loop is re-adding props
    let restoring = false;
    // bumped by Clear to invalidate in-flight GLB loads
    let epoch = 0;

    const containerCache = new Map();   // url -> Promise<asset>

    const loadContainer = (url) => {
        if (!containerCache.has(url)) {
            const promise = new Promise((resolve, reject) => {
                app.assets.loadFromUrl(url, 'container', (err, asset) => {
                    if (err) {
                        containerCache.delete(url);   // allow retry after transient failure
                        reject(new Error(err));
                    } else {
                        resolve(asset);
                    }
                });
            });
            containerCache.set(url, promise);
        }
        return containerCache.get(url);
    };

    // world-space AABB (min/max y and full box) across all render components
    const worldAabb = (entity) => {
        const renders = entity.findComponents('render');
        let min = null;
        let max = null;
        for (const render of renders) {
            for (const mi of render.meshInstances) {
                const c = mi.aabb.center;
                const h = mi.aabb.halfExtents;
                if (!min) {
                    min = { x: c.x - h.x, y: c.y - h.y, z: c.z - h.z };
                    max = { x: c.x + h.x, y: c.y + h.y, z: c.z + h.z };
                } else {
                    min.x = Math.min(min.x, c.x - h.x); min.y = Math.min(min.y, c.y - h.y); min.z = Math.min(min.z, c.z - h.z);
                    max.x = Math.max(max.x, c.x + h.x); max.y = Math.max(max.y, c.y + h.y); max.z = Math.max(max.z, c.z + h.z);
                }
            }
        }
        return min ? { min, max } : null;
    };

    const getSelected = () => state.props.find((p) => p.uid === state.selected) ?? null;

    // ---- persistence ------------------------------------------------------
    const layoutJson = () => JSON.stringify({
        scene,
        groundY: state.groundY,
        props: [
            ...state.props.map((p) => {
                const pos = p.entity.getLocalPosition();
                return {
                    id: p.id,
                    name: p.name,
                    url: p.url,
                    position: [pos.x, pos.y, pos.z].map((v) => +v.toFixed(4)),
                    eulerAngles: p.euler.map((v) => +v.toFixed(2)),
                    scale: +p.scale.toFixed(4),
                    visible: p.entity.enabled
                };
            }),
            ...unrestored
        ]
    }, null, 2);

    const saveLayout = () => {
        if (restoring) return;
        try { localStorage.setItem(storageKey, layoutJson()); } catch { /* ignore */ }
    };

    const loadSavedLayout = async () => {
        // the committed default layout is the source of truth on load; the
        // localStorage autosave only fills in when no file is shipped (use
        // Download to bake in-browser arrangements into the file)
        try {
            const resp = await fetch(`./layout-${scene}.json`);
            if (resp.ok) return await resp.json();
        } catch { /* ignore */ }
        try {
            const stored = localStorage.getItem(storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                if (parsed?.props?.length) return parsed;
            }
        } catch { /* ignore */ }
        return null;
    };

    // ---- prop lifecycle ---------------------------------------------------
    const spawnPose = () => {
        // 1.5m in front of the camera
        if (cameraEntity) {
            const p = cameraEntity.getPosition();
            const f = cameraEntity.forward;
            return { x: p.x + f.x * 1.5, y: p.y + f.y * 1.5, z: p.z + f.z * 1.5 };
        }
        return { x: 0, y: 0, z: 0 };
    };

    const addProp = async (def, restore = null) => {
        setStatus(`Loading ${def.name}… (large file, hold on)`);
        const myEpoch = epoch;
        let asset;
        try {
            asset = await loadContainer(def.url);
        } catch (err) {
            console.error('props-lab: failed to load', def.url, err);
            setStatus(`Failed to load ${def.name}: ${err.message}`, true);
            return null;
        }
        if (myEpoch !== epoch) return null;   // cleared while loading
        const entity = asset.resource.instantiateRenderEntity();
        entity.name = `prop:${def.id}`;
        sourceFrame.addChild(entity);

        const prop = {
            uid: state.nextUid++,
            id: def.id,
            name: def.name,
            url: def.url,
            entity,
            euler: [0, 0, 0],   // source of truth — entity euler read-back is ambiguous past ±90° yaw
            scale: 1,
            falling: false,
            vy: 0
        };

        if (restore) {
            prop.scale = restore.scale ?? 1;
            entity.setLocalScale(prop.scale, prop.scale, prop.scale);
            const [px, py, pz] = restore.position ?? [0, 0, 0];
            entity.setLocalPosition(px, py, pz);
            prop.euler = (restore.eulerAngles ?? [0, 0, 0]).slice();
            entity.setLocalEulerAngles(...prop.euler);
            entity.enabled = restore.visible !== false;
        } else {
            const pos = spawnPose();
            entity.setPosition(pos.x, pos.y, pos.z);
            if (hasSceneRotation) {
                entity.setEulerAngles(0, 0, 0);
                const local = entity.getLocalEulerAngles();
                prop.euler = [local.x, local.y, local.z];
            }
        }

        state.props.push(prop);
        if (!restore || state.selected === null) state.selected = prop.uid;
        setStatus(`${def.name} added.`);
        refreshUI();
        saveLayout();
        requestRender();
        return prop;
    };

    const removeProp = (uid) => {
        const i = state.props.findIndex((p) => p.uid === uid);
        if (i < 0) return;
        state.props[i].entity.destroy();
        state.props.splice(i, 1);
        if (state.selected === uid) state.selected = state.props[i]?.uid ?? state.props[i - 1]?.uid ?? null;
        refreshUI();
        saveLayout();
        requestRender();
    };

    // ---- gravity + collision (scene surfaces, manual plane fallback) -------
    app.on('update', (rawDt) => {
        const dt = Math.min(rawDt, MAX_DT);
        let active = false;
        for (const p of state.props) {
            if (!p.falling) continue;
            active = true;
            p.vy += GRAVITY * dt;
            const pos = p.entity.getPosition();
            if (!p.dropStart) p.dropStart = [pos.x, pos.y, pos.z];

            // fall-distance safety: nothing to land on — restore and stop
            if (p.dropStart[1] - pos.y > MAX_FALL) {
                p.entity.setPosition(...p.dropStart);
                p.falling = false;
                p.vy = 0;
                p.dropStart = null;
                setStatus(`${p.name}: nothing to land on below — position restored. Set ground Y or check collision.`, true);
                saveLayout();
                refreshUI();
                continue;
            }

            const aabb = worldAabb(p.entity);
            const dy = p.vy * dt;   // negative while falling
            if (!aabb) {
                p.entity.setPosition(pos.x, pos.y + dy, pos.z);
                continue;
            }
            const bottom = aabb.min.y;

            // find the highest support under the prop this frame
            let support = null;
            let supportSrc = '';

            // (a) scene collision: rays cast down from just above the prop's
            // bottom footprint (center + 4 inset corners, like walk mode)
            const col = getCollision();
            if (col && p.vy < 0) {
                const cx = (aabb.min.x + aabb.max.x) / 2;
                const cz = (aabb.min.z + aabb.max.z) / 2;
                const ix = (aabb.max.x - aabb.min.x) * 0.3;
                const iz = (aabb.max.z - aabb.min.z) * 0.3;
                // Long rays: the surface below is the standing target even when
                // it is still metres away — landing happens only when the prop
                // actually crosses it. The center ray decides where the prop
                // rests; corner rays matter only when the center hangs over a
                // void (else a corner nicking nearby clutter voxels would leave
                // the prop floating mid-air).
                const landBias = p.landBias ?? 0;
                const castDown = (ox, oz) => {
                    const hit = sceneSurfaceBelow(col, ox, bottom + RAY_LIFT, oz, RAY_LIFT + 3);
                    if (!hit) return null;
                    const surfaceY = hit.y + landBias;
                    return surfaceY <= bottom + SNAP_UP ? surfaceY : null;
                };
                let sceneY = castDown(cx, cz);
                if (sceneY === null) {
                    for (const [ox, oz] of [[cx - ix, cz - iz], [cx - ix, cz + iz], [cx + ix, cz - iz], [cx + ix, cz + iz]]) {
                        const y = castDown(ox, oz);
                        if (y !== null && (sceneY === null || y > sceneY)) sceneY = y;
                    }
                }
                if (sceneY !== null) {
                    support = sceneY;
                    supportSrc = 'scene surface';
                }
            }

            // (b) manual ground plane, unless the prop is already well below it
            if (state.groundY !== null && bottom >= state.groundY - PLANE_GRACE && (support === null || state.groundY > support)) {
                support = state.groundY;
                supportSrc = 'ground plane';
            }

            if (support !== null && bottom + dy <= support && p.vy < 0) {
                // land: lift bottom onto the support, then bounce or rest
                p.entity.setPosition(pos.x, pos.y + (support - bottom), pos.z);
                p.vy = -p.vy * RESTITUTION;
                if (p.vy < REST_SPEED) {
                    p.vy = 0;
                    p.falling = false;
                    p.dropStart = null;
                    setStatus(`${p.name} landed on ${supportSrc} (Y = ${support.toFixed(3)}).`);
                    saveLayout();
                    refreshUI();
                }
            } else {
                p.entity.setPosition(pos.x, pos.y + dy, pos.z);
            }
        }
        if (active) {
            requestRender();
            refreshTransformInputs();
        }
    });

    // ---- UI ---------------------------------------------------------------
    const panel = document.createElement('div');
    panel.id = 'propsPanel';
    if (config?.noui) panel.style.display = 'none';

    // Keep panel interaction away from the viewer's camera / hotkey handlers.
    // 'keyup' is deliberately NOT stopped: if a movement key was pressed over the
    // canvas and released over the panel, the viewer must see the release or the
    // camera keeps flying (its key state is absolute 0/1, so a stray keyup is safe).
    for (const type of ['keydown', 'keypress', 'pointerdown', 'pointerup', 'pointermove', 'mousedown', 'mouseup', 'dblclick', 'wheel', 'contextmenu', 'touchstart', 'touchmove', 'touchend']) {
        panel.addEventListener(type, (e) => e.stopPropagation());
    }

    const el = (tag, attrs = {}, ...children) => {
        const node = document.createElement(tag);
        for (const [k, v] of Object.entries(attrs)) {
            if (k === 'class') node.className = v;
            else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
            else if (k === 'text') node.textContent = v;
            else node.setAttribute(k, v);
        }
        for (const c of children) node.append(c);
        return node;
    };

    // header
    const header = el('div', { class: 'pl-header', onclick: () => panel.classList.toggle('collapsed') },
        el('span', { text: '\u{1FA7A} Props Lab' }),
        el('span', { class: 'pl-scene', text: scene }),
        el('span', { text: '▾' })
    );

    // add-prop row
    const propSelect = el('select');
    for (const def of manifest.props) {
        propSelect.append(el('option', { value: def.id, text: def.name }));
    }
    const addBtn = el('button', { class: 'pl-primary', text: 'Add', onclick: async () => {
        const def = manifest.props.find((d) => d.id === propSelect.value);
        if (!def) return;
        addBtn.disabled = true;
        try { await addProp(def); } finally { addBtn.disabled = false; }
    } });
    const addRow = el('div', { class: 'pl-row' }, propSelect, addBtn);

    // prop list
    const listBox = el('div', { class: 'pl-list' });

    // transform inputs
    const numInput = (step, onchange) => el('input', { type: 'number', step: String(step), onchange, onfocus: (e) => e.target.select() });
    const inX = numInput(0.01, () => applyInputs());
    const inY = numInput(0.01, () => applyInputs());
    const inZ = numInput(0.01, () => applyInputs());
    const inRX = numInput(5, () => applyInputs());
    const inRY = numInput(5, () => applyInputs());
    const inRZ = numInput(5, () => applyInputs());
    const inScale = numInput(0.05, () => applyInputs());

    const applyInputs = () => {
        const p = getSelected();
        if (!p) return;
        p.falling = false;
        p.vy = 0;
        p.dropStart = null;
        p.entity.setPosition(+inX.value || 0, +inY.value || 0, +inZ.value || 0);
        p.euler = [+inRX.value || 0, +inRY.value || 0, +inRZ.value || 0];
        p.entity.setLocalEulerAngles(...p.euler);
        const s = Math.max(0.001, +inScale.value || 1);
        p.scale = s;
        p.entity.setLocalScale(s, s, s);
        saveLayout();
        requestRender();
    };

    const refreshTransformInputs = () => {
        const p = getSelected();
        if (!p) return;
        if (document.activeElement && panel.contains(document.activeElement) && document.activeElement.tagName === 'INPUT') return;
        const pos = p.entity.getPosition();
        inX.value = pos.x.toFixed(3); inY.value = pos.y.toFixed(3); inZ.value = pos.z.toFixed(3);
        inRX.value = p.euler[0].toFixed(1); inRY.value = p.euler[1].toFixed(1); inRZ.value = p.euler[2].toFixed(1);
        inScale.value = p.scale.toFixed(3);
    };

    // nudge helpers (hold to repeat)
    let stepSize = 0.05;
    const nudge = (dx, dy, dz) => {
        const p = getSelected();
        if (!p) return;
        const pos = p.entity.getPosition();
        p.entity.setPosition(pos.x + dx * stepSize, pos.y + dy * stepSize, pos.z + dz * stepSize);
        refreshTransformInputs();
        saveLayout();
        requestRender();
    };
    const rotate = (axis, dir) => {
        const p = getSelected();
        if (!p) return;
        // mutate the JS-side euler (never read back from the entity — the
        // engine's decomposition flips triples past ±90° yaw)
        const i = { x: 0, y: 1, z: 2 }[axis];
        const norm = (v) => ((v + 180) % 360 + 360) % 360 - 180;
        p.euler[i] = norm(p.euler[i] + 15 * dir);
        p.entity.setLocalEulerAngles(...p.euler);
        refreshTransformInputs();
        saveLayout();
        requestRender();
    };
    const holdable = (label, title, fn) => {
        let timer = null;
        const stop = () => { if (timer) { clearInterval(timer); timer = null; } };
        const b = el('button', { text: label, title });
        b.addEventListener('pointerdown', (e) => {
            if (e.button !== 0) return;   // right-click opens the context menu and would leak the repeat timer
            fn();
            stop();
            timer = setInterval(fn, 120);
        });
        for (const t of ['pointerup', 'pointerleave', 'pointercancel']) b.addEventListener(t, stop);
        return b;
    };

    const stepSelect = el('select', { onchange: (e) => { stepSize = +e.target.value; } },
        el('option', { value: '0.01', text: 'step 1 cm' }),
        el('option', { value: '0.05', text: 'step 5 cm', selected: '' }),
        el('option', { value: '0.25', text: 'step 25 cm' })
    );

    const nudgeGrid = el('div', { class: 'pl-nudge' },
        holdable('X−', 'move -X', () => nudge(-1, 0, 0)), holdable('X+', 'move +X', () => nudge(1, 0, 0)),
        holdable('Y−', 'move down', () => nudge(0, -1, 0)), holdable('Y+', 'move up', () => nudge(0, 1, 0)),
        holdable('Z−', 'move -Z', () => nudge(0, 0, -1)), holdable('Z+', 'move +Z', () => nudge(0, 0, 1)),
        holdable('RX−', 'pitch -15°', () => rotate('x', -1)), holdable('RX+', 'pitch +15°', () => rotate('x', 1)),
        holdable('RY−', 'yaw -15°', () => rotate('y', -1)), holdable('RY+', 'yaw +15°', () => rotate('y', 1)),
        holdable('RZ−', 'roll -15°', () => rotate('z', -1)), holdable('RZ+', 'roll +15°', () => rotate('z', 1))
    );

    const scaleTimes = (f) => {
        const p = getSelected();
        if (!p) return;
        p.scale = Math.max(0.001, p.scale * f);
        p.entity.setLocalScale(p.scale, p.scale, p.scale);
        refreshTransformInputs();
        saveLayout();
        requestRender();
    };

    // ---- walk-mode spawn fix ----------------------------------------------
    // Entering walk mode seeds the controller's spawn search from the current
    // camera, so an orbit camera above the scene lands the player on the roof.
    // The collision voxels are built with a standing pocket carved at the
    // curated initial ("video") viewpoint, so seeding the camera there makes
    // the spawn search accept that exact spot. Entries from anim mode defer
    // and re-enter via orbit, whose exit does not clobber the seeded pose.
    const videoPose = () => {
        const cam0 = viewer.global.settings?.cameras?.[0]?.initial;
        if (!Array.isArray(cam0?.position) || !Array.isArray(cam0?.target)) return null;
        const [px, py, pz] = cam0.position;
        const [tx, , tz] = cam0.target;
        return { px, py, pz, yawDeg: Math.atan2(-(tx - px), -(tz - pz)) * 180 / Math.PI };
    };
    const seedWalkPose = () => {
        const cm = viewer.cameraManager;
        const pose = videoPose();
        if (!cm?.camera?.position || !pose) return false;
        cm.camera.position.set(pose.px, pose.py, pose.pz);
        cm.camera.angles.set(0, pose.yawDeg, 0);
        return true;
    };
    // default to fly ("drone") mode once the scene is up — the stock viewer
    // would otherwise auto-pick walk/orbit
    const flyDefault = setInterval(() => {
        if (!viewer.cameraManager) return;
        clearInterval(flyDefault);
        try { viewer.global.state.cameraMode = 'fly'; } catch { /* ignore */ }
    }, 200);

    let reseeding = false;
    viewer.global?.events?.on('cameraMode:changed', (mode, prev) => {
        if (mode !== 'walk' || reseeding) return;
        try {
            if (!viewer.cameraManager?.camera?.position || prev === 'anim') {
                // initial entry at load (cameraManager not assigned yet) or an
                // entry from anim, whose onExit restores the track pose
                reseeding = true;
                setTimeout(() => { reseeding = false; }, 500);
                setTimeout(() => {
                    try {
                        if (viewer.global.state.cameraMode !== 'walk') return;
                        viewer.global.state.cameraMode = 'orbit';
                        if (seedWalkPose()) viewer.global.state.cameraMode = 'walk';
                    } catch { /* ignore */ }
                }, 100);
                return;
            }
            seedWalkPose();
        } catch { /* ignore */ }
    });

    // ---- click-picking on the splat surface -------------------------------
    // viewer.picker is created once the splat finishes loading; access lazily.
    let pickMode = null; // null | 'place' | 'ground'
    const canvas = app.graphicsDevice.canvas;

    const armPick = (mode) => {
        if (pickMode === mode) { disarmPick(); return; }
        if (!viewer.picker) {
            setStatus('Scene still loading — try again in a moment.', true);
            return;
        }
        pickMode = mode;
        canvas.style.cursor = 'crosshair';
        placeBtn.classList.toggle('pl-active', mode === 'place');
        groundPickBtn.classList.toggle('pl-active', mode === 'ground');
        setStatus(mode === 'place' ? 'Click the scene to place the selected prop.' : 'Click the scene to set the ground plane height.');
    };
    const disarmPick = () => {
        pickMode = null;
        canvas.style.cursor = '';
        placeBtn.classList.remove('pl-active');
        groundPickBtn.classList.remove('pl-active');
    };

    // capture-phase on window: runs before the viewer's canvas listeners, so a
    // pick click never starts a camera drag / fly-to
    window.addEventListener('pointerdown', async (e) => {
        if (!pickMode || e.target !== canvas || !viewer.picker) return;
        if (e.pointerType === 'mouse' && e.button !== 0) return;   // let right/middle-drag camera moves work while armed
        e.stopPropagation();
        e.preventDefault();
        // also swallow the matching pointerup/cancel: the viewer's touch input
        // counts pointers with +1/-1 deltas, and seeing only the release would
        // desync its touch count permanently
        const { pointerId } = e;
        const swallowEnd = (ev) => {
            if (ev.pointerId !== pointerId) return;
            ev.stopPropagation();
            window.removeEventListener('pointerup', swallowEnd, true);
            window.removeEventListener('pointercancel', swallowEnd, true);
        };
        window.addEventListener('pointerup', swallowEnd, true);
        window.addEventListener('pointercancel', swallowEnd, true);
        const mode = pickMode;
        disarmPick();
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) / rect.width;
        const y = (e.clientY - rect.top) / rect.height;
        let hit = null;
        try { hit = await viewer.picker.pickSurface(x, y); } catch (err) { console.warn('props-lab: pick failed', err); }
        if (!hit?.position) {
            setStatus('No surface under that click.', true);
            return;
        }
        if (mode === 'ground') {
            state.groundY = hit.position.y;
            groundInput.value = state.groundY.toFixed(3);
            saveLayout();
            setStatus(`Ground Y set to ${state.groundY.toFixed(3)} from picked surface.`);
        } else {
            const p = getSelected();
            if (!p) return;
            const pos = p.entity.getPosition();
            const aabb = worldAabb(p.entity);
            const lift = aabb ? pos.y - aabb.min.y : 0; // keep bottom on the surface
            p.falling = false;
            p.vy = 0;
            p.dropStart = null;
            p.entity.setPosition(hit.position.x, hit.position.y + lift, hit.position.z);
            refreshTransformInputs();
            saveLayout();
            setStatus(`${p.name} placed at picked surface.`);
        }
        requestRender();
    }, true);

    const placeBtn = el('button', { text: 'Place at click', title: 'Then click a spot in the scene', onclick: () => armPick('place') });
    const groundPickBtn = el('button', { text: 'Ground from click', title: 'Click a floor point to set the ground plane', onclick: () => armPick('ground') });

    // physics controls; empty ground field = rely on scene collision only
    const groundInput = numInput(0.05, () => {
        state.groundY = groundInput.value === '' ? null : (+groundInput.value || 0);
        saveLayout();
    });
    groundInput.setAttribute('placeholder', 'auto');
    const dropBtn = el('button', { class: 'pl-primary', text: 'Drop (gravity)', onclick: async () => {
        const p = getSelected();
        if (!p) return;
        if (p.falling) {
            p.falling = false;
            p.vy = 0;
            p.dropStart = null;
            saveLayout();   // persist a deliberately frozen mid-air pose
        } else {
            p.landBias = await calibrateDropBias(p);
            p.falling = true;
            p.vy = 0;
            p.dropStart = null;
        }
        refreshUI();
        requestRender();
    } });
    const groundFromPropBtn = el('button', { text: 'Ground = prop bottom', title: 'Set ground plane to the selected prop’s current lowest point', onclick: () => {
        const p = getSelected();
        if (!p) return;
        const aabb = worldAabb(p.entity);
        if (aabb) {
            state.groundY = aabb.min.y;
            groundInput.value = state.groundY.toFixed(3);
            saveLayout();
        }
    } });

    // save / export
    const statusLine = el('div', { class: 'pl-status' });
    const setStatus = (msg, isError = false) => {
        statusLine.textContent = msg;
        statusLine.classList.toggle('error', isError);
    };

    const copyBtn = el('button', { text: 'Copy JSON', onclick: async () => {
        try {
            await navigator.clipboard.writeText(layoutJson());
            setStatus('Layout JSON copied to clipboard.');
        } catch {
            console.log(layoutJson());
            setStatus('Clipboard blocked — layout JSON printed to console.', true);
        }
    } });
    const downloadBtn = el('button', { text: 'Download', onclick: () => {
        const blob = new Blob([layoutJson()], { type: 'application/json' });
        const a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = `layout-${scene}.json`;
        a.click();
        URL.revokeObjectURL(a.href);
        setStatus(`Saved layout-${scene}.json — drop it into viewers/props-lab/ to auto-load.`);
    } });
    const resetBtn = el('button', { class: 'pl-danger', text: 'Clear', onclick: () => {
        if (!confirm('Remove all props and clear the saved layout for this scene?')) return;
        epoch++;               // cancel any GLB loads still in flight
        restoring = false;
        unrestored = [];
        for (const p of [...state.props]) removeProp(p.uid);
        try { localStorage.removeItem(storageKey); } catch { /* ignore */ }
        setStatus('Layout cleared.');
    } });

    const selHeader = el('div', { class: 'pl-section-title', text: 'No prop selected' });

    panel.append(
        header,
        addRow,
        listBox,
        el('div', { class: 'pl-section' },
            selHeader,
            el('div', { class: 'pl-row' }, el('label', { text: 'pos' }), inX, inY, inZ),
            el('div', { class: 'pl-row' }, el('label', { text: 'rot' }), inRX, inRY, inRZ),
            el('div', { class: 'pl-row' },
                el('label', { text: 'scl' }), inScale,
                el('button', { text: '×0.8', onclick: () => scaleTimes(0.8) }),
                el('button', { text: '×1.25', onclick: () => scaleTimes(1.25) })
            ),
            el('div', { class: 'pl-row' }, stepSelect, placeBtn),
            nudgeGrid
        ),
        el('div', { class: 'pl-section' },
            el('div', { class: 'pl-section-title', text: 'Gravity / collision' }),
            el('div', { class: 'pl-hint', text: 'Drop lands on the scene’s collision surfaces. Ground Y is an optional override plane — leave it on "auto".' }),
            el('div', { class: 'pl-row' }, el('label', { text: 'ground Y' }), groundInput),
            el('div', { class: 'pl-row' }, groundPickBtn, groundFromPropBtn),
            el('div', { class: 'pl-row' }, dropBtn)
        ),
        el('div', { class: 'pl-section' },
            el('div', { class: 'pl-section-title', text: 'Layout' }),
            el('div', { class: 'pl-row' }, copyBtn, downloadBtn, resetBtn),
            statusLine,
            el('div', { class: 'pl-hint', text: 'Layout autosaves to this browser (localStorage) per scene. Download and place layout-<scene>.json next to this page to make it the default.' })
        )
    );

    document.body.append(panel);

    const refreshUI = () => {
        // prop list
        listBox.textContent = '';
        for (const p of state.props) {
            const item = el('div', { class: `pl-list-item${p.uid === state.selected ? ' selected' : ''}`, onclick: () => { state.selected = p.uid; refreshUI(); } },
                el('span', { class: 'pl-name', text: `${p.name}${p.falling ? ' ↓' : ''}` }),
                el('button', { text: p.entity.enabled ? '\u{1F441}' : '–', title: 'toggle visibility', onclick: (e) => {
                    e.stopPropagation();
                    p.entity.enabled = !p.entity.enabled;
                    saveLayout();
                    refreshUI();
                    requestRender();
                } }),
                el('button', { text: '✕', title: 'remove', onclick: (e) => { e.stopPropagation(); removeProp(p.uid); } })
            );
            listBox.append(item);
        }
        // selection section
        const p = getSelected();
        selHeader.textContent = p ? `Selected: ${p.name}` : 'No prop selected';
        for (const input of [inX, inY, inZ, inRX, inRY, inRZ, inScale]) input.disabled = !p;
        dropBtn.disabled = !p;
        dropBtn.classList.toggle('pl-active', !!p?.falling);
        dropBtn.textContent = p?.falling ? 'Falling… (click to stop)' : 'Drop (gravity)';
        groundFromPropBtn.disabled = !p;
        placeBtn.disabled = !p;
        refreshTransformInputs();
    };

    // ---- restore saved layout --------------------------------------------
    // While restoring, saveLayout() is a no-op so a reload / failed load can
    // never overwrite the saved layout with a partial one. Entries that fail
    // to load are parked in `unrestored` and survive future saves verbatim.
    const saved = await loadSavedLayout();
    const restoreEpoch = epoch;
    if (saved) {
        state.groundY = saved.groundY ?? null;
        groundInput.value = state.groundY === null ? '' : String(state.groundY);
        restoring = true;
        try {
            for (const sp of saved.props ?? []) {
                const def = manifest.props.find((d) => d.id === sp.id) ?? { id: sp.id, name: sp.name ?? sp.id, url: sp.url };
                // sequential on purpose: same-URL props share one container fetch
                const prop = await addProp(def, sp);
                if (epoch !== restoreEpoch) break;   // user hit Clear mid-restore
                if (!prop) unrestored.push(sp);
            }
        } finally {
            restoring = false;
        }
        if (epoch === restoreEpoch) saveLayout();
    }

    refreshUI();
    if (saved) {
        const failed = unrestored.length ? ` (${unrestored.length} failed to load — kept in the layout)` : '';
        setStatus(`Restored ${state.props.length} prop(s) from saved layout${failed}.`, unrestored.length > 0);
    } else if (manifest.props.length === 0) {
        setStatus('No props available — manifest.json missing or empty (see console).', true);
    } else {
        setStatus('Add a prop to get started.');
    }
    console.log('props-lab: ready', { scene, app });

    return { state, addProp, removeProp, manifest, layoutJson };
};
