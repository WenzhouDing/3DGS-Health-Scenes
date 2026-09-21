import * as pc from '../mannequin-rig/vendor/playcanvas.mjs';

const FIELDS = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
  'scale_0', 'scale_1', 'scale_2', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
const MANIFEST_URL = new URL('../mannequin-fusion/fusion.json', import.meta.url);
const DEG = Math.PI / 180;
const FOV = 45;
const VIEWS = { perspective: [55, 12], front: [90, 0], back: [-90, 0],
  left: [0, 0], right: [180, 0], top: [90, 90], bottom: [90, -90] };
const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
const boundsEmpty = () => ({ min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] });
function include(bounds, xyz) {
  for (let i = 0; i < 3; i++) {
    bounds.min[i] = Math.min(bounds.min[i], xyz[i]);
    bounds.max[i] = Math.max(bounds.max[i], xyz[i]);
  }
}
function validateManifest(value) {
  if (value?.version !== 1 || !Array.isArray(value.parts) || !value.parts.length || value.parts.length > 256) {
    throw new Error('The fusion manifest must contain version 1 and 1–256 body regions.');
  }
  if (value.parts.some(p => typeof p.id !== 'string' || !p.id) || new Set(value.parts.map(p => p.id)).size !== value.parts.length) {
    throw new Error('The fusion manifest contains invalid body region IDs.');
  }
  if (!Array.isArray(value.captures) || value.captures.length !== 2 || new Set(value.captures.map(c => c.id)).size !== 2
    || value.captures.some(c => !['front', 'back'].includes(c.id) || !Number.isInteger(c.count) || c.count < 0
      || typeof c.url !== 'string' || typeof c.labelsUrl !== 'string')) {
    throw new Error('The fusion manifest must describe both prepared captures.');
  }
  if (['min', 'max'].some(key => !Array.isArray(value.bounds?.[key]) || value.bounds[key].length !== 3
    || !value.bounds[key].every(Number.isFinite)) || value.bounds.min.some((n, i) => n > value.bounds.max[i])) {
    throw new Error('The fusion manifest contains invalid scene bounds.');
  }
}

/** Render the existing fused scan. World coordinates are the exported viewer XYZ frame, with +Z up.
 * Screen coordinates are CSS pixels relative to container, independent of device pixel ratio.
 * Joint overlays belong to the caller; this renderer never edits the underlying scan.
 */
export async function createScanView({ canvas, container, onStatus = () => {}, onCameraChange = () => {} }) {
  if (!canvas || !container) throw new Error('A canvas and its viewport container are required.');
  const abort = new AbortController();
  const entries = [];
  const listeners = [];
  let app, camera, observer, manifest, disposed = false, loading = true, drag = null;
  let width = 1, height = 1, radius = 1, source = 'both', visibleParts = null, navigationEnabled = true;
  let view = 'perspective', orthographic = false, fitted = true, fittedParts = null;
  const orbit = { yaw: 55, pitch: 12, distance: 3, height: 1, target: new pc.Vec3() };
  const basis = { outward: new pc.Vec3(), forward: new pc.Vec3(), right: new pc.Vec3(), up: new pc.Vec3() };
  const position = new pc.Vec3();
  const addListener = (target, event, handler, options) => {
    target.addEventListener(event, handler, options);
    listeners.push(() => target.removeEventListener(event, handler, options));
  };
  const assertAlive = () => { if (disposed) throw new Error('The scan viewer has been disposed.'); };
  const checkIds = ids => {
    if (ids == null) return null;
    const result = new Set(typeof ids === 'string' ? [ids] : ids);
    for (const id of result) if (!manifest.parts.some(part => part.id === id)) throw new Error(`Unknown body region: ${id}`);
    return result;
  };
  const announceCamera = () => onCameraChange({ view, projection: orthographic ? 'orthographic' : 'perspective', width, height });
  const fetchFile = async (url, format = 'arrayBuffer') => {
    const response = await fetch(url, { signal: abort.signal });
    if (!response.ok) throw new Error(`Cannot load ${new URL(url).pathname}: HTTP ${response.status}.`);
    return response[format]();
  };

  function updateBasis() {
    const yaw = orbit.yaw * DEG, pitch = orbit.pitch * DEG;
    basis.outward.set(Math.cos(yaw) * Math.cos(pitch), Math.sin(yaw) * Math.cos(pitch), Math.sin(pitch));
    basis.forward.copy(basis.outward).mulScalar(-1);
    // Exact top/bottom views need a second up vector; +Z would be parallel to the viewing ray.
    const preferredUp = Math.abs(basis.outward.z) > .99999 ? new pc.Vec3(0, 1, 0) : new pc.Vec3(0, 0, 1);
    basis.right.cross(basis.forward, preferredUp).normalize();
    basis.up.cross(basis.right, basis.forward).normalize();
  }
  function updateCamera() {
    if (!camera || disposed) return;
    updateBasis();
    position.copy(basis.outward).mulScalar(orbit.distance).add(orbit.target);
    camera.setPosition(position);
    camera.lookAt(orbit.target, basis.up);
    camera.camera.projection = orthographic ? pc.PROJECTION_ORTHOGRAPHIC : pc.PROJECTION_PERSPECTIVE;
    camera.camera.aspectRatio = width / height;
    camera.camera.orthoHeight = orbit.height;
    camera.camera.nearClip = Math.max(radius * .0001, .000001);
    camera.camera.farClip = Math.max(radius * 100, orbit.distance + radius * 10);
    announceCamera();
  }
  function selectionBounds(ids) {
    if (ids == null) return manifest.bounds;
    const result = boundsEmpty();
    for (const entry of entries) if (ids.has(entry.part)) {
      include(result, entry.bounds.min); include(result, entry.bounds.max);
    }
    return Number.isFinite(result.min[0]) ? result : manifest.bounds;
  }
  function fitInternal(ids) {
    const bounds = selectionBounds(ids);
    orbit.target.set(...bounds.min.map((n, i) => (n + bounds.max[i]) / 2));
    updateBasis();
    let halfHeight = radius * .005, distance = radius * .025;
    const tanY = Math.tan(FOV * DEG / 2), aspect = width / height;
    for (let mask = 0; mask < 8; mask++) {
      const offset = new pc.Vec3(...bounds.min.map((n, i) => ((mask >> i) & 1 ? bounds.max[i] : n)))
        .sub(orbit.target);
      const projectedHeight = Math.max(Math.abs(offset.dot(basis.up)), Math.abs(offset.dot(basis.right)) / aspect);
      halfHeight = Math.max(halfHeight, projectedHeight * 1.17);
      distance = Math.max(distance, offset.dot(basis.outward) + projectedHeight * 1.17 / tanY);
    }
    orbit.height = halfHeight;
    orbit.distance = orthographic ? Math.max(radius * 3, halfHeight * 2) : distance;
    updateCamera();
  }
  function fit(ids = null) {
    assertAlive();
    fittedParts = ids == null ? visibleParts : checkIds(ids);
    fitted = true;
    fitInternal(fittedParts);
  }
  function setView(name) {
    assertAlive();
    name = ({ head: 'top', feet: 'bottom' })[name] || name;
    if (!VIEWS[name]) throw new Error(`Unknown camera view: ${name}`);
    const wasOrthographic = orthographic;
    view = name; orthographic = name !== 'perspective';
    [orbit.yaw, orbit.pitch] = VIEWS[name];
    if (fitted) fitInternal(fittedParts);
    else {
      // Preserve the target and apparent scale when changing projection during close annotation work.
      if (orthographic && !wasOrthographic) orbit.height = orbit.distance * Math.tan(FOV * DEG / 2);
      if (!orthographic && wasOrthographic) orbit.distance = orbit.height / Math.tan(FOV * DEG / 2);
      if (orthographic) orbit.distance = Math.max(radius * 3, orbit.height * 2);
      updateCamera();
    }
  }
  function updateVisibility() {
    let count = 0;
    for (const entry of entries) {
      entry.entity.enabled = (!visibleParts || visibleParts.has(entry.part)) && (source === 'both' || source === entry.capture);
      if (entry.entity.enabled) count += entry.count;
    }
    if (!loading) onStatus(`${count.toLocaleString()} Gaussians · ${source === 'both' ? 'Both captures' : `${source === 'front' ? 'Front' : 'Back'} capture`}`);
  }
  function setVisibleParts(ids = null) {
    assertAlive(); visibleParts = checkIds(ids); updateVisibility();
  }
  function setSource(value) {
    assertAlive();
    if (!['both', 'front', 'back'].includes(value)) throw new Error(`Unknown capture: ${value}`);
    source = value; updateVisibility();
  }
  function project(point) {
    assertAlive();
    const relative = new pc.Vec3(...point).sub(position);
    const depth = relative.dot(basis.forward);
    const halfHeight = orthographic ? orbit.height : depth * Math.tan(FOV * DEG / 2);
    const x = width / 2 + relative.dot(basis.right) * height / (2 * halfHeight);
    const y = height / 2 - relative.dot(basis.up) * height / (2 * halfHeight);
    return { x, y, visible: Number.isFinite(x) && Number.isFinite(y) && depth > camera.camera.nearClip
      && depth < camera.camera.farClip && x >= 0 && x <= width && y >= 0 && y <= height };
  }
  function ray(x, y) {
    assertAlive();
    const scale = 2 * (orthographic ? orbit.height : Math.tan(FOV * DEG / 2)) / height;
    const offset = basis.right.clone().mulScalar((x - width / 2) * scale)
      .add(basis.up.clone().mulScalar((height / 2 - y) * scale));
    const origin = position.clone();
    const direction = basis.forward.clone();
    if (orthographic) origin.add(offset);
    else direction.add(offset).normalize();
    return { origin: origin.toArray(), direction: direction.toArray() };
  }
  function clearDrag() {
    if (drag && canvas.hasPointerCapture(drag.id)) canvas.releasePointerCapture(drag.id);
    drag = null;
  }
  function dispose() {
    if (disposed) return;
    disposed = true; abort.abort(); observer?.disconnect(); clearDrag();
    listeners.splice(0).forEach(remove => remove());
    for (const entry of entries) { entry.entity.destroy(); entry.resource.destroy(); }
    entries.length = 0;
    app?.destroy();
  }
  async function loadCapture(capture) {
    onStatus(`Loading ${capture.label || capture.id} · ${capture.count.toLocaleString()} Gaussians…`);
    const [binary, labelsBinary] = await Promise.all([
      fetchFile(new URL(capture.url, MANIFEST_URL)), fetchFile(new URL(capture.labelsUrl, MANIFEST_URL))
    ]);
    if (binary.byteLength !== capture.count * FIELDS.length * 4 || labelsBinary.byteLength !== capture.count) {
      throw new Error(`The ${capture.id} capture does not match its manifest. Rebuild the fused export first.`);
    }
    const packed = new Float32Array(binary), labels = new Uint8Array(labelsBinary);
    const counts = new Uint32Array(manifest.parts.length);
    for (const label of labels) {
      if (label >= counts.length) throw new Error(`Unknown body region ${label} in the ${capture.id} capture.`);
      counts[label]++;
    }
    const arrays = Array.from(counts, count => count ? FIELDS.map(() => new Float32Array(count)) : null);
    const offsets = new Uint32Array(counts.length), bounds = Array.from(counts, boundsEmpty);
    for (let i = 0; i < capture.count; i++) {
      const label = labels[i], at = offsets[label]++, k = i * FIELDS.length;
      for (let field = 0; field < FIELDS.length; field++) arrays[label][field][at] = packed[k + field];
      include(bounds[label], [packed[k], packed[k + 1], packed[k + 2]]);
      if (i && i % 150000 === 0) {
        onStatus(`Preparing ${capture.label || capture.id} · ${Math.round(i / capture.count * 100)}%…`);
        await nextFrame();
      }
    }
    for (let index = 0; index < counts.length; index++) {
      if (!counts[index]) continue;
      const data = new pc.GSplatData([{ name: 'vertex', count: counts[index], properties: FIELDS.map((name, field) =>
        ({ name, type: 'float', byteSize: 4, storage: arrays[index][field] })) }]);
      const resource = new pc.GSplatResource(app.graphicsDevice, data);
      const part = manifest.parts[index];
      const entity = new pc.Entity(`${capture.id}:${part.id}`);
      entries.push({ capture: capture.id, part: part.id, count: counts[index], bounds: bounds[index], entity, resource });
      entity.addComponent('gsplat', { resource }); app.root.addChild(entity);
      onStatus(`Preparing ${capture.label || capture.id} · ${part.label || part.id}…`);
      await nextFrame();
    }
  }

  try {
    onStatus('Opening the prepared mannequin scan…');
    manifest = await fetchFile(MANIFEST_URL, 'json'); validateManifest(manifest);
    radius = Math.max(Math.hypot(...manifest.bounds.min.map((n, i) => manifest.bounds.max[i] - n)) / 2, .001);
    app = new pc.Application(canvas, { graphicsDeviceOptions: {
      deviceTypes: ['webgl2'], antialias: false, alpha: false, preserveDrawingBuffer: true
    } });
    app.graphicsDevice.maxPixelRatio = Math.min(window.devicePixelRatio || 1, 1.5);
    app.scene.toneMapping = pc.TONEMAP_LINEAR;
    camera = new pc.Entity('Joint annotation camera');
    camera.addComponent('camera', { clearColor: new pc.Color(.045, .066, .064, 1), fov: FOV,
      aspectRatioMode: pc.ASPECT_MANUAL });
    app.root.addChild(camera);
    app.setCanvasFillMode(pc.FILLMODE_NONE); app.setCanvasResolution(pc.RESOLUTION_AUTO);
    const resize = () => {
      width = Math.max(1, container.clientWidth); height = Math.max(1, container.clientHeight);
      app.resizeCanvas(width, height);
      if (fitted) fitInternal(fittedParts); else updateCamera();
    };
    observer = new ResizeObserver(resize); observer.observe(container); resize();
    app.start();
    for (const capture of manifest.captures) await loadCapture(capture);
    loading = false; updateVisibility();
    addListener(canvas, 'contextmenu', event => event.preventDefault());
    addListener(canvas, 'pointerdown', event => {
      if (!navigationEnabled || ![0, 1, 2].includes(event.button)) return;
      canvas.setPointerCapture(event.pointerId);
      drag = { id: event.pointerId, type: event.button !== 0 || event.shiftKey ? 'pan' : 'orbit',
        x: event.clientX, y: event.clientY, yaw: orbit.yaw, pitch: orbit.pitch,
        target: orbit.target.clone(), right: basis.right.clone(), up: basis.up.clone() };
    });
    addListener(canvas, 'pointermove', event => {
      if (!drag || drag.id !== event.pointerId || !navigationEnabled) return;
      const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
      if (dx === 0 && dy === 0) return;
      fitted = false;
      if (drag.type === 'orbit') {
        orbit.yaw = drag.yaw - dx * .28;
        orbit.pitch = Math.max(-89.5, Math.min(89.5, drag.pitch + dy * .28));
        view = 'orbit';
      } else {
        const scale = 2 * (orthographic ? orbit.height : orbit.distance * Math.tan(FOV * DEG / 2)) / height;
        orbit.target.copy(drag.target).add(drag.right.clone().mulScalar(-dx * scale))
          .add(drag.up.clone().mulScalar(dy * scale));
      }
      updateCamera();
    });
    for (const event of ['pointerup', 'pointercancel', 'lostpointercapture']) addListener(canvas, event, clearDrag);
    addListener(canvas, 'wheel', event => {
      if (!navigationEnabled) return;
      event.preventDefault(); fitted = false;
      const delta = event.deltaY * (event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1);
      const factor = Math.exp(Math.max(-2, Math.min(2, delta * .001)));
      if (orthographic) orbit.height = Math.max(radius * .002, Math.min(radius * 20, orbit.height * factor));
      else orbit.distance = Math.max(radius * .008, Math.min(radius * 40, orbit.distance * factor));
      updateCamera();
    }, { passive: false });
    addListener(canvas, 'webglcontextlost', () => onStatus('The graphics context was interrupted. Reload the page to reopen the scan.'));
    announceCamera();
    return { manifest, setView, setVisibleParts, setSource, fit, project, ray,
      cameraDirection: () => basis.forward.toArray(),
      setNavigationEnabled(value) { navigationEnabled = Boolean(value); if (!navigationEnabled) clearDrag(); },
      dispose };
  } catch (error) {
    dispose(); onStatus(`Scan could not be opened: ${error.message}`); throw error;
  }
}
