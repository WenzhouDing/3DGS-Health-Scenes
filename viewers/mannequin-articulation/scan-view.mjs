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
const IDENTITY_POSE = Object.freeze({ position: Object.freeze([0, 0, 0]), rotation: Object.freeze([0, 0, 0, 1]) });
const vector = (value, length) => (Array.isArray(value) || ArrayBuffer.isView(value))
  && value.length === length && Array.from(value).every(Number.isFinite);
function include(bounds, xyz) {
  for (let i = 0; i < 3; i++) {
    bounds.min[i] = Math.min(bounds.min[i], xyz[i]);
    bounds.max[i] = Math.max(bounds.max[i], xyz[i]);
  }
}

/** A pose maps original fused-viewer points to their posed world coordinates.
 * Exported for numeric checks without constructing a GPU viewer.
 */
export function normalizePartTransform(value) {
  if (!value || !vector(value.position, 3) || !vector(value.rotation, 4)) {
    throw new TypeError('Each body transform requires a finite position and an XYZW quaternion.');
  }
  const length = Math.hypot(...value.rotation);
  if (!Number.isFinite(length) || length < 1e-12) throw new TypeError('A body rotation must be a nonzero finite quaternion.');
  return { position: Array.from(value.position), rotation: Array.from(value.rotation, n => n / length) };
}

export function transformPartBounds(bounds, pose) {
  const result = boundsEmpty(), point = new pc.Vec3(), rotation = new pc.Quat(...pose.rotation);
  for (let mask = 0; mask < 8; mask++) {
    point.set(...bounds.min.map((n, i) => (mask >> i) & 1 ? bounds.max[i] : n));
    rotation.transformVector(point, point);
    include(result, [point.x + pose.position[0], point.y + pose.position[1], point.z + pose.position[2]]);
  }
  return result;
}

/** A purely illustrative bed in viewer coordinates, separate from all scan resources.
 * The mattress lies in XY; center is its volume center and size is [width, length, thickness].
 * An optional backrest has its own world center and XYZW orientation.
 */
export function normalizeBedConfig(value) {
  if (value == null) return null;
  const size = (input, name) => {
    if (!vector(input, 3) || input.some(n => n <= 0)) throw new TypeError(`${name} needs three positive finite dimensions.`);
    return Array.from(input);
  };
  if (!vector(value.center, 3) || !Number.isFinite(value.frameHeight) || value.frameHeight < 0) {
    throw new TypeError('The illustrative bed needs a finite center and a nonnegative frame height.');
  }
  const result = { center: Array.from(value.center), size: size(value.size, 'Mattress'), frameHeight: value.frameHeight };
  if (value.backrest != null) {
    const pose = normalizePartTransform({ position: value.backrest.center, rotation: value.backrest.rotation });
    result.backrest = { center: pose.position, rotation: pose.rotation, size: size(value.backrest.size, 'Backrest') };
  }
  return result;
}
function validateManifest(value) {
  if (value?.version !== 1 || !Array.isArray(value.parts) || !value.parts.length || value.parts.length > 256) {
    throw new Error('The fusion manifest must contain version 1 and 1–256 body regions.');
  }
  if (value.parts.some(p => typeof p.id !== 'string' || !p.id) || new Set(value.parts.map(p => p.id)).size !== value.parts.length) {
    throw new Error('The fusion manifest contains invalid body region IDs.');
  }
  if (value.parts.some(part => !vector(part.color, 3) || part.color.some(n => n < 0 || n > 1))) {
    throw new Error('Every body region needs a finite RGB segmentation color.');
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

/** Articulate the complete fused scan. World coordinates use the exported viewer XYZ frame, +Z up.
 * Screen coordinates are CSS pixels relative to container, independent of device pixel ratio.
 * Every capture/part entity is a direct scene-root child: callers supply complete world poses,
 * so a parent's rotation is never applied twice. Resource means/covariances/colors remain original.
 */
export async function createArticulationView({ canvas, container, onStatus = () => {}, onCameraChange = () => {} }) {
  if (!canvas || !container) throw new Error('A canvas and its viewport container are required.');
  const abort = new AbortController();
  const entries = [];
  const partTransforms = new Map();
  const listeners = [];
  let app, camera, observer, manifest, disposed = false, loading = true, drag = null;
  let colorMode = 'natural', theme = 'light', bedRoot = null, bedConfig = null, bedBounds = null;
  const bedMaterials = new Map();
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
    updateClipping();
    announceCamera();
  }
  function updateClipping() {
    if (!camera) return;
    let farthest = orbit.distance + radius;
    const bounds = selectionBounds(null, true);
    for (let mask = 0; mask < 8; mask++) {
      const point = new pc.Vec3(...bounds.min.map((n, i) => (mask >> i) & 1 ? bounds.max[i] : n));
      farthest = Math.max(farthest, point.sub(position).dot(basis.forward));
    }
    camera.camera.nearClip = Math.max(Math.min(radius, 1) * .0001, .000001);
    camera.camera.farClip = Math.max(radius * 100, farthest + radius * 2, 1);
  }
  function selectionBounds(ids, includeBed = false) {
    const result = boundsEmpty();
    for (const entry of entries) if ((!ids || ids.has(entry.part)) && (source === 'both' || source === entry.capture)) {
      include(result, entry.worldBounds.min); include(result, entry.worldBounds.max);
    }
    if (!Number.isFinite(result.min[0])) {
      for (const entry of entries) { include(result, entry.worldBounds.min); include(result, entry.worldBounds.max); }
    }
    if (!Number.isFinite(result.min[0])) {
      include(result, manifest.bounds.min); include(result, manifest.bounds.max);
    }
    if (includeBed && bedBounds) { include(result, bedBounds.min); include(result, bedBounds.max); }
    return result;
  }
  function getBounds(ids = null) {
    assertAlive();
    const bounds = selectionBounds(checkIds(ids));
    return { min: bounds.min.slice(), max: bounds.max.slice() };
  }
  function setPartTransforms(transforms) {
    assertAlive();
    if (!(transforms instanceof Map) && (!transforms || typeof transforms !== 'object' || Array.isArray(transforms))) {
      throw new TypeError('Body transforms must be a Map or an object keyed by body region.');
    }
    const next = new Map();
    for (const [id, value] of transforms instanceof Map ? transforms : Object.entries(transforms)) {
      checkIds([id]); next.set(id, normalizePartTransform(value));
    }
    // Validate the entire incoming pose before changing anything. Missing parts return to rest.
    let changed = false;
    for (const part of manifest.parts) {
      const before = partTransforms.get(part.id) || IDENTITY_POSE, after = next.get(part.id) || IDENTITY_POSE;
      if (before.position.every((n, i) => n === after.position[i])
        && before.rotation.every((n, i) => n === after.rotation[i])) continue;
      changed = true;
      partTransforms.set(part.id, after);
      for (const entry of entries) if (entry.part === part.id) {
        entry.entity.setPosition(...after.position);
        entry.entity.setRotation(...after.rotation);
        entry.worldBounds = transformPartBounds(entry.bounds, after);
      }
    }
    if (!changed) return false;
    const bounds = selectionBounds(null);
    radius = Math.max(Math.hypot(...bounds.min.map((n, i) => bounds.max[i] - n)) / 2, .001);
    // Keep the current camera stable while a limb moves. Fit and view changes use the posed bounds.
    // PlayCanvas's unified GSplatInfo.update detects world-matrix changes; bake() then uploads
    // rotated Gaussian covariance and marks sorting dirty, including when the camera is stationary.
    // AUTO avoids touching untouched body regions and preserves ordering across both captures.
    updateClipping();
    app.renderNextFrame = true;
    return true;
  }
  function fitInternal(ids) {
    // A full scene fit includes the illustrative bed; focusing a region remains a close body view.
    const bounds = selectionBounds(ids, ids == null);
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
  function focusPoint(point, { span = radius * .3 } = {}) {
    assertAlive();
    if (!vector(point, 3) || !Number.isFinite(span) || span <= 0) {
      throw new TypeError('Focus needs a finite world point and a positive span in scan units.');
    }
    fitted = false;
    orbit.target.set(...point);
    orbit.height = span / 2;
    orbit.distance = orthographic ? Math.max(radius * 3, span * 2) : orbit.height / Math.tan(FOV * DEG / 2);
    updateCamera();
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
  function setCamera({ yaw, pitch }) {
    assertAlive();
    if (!Number.isFinite(yaw) || !Number.isFinite(pitch) || Math.abs(pitch) >= 90) {
      throw new TypeError('A presentation camera needs a finite yaw and a pitch between -90 and 90 degrees.');
    }
    view = 'perspective'; orthographic = false; fitted = true;
    orbit.yaw = yaw; orbit.pitch = pitch;
    fitInternal(fittedParts);
  }
  function setColorMode(value) {
    assertAlive();
    if (!['natural', 'segments'].includes(value)) throw new Error(`Unknown scan color mode: ${value}`);
    if (colorMode === value) return;
    colorMode = value;
    for (const entry of entries) {
      const color = manifest.parts.find(part => part.id === entry.part).color;
      // Only RGB changes. Centers, covariance, opacity and the original resource all stay intact.
      // Both captures share the manifest's body-region palette, including while articulated.
      entry.entity.gsplat.setWorkBufferModifier(value === 'natural' ? null : {
        glsl: `void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}
void modifySplatColor(vec3 center, inout vec4 color) { color.rgb = vec3(${color.map(n => n.toFixed(5)).join(',')}); }`
      });
    }
    app.renderNextFrame = true;
  }
  function updateBedMaterials() {
    const colors = theme === 'light'
      ? { mattress: [.76, .81, .84], frame: [.44, .51, .55], legs: [.53, .59, .62] }
      : { mattress: [.43, .49, .53], frame: [.23, .29, .33], legs: [.31, .38, .42] };
    for (const [name, material] of bedMaterials) {
      material.diffuse.set(...colors[name]);
      material.update();
    }
  }
  function setTheme(value) {
    assertAlive();
    if (!['light', 'dark'].includes(value)) throw new Error(`Unknown viewer theme: ${value}`);
    theme = value;
    camera.camera.clearColor = value === 'light' ? new pc.Color(.957, .969, .98, 1) : new pc.Color(.059, .075, .094, 1);
    updateBedMaterials();
    app.renderNextFrame = true;
  }
  function bedMaterial(name) {
    if (!bedMaterials.has(name)) {
      const material = new pc.StandardMaterial();
      material.name = `Illustrative bed ${name}`;
      material.gloss = .15; material.useMetalness = true; material.metalness = 0;
      bedMaterials.set(name, material); updateBedMaterials();
    }
    return bedMaterials.get(name);
  }
  function setBed(value) {
    assertAlive();
    const next = normalizeBedConfig(value);
    if (JSON.stringify(next) === JSON.stringify(bedConfig)) return false;
    const root = next ? new pc.Entity('Illustrative bed — not scanned geometry') : null;
    const bounds = next ? boundsEmpty() : null;
    const box = (name, center, size, material, rotation = [0, 0, 0, 1]) => {
      const entity = new pc.Entity(name);
      entity.addComponent('render', { type: 'box', material: bedMaterial(material), castShadows: false, receiveShadows: false });
      entity.setPosition(...center); entity.setRotation(...rotation); entity.setLocalScale(...size);
      root.addChild(entity);
      const boxBounds = transformPartBounds({ min: size.map(n => -n / 2), max: size.map(n => n / 2) },
        { position: center, rotation });
      include(bounds, boxBounds.min); include(bounds, boxBounds.max);
    };
    if (next) {
      const [x, y, z] = next.center, [w, l, t] = next.size, bottom = z - t / 2;
      box('Reference mattress', next.center, next.size, 'mattress');
      if (next.frameHeight > 0) {
        const frameThickness = Math.min(t * .5, next.frameHeight * .22);
        const legWidth = Math.min(w, l) * .045;
        box('Reference frame', [x, y, bottom - frameThickness / 2], [w * .94, l * .98, frameThickness], 'frame');
        for (const side of [-1, 1]) for (const end of [-1, 1]) {
          box('Reference leg', [x + side * w * .41, y + end * l * .43, bottom - next.frameHeight / 2],
            [legWidth, legWidth, next.frameHeight], 'legs');
        }
      }
      if (next.backrest) box('Reference raised backrest', next.backrest.center, next.backrest.size, 'mattress', next.backrest.rotation);
      app.root.addChild(root);
    }
    bedRoot?.destroy(); bedRoot = root; bedConfig = next; bedBounds = bounds;
    if (fitted && fittedParts == null) fitInternal(null); else updateClipping();
    app.renderNextFrame = true;
    return true;
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
    bedRoot?.destroy(); bedRoot = null;
    for (const material of bedMaterials.values()) material.destroy();
    bedMaterials.clear();
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
    const offsets = new Uint32Array(counts.length);
    for (let i = 0; i < capture.count; i++) {
      const label = labels[i], at = offsets[label]++, k = i * FIELDS.length;
      for (let field = 0; field < FIELDS.length; field++) arrays[label][field][at] = packed[k + field];
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
      // Resource bounds include Gaussian extents, not just center positions.
      const localBounds = { min: resource.aabb.getMin().toArray(), max: resource.aabb.getMax().toArray() };
      entries.push({ capture: capture.id, part: part.id, count: counts[index], bounds: localBounds,
        worldBounds: transformPartBounds(localBounds, IDENTITY_POSE), entity, resource });
      entity.addComponent('gsplat', { resource, unified: true, workBufferUpdate: pc.WORKBUFFER_UPDATE_AUTO });
      app.root.addChild(entity);
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
    app.scene.ambientLight = new pc.Color(.7, .7, .7);
    const bedLight = new pc.Entity('Reference bed lighting');
    bedLight.addComponent('light', { type: 'directional', color: new pc.Color(1, 1, 1), intensity: .55, castShadows: false });
    bedLight.setEulerAngles(38, -28, -38); app.root.addChild(bedLight);
    camera = new pc.Entity('Articulation camera');
    camera.addComponent('camera', { clearColor: new pc.Color(.045, .066, .064, 1), fov: FOV,
      aspectRatioMode: pc.ASPECT_MANUAL });
    app.root.addChild(camera);
    setTheme(theme);
    app.setCanvasFillMode(pc.FILLMODE_NONE); app.setCanvasResolution(pc.RESOLUTION_AUTO);
    const resize = () => {
      width = Math.max(1, container.clientWidth); height = Math.max(1, container.clientHeight);
      app.resizeCanvas(width, height);
      if (fitted) fitInternal(fittedParts); else updateCamera();
    };
    observer = new ResizeObserver(resize); observer.observe(container); resize();
    app.start();
    for (const capture of manifest.captures) await loadCapture(capture);
    loading = false; updateVisibility(); fitInternal(null);
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
    return { manifest, setPartTransforms, setView, setCamera, setVisibleParts, setSource, setColorMode, setTheme, setBed,
      fit, focusPoint, getBounds, project, ray,
      cameraDirection: () => basis.forward.toArray(),
      setNavigationEnabled(value) { navigationEnabled = Boolean(value); if (!navigationEnabled) clearDrag(); },
      dispose };
  } catch (error) {
    dispose(); onStatus(`Scan could not be opened: ${error.message}`); throw error;
  }
}
