import * as pc from '../mannequin-rig/vendor/playcanvas.mjs';

const FIELDS = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
  'scale_0', 'scale_1', 'scale_2', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
const SOURCE_COLORS = { front: [.22, .66, .91], back: [.95, .53, .16] };
const VIEW_NAMES = { perspective: 'Perspective', front: 'Front · ventral (+Y)',
  back: 'Back · dorsal (−Y)', left: 'Left (+X)', right: 'Right (−X)',
  head: 'Head (+Z)', feet: 'Feet (−Z)' };
const VIEW_ANGLES = { perspective: [55, 12], front: [90, 0], back: [-90, 0],
  left: [0, 0], right: [180, 0], head: [90, 89.99], feet: [90, -89.99] };
const $ = id => document.getElementById(id);
const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
const radians = Math.PI / 180;
const state = { source: 'both', part: 'all', color: 'natural', view: 'perspective', busy: true };
const orbit = { yaw: 55, pitch: 12, distance: 3, target: new pc.Vec3() };
const headUp = new pc.Vec3(0, 0, 1);
let manifest, app, camera, sceneRadius = 1, drag = null;
let entries = [], frameCount = 0, fpsTime = performance.now();

function status(message) { $('status').textContent = message; }
function validateManifest(value) {
  if (value?.version !== 1) throw new Error('fusion.json must have version: 1.');
  if (!Array.isArray(value.parts) || !value.parts.length || value.parts.length > 256) {
    throw new Error('fusion.json must define between 1 and 256 parts.');
  }
  const ids = new Set();
  for (const part of value.parts) {
    if (typeof part.id !== 'string' || !part.id || ids.has(part.id) || part.id === 'all') {
      throw new Error('Every part needs a unique id; "all" is reserved.');
    }
    ids.add(part.id);
    if (typeof part.label !== 'string' || !Array.isArray(part.color) || part.color.length !== 3
      || !part.color.every(x => Number.isFinite(x) && x >= 0 && x <= 1)) {
      throw new Error(`Invalid label or RGB color for part ${part.id}.`);
    }
  }
  if (!Array.isArray(value.captures) || value.captures.length !== 2
    || new Set(value.captures.map(c => c.id)).size !== 2) {
    throw new Error('fusion.json must define the front and back captures.');
  }
  for (const capture of value.captures) {
    if (!['front', 'back'].includes(capture.id) || !Number.isInteger(capture.count) || capture.count < 0
      || typeof capture.url !== 'string' || typeof capture.labelsUrl !== 'string') {
      throw new Error('Each capture needs id, url, labelsUrl and a nonnegative integer count.');
    }
  }
  for (const key of ['min', 'max']) {
    if (!Array.isArray(value.bounds?.[key]) || value.bounds[key].length !== 3
      || !value.bounds[key].every(Number.isFinite)) throw new Error('fusion.json needs finite XYZ bounds.');
  }
  if (value.bounds.min.some((x, i) => x > value.bounds.max[i])) throw new Error('Invalid scene bounds.');
}

async function fetchFile(url, type = 'arrayBuffer') {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}. Prepare the local fusion data first.`);
  return response[type]();
}

function emptyBounds() { return { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] }; }
function includePoint(bounds, x, y, z) {
  bounds.min[0] = Math.min(bounds.min[0], x); bounds.max[0] = Math.max(bounds.max[0], x);
  bounds.min[1] = Math.min(bounds.min[1], y); bounds.max[1] = Math.max(bounds.max[1], y);
  bounds.min[2] = Math.min(bounds.min[2], z); bounds.max[2] = Math.max(bounds.max[2], z);
}

async function loadCapture(capture) {
  $('load-detail').textContent = `Reading ${capture.label || capture.id} · ${capture.count.toLocaleString()} Gaussians`;
  const [binary, labelsBinary] = await Promise.all([fetchFile(capture.url), fetchFile(capture.labelsUrl)]);
  if (binary.byteLength !== capture.count * FIELDS.length * 4) {
    throw new Error(`${capture.url} size does not match its count and 14-float layout.`);
  }
  if (labelsBinary.byteLength !== capture.count) throw new Error(`${capture.labelsUrl} must contain one Uint8 label per Gaussian.`);
  const packed = new Float32Array(binary), labels = new Uint8Array(labelsBinary);
  const counts = new Uint32Array(manifest.parts.length);
  for (const label of labels) {
    if (label >= counts.length) throw new Error(`${capture.labelsUrl} contains an unknown part index: ${label}.`);
    counts[label]++;
  }
  const arrays = Array.from(counts, count => count ? FIELDS.map(() => new Float32Array(count)) : null);
  const offsets = new Uint32Array(counts.length);
  const bounds = Array.from(counts, emptyBounds);
  for (let i = 0; i < capture.count; i++) {
    const label = labels[i], at = offsets[label]++, k = i * FIELDS.length;
    const partArrays = arrays[label];
    for (let field = 0; field < FIELDS.length; field++) partArrays[field][at] = packed[k + field];
    includePoint(bounds[label], packed[k], packed[k + 1], packed[k + 2]);
    if (i && i % 150000 === 0) {
      $('load-detail').textContent = `Organizing ${capture.label || capture.id} · ${Math.round(i / capture.count * 100)}%`;
      await nextFrame();
    }
  }
  for (let partIndex = 0; partIndex < counts.length; partIndex++) {
    if (!counts[partIndex]) continue;
    const part = manifest.parts[partIndex];
    const data = new pc.GSplatData([{ name: 'vertex', count: counts[partIndex],
      properties: FIELDS.map((name, f) => ({ name, type: 'float', byteSize: 4, storage: arrays[partIndex][f] })) }]);
    const resource = new pc.GSplatResource(app.graphicsDevice, data);
    const entity = new pc.Entity(`${capture.id}:${part.id}`);
    entity.addComponent('gsplat', { resource });
    app.root.addChild(entity);
    entries.push({ capture: capture.id, part: part.id, partIndex, count: counts[partIndex], bounds: bounds[partIndex], entity, resource });
    $('load-detail').textContent = `Preparing ${capture.label || capture.id} · ${part.label}`;
    await nextFrame();
  }
}

function updateCamera() {
  const yaw = orbit.yaw * radians, pitch = orbit.pitch * radians;
  camera.setPosition(orbit.target.x + orbit.distance * Math.cos(yaw) * Math.cos(pitch),
    orbit.target.y + orbit.distance * Math.sin(yaw) * Math.cos(pitch),
    orbit.target.z + orbit.distance * Math.sin(pitch));
  camera.lookAt(orbit.target, headUp);
  camera.camera.nearClip = Math.max(sceneRadius * .0005, .00001);
  camera.camera.farClip = Math.max(sceneRadius * 100, orbit.distance + sceneRadius * 5);
}

function selectedBounds() {
  if (state.part === 'all') return manifest.bounds;
  const bounds = emptyBounds();
  // Keep framing constant while alternating sources to make disagreement visible.
  for (const entry of entries) {
    if (entry.part !== state.part) continue;
    includePoint(bounds, ...entry.bounds.min);
    includePoint(bounds, ...entry.bounds.max);
  }
  return Number.isFinite(bounds.min[0]) ? bounds : manifest.bounds;
}

function fitSelection() {
  const bounds = selectedBounds();
  orbit.target.set(...bounds.min.map((x, i) => (x + bounds.max[i]) / 2));
  const radius = Math.max(Math.hypot(...bounds.min.map((x, i) => bounds.max[i] - x)) / 2, sceneRadius * .015);
  const aspect = Math.max(.1, $('viewport').clientWidth / Math.max(1, $('viewport').clientHeight));
  const vertical = camera.camera.fov * radians / 2;
  const halfFov = Math.min(vertical, Math.atan(Math.tan(vertical) * aspect));
  orbit.distance = radius / Math.sin(halfFov) * 1.12;
  updateCamera();
}

function setView(name) {
  state.view = name;
  [orbit.yaw, orbit.pitch] = VIEW_ANGLES[name];
  document.querySelectorAll('[data-view]').forEach(button => {
    const active = button.dataset.view === name;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
  $('view-label').textContent = VIEW_NAMES[name];
  updateCamera();
}

function updateVisibility() {
  let count = 0;
  const partCounts = { front: 0, back: 0 };
  for (const entry of entries) {
    const matchesPart = state.part === 'all' || entry.part === state.part;
    entry.entity.enabled = matchesPart && (state.source === 'both' || state.source === entry.capture);
    if (matchesPart) partCounts[entry.capture] += entry.count;
    if (entry.entity.enabled) count += entry.count;
  }
  for (const id of ['front', 'back']) {
    $(`${id}-count`).textContent = partCounts[id].toLocaleString();
    $(`${id}-count`).parentElement.classList.toggle('inactive', state.source !== 'both' && state.source !== id);
  }
  const selectedPart = manifest.parts.find(part => part.id === state.part);
  const title = selectedPart?.label || 'All body parts';
  $('part-summary').textContent = `${(partCounts.front + partCounts.back).toLocaleString()} Gaussians across both captures`;
  $('visible-indicator').textContent = state.source === 'both' ? 'Both captures' : `${state.source === 'front' ? 'Front' : 'Back'} capture only`;
  status(`${count.toLocaleString()} visible Gaussians · ${title} · ${state.source === 'both' ? 'Both captures' : state.source + ' capture'} · prepared local comparison`);
  return count;
}

function setSource(id) {
  state.source = id;
  document.querySelectorAll('[data-source]').forEach(button => {
    const active = button.dataset.source === id;
    button.classList.toggle('active', active); button.setAttribute('aria-pressed', String(active));
  });
  updateVisibility();
}

function tintCaptures() {
  for (const entry of entries) {
    const color = state.color === 'source' ? SOURCE_COLORS[entry.capture] : manifest.parts[entry.partIndex].color;
    entry.entity.gsplat.setWorkBufferModifier(state.color === 'natural' ? null : {
      glsl: `void modifySplatCenter(inout vec3 center) {}
void modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}
void modifySplatColor(vec3 center, inout vec4 color) { color.rgb = vec3(${color.map(x => x.toFixed(5)).join(',')}); }`
    });
  }
  $('source-legend').hidden = state.color !== 'source';
}

function metricLabel(key) {
  return key.replace(/([a-z])([A-Z])/g, '$1 $2').replace(/[_-]/g, ' ').replace(/^./, char => char.toUpperCase());
}
function metricValue(value) {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return Number.isInteger(value) ? value.toLocaleString() : Number(value.toPrecision(5)).toString();
  return String(value);
}
function collectMetrics(object, prefix = '', depth = 0) {
  if (!object || typeof object !== 'object' || Array.isArray(object)) return [];
  const result = [];
  for (const [key, value] of Object.entries(object)) {
    if (['id', 'label', 'part', 'parts', 'regions', 'transform', 'transformation', 'matrix', 'color'].includes(key)) continue;
    const label = prefix ? `${prefix} · ${metricLabel(key)}` : metricLabel(key);
    if (value !== null && ['string', 'number', 'boolean'].includes(typeof value)) result.push([label, metricValue(value)]);
    else if (depth < 1 && value && typeof value === 'object' && !Array.isArray(value)) result.push(...collectMetrics(value, label, depth + 1));
  }
  return result;
}
function drawMetrics() {
  $('metrics').replaceChildren();
  const report = manifest.report;
  $('report-details').hidden = !report;
  if (!report) {
    $('metrics-note').textContent = 'No registration metrics were included with this prepared comparison.';
    return;
  }
  $('raw-report').textContent = JSON.stringify(report, null, 2);
  const parts = report.parts || report.regions;
  let selectedReport = report.summary && typeof report.summary === 'object' ? report.summary : report;
  if (state.part !== 'all') selectedReport = Array.isArray(parts)
    ? parts.find(part => part.id === state.part || part.part === state.part)
    : parts?.[state.part];
  const metrics = collectMetrics(selectedReport).slice(0, 18);
  const name = manifest.parts.find(part => part.id === state.part)?.label;
  $('metrics-title').textContent = name ? `${name} · alignment` : 'Registration report';
  $('metrics-note').textContent = typeof selectedReport?.note === 'string' ? selectedReport.note
    : typeof selectedReport?.notes === 'string' ? selectedReport.notes
    : metrics.length ? 'Values are reported by the preparation pipeline in scan units unless stated otherwise.'
    : name ? 'No per-part metrics were recorded for this region. The full report remains available below.'
    : 'Select a body part to inspect its recorded alignment metrics.';
  for (const [key, value] of metrics) {
    if (key === 'Note' || key === 'Notes') continue;
    const term = document.createElement('dt'), detail = document.createElement('dd');
    term.textContent = key; detail.textContent = value;
    $('metrics').append(term, detail);
  }
}

function setupEvents() {
  document.querySelectorAll('[data-source]').forEach(button => button.onclick = () => setSource(button.dataset.source));
  document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => setView(button.dataset.view));
  $('color-mode').onchange = event => { state.color = event.target.value; tintCaptures(); };
  $('part-select').onchange = event => { state.part = event.target.value; updateVisibility(); drawMetrics(); };
  $('fit').onclick = $('fit-part').onclick = fitSelection;
  const canvas = $('canvas');
  canvas.oncontextmenu = event => event.preventDefault();
  canvas.onpointerdown = event => {
    if (state.busy) return;
    canvas.setPointerCapture(event.pointerId);
    drag = { type: event.button === 2 || event.shiftKey ? 'pan' : 'orbit', x: event.clientX, y: event.clientY,
      yaw: orbit.yaw, pitch: orbit.pitch, target: orbit.target.clone() };
  };
  canvas.onpointermove = event => {
    if (!drag) return;
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    if (drag.type === 'orbit') {
      orbit.yaw = drag.yaw - dx * .28;
      orbit.pitch = Math.max(-89.99, Math.min(89.99, drag.pitch + dy * .28));
      state.view = 'orbit';
      document.querySelectorAll('[data-view]').forEach(button => { button.classList.remove('active'); button.setAttribute('aria-pressed', 'false'); });
      $('view-label').textContent = 'Orbit view';
    } else {
      const scale = 2 * orbit.distance * Math.tan(camera.camera.fov * radians / 2) / Math.max(1, canvas.clientHeight);
      orbit.target.copy(drag.target).add(camera.right.clone().mulScalar(-dx * scale)).add(camera.up.clone().mulScalar(dy * scale));
    }
    updateCamera();
  };
  canvas.onpointerup = canvas.onpointercancel = () => { drag = null; };
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    orbit.distance = Math.max(sceneRadius * .025, Math.min(sceneRadius * 30, orbit.distance * Math.exp(event.deltaY * .001)));
    updateCamera();
  }, { passive: false });
  window.addEventListener('keydown', event => {
    if (state.busy || event.ctrlKey || event.metaKey || event.altKey || ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName)) return;
    const source = { '1': 'front', '2': 'back', '3': 'both' }[event.key];
    if (source) { setSource(source); event.preventDefault(); }
    if (event.key.toLowerCase() === 'f') fitSelection();
  });
}

async function start() {
  $('local-reviews').hidden=!(location.protocol==='http:'&&(location.hostname==='localhost'||/^127(?:\.\d{1,3}){3}$/.test(location.hostname)));
  manifest = await fetchFile('./fusion.json', 'json');
  validateManifest(manifest);
  sceneRadius = Math.max(Math.hypot(...manifest.bounds.min.map((x, i) => manifest.bounds.max[i] - x)) / 2, .001);
  app = new pc.Application($('canvas'), { graphicsDeviceOptions: {
    deviceTypes: ['webgl2'], antialias: false, alpha: false, preserveDrawingBuffer: true
  } });
  app.graphicsDevice.maxPixelRatio = Math.min(devicePixelRatio, 1.5);
  app.scene.toneMapping = pc.TONEMAP_LINEAR;
  camera = new pc.Entity('Review camera');
  camera.addComponent('camera', { clearColor: new pc.Color(.054, .084, .071, 1), fov: 45 });
  app.root.addChild(camera);
  app.setCanvasFillMode(pc.FILLMODE_NONE);
  app.setCanvasResolution(pc.RESOLUTION_AUTO);
  const resize = () => { app.resizeCanvas($('viewport').clientWidth, $('viewport').clientHeight); };
  new ResizeObserver(resize).observe($('viewport'));
  resize(); fitSelection();
  app.on('update', () => {
    frameCount++;
    const now = performance.now();
    if (now - fpsTime >= 1000) {
      $('fps').textContent = `${Math.round(frameCount * 1000 / (now - fpsTime))} fps`;
      fpsTime = now; frameCount = 0;
    }
  });
  app.start();
  $('part-select').append(...manifest.parts.map(part => new Option(part.label, part.id)));
  for (const capture of manifest.captures) await loadCapture(capture);
  const total = manifest.captures.reduce((count, capture) => count + capture.count, 0);
  const fullData = total === manifest.report?.fusedCount;
  $('scene-summary').textContent = `${total.toLocaleString()} Gaussians · ${manifest.parts.length} regions · ${fullData ? 'full dataset' : 'preview'} · feature aligned`;
  setupEvents(); tintCaptures(); updateVisibility(); drawMetrics();
  state.busy = false;
  $('loading').hidden = true;
  // Read-only state for local quality checks; no mutation API or automatic editing.
  window.mannequinFusion = {
    get manifest() { return structuredClone(manifest); },
    get stats() { return { source: state.source, part: state.part, color: state.color, view: state.view,
      count: total, visible: entries.reduce((count, e) => count + (e.entity.enabled ? e.count : 0), 0),
      captures: entries.map(e => ({ capture: e.capture, part: e.part, count: e.count, visible: e.entity.enabled })),
      camera: { position: camera.getPosition().toArray(), target: orbit.target.toArray(), distance: orbit.distance } }; }
  };
}

start().catch(error => {
  console.error(error);
  $('load-title').textContent = 'Unable to load the fusion review';
  $('load-detail').textContent = `${error.message} Serve the repository over HTTP with python3 tools/serve.py 8766.`;
  $('loading').querySelector('.loader').hidden = true;
  status(error.message);
});
