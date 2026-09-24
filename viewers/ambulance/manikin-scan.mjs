import { validateAnnotations } from '../mannequin-joints/annotation-core.mjs';

const MANIFEST_URL = new URL('../mannequin-fusion/fusion.json', import.meta.url);
const FIELDS = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3',
  'scale_0', 'scale_1', 'scale_2', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
const IDENTITY = Object.freeze({ position: Object.freeze([0, 0, 0]), rotation: Object.freeze([0, 0, 0, 1]) });
const CELL_SIZE = .02;
const MAX_SAMPLE_SIGMA = .025;
const MAX_SAMPLES = 12000;
const isVector = (value, size) => (Array.isArray(value) || ArrayBuffer.isView(value))
  && value.length === size && Array.from(value).every(Number.isFinite);
const yieldThread = () => new Promise(resolve => setTimeout(resolve, 0));

export function normalizeManikinTransform(value) {
  if (!value || !isVector(value.position, 3) || !isVector(value.rotation, 4)) {
    throw new TypeError('A manikin transform needs a finite position and an XYZW quaternion.');
  }
  const length = Math.hypot(...value.rotation);
  if (!Number.isFinite(length) || length < 1e-12) throw new TypeError('A manikin quaternion must be nonzero.');
  return { position: Array.from(value.position), rotation: Array.from(value.rotation, n => n / length) };
}

function normalizePlacement(value) {
  const pose = normalizeManikinTransform(value);
  if (!Number.isFinite(value.scale) || value.scale <= 0) throw new TypeError('Manikin placement needs a positive uniform scale.');
  return { ...pose, scale: value.scale };
}

function rotate(q, x, y, z) {
  const tx = 2 * (q[1] * z - q[2] * y), ty = 2 * (q[2] * x - q[0] * z), tz = 2 * (q[0] * y - q[1] * x);
  return [x + q[3] * tx + q[1] * tz - q[2] * ty,
    y + q[3] * ty + q[2] * tx - q[0] * tz,
    z + q[3] * tz + q[0] * ty - q[1] * tx];
}

function validateManifest(manifest) {
  if (manifest?.version !== 1 || !Array.isArray(manifest.parts) || manifest.parts.length !== 16
    || manifest.parts.some(part => typeof part?.id !== 'string' || !part.id)
    || new Set(manifest.parts.map(part => part.id)).size !== 16) {
    throw new Error('The fused manikin manifest must describe all 16 body regions.');
  }
  if (!Array.isArray(manifest.captures) || manifest.captures.length !== 2
    || new Set(manifest.captures.map(capture => capture.id)).size !== 2
    || manifest.captures.some(capture => !['front', 'back'].includes(capture.id)
      || !Number.isSafeInteger(capture.count) || capture.count < 1
      || typeof capture.url !== 'string' || !capture.url
      || typeof capture.labelsUrl !== 'string' || !capture.labelsUrl)) {
    throw new Error('The fused manikin manifest must describe both complete prepared captures.');
  }
  if (!isVector(manifest.bounds?.min, 3) || !isVector(manifest.bounds?.max, 3)
    || manifest.bounds.min.some((n, i) => n > manifest.bounds.max[i])) {
    throw new Error('The fused manikin manifest has invalid scene bounds.');
  }
}

/** Mirrors the articulation page: local saved map first on localhost, then only
 * a refinement matching the saved bytes, scene revision and annotation schema.
 * The optional refinement never prevents a valid saved map from opening.
 */
async function loadAnnotations(manifest, signal) {
  const localHost = ['localhost', '127.0.0.1', '[::1]'].includes(globalThis.location?.hostname);
  let response, local = false;
  const load = path => fetch(new URL(path, import.meta.url), { cache: 'no-store', signal });
  if (localHost) {
    response = await load('../../raw/mannequin-fused/joint-annotations.json');
    if (response.ok) local = true;
    else if (response.status !== 404) throw new Error('The local manikin joint map could not be read.');
  }
  if (!local) response = await load('../mannequin-articulation/joint-annotations.json');
  if (!response.ok) throw new Error(`The manikin joint map could not be loaded: HTTP ${response.status}.`);
  const text = await response.text(), saved = JSON.parse(text);
  const validation = validateAnnotations(saved, manifest);
  if (!validation.valid) throw new Error(validation.errors.join(' '));
  const digest = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  const annotationSha256 = Array.from(new Uint8Array(digest), n => n.toString(16).padStart(2, '0')).join('');
  let annotations = saved, refinement = null;
  try {
    const fitted = await load(local ? '../../raw/mannequin-fused/joint-refinement.json'
      : '../mannequin-articulation/joint-refinement.json');
    if (fitted.ok) {
      const candidate = await fitted.json();
      if (candidate.schema === 'mannequin-joint-refinement' && candidate.version === 1
        && Array.isArray(candidate.joints) && candidate.joints.every(item => item && typeof item.id === 'string')
        && candidate.inputAnnotationSha256 === annotationSha256
        && candidate.inputSceneRevision === saved.scene.revision
        && validateAnnotations(candidate.annotations, manifest).valid) {
        annotations = candidate.annotations;
        refinement = candidate;
      }
    }
  } catch { /* An optional refinement is allowed to be absent, stale or invalid. */ }
  return { annotations, refinement, annotationSha256, annotationSource: local ? 'local' : 'published' };
}

/** Loads the full fused scan into the existing ambulance app and engine instance.
 * Native part poses map fused-rest XYZ to posed XYZ. The placement parent then
 * maps those coordinates to ambulance world; no implicit axis flip is applied.
 * Resource values are copied bit-for-bit, with only rigid entity transforms and
 * uniform parent scale. Unified GSplat sorting includes the ambulance scene.
 */
export async function createAmbulanceManikin({ app, pc, onStatus = () => {} }) {
  if (!app?.root || !app.graphicsDevice || !pc?.Entity || !pc.GSplatData || !pc.GSplatResource
    || pc.WORKBUFFER_UPDATE_AUTO === undefined) {
    throw new TypeError('The existing ambulance app and its GSplat engine exports are required.');
  }
  const abort = new AbortController(), entries = [], sampleCells = new Map();
  const root = new pc.Entity('Articulated fused manikin');
  root.enabled = false;
  app.root.addChild(root);
  let disposed = false, manifest, partIds, restSamples = [], partTransforms = new Map();
  let placement = { ...IDENTITY, scale: 1 }, sampleCandidates = 0;
  const assertAlive = () => { if (disposed) throw new Error('The ambulance manikin has been disposed.'); };
  const requestRender = () => { app.renderNextFrame = true; };

  function normalizeTransforms(transforms) {
    if (!(transforms instanceof Map) && (!transforms || typeof transforms !== 'object' || Array.isArray(transforms))) {
      throw new TypeError('Body transforms must be a Map or an object keyed by body region.');
    }
    const normalized = new Map();
    for (const [part, pose] of transforms instanceof Map ? transforms : Object.entries(transforms)) {
      if (!partIds.has(part)) throw new Error(`Unknown manikin body region: ${part}`);
      normalized.set(part, normalizeManikinTransform(pose));
    }
    return normalized;
  }

  function setPartTransforms(transforms) {
    assertAlive();
    const next = normalizeTransforms(transforms);
    let changed = false;
    // Validate the whole map first. Missing parts return to their captured rest pose.
    for (const entry of entries) {
      const before = partTransforms.get(entry.part) || IDENTITY, after = next.get(entry.part) || IDENTITY;
      if (before.position.every((n, i) => n === after.position[i])
        && before.rotation.every((n, i) => n === after.rotation[i])) continue;
      entry.entity.setLocalPosition(...after.position);
      entry.entity.setLocalRotation(...after.rotation);
      changed = true;
    }
    partTransforms = next;
    if (changed) requestRender();
    return changed;
  }

  function setPlacement(value) {
    assertAlive();
    const next = normalizePlacement(value);
    root.setLocalPosition(...next.position);
    root.setLocalRotation(...next.rotation);
    root.setLocalScale(next.scale, next.scale, next.scale);
    placement = next;
    requestRender();
  }

  /** Stable surface representatives, not a calibrated or watertight collider.
   * Proposed transforms/placement can be checked without changing visible state.
   * Samples and radii are returned in ambulance world units, independent of visibility.
   */
  function getCollisionSamples(transforms = partTransforms, proposedPlacement = placement, {parts = null} = {}) {
    assertAlive();
    const poses = transforms === partTransforms ? partTransforms : normalizeTransforms(transforms);
    const parent = proposedPlacement === placement ? placement : normalizePlacement(proposedPlacement);
    const filter = parts === null ? null : parts instanceof Set ? parts : new Set(parts);
    return (filter ? restSamples.filter(sample => filter.has(sample.part)) : restSamples).map(sample => {
      const pose = poses.get(sample.part) || IDENTITY;
      const local = rotate(pose.rotation, ...sample.position);
      const world = rotate(parent.rotation, local[0] + pose.position[0], local[1] + pose.position[1], local[2] + pose.position[2]);
      return { id: sample.id, part: sample.part,
        position: world.map((n, axis) => parent.position[axis] + parent.scale * n), radius: sample.radius * parent.scale };
    });
  }

  function dispose() {
    if (disposed) return;
    disposed = true;
    abort.abort();
    root.destroy();
    for (const entry of entries) entry.resource.destroy();
    entries.length = 0;
    restSamples = [];
    sampleCells.clear();
    partTransforms.clear();
    requestRender();
  }

  function considerSample(source, offset, partIndex) {
    const logit = source[offset + 10], maxLogScale = Math.max(source[offset + 7], source[offset + 8], source[offset + 9]);
    if (logit < 0 || !Number.isFinite(logit) || !Number.isFinite(maxLogScale) || maxLogScale > Math.log(MAX_SAMPLE_SIGMA)) return;
    const position = [source[offset], source[offset + 1], source[offset + 2]];
    if (!position.every(Number.isFinite) || position.some((n, i) => n < manifest.bounds.min[i] || n > manifest.bounds.max[i])) return;
    sampleCandidates++;
    const cell = position.map(n => Math.floor(n / CELL_SIZE));
    const id = `${manifest.parts[partIndex].id}:${cell.join(':')}`, previous = sampleCells.get(id);
    if (previous && (previous.logit > logit || (previous.logit === logit && previous.maxLogScale <= maxLogScale))) return;
    sampleCells.set(id, { id, part: manifest.parts[partIndex].id, partIndex, cell, position,
      radius: Math.min(CELL_SIZE * .5, 2 * Math.exp(maxLogScale)), logit, maxLogScale });
  }

  async function loadCapture(capture) {
    onStatus(`Loading the complete ${capture.id} manikin capture…`);
    const fetchBinary = async path => {
      const response = await fetch(new URL(path, MANIFEST_URL), { signal: abort.signal });
      if (!response.ok) throw new Error(`Cannot load manikin ${path}: HTTP ${response.status}.`);
      return response.arrayBuffer();
    };
    const [buffer, labelBuffer] = await Promise.all([fetchBinary(capture.url), fetchBinary(capture.labelsUrl)]);
    if (buffer.byteLength !== capture.count * FIELDS.length * 4 || labelBuffer.byteLength !== capture.count) {
      throw new Error(`The prepared ${capture.id} capture does not match its manifest byte counts.`);
    }
    const source = new Float32Array(buffer), labels = new Uint8Array(labelBuffer), counts = new Uint32Array(manifest.parts.length);
    for (const label of labels) {
      if (label >= counts.length) throw new Error(`The ${capture.id} capture contains an invalid region label.`);
      counts[label]++;
    }
    const arrays = Array.from(counts, count => FIELDS.map(() => new Float32Array(count)));
    const offsets = new Uint32Array(counts.length);
    for (let i = 0; i < capture.count; i++) {
      const label = labels[i], index = offsets[label]++, offset = i * FIELDS.length;
      for (let field = 0; field < FIELDS.length; field++) arrays[label][field][index] = source[offset + field];
      considerSample(source, offset, label);
      if (i > 0 && i % 150000 === 0) await yieldThread();
    }
    for (let index = 0; index < counts.length; index++) {
      if (!counts[index]) continue;
      const part = manifest.parts[index].id;
      const data = new pc.GSplatData([{ name: 'vertex', count: counts[index],
        properties: FIELDS.map((name, field) => ({ name, type: 'float', byteSize: 4, storage: arrays[index][field] })) }]);
      const resource = new pc.GSplatResource(app.graphicsDevice, data), entity = new pc.Entity(`manikin:${capture.id}:${part}`);
      // Register ownership before component creation so a partial GPU failure is cleaned up.
      entries.push({ part, capture: capture.id, entity, resource });
      root.addChild(entity);
      entity.addComponent('gsplat', { resource, unified: true, workBufferUpdate: pc.WORKBUFFER_UPDATE_AUTO });
      await yieldThread();
    }
  }

  try {
    onStatus('Loading the fused manikin manifest and joint map…');
    const response = await fetch(MANIFEST_URL, { signal: abort.signal });
    if (!response.ok) throw new Error(`Cannot load the fused manikin manifest: HTTP ${response.status}.`);
    manifest = await response.json();
    validateManifest(manifest);
    partIds = new Set(manifest.parts.map(part => part.id));
    const annotationInfo = await loadAnnotations(manifest, abort.signal);
    for (const capture of manifest.captures) await loadCapture(capture);
    restSamples = Array.from(sampleCells.values()).sort((a, b) => a.partIndex - b.partIndex
      || a.cell[0] - b.cell[0] || a.cell[1] - b.cell[1] || a.cell[2] - b.cell[2]);
    const occupiedCells = restSamples.length;
    // A deterministic cap keeps future denser revisions from making pose sweeps unbounded.
    if (restSamples.length > MAX_SAMPLES) {
      const source = restSamples;
      restSamples = Array.from({ length: MAX_SAMPLES }, (_, i) => source[Math.floor(i * source.length / MAX_SAMPLES)]);
    }
    sampleCells.clear();
    const count = manifest.captures.reduce((sum, capture) => sum + capture.count, 0);
    const sampleCount = restSamples.length;
    const collisionSampleInfo = Object.freeze({ count: sampleCount, candidates: sampleCandidates, occupiedCells,
      cellSize: CELL_SIZE, minimumOpacity: .5, maximumSigma: MAX_SAMPLE_SIGMA, units: 'scan units before placement',
      byPart: Object.freeze(Object.fromEntries(manifest.parts.map(part => [part.id, restSamples.filter(sample => sample.part === part.id).length]))) });
    root.enabled = true;
    requestRender();
    onStatus(`Loaded ${count.toLocaleString()} captured Gaussians in 16 articulated regions.`);
    return { manifest, ...annotationInfo, count, sampleCount, collisionSampleInfo,
      setPartTransforms, setPlacement, getCollisionSamples,
      setVisible(value) { assertAlive(); root.enabled = Boolean(value); requestRender(); }, dispose };
  } catch (error) {
    dispose();
    onStatus(`The manikin could not be loaded: ${error.message}`);
    throw error;
  }
}
