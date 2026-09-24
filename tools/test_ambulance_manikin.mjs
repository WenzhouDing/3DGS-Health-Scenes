// Run with Node; --full additionally checks every field of both prepared captures.
import assert from 'node:assert/strict';
import { readFile, mkdir, writeFile } from 'node:fs/promises';
import { createHash, webcrypto } from 'node:crypto';
import { createAmbulanceManikin } from '../viewers/ambulance/manikin-scan.mjs';
import { createMotion, transformPoint } from '../viewers/mannequin-articulation/motion-core.mjs';
import { buildPreset } from '../viewers/mannequin-articulation/pose-presets.mjs';
import { createContactChecker } from '../viewers/ambulance/contact-check.mjs';

const base = new URL('../', import.meta.url);
const read = path => readFile(new URL(path, base));
const realManifest = JSON.parse(await read('viewers/mannequin-fusion/fusion.json'));
const savedText = (await read('viewers/mannequin-articulation/joint-annotations.json')).toString();
const saved = JSON.parse(savedText);
const fitted = JSON.parse(await read('viewers/mannequin-articulation/joint-refinement.json'));
const savedHash = createHash('sha256').update(savedText).digest('hex');
assert.equal(fitted.inputAnnotationSha256, savedHash, 'local refinement corresponds to the current annotation bytes');
const fields = ['x', 'y', 'z', 'rot_0', 'rot_1', 'rot_2', 'rot_3', 'scale_0', 'scale_1', 'scale_2', 'opacity', 'f_dc_0', 'f_dc_1', 'f_dc_2'];
const savedGlobals = Object.fromEntries(['fetch', 'location', 'crypto'].map(key => [key, Object.getOwnPropertyDescriptor(globalThis, key)]));
const near = (a, b) => a.forEach((n, i) => assert.ok(Math.abs(n - b[i]) < 1e-10, `${a} differs from ${b}`));

function mockEngine() {
  const resources = [];
  class Entity {
    constructor(name) { this.name = name; this.children = []; this.position = [0, 0, 0]; this.rotation = [0, 0, 0, 1]; this.scale = [1, 1, 1]; }
    addChild(child) { child.parent = this; this.children.push(child); }
    addComponent(type, data) { assert.equal(type, 'gsplat'); this.gsplat = data; }
    setLocalPosition(...v) { this.position = v; }
    setLocalRotation(...v) { this.rotation = v; }
    setLocalScale(...v) { this.scale = v; }
    destroy() { this.destroyed = true; for (const child of [...this.children]) child.destroy(); if (this.parent) this.parent.children.splice(this.parent.children.indexOf(this), 1); }
  }
  class GSplatData { constructor(elements) { this.elements = elements; } }
  class GSplatResource {
    constructor(device, data) { assert.ok(device); this.data = data; this.destroyCount = 0; resources.push(this); }
    destroy() { this.destroyCount++; }
  }
  const app = { root: new Entity('Existing ambulance'), graphicsDevice: {}, renderNextFrame: false };
  return { app, resources, pc: { Entity, GSplatData, GSplatResource, WORKBUFFER_UPDATE_AUTO: 19 } };
}

function syntheticCapture(capture) {
  const source = new Float32Array(16 * 3 * 14), labels = new Uint8Array(16 * 3);
  for (let part = 0; part < 16; part++) for (let row = 0; row < 3; row++) {
    const index = part * 3 + row;
    labels[index] = part;
    source.set([-.693 + .08 * part + (capture.id === 'back' ? .001 : 0), .04 + row * .03, -.2,
      1, 0, 0, 0, Math.log(row === 2 ? .1 : .004), Math.log(.002), Math.log(.001),
      row === 1 ? -2 : (capture.id === 'back' ? 2 : 1), .125, -.25, .75], index * 14);
  }
  return { source, labels };
}

async function fixture({ full = false, localStatus = 200, refinement = fitted, badLabels = false } = {}) {
  const manifest = structuredClone(realManifest), binaries = new Map(), requests = [];
  for (const capture of manifest.captures) {
    if (full) {
      const bytes = await read(`viewers/mannequin-fusion/${capture.url}`), labels = await read(`viewers/mannequin-fusion/${capture.labelsUrl}`);
      binaries.set(capture.url, new Uint8Array(bytes).buffer);
      binaries.set(capture.labelsUrl, new Uint8Array(labels).buffer);
    } else {
      capture.count = 48;
      const data = syntheticCapture(capture);
      if (badLabels && capture.id === 'back') data.labels[0] = 16;
      binaries.set(capture.url, data.source.buffer); binaries.set(capture.labelsUrl, data.labels.buffer);
    }
  }
  Object.defineProperty(globalThis, 'location', { configurable: true, value: { hostname: 'localhost' } });
  Object.defineProperty(globalThis, 'crypto', { configurable: true, value: webcrypto });
  Object.defineProperty(globalThis, 'fetch', { configurable: true, value: async (url, options) => {
    const path = new URL(url).pathname; requests.push({ path, options });
    if (path.endsWith('/fusion.json')) return new Response(JSON.stringify(manifest));
    if (path.endsWith('/joint-annotations.json')) {
      if (path.includes('/raw/') && localStatus !== 200) return new Response('', { status: localStatus });
      return new Response(savedText);
    }
    if (path.endsWith('/joint-refinement.json')) return refinement ? new Response(JSON.stringify(refinement)) : new Response('', { status: 404 });
    const binary = binaries.get(path.split('/').at(-1));
    assert.ok(binary, `Unexpected fetch ${path}`);
    return new Response(binary);
  } });
  return { manifest, binaries, requests };
}

function assertPreserved(scan, app, inputs) {
  const root = app.root.children[0];
  assert.equal(root.children.length, 32, 'both contributions remain split into all sixteen body regions');
  for (const capture of scan.manifest.captures) {
    const raw = new Uint32Array(inputs.binaries.get(capture.url));
    const labels = new Uint8Array(inputs.binaries.get(capture.labelsUrl));
    const children = scan.manifest.parts.map(part => root.children.find(child => child.name === `manikin:${capture.id}:${part.id}`));
    const output = children.map(child => {
      assert.equal(child.gsplat.unified, true); assert.equal(child.gsplat.workBufferUpdate, 19);
      const props = child.gsplat.resource.data.elements[0].properties;
      assert.deepEqual(props.map(prop => prop.name), fields);
      return props.map(prop => new Uint32Array(prop.storage.buffer));
    });
    const cursor = new Uint32Array(16), reconstructed = new Uint32Array(raw.length);
    for (let row = 0; row < labels.length; row++) {
      const part = labels[row], index = cursor[part]++;
      for (let field = 0; field < 14; field++) reconstructed[row * 14 + field] = output[part][field][index];
    }
    assert.ok(Buffer.from(raw.buffer).equals(Buffer.from(reconstructed.buffer)), `${capture.id}: every Gaussian field remains bit-identical`);
  }
}

try {
  let inputs = await fixture(), engine = mockEngine();
  let scan = await createAmbulanceManikin(engine);
  assertPreserved(scan, engine.app, inputs);
  assert.equal(scan.count, 96);
  assert.equal(scan.annotationSource, 'local');
  assert.equal(scan.annotationSha256, savedHash);
  assert.deepEqual(scan.annotations, fitted.annotations);
  assert.equal(scan.sampleCount, 16, 'low-alpha and oversized records excluded; duplicate capture cells merge');
  const rest = scan.getCollisionSamples();
  assert.equal(new Set(rest.map(s => s.id)).size, 16);
  assert.ok(rest.every(s => s.radius > 0 && s.radius <= .01));
  assert.ok(rest.every(s => Math.abs(s.position[1] - .04) < 1e-8));
  const motion = createMotion(scan.annotations, scan.manifest), pose = motion.evaluate().parts;
  assert.ok(pose instanceof Map); scan.setPartTransforms(pose);
  const first = rest[0], q = [0, 0, Math.SQRT1_2, Math.SQRT1_2];
  const proposed = new Map([[first.part, { position: [1, 2, 3], rotation: q }]]);
  const placement = { position: [7, 8, 9], rotation: [Math.SQRT1_2, 0, 0, Math.SQRT1_2], scale: 2 };
  const posed = transformPoint(proposed.get(first.part), first.position);
  const expected = transformPoint(placement, posed.map(n => n * 2));
  const checked = scan.getCollisionSamples(proposed, placement)[0];
  near(checked.position, expected); assert.equal(checked.radius, first.radius * 2);
  assert.equal(engine.app.root.children[0].scale[0], 1, 'proposal check does not mutate placement');
  scan.setPartTransforms(proposed); scan.setPlacement(placement);
  near(scan.getCollisionSamples()[0].position, expected);
  const child = engine.app.root.children[0].children[0], before = child.position.slice();
  assert.throws(() => scan.setPartTransforms(new Map([[first.part, { position: [0, 0, 0], rotation: q }], ['bad-part', pose.values().next().value]])), /Unknown/);
  assert.deepEqual(child.position, before, 'an invalid late transform cannot partially mutate the scene');
  assert.throws(() => scan.setPlacement({ ...placement, scale: -1 }), /positive/);
  assert.throws(() => scan.getCollisionSamples(new Map([[first.part, { position: [0, 0, 0], rotation: [0, 0, 0, 0] }]])), /nonzero/);
  scan.setPartTransforms(new Map()); near(child.position, [0, 0, 0]);
  scan.setVisible(false); assert.equal(engine.app.root.children[0].enabled, false);
  assert.equal(scan.getCollisionSamples().length, 16, 'hidden render state does not invalidate proposed contact checks');
  scan.dispose(); scan.dispose();
  assert.equal(engine.app.root.children.length, 0); assert.ok(engine.resources.every(r => r.destroyCount === 1));
  assert.equal(engine.app.root.destroyed, undefined, 'existing ambulance app survives disposal');
  assert.throws(() => scan.getCollisionSamples(), /disposed/);

  inputs = await fixture({ localStatus: 404, refinement: { ...fitted, inputAnnotationSha256: '0'.repeat(64) } });
  engine = mockEngine(); scan = await createAmbulanceManikin(engine);
  assert.equal(scan.annotationSource, 'published'); assert.equal(scan.refinement, null); assert.deepEqual(scan.annotations, saved);
  assert.ok(inputs.requests.some(r => r.path.includes('/mannequin-articulation/joint-annotations.json'))); scan.dispose();
  await fixture({ localStatus: 403 }); engine = mockEngine();
  await assert.rejects(createAmbulanceManikin(engine), /local manikin joint map/);
  assert.equal(engine.app.root.children.length, 0);
  await fixture({ badLabels: true }); engine = mockEngine();
  await assert.rejects(createAmbulanceManikin(engine), /invalid region label/);
  assert.equal(engine.app.root.children.length, 0); assert.equal(engine.resources.length, 16);
  assert.ok(engine.resources.every(r => r.destroyCount === 1), 'partial load resources released');

  if (process.argv.includes('--full') || process.argv.includes('--contacts')) {
    inputs = await fixture({ full: true }); engine = mockEngine();
    scan = await createAmbulanceManikin(engine); assertPreserved(scan, engine.app, inputs);
    assert.equal(scan.count, 2698682); assert.equal(scan.sampleCount, 11513);
    assert.equal(scan.manifest.report.parameters.revision, 'feature-guided-v11-finger-side-coverage');
    assert.ok(Object.values(scan.collisionSampleInfo.byPart).every(n => n > 0));
    console.log(JSON.stringify({ fullCapturedGaussians: scan.count, collisionSamples: scan.sampleCount,
      sampleCountsByPart: scan.collisionSampleInfo.byPart, fullFieldsBitIdentical: true }));
    if (process.argv.includes('--contacts')) {
      const configBytes = await read('viewers/ambulance/bed-placement.json'), config = JSON.parse(configBytes);
      const posedMotion = createMotion(scan.annotations, scan.manifest);
      const preset = buildPreset('lying', posedMotion, scan.manifest);
      for (const [id, angles] of Object.entries({ ...preset.targets, ...config.jointOverrides })) posedMotion.setTarget(id, angles, { immediate: true });
      const samples = scan.getCollisionSamples(posedMotion.evaluate().parts, config.placement);
      const checker = createContactChecker(config.collision), start = performance.now(), report = checker.evaluate(samples);
      const elapsedMs = performance.now() - start;
      const sampleById = new Map(samples.map(sample => [sample.id, sample]));
      const review = { configSha256: createHash('sha256').update(configBytes).digest('hex'),
        sceneSha256: config.scene.sha256, annotationSha256: scan.annotationSha256,
        refinementUsed: Boolean(scan.refinement), sampleInfo: scan.collisionSampleInfo,
        placement: config.placement, jointOverrides: config.jointOverrides, elapsedMs,
        status: report.status, blocked: report.blocked, maxPenetration: report.maxPenetration,
        contactCount: report.contacts.length, violationCount: report.violations.length, byPart: report.byPart,
        violations: report.violations.map(hit => ({ ...hit, sample: sampleById.get(hit.sampleId) })) };
      await mkdir(new URL('raw/ambulance-manikin/', base), { recursive: true });
      await writeFile(new URL('raw/ambulance-manikin/contact-review.json', base), JSON.stringify(review, null, 2) + '\n');
      console.log(JSON.stringify({ contactReview: 'raw/ambulance-manikin/contact-review.json',
        configSha256: review.configSha256, status: report.status, violations: report.violations.length,
        maxPenetration: report.maxPenetration, elapsedMs, byPart: report.byPart }));
    }
    scan.dispose();
  }
  console.log('Ambulance manikin loader checks passed.');
} finally {
  for (const [key, descriptor] of Object.entries(savedGlobals)) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
}
