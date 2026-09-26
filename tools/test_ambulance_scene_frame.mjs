import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { createSceneFrame, AMBULANCE_LEVEL_ROTATION, AMBULANCE_SOURCE_UP } from '../viewers/ambulance/scene-frame.mjs';
import { createContactChecker } from '../viewers/ambulance/contact-check.mjs';
import { transformPoint, createMotion } from '../viewers/mannequin-articulation/motion-core.mjs';
import { buildPreset } from '../viewers/mannequin-articulation/pose-presets.mjs';

const near = (a, b, e = 1e-10) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) <= e, `${a} != ${b}`));
const frame = createSceneFrame(AMBULANCE_LEVEL_ROTATION);
const placed = (p, v) => transformPoint(p, v.map(x => x * p.scale));

test('measured cabin up maps to world Y, without scale, translation, or source mutation', () => {
  near(frame.direction(AMBULANCE_SOURCE_UP), [0, 1, 0], 1e-14);
  for (const point of [[2, -.7, .9], [-1.4, 1.1, -.6], [0, 0, 0]]) {
    const before = [...point], world = frame.point(point);
    near(frame.inversePoint(world), point); near([Math.hypot(...world)], [Math.hypot(...point)]);
    assert.deepEqual(point, before);
  }
  near(frame.sourceUp, AMBULANCE_SOURCE_UP);
});

test('placement composes the world rotation on the left and inverse restores saved source coordinates', () => {
  const source = { position: [.3, -.196, -.3185], rotation: [0, Math.sin(.4), 0, Math.cos(.4)], scale: .75, id: 'saved' };
  const original = structuredClone(source), runtime = frame.placement(source);
  for (const p of [[1, 2, 3], [-.4, 0, -.8]]) near(placed(runtime, p), frame.point(placed(source, p)));
  const back = frame.inversePlacement(runtime); near(back.position, source.position); near(back.rotation, source.rotation);
  assert.equal(back.scale, source.scale); assert.equal(back.id, source.id); assert.deepEqual(source, original);
  // Legacy SOG format flip is applied first; leveling is world-space Q * F.
  const legacy = { position: [0, 0, 0], rotation: [0, 0, 1, 0], scale: 1 };
  const composed = frame.placement(legacy), p = [.2, .5, .9];
  near(placed(composed, p), frame.point([-.2, -.5, .9]));
  assert.ok(Math.hypot(...placed(composed, p).map((v, i) => v - placed(legacy, frame.point(p))[i])) > .005);
});

test('cameras and both settings versions rotate once and leave saved settings untouched', () => {
  const camera = { position: [-1.1, .25, .45], target: [.3, -.05, .1], fov: 70 };
  const settings = { version: 2, cameras: [{ initial: camera }], annotations: [{ position: [1, 0, 0], camera }],
    animTracks: [{ keyframes: { values: { position: [1, 2, 3, 4, 5, 6], target: [0, 1, 2, 3, 4, 5] } } }] };
  const original = structuredClone(settings), result = frame.settings(settings);
  near(result.cameras[0].initial.position, frame.point(camera.position)); near(result.cameras[0].initial.target, frame.point(camera.target));
  assert.equal(result.cameras[0].initial.fov, 70); near(result.annotations[0].position, frame.point([1, 0, 0]));
  near(result.animTracks[0].keyframes.values.position.slice(3), frame.point([4, 5, 6])); assert.deepEqual(settings, original);
  near(frame.settings({ camera }).camera.position, frame.point(camera.position));
  assert.deepEqual(createSceneFrame().settings(settings), settings);
});

test('wrapped contact reports keep physical distances and rotate points/normals into the runtime frame', () => {
  const source = createContactChecker({ surfaces: [{ id: 'floor', normal: [0, 1, 0], offset: -.7, bounds: [[-2, -.7, -2], [2, -.7, 2]] }] });
  const wrapped = frame.wrapContactChecker(source), samples = [{ id: 3, part: 'heel', position: [.2, -.702, .4], radius: .003 }];
  const original = structuredClone(samples), a = source.evaluate(samples), b = wrapped.evaluate(samples.map(s => ({ ...s, position: frame.point(s.position) })));
  assert.equal(b.blocked, a.blocked); near([b.maxPenetration], [a.maxPenetration]); near(b.contacts[0].point, frame.point(a.contacts[0].point));
  near(b.contacts[0].normal, frame.direction(a.contacts[0].normal)); assert.deepEqual(samples, original);
  const path = t => [{ ...samples[0], position: [.2, -.60 - .2 * t, .4] }];
  const aa = source.evaluatePath(path), bb = wrapped.evaluatePath(t => path(t).map(s => ({ ...s, position: frame.point(s.position) })));
  near([bb.acceptedFraction], [aa.acceptedFraction]); assert.equal(bb.report.blocked, false);
  near(bb.blockingReport.violations[0].point, frame.point(aa.blockingReport.violations[0].point));
});

test('identity/default content stays unchanged and invalid frame inputs fail explicitly', () => {
  const f = createSceneFrame(); assert.equal(f.identity, true); near(f.point([1, 2, 3]), [1, 2, 3]);
  assert.equal(createSceneFrame(null).identity, true); near(createSceneFrame(null).point([1, 2, 3]), [1, 2, 3]);
  assert.throws(() => createSceneFrame([0, 0, 0, 0]), /zero/);
  assert.throws(() => createSceneFrame([NaN, 0, 0, 1]), /finite/);
  assert.throws(() => f.point([1, 2]), /finite/);
  assert.throws(() => f.placement({ position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: 0 }), /scale/);
});

test('actual 11513-sample fitted manikin contacts are invariant under one shared rigid frame', async t => {
  const { createAmbulanceManikin } = await import('../viewers/ambulance/manikin-scan.mjs');
  const priorFetch = globalThis.fetch, priorLocation = globalThis.location;
  globalThis.location = { hostname: '127.0.0.1' };
  globalThis.fetch = async url => {
    try {
      const b = await readFile(fileURLToPath(url)); return { ok: true, status: 200, text: async () => b.toString(),
        json: async () => JSON.parse(b), arrayBuffer: async () => b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) };
    } catch (e) { return { ok: false, status: e.code === 'ENOENT' ? 404 : 500 }; }
  };
  t.after(() => { globalThis.fetch = priorFetch; globalThis.location = priorLocation; });
  class Entity { addChild() {} addComponent() {} setLocalPosition() {} setLocalRotation() {} setLocalScale() {} destroy() {} }
  const pc = { Entity, GSplatData: class {}, GSplatResource: class { destroy() {} }, WORKBUFFER_UPDATE_AUTO: 0 };
  const scan = await createAmbulanceManikin({ app: { root: new Entity(), graphicsDevice: {} }, pc }); t.after(() => scan.dispose());
  const config = JSON.parse(await readFile(new URL('../viewers/ambulance/bed-placement.json', import.meta.url)));
  const motion = createMotion(scan.annotations, scan.manifest);
  for (const [id, angles] of Object.entries({ ...buildPreset('lying', motion, scan.manifest).targets, ...config.jointOverrides })) motion.setTarget(id, angles, { immediate: true });
  const parts = motion.evaluate().parts, source = createContactChecker(config), runtime = frame.wrapContactChecker(source);
  const a = source.evaluate(scan.getCollisionSamples(parts, config.placement));
  const b = runtime.evaluate(scan.getCollisionSamples(parts, frame.placement(config.placement)));
  assert.equal(a.sampleCount, 11513); assert.equal(a.blocked, false); assert.equal(b.blocked, false);
  assert.equal(b.contacts.length, a.contacts.length); near([b.maxPenetration], [a.maxPenetration]);
  for (let i = 0; i < a.contacts.length; i++) {
    assert.equal(b.contacts[i].sampleId, a.contacts[i].sampleId); near(b.contacts[i].point, frame.point(a.contacts[i].point));
    near(b.contacts[i].normal, frame.direction(a.contacts[i].normal));
  }
});
