import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { createMotion, transformPoint } from '../viewers/mannequin-articulation/motion-core.mjs';
import { createArmGravity } from '../viewers/ambulance/arm-gravity.mjs';

const manifest = JSON.parse(await readFile(new URL('../viewers/mannequin-fusion/fusion.json', import.meta.url)));
const annotations = JSON.parse(await readFile(new URL('../viewers/mannequin-articulation/joint-refinement.json', import.meta.url))).annotations;
const DEG = Math.PI / 180, identity = { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: 1 };
const clear = () => ({ status: 'clear', blocked: false, contacts: [], violations: [], maxPenetration: 0 });
const near = (a, b, epsilon = 1e-8) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
function restCloud() {
  const samples = [];
  for (const side of ['left', 'right']) {
    const shoulder = annotations.joints.find(j => j.id === `${side}_shoulder`).pivot;
    const elbow = annotations.joints.find(j => j.id === `${side}_arm_swivel`).pivot, sign = side === 'left' ? 1 : -1;
    const centers = { upper_arm: shoulder.map((v, i) => (v + elbow[i]) / 2),
      forearm: elbow.map((v, i) => v + [sign * .08, .005, -.20][i]),
      hand: elbow.map((v, i) => v + [sign * .15, .01, -.35][i]) };
    for (const [part, center] of Object.entries(centers)) for (const x of [-1, 1]) for (const y of [-1, 1]) for (const z of [-1, 1]) {
      samples.push({ id: samples.length, part: `${side}_${part}`, position: center.map((v, i) => v + [x, y, z][i] * .015), radius: .005 });
    }
  }
  return samples;
}
function fixture(options = {}) {
  const motion = createMotion(annotations, manifest), restSamples = restCloud(), updates = [], settled = [];
  let placement = structuredClone(identity);
  const c = createArmGravity({ motion, restSamples, getPlacement: () => placement, evaluate: clear,
    onUpdate: s => updates.push(s), onSettle: s => settled.push(s), ...options });
  return { c, motion, restSamples, updates, settled, setPlacement: p => { placement = p; } };
}
function energy(f, side = 'left', placement = identity) {
  const props = f.c.getMassProperties(), parts = f.motion.evaluate().parts;
  return Object.values(props).filter(p => p.part.startsWith(side)).reduce((sum, p) => {
    const native = transformPoint(parts.get(p.part), p.center).map(v => v * placement.scale);
    const world = transformPoint(placement, native);
    return sum + p.mass * 9.81 * world[1];
  }, 0);
}

test('starts asleep; holding either arm joint freezes shoulder and swivel while the other arm can fall', () => {
  const f = fixture(); assert.equal(f.c.getState().active, false);
  f.c.hold('left_arm_swivel'); f.c.wake('right');
  const before = f.motion.getState('left_shoulder').angles;
  for (let i = 0; i < 20; i++) f.c.update(1 / 60);
  assert.deepEqual(f.motion.getState('left_shoulder').angles, before);
  assert.deepEqual(f.motion.getState('left_arm_swivel').angles, [0, 0, 0]);
  assert.notDeepEqual(f.motion.getState('right_shoulder').angles, [0, 0, 0]);
  assert.equal(f.c.getState().sides.left.status, 'held');
});

test('release lowers gravitational potential without returning to a preset or moving non-arm joints', () => {
  const f = fixture(); f.motion.setTarget('left_shoulder', [40, 0, 0]);
  const initialEnergy = energy(f), untouched = f.motion.exportPose().joints.filter(j => !j.id.startsWith('left_shoulder') && !j.id.startsWith('left_arm_swivel'));
  f.c.hold('left_shoulder'); f.c.release('left_shoulder');
  for (let i = 0; i < 60; i++) f.c.update(1 / 120);
  assert.ok(energy(f) < initialEnergy - .025);
  assert.ok(f.motion.getState('left_shoulder').angles[0] < 0, 'The old +40 preset must not pull the arm upward');
  for (const j of untouched) assert.deepEqual(f.motion.getState(j.id).angles, j.angles);
});

test('gravity torque matches the finite difference of potential for all instantaneous shoulder Euler axes and swivel', () => {
  const f = fixture(); f.motion.setTarget('left_shoulder', [20, 35, -15], { immediate: true });
  f.motion.setTarget('left_arm_swivel', [23, 0, 0], { immediate: true });
  const original = Object.fromEntries(['left_shoulder', 'left_arm_swivel'].map(id => [id, f.motion.getState(id).angles]));
  const derivatives = {};
  for (const id of Object.keys(original)) {
    derivatives[id] = [];
    for (let axis = 0; axis < (id.endsWith('shoulder') ? 3 : 1); axis++) {
      const plus = [...original[id]], minus = [...original[id]], epsilon = 1e-5;
      plus[axis] += epsilon / DEG; minus[axis] -= epsilon / DEG;
      f.motion.setTarget(id, plus, { immediate: true }); const up = energy(f);
      f.motion.setTarget(id, minus, { immediate: true }); const down = energy(f);
      f.motion.setTarget(id, original[id], { immediate: true }); derivatives[id].push(-(up - down) / (2 * epsilon));
    }
  }
  f.c.wake('left'); f.c.update(1 / 120);
  const dynamics = f.c.getState().sides.left.dynamics;
  for (const [id, values] of Object.entries(derivatives)) values.forEach((value, i) => near(dynamics[id][i].torque, value, 1e-8));
});

test('per-part relative mass and inertia are independent of repeated sample density', () => {
  const a = fixture(), samples = restCloud(), dense = [...samples, ...samples.filter(s => s.part === 'left_upper_arm'), ...samples.filter(s => s.part === 'left_upper_arm')];
  const b = fixture({ restSamples: dense }); a.c.wake('left'); b.c.wake('left'); a.c.update(1 / 120); b.c.update(1 / 120);
  near(a.c.getMassProperties().left_upper_arm.mass, b.c.getMassProperties().left_upper_arm.mass);
  const aa = a.c.getState().sides.left.dynamics, bb = b.c.getState().sides.left.dynamics;
  for (const id of Object.keys(aa)) aa[id].forEach((v, i) => { near(v.torque, bb[id][i].torque); near(v.inertia, bb[id][i].inertia); });
});

test('world placement rotates gravity direction and uniform scale gives correct torque/inertia scaling', () => {
  const a = fixture(), b = fixture(); b.setPlacement({ ...identity, scale: .5 });
  a.c.wake('left'); b.c.wake('left'); a.c.update(1 / 120); b.c.update(1 / 120);
  const da = a.c.getState().sides.left.dynamics.left_shoulder[0], db = b.c.getState().sides.left.dynamics.left_shoulder[0];
  near(db.torque, da.torque * .5); near(db.inertia, da.inertia * .25);
  const c = fixture(); c.setPlacement({ ...identity, rotation: [0, 0, 1, 0] }); c.c.wake('left'); c.c.update(1 / 120);
  near(c.c.getState().sides.left.dynamics.left_shoulder[0].torque, -da.torque);
});

test('swept collision refinement cannot jump a narrow angular obstacle and only commits a safe pose', () => {
  let f; const probes = [];
  f = fixture({ maxAngularSpeed: 20, maxAngularAcceleration: 500, fixedStep: .05, maxAngularStep: .25,
    evaluate: (_pose, context) => {
      assert.equal(context.side, 'left'); assert.deepEqual(context.movingParts, ['left_upper_arm', 'left_forearm', 'left_hand']);
      const angle = f.motion.getState('left_shoulder').angles[0]; probes.push(angle);
      const blocked = angle < -2 && angle > -2.6;
      return { ...clear(), blocked, status: blocked ? 'penetrating' : 'clear' };
    } });
  f.c.release('left_shoulder'); for (let i = 0; i < 20 && f.c.getState().active; i++) f.c.update(.05);
  assert.equal(f.c.getState().sides.left.reason, 'contact'); assert.ok(f.motion.getState('left_shoulder').angles[0] >= -2);
  assert.ok(probes.some(v => v < -2)); assert.equal(f.settled.length, 1);
  f.c.update(1); assert.equal(f.settled.length, 1);
});

test('initial collision and callback failure leave the original pose intact', () => {
  const blocked = fixture({ evaluate: () => ({ ...clear(), blocked: true }) });
  blocked.c.release('left_shoulder'); blocked.c.update(.02);
  assert.deepEqual(blocked.motion.getState('left_shoulder').angles, [0, 0, 0]); assert.equal(blocked.c.getState().sides.left.reason, 'initial-overlap');
  let n = 0; const failed = fixture({ evaluate: () => { if (++n > 1) throw new Error('unavailable'); return clear(); } });
  failed.c.release('left_shoulder'); failed.c.update(.02);
  assert.deepEqual(failed.motion.getState('left_shoulder').angles, [0, 0, 0]); assert.match(failed.c.getState().sides.left.reason, /evaluation-error/);
});

test('pause, capped time, stop and reset never catch up or change pose', () => {
  const f = fixture(); f.c.release('left_shoulder'); f.c.update(.001); f.c.pause();
  const angles = f.motion.getState('left_shoulder').angles; f.c.update(100); assert.deepEqual(f.motion.getState('left_shoulder').angles, angles);
  f.c.resume(); f.c.update(.001); near(f.c.getState().elapsed, 0);
  f.c.update(100); assert.ok(f.c.getState().elapsed <= .05 + 1e-9); assert.ok(f.c.getState().discardedTime > 99);
  f.c.stop(); const end = f.motion.getState('left_shoulder').angles; f.c.update(.1); assert.deepEqual(f.motion.getState('left_shoulder').angles, end);
  f.c.reset(); assert.equal(f.c.getState().active, false); assert.deepEqual(f.motion.getState('left_shoulder').angles, end);
});

test('joint limits bound every accepted angle and held nested handles require matching release', () => {
  const f = fixture(); f.c.hold('left_shoulder'); f.c.hold('left_arm_swivel'); f.c.release('left_shoulder');
  assert.equal(f.c.getState().sides.left.status, 'held'); f.c.release('left_arm_swivel');
  for (let i = 0; i < 600; i++) f.c.update(1 / 60);
  for (const id of ['left_shoulder', 'left_arm_swivel']) {
    const j = f.motion.joints.find(j => j.id === id);
    f.motion.getState(id).angles.forEach((v, i) => assert.ok(v >= j.limits[i][0] && v <= j.limits[i][1]));
  }
  assert.ok(f.updates.some(s => s.changed));
});

test('missing captured parts and invalid input fail explicitly; non-arm hold is rejected', () => {
  assert.throws(() => fixture({ restSamples: [] }), /Missing captured/);
  const f = fixture(); assert.throws(() => f.c.hold('left_hip'), /Not an arm/); assert.throws(() => f.c.update(-1), /dt/);
  assert.throws(() => f.c.update(NaN), /dt/);
});
