import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createGravityTest } from '../viewers/ambulance/gravity-test.mjs';
import { createContactChecker } from '../viewers/ambulance/contact-check.mjs';

const floor = { id: 'floor', normal: [0, 1, 0], offset: 0, bounds: [[-2, 0, -2], [2, 0, 2]] };
const near = (a, b, epsilon = 1e-8) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
function fixture({ start = [0, .05, 0], radius = .05, collision = { surfaces: [floor] }, ...options } = {}) {
  let p = [...start]; const accepted = [], stops = [], updates = [], probes = [];
  const checker = createContactChecker(collision);
  const evaluateAt = q => { probes.push([...q]); return checker.evaluate([{ id: 'body', part: 'body', position: q, radius }]); };
  const controller = createGravityTest({ getPosition: () => p, setPosition: q => { p = q; accepted.push([...q]); }, evaluateAt,
    onStop: s => stops.push(s), onUpdate: s => updates.push(s), ...options });
  return { controller, accepted, stops, updates, probes, evaluateAt, getPosition: () => p, move: q => { p = q; } };
}
function run(controller, dt = 1 / 60, count = 1000) {
  for (let i = 0; i < count && controller.getState().status === 'running'; i++) controller.update(dt);
  return controller.getState();
}

test('lift and accelerated descent land at actual contact, not the near-contact band', () => {
  const f = fixture(), c = f.controller;
  const started = c.start({ lift: .25 }); near(started.position[1], .3);
  c.update(1 / 120); let s = c.getState(); near(s.velocity, -9.81 / 120); near(s.position[1], .3 - .5 * 9.81 / 120 ** 2);
  s = run(c); assert.equal(s.status, 'landed'); assert.equal(s.contact.proxy, 'floor');
  assert.ok(s.position[1] <= .050001 && s.position[1] >= .049999); assert.equal(s.lastReport.blocked, false);
  assert.equal(f.stops.length, 1); c.stop(); c.update(2); assert.equal(f.stops.length, 1);
  for (const p of f.accepted) assert.equal(f.evaluateAt(p).blocked, false);
});

test('held-pose sweeps catch a thin mattress even when the coarse step ends below it', () => {
  const f = fixture({ start: [0, .5, 0], radius: .001, fixedStep: .1, maxFrameDelta: .5, gravity: 100,
    collision: { boxes: [{ id: 'thin', center: [0, 0, 0], halfExtents: [1, .001, 1], kind: 'mattress' }] } });
  f.controller.start({ lift: 0 }); const s = f.controller.update(.1);
  assert.equal(s.status, 'landed'); near(s.position[1], .002, 1e-6); assert.equal(s.lastReport.blocked, false);
});

test('overlapping start and obstructed lift never move the initial body', () => {
  const overlap = fixture({ start: [0, .02, 0] });
  assert.equal(overlap.controller.start().status, 'rejected'); assert.equal(overlap.accepted.length, 0);
  const lift = fixture({ collision: { surfaces: [floor, { id: 'roof', normal: [0, -1, 0], offset: -.2, bounds: [[-1, .2, -1], [1, .2, 1]] }] } });
  const s = lift.controller.start({ lift: .25 });
  assert.equal(s.status, 'obstructed'); assert.equal(s.reason, 'lift-obstructed'); assert.equal(s.contact.proxy, 'roof');
  near(lift.getPosition()[1], .05); assert.equal(lift.accepted.length, 0); assert.equal(lift.stops.length, 1);
});

test('pause/resume discards fractional accumulated time and hidden-tab delay', () => {
  const f = fixture({ start: [0, 2, 0] }), c = f.controller;
  c.start({ lift: 0 }); c.update(1 / 240); c.pause(); const paused = c.getState();
  c.update(100); assert.deepEqual(c.getState(), paused); c.resume(); c.update(1 / 240); near(c.getState().elapsed, 0);
  c.update(1 / 240); near(c.getState().elapsed, 1 / 120);
  c.update(100); assert.ok(c.getState().elapsed < .12); assert.ok(c.getState().discardedTime > 99);
});

test('large dt stays bounded, terminal callback fires once, and notifications are per public update', () => {
  const f = fixture({ start: [0, 2, 0], maxFallDistance: .05 }); const c = f.controller;
  c.start({ lift: 0 }); const before = f.updates.length; const s = c.update(100);
  assert.equal(f.updates.length, before + 1); assert.equal(s.status, 'running');
  const end = run(c); assert.equal(end.status, 'out-of-bounds'); assert.equal(end.reason, 'maximum-fall-distance');
  near(end.fallDistance, .05); assert.equal(f.stops.length, 1); c.stop(); assert.equal(f.stops.length, 1);
});

test('bounds and duration cap terminate without placing below a limit', () => {
  const low = fixture({ start: [3, .2, 0], minY: .1 });
  low.controller.start({ lift: 0 }); const a = run(low.controller); assert.equal(a.status, 'out-of-bounds'); near(a.position[1], .1);
  const timed = fixture({ start: [0, 2, 0], maxDuration: .02 }); timed.controller.start({ lift: 0 });
  const b = run(timed.controller); assert.equal(b.status, 'timeout'); near(b.elapsed, .02);
});

test('side contact is obstruction, upward normals are landing, and touching start can finish immediately', () => {
  const resting = fixture(); assert.equal(resting.controller.start({ lift: 0 }).status, 'landed');
  const side = fixture({ start: [.045, .04, 0], collision: { surfaces: [{ id: 'side', normal: [1, 0, 0], offset: 0, bounds: [[0, -.2, -1], [0, .2, 1]], allowedPenetration: .01 }] } });
  assert.equal(side.controller.start({ lift: 0 }).status, 'obstructed');
});

test('invalid or missing contact data fails closed without committing unchecked movement', () => {
  let p = [0, 1, 0]; const changes = [];
  const c = createGravityTest({ getPosition: () => p, setPosition: q => { changes.push(q); p = q; }, evaluateAt: () => ({ status: 'unavailable', blocked: true, contacts: [], sampleCount: 0 }) });
  assert.equal(c.start().status, 'rejected'); assert.equal(changes.length, 0);
  const f = fixture(); assert.throws(() => f.controller.start({ lift: -1 }), /lift/);
  assert.throws(() => f.controller.update(Infinity), /dt/); assert.throws(() => f.controller.update(-1), /dt/);
});

test('external placement changes cancel instead of overwriting user movement; state snapshots are isolated', () => {
  const f = fixture({ start: [0, 1, 0] }), c = f.controller; c.start({ lift: 0 });
  const s = c.getState(); s.position[1] = 100; near(c.getState().position[1], 1);
  f.move([1, 2, 3]); const end = c.update(.01); assert.equal(end.reason, 'external-position-change'); assert.deepEqual(f.getPosition(), [1, 2, 3]);
});

test('evaluation failures during a fall stop on previously verified geometry', () => {
  let calls = 0, p = [0, 1, 0]; const checker = createContactChecker({ surfaces: [floor] });
  const c = createGravityTest({ getPosition: () => p, setPosition: q => { p = q; }, evaluateAt: q => {
    if (++calls > 3) throw new Error('offline');
    return checker.evaluate([{ position: q, radius: .05 }]);
  } });
  c.start({ lift: 0 }); const s = c.update(.1); assert.equal(s.status, 'error'); assert.match(s.error, /offline/);
  assert.equal(checker.evaluate([{ position: p, radius: .05 }]).blocked, false);
});
