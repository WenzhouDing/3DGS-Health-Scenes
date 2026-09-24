#!/usr/bin/env node
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createContactChecker, planePolygon } from '../viewers/ambulance/contact-check.mjs';

const point = (position, radius = 0, part = 'forearm') => [{ id: 7, part, position, radius }];
const floor = { id: 'floor', normal: [0, 1, 0], offset: 0, bounds: [[-2, 0, -2], [2, 0, 2]] };
const near = (a, b, epsilon = 1e-9) => assert.ok(Math.abs(a - b) < epsilon, `${a} != ${b}`);

test('bounded floor detects contact and penetration without blocking distant points', () => {
  const c = createContactChecker({ surfaces: [floor] });
  assert.equal(c.evaluate(point([0, .2, 0], .05)).status, 'clear');
  assert.equal(c.evaluate(point([0, .05, 0], .05)).status, 'contact');
  const hit = c.evaluate(point([0, .04, 0], .05));
  assert.equal(hit.blocked, true); near(hit.maxPenetration, .01);
  assert.equal(hit.violations[0].proxy, 'floor');
  assert.equal(hit.byPart.forearm.violations, 1);
  assert.equal(c.evaluate(point([3, -.5, 0], .05)).blocked, false);
  // Footprint outside, but a finite sphere still touches the panel rim.
  assert.equal(c.evaluate(point([2.02, .02, 0], .05)).blocked, true);
  assert.equal(c.evaluate(point([0, -.1, 0])).blocked, true);
});

test('sloped surface normal/offset normalization and signed free side agree', () => {
  const c = createContactChecker({ surfaces: [{ id: 'ramp', normal: [-1, 2, 0], offset: 2,
    bounds: [[-1, .5, -1], [1, 1.5, 1]] }] });
  assert.equal(c.evaluate(point([0, 1.02, 0])).blocked, false);
  assert.equal(c.evaluate(point([0, .98, 0])).blocked, true);
  near(c.evaluate(point([0, .98, 0])).maxPenetration, .04 / Math.sqrt(5));
  const polygon = planePolygon([0, 1, 0], 0, floor.bounds);
  assert.equal(polygon.length, 4);
  // Includes the box's final vertex when it is the b endpoint of every edge.
  assert.equal(planePolygon([1, 1, -1], 1, [[-1, -1, -1], [1, 1, 1]]).length, 3);
});

test('mattress allows shallow top compression, never side or bottom overlap', () => {
  const c = createContactChecker({ boxes: [{ id: 'bed', label: 'Mattress', center: [0, 0, 0],
    halfExtents: [1, .1, .4], kind: 'mattress', allowedPenetration: .012 }],
    tolerances: { penetration: .002, contact: .006 } });
  const resting = c.evaluate(point([0, .14, 0], .05, 'back'));
  assert.equal(resting.blocked, false); assert.equal(resting.status, 'contact');
  assert.equal(resting.contacts[0].support, true); near(resting.maxPenetration, .01);
  assert.equal(c.evaluate(point([0, .13, 0], .05)).blocked, true);
  const side = c.evaluate(point([1.04, 0, 0], .05));
  assert.equal(side.blocked, true); assert.equal(side.contacts[0].support, false);
  assert.equal(c.evaluate(point([0, -.14, 0], .05)).blocked, true);
  const deeplyInside = c.evaluate(point([0, .05, 0]));
  assert.equal(deeplyInside.blocked, true); near(deeplyInside.maxPenetration, .05);
});

test('box quaternion applies to nearest face, support normal and world point', () => {
  const c = createContactChecker({ boxes: [{ id: 'tilted-bed', center: [3, 0, 0],
    halfExtents: [1, .1, .3], rotation: [0, 0, Math.SQRT1_2, Math.SQRT1_2],
    kind: 'mattress', allowedPenetration: .012 }] });
  const hit = c.evaluate(point([2.86, 0, 0], .05));
  assert.equal(hit.blocked, false); assert.equal(hit.contacts[0].support, true);
  near(hit.contacts[0].normal[0], -1); near(hit.contacts[0].point[0], 2.9);
  assert.equal(c.evaluate(point([3, 1.04, 0], .05)).blocked, true);
});

test('solid sphere-box corner uses Euclidean distance, not expanded-AABB false positives', () => {
  const c = createContactChecker({ boxes: [{ id: 'cabinet', center: [0, 0, 0], halfExtents: [1, 1, 1] }] });
  assert.equal(c.evaluate(point([1.09, 1.09, 0], .1)).status, 'clear');
  assert.equal(c.evaluate(point([1.06, 1.06, 0], .1)).blocked, true);
  const inside = c.evaluate(point([0, 0, 0]));
  assert.equal(inside.blocked, true); near(inside.maxPenetration, 1);
});

test('multiple parts and coincident constraints remain separately attributable', () => {
  const c = createContactChecker({ surfaces: [floor, { id: 'wall', normal: [1, 0, 0], offset: 0,
    bounds: [[0, -1, -1], [0, 2, 1]] }] });
  const hit = c.evaluate([...point([-.1, -.1, 0], 0, 'hand'), ...point([1, 1, 0], 0, 'head')]);
  assert.equal(hit.sampleCount, 2); assert.equal(hit.violations.length, 2);
  assert.equal(hit.byPart.hand.violations, 2); assert.equal(hit.byPart.head.violations, 0);
});

test('swept articulated samples stop at the first obstacle even when destination is clear', () => {
  const c = createContactChecker({ boxes: [{ id: 'thin-wall', center: [0, 0, 0], halfExtents: [.04, 1, 1] }],
    tolerances: { penetration: 0 } });
  const trajectory = t => point([-1 + 2 * t, 0, 0], .01);
  assert.equal(c.evaluate(trajectory(0)).blocked, false);
  assert.equal(c.evaluate(trajectory(1)).blocked, false);
  const result = c.evaluatePath(trajectory, { steps: 100, refinementSteps: 12 });
  assert.equal(result.blocked, true); assert.equal(result.report.blocked, false);
  assert.ok(result.acceptedFraction > .4749 && result.acceptedFraction <= .475);
  assert.equal(result.blockingReport.violations[0].proxy, 'thin-wall');
});

test('swept path handles initial penetration and safe complete motion', () => {
  const c = createContactChecker({ surfaces: [floor] });
  assert.equal(c.evaluatePath(t => point([0, -1 + 2 * t, 0])).acceptedFraction, 0);
  const safe = c.evaluatePath(t => point([0, .1 + t, 0]));
  assert.equal(safe.acceptedFraction, 1); assert.equal(safe.blocked, false);
});

test('input validation fails closed for missing samples and rejects invalid geometry', () => {
  const c = createContactChecker({ collision: { surfaces: [floor] } });
  assert.equal(c.evaluate([]).status, 'unavailable'); assert.equal(c.evaluate([]).blocked, true);
  assert.throws(() => c.evaluate(point([NaN, 0, 0])), /finite/);
  assert.throws(() => c.evaluate(point([0, 0, 0], -1)), /nonnegative/);
  assert.throws(() => createContactChecker({ surfaces: [{ ...floor, normal: [0, 0, 0] }] }), /zero/);
  assert.throws(() => createContactChecker({ surfaces: [floor, floor] }), /unique/);
  assert.throws(() => createContactChecker({ boxes: [{ id: 'bad', center: [0, 0, 0], halfExtents: [0, 1, 1] }] }), /positive/);
  assert.throws(() => createContactChecker({ surfaces: [{ ...floor, offset: 5 }] }), /intersect/);
  assert.throws(() => createContactChecker({}), /At least one/);
});

test('checker does not mutate input config or sample vectors', () => {
  const config = { surfaces: [structuredClone(floor)] }, samples = point([0, .001, 0]);
  const before = JSON.stringify({ config, samples });
  createContactChecker(config).evaluate(samples);
  assert.equal(JSON.stringify({ config, samples }), before);
});

test('explicit plane tolerance is honored and early rejection remains attributable', () => {
  const c = createContactChecker({ surfaces: [{ ...floor, allowedPenetration: .006 }] });
  assert.equal(c.evaluate(point([0, -.005, 0])).blocked, false);
  const samples = [...point([0, .1, 0], 0, 'head'), ...point([0, -.02, 0], 0, 'foot'), ...point([0, .2, 0], 0, 'hand')];
  const short = c.evaluate(samples, { stopOnBlock: true });
  assert.equal(short.blocked, true); assert.equal(short.complete, false);
  assert.equal(short.sampleCount, 2); assert.equal(short.violations[0].part, 'foot');
  assert.equal(c.evaluate(samples).complete, true);
});
