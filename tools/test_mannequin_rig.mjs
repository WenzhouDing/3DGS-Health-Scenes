#!/usr/bin/env node
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import {
  validateRig, classifyPoint, classifyPoints, clampJointAngles,
  getJointQuaternion, evaluateRig, stepSpring,
} from '../viewers/mannequin-rig/rig-core.mjs';

const tolerance = 1e-10;
function near(actual, expected, epsilon = tolerance) {
  if (Array.isArray(expected)) {
    assert.equal(actual.length, expected.length);
    actual.forEach((value, i) => near(value, expected[i], epsilon));
  } else {
    assert.ok(Math.abs(actual - expected) <= epsilon, `${actual} differs from ${expected}`);
  }
}
function transformPoint(transform, point) {
  const m = transform.matrix;
  return [0, 1, 2].map(row => m[row] * point[0] + m[4 + row] * point[1] + m[8 + row] * point[2] + m[12 + row]);
}
function joint(id, parent = null, pivot = [0, 0, 0], type = 'ball') {
  return { id, label: id, parent, pivot, type, axis: [0, 0, 1], frame: [0, 0, 0],
    angles: [0, 0, 0], target: [0, 0, 0], limits: [[-180, 180], [-180, 180], [-180, 180]],
    stiffness: 30, damping: 7, inertia: 1 };
}
function segment(id, jointId = null, shapes = []) {
  return { id, label: id, joint: jointId, color: [0.5, 0.5, 0.5], shapes };
}
function rig(joints = [joint('shoulder'), joint('elbow', 'shoulder', [1, 0, 0], 'hinge')]) {
  return { version: 1, joints, segments: [segment('environment'), segment('arm', joints[0]?.id ?? null)] };
}
function sphere(center, radius) { return { type: 'capsule', a: center, b: center, radius }; }

test('rest transforms are identity even for off-origin nested pivots and frames', () => {
  const model = rig([joint('upper', null, [3, 4, 5]), joint('lower', 'upper', [8, 9, 10])]);
  model.joints[0].frame = [23, 41, -37];
  model.joints[1].frame = [-15, 68, 32];
  for (const [id, transform] of evaluateRig(model)) {
    near(transform.rotation, [0, 0, 0, 1]);
    near(transform.position, [0, 0, 0]);
    near(transform.matrix, [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]);
    near(transformPoint(transform, [6, -2, 7]), [6, -2, 7]);
    near(transform.pivot, model.joints.find(j => j.id === id).pivot);
  }
});

test('shoulder motion carries elbow, whose local motion preserves its posed pivot', () => {
  const model = rig();
  const posed = evaluateRig(model, { shoulder: [0, 0, 90], elbow: [90, 0, 0] });
  near(transformPoint(posed.get('shoulder'), [1, 0, 0]), [0, 1, 0]);
  near(posed.get('elbow').pivot, [0, 1, 0]);
  near(transformPoint(posed.get('elbow'), [1, 0, 0]), [0, 1, 0]);
  near(transformPoint(posed.get('elbow'), [2, 0, 0]), [-1, 1, 0]);
  model.joints.reverse();
  near(transformPoint(evaluateRig(model, new Map([['shoulder', [0, 0, 90]], ['elbow', [90, 0, 0]]])).get('elbow'), [2, 0, 0]), [-1, 1, 0]);
});

test('arbitrary hinge axis and joint frame orient rotations', () => {
  const hinge = joint('hinge', null, [2, 3, 4], 'hinge');
  hinge.axis = [1, 0, 0];
  hinge.frame = [0, 0, 90]; // Hinge's X axis becomes world Y.
  const posed = evaluateRig(rig([hinge]), { hinge: [90, 25, -70] }).get('hinge');
  near(posed.pivot, [2, 3, 4]);
  near(transformPoint(posed, [2, 3, 5]), [3, 3, 4]);
  near(transformPoint(posed, [2, 4, 4]), [2, 4, 4]);
  near(getJointQuaternion(hinge, [90, 0, 0]), [0, Math.SQRT1_2, 0, Math.SQRT1_2]);
});

test('ball frame, Euler order, limits, and fixed joints are enforced', () => {
  const ball = joint('ball');
  ball.frame = [0, 0, 90];
  near(transformPoint(evaluateRig(rig([ball]), { ball: [90, 0, 0] }).get('ball'), [0, 0, 1]), [1, 0, 0]);
  ball.frame = [0, 0, 0];
  near(transformPoint(evaluateRig(rig([ball]), { ball: [90, 90, 0] }).get('ball'), [0, 1, 0]), [1, 0, 0]);
  ball.limits = [[-10, 20], [-30, 40], [-50, 60]];
  near(clampJointAngles(ball, [-90, 90, 20]), [-10, 40, 20]);
  ball.type = 'hinge';
  near(clampJointAngles(ball, [90, 90, 20]), [20, 0, 0]);
  ball.type = 'fixed';
  near(clampJointAngles(ball, [90, 90, 20]), [0, 0, 0]);
  near(getJointQuaternion(ball, [90, 90, 20]), [0, 0, 0, 1]);
});

test('segmentation uses capsule distance, sphere endpoints, boxes, ties, and environment', () => {
  const segments = [segment('environment'),
    segment('wide', null, [{ type: 'capsule', a: [0, 0, 0], b: [0, 2, 0], radius: 1 }]),
    segment('narrow', null, [sphere([0.5, 1, 0], 0.2)]),
    segment('box', null, [{ type: 'box', min: [-2, 0, -1], max: [2, 2, 1] }]),
    segment('tie', null, [sphere([0.5, 1, 0], 0.2)])];
  assert.equal(classifyPoint([0.5, 1, 0], segments), 2);
  assert.equal(classifyPoint([0.3, 1, 0], segments), 1);
  assert.equal(classifyPoint([1.5, 1, 0], segments), 3);
  assert.equal(classifyPoint([0, -0.9, 0], segments), 1);
  assert.equal(classifyPoint([0, -1.1, 0], segments), 0);
  assert.equal(classifyPoint([10, 10, 10], segments), 0);
  assert.deepEqual([...classifyPoints(new Float32Array([99, 0.5, 1, 0, 7, 10, 10, 10]), segments, { offset: 1, stride: 4 })], [2, 0]);
  assert.throws(() => classifyPoints([0, NaN, 0], segments), /finite/);
  assert.throws(() => classifyPoints([0, 0, 0], segments, { count: 2 }), /count/);
});

test('spring supports free impulses, drag, hard stops, and invalid input rejection', () => {
  assert.deepEqual(stepSpring(0.2, 1.5, 0, 0, 0, 2, 0.4, -10, 10), { angle: 0.8, velocity: 1.5 });
  const dragged = stepSpring(0, 2, 0, 0, 2, 1, 1, -10, 10);
  near(dragged.angle, 1 - Math.exp(-2));
  near(dragged.velocity, 2 * Math.exp(-2));
  assert.deepEqual(stepSpring(0.9, 2, 0, 0, 0, 1, 0.1, -1, 1), { angle: 1, velocity: 0 });
  assert.deepEqual(stepSpring(0, 1, 5, 3, 4, 1, 0.1, 0, 0), { angle: 0, velocity: 0 });
  const input = [0, 0, 0, 1, 1, 1, 1 / 120, -1, 1];
  for (let i = 0; i < input.length; i++) {
    const invalid = [...input]; invalid[i] = null;
    assert.throws(() => stepSpring(...invalid), /finite/);
  }
  assert.throws(() => stepSpring(0, 0, 0, -1, 1, 1, 0.1, -1, 1), /nonnegative/);
  assert.throws(() => stepSpring(0, 0, 0, 1, 1, 0, 0.1, -1, 1), /inertia/);
});

test('spring solutions converge in all damping regimes and are step-size invariant', () => {
  for (const damping of [0.5, 4, 9]) { // underdamped, critical, overdamped for k=4, I=1
    let state = { angle: -0.4, velocity: 0.2 };
    for (let i = 0; i < 120; i++) state = stepSpring(state.angle, state.velocity, 0.7, 4, damping, 1, 1 / 120, -10, 10);
    const once = stepSpring(-0.4, 0.2, 0.7, 4, damping, 1, 1, -10, 10);
    near(state.angle, once.angle);
    near(state.velocity, once.velocity);
    const settled = stepSpring(-0.4, 0.2, 0.7, 4, damping, 1, 100, -10, 10);
    near(settled.angle, 0.7, 1e-9);
    near(settled.velocity, 0, 1e-9);
  }
  const undamped = stepSpring(1, 0, 0, 100, 0, 1, 1000, -10, 10);
  near(100 * undamped.angle ** 2 + undamped.velocity ** 2, 100, 1e-8);
});

test('rig imports reject broken hierarchy, malformed masks, and nonfinite parameters', () => {
  assert.equal(validateRig(rig()).version, 1);
  const invalidCases = [
    [model => { model.version = 2; }, /version/],
    [model => { model.joints[0].parent = 'elbow'; }, /cycle/],
    [model => { model.joints[1].parent = 'missing'; }, /unknown joint/],
    [model => { model.joints[1].id = 'shoulder'; }, /duplicate joint/],
    [model => { model.joints[0].pivot[0] = NaN; }, /pivot/],
    [model => { model.joints[0].stiffness = -1; }, /stiffness/],
    [model => { model.joints[0].inertia = 0; }, /inertia/],
    [model => { model.joints[1].axis = [0, 0, 0]; }, /unit length/],
    [model => { model.joints[0].limits[0] = [10, -10]; }, /minimum/],
    [model => { model.segments[0].joint = 'shoulder'; }, /first segment/],
    [model => { model.segments[1].joint = 'missing'; }, /unknown joint/],
    [model => { model.segments[1].shapes = [sphere([0, 0, 0], 0)]; }, /radius/],
    [model => { model.segments[1].shapes = [{ type: 'mesh' }]; }, /type/],
    [model => { model.segments[1].shapes = [{ type: 'box', min: [2, 0, 0], max: [1, 1, 1] }]; }, /minimum/],
  ];
  for (const [mutate, expected] of invalidCases) {
    const model = rig(); mutate(model);
    assert.throws(() => validateRig(model), expected);
  }
  assert.throws(() => evaluateRig(rig(), { shoulder: null }), /angles/);
  assert.throws(() => evaluateRig(rig(), null), /poseById/);
});


test('shipped scan rig validates and all rest transforms preserve the original scan', () => {
  const shipped = JSON.parse(readFileSync(new URL('../viewers/mannequin-rig/default-rig.json', import.meta.url)));
  validateRig(shipped);
  const transforms = evaluateRig(shipped);
  assert.equal(shipped.joints.filter(joint => joint.type !== 'fixed').length, 13);
  for (const joint of shipped.joints) {
    near(transformPoint(transforms.get(joint.id), joint.pivot), joint.pivot);
    near(transformPoint(transforms.get(joint.id), [0.2, -0.1, 0.8]), [0.2, -0.1, 0.8]);
  }
});

for (const side of ['left', 'right']) test(`measured ${side} arm swivel rotates around its long axis and follows the shoulder`, () => {
  const shipped = JSON.parse(readFileSync(new URL('../viewers/mannequin-rig/default-rig.json', import.meta.url)));
  const swivelId = `${side}_elbow`, shoulderId = `${side}_shoulder`;
  const swivel = shipped.joints.find(joint => joint.id === swivelId);
  assert.equal(swivel.motion, 'axial');
  assert.equal(swivel.type, 'hinge');
  for (const angle of [-90, 35, 90]) {
    const pose = { [swivelId]: [angle, 0, 0] };
    const transform = evaluateRig(shipped, pose).get(swivel.id);
    for (const distance of [-0.2, 0, 0.4]) {
      const point = swivel.pivot.map((value, i) => value + distance * swivel.axis[i]);
      near(transformPoint(transform, point), point);
      const parent = evaluateRig(shipped, { [shoulderId]: [15, 20, -10] }).get(shoulderId);
      const carried = evaluateRig(shipped, { ...pose, [shoulderId]: [15, 20, -10] }).get(swivel.id);
      near(transformPoint(carried, point), transformPoint(parent, point));
    }
    const offset = [-swivel.axis[1], swivel.axis[0], 0];
    const point = swivel.pivot.map((value, i) => value + offset[i]);
    const rotated = transformPoint(transform, point);
    near(Math.hypot(...rotated.map((value, i) => value - swivel.pivot[i])), Math.hypot(...offset));
    near(transformPoint(evaluateRig(shipped, { [swivelId]: [angle, 80, -60] }).get(swivel.id), point), rotated);
  }
});
