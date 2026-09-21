import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {createAnnotations} from '../viewers/mannequin-joints/annotation-core.mjs';
import {createMotion, transformPoint, validatePose, annotationSignature} from '../viewers/mannequin-articulation/motion-core.mjs';

const manifest = JSON.parse(fs.readFileSync(new URL('../viewers/mannequin-fusion/fusion.json', import.meta.url)));
const landmarks = JSON.parse(fs.readFileSync(new URL('./fusion/front-landmarks.json', import.meta.url)));
const document = () => createAnnotations(manifest, landmarks);
const close = (a, b, epsilon = 1e-10) => a.forEach((value, i) => assert.ok(Math.abs(value - b[i]) <= epsilon, `${a} != ${b}`));
const add = (a, b) => a.map((value, i) => value + b[i]);
const subtract = (a, b) => a.map((value, i) => value - b[i]);
const identity = {position: [0, 0, 0], rotation: [0, 0, 0, 1], matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]};

// Independent Rodrigues calculation serves as the hierarchy/axis oracle.
function rotateAbout(point, pivot, axis, degrees) {
  const p = subtract(point, pivot), [x, y, z] = p, [ax, ay, az] = axis;
  const dot = x * ax + y * ay + z * az, c = Math.cos(degrees * Math.PI / 180), s = Math.sin(degrees * Math.PI / 180);
  const cross = [ay * z - az * y, az * x - ax * z, ax * y - ay * x];
  return add(p.map((v, i) => v * c + cross[i] * s + axis[i] * dot * (1 - c)), pivot);
}

test('all 16 fused parts start and reset at exact identity without changing the scan map', () => {
  const doc = document(), untouched = structuredClone(doc), motion = createMotion(doc, manifest);
  assert.equal(motion.partIds.length, 16);
  for (const value of motion.evaluate().parts.values()) {
    assert.deepEqual(value.position, identity.position); assert.deepEqual(value.rotation, identity.rotation); assert.deepEqual(value.matrix, identity.matrix);
  }
  motion.setTarget('left_shoulder', [23, -15, 41], {immediate: true});
  motion.setTarget('left_arm_swivel', [52, 0, 0], {immediate: true});
  motion.nudge('neck'); motion.step(.03); motion.reset(); motion.step(.05);
  for (const value of motion.evaluate().parts.values()) {
    assert.deepEqual(value.position, identity.position); assert.deepEqual(value.rotation, identity.rotation); assert.deepEqual(value.matrix, identity.matrix);
  }
  assert.deepEqual(doc, untouched);
});

test('shoulder and axial arm motions accumulate once and keep forearm, hand and wrist pad rigid', () => {
  const motion = createMotion(document(), manifest);
  motion.setTarget('left_shoulder', [0, 0, 38], {immediate: true});
  motion.setTarget('left_arm_swivel', [64, 0, 0], {immediate: true});
  const scene = motion.evaluate();
  const shoulder = motion.joints.find(j => j.id === 'left_shoulder'), swivel = motion.joints.find(j => j.id === 'left_arm_swivel');
  const hand = scene.parts.get('left_hand'), forearm = scene.parts.get('left_forearm');
  assert.equal(hand, forearm, 'the paired captures can share one identical full rigid transform');
  for (const p of [[.7, -.065, -.584], [.64, -.08, -.63], [...swivel.pivot]]) {
    const expected = rotateAbout(rotateAbout(p, swivel.pivot, swivel.axis, 64), shoulder.pivot, [0, 0, 1], 38);
    close(transformPoint(hand, p), expected);
  }
  close(scene.joints.get(swivel.id).pivot, rotateAbout(swivel.pivot, shoulder.pivot, [0, 0, 1], 38));
  close(scene.joints.get(swivel.id).axis, rotateAbout(swivel.axis, [0, 0, 0], [0, 0, 1], 38));
  assert.deepEqual(scene.parts.get('right_hand').matrix, identity.matrix);
});

test('leg chain and reordered annotation maps yield the same full accumulated transforms', () => {
  const doc = document(), reverse = structuredClone(doc); reverse.joints.reverse();
  const a = createMotion(doc, manifest), b = createMotion(reverse, manifest);
  for (const model of [a, b]) {
    model.setTarget('right_hip', [31, -19, 22], {immediate: true});
    model.setTarget('right_knee', [57, 0, 0], {immediate: true});
    model.setTarget('right_ankle', [-18, 0, 0], {immediate: true});
  }
  const sa = a.evaluate(), sb = b.evaluate();
  for (const part of a.partIds) close(sa.parts.get(part).matrix, sb.parts.get(part).matrix);
  const ankle = a.joints.find(j => j.id === 'right_ankle');
  close(sa.joints.get(ankle.id).pivot, transformPoint(sa.parts.get('right_shin'), ankle.pivot));
  close(sa.joints.get(ankle.id).pivot, transformPoint(sa.parts.get('right_foot'), ankle.pivot));
  assert.equal(a.signature, b.signature);
});

test('ball rotations use X then Y then Z and preserve distances without scaling', () => {
  const motion = createMotion(document(), manifest), neck = motion.joints.find(j => j.id === 'neck');
  motion.setTarget('neck', [24, -31, 52], {immediate: true});
  const transform = motion.evaluate().parts.get('head'), p = [.12, .21, .14];
  const expected = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].reduce((value, axis, i) => rotateAbout(value, neck.pivot, axis, [24, -31, 52][i]), p);
  close(transformPoint(transform, p), expected);
  assert.ok(Math.abs(Math.hypot(...subtract(p, neck.pivot)) - Math.hypot(...subtract(expected, neck.pivot))) < 1e-12);
  const m = transform.matrix;
  close([m[0] * p[0] + m[4] * p[1] + m[8] * p[2] + m[12], m[1] * p[0] + m[5] * p[1] + m[9] * p[2] + m[13], m[2] * p[0] + m[6] * p[1] + m[10] * p[2] + m[14]], expected);
});

test('part quaternion rotates anisotropic Gaussian covariance as well as its center', () => {
  const motion = createMotion(document(), manifest); motion.setTarget('left_shoulder', [0, 0, 90], {immediate: true});
  const transform = motion.evaluate().parts.get('left_forearm');
  const basis = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(p => subtract(transformPoint(transform, p), transform.position));
  const covariance = [0, 1, 2].map(i => [0, 1, 2].map(j => basis.reduce((sum, axis, k) => sum + axis[i] * [9, 4, 1][k] * axis[j], 0)));
  close(covariance.flat(), [4, 0, 0, 0, 9, 0, 0, 0, 1]);
  assert.ok(Math.abs(Math.hypot(...transform.rotation) - 1) < 1e-12);
});

test('limits clamp targets, inactive axes stay zero, and fixed connections cannot rotate', () => {
  const motion = createMotion(document(), manifest);
  close(motion.setTarget('left_knee', [900, 24, 42], {immediate: true}).angles, [120, 0, 0]);
  close(motion.setTarget('left_shoulder', [-999, 180, 20], {immediate: true}).angles, [-90, 90, 20]);
  close(motion.setTarget('waist', [30, 20, 10], {immediate: true}).angles, [0, 0, 0]);
  motion.nudge('waist'); close(motion.getState('waist').velocity, [0, 0, 0]);
  motion.nudge('left_knee', 0, 500); motion.step(.05);
  assert.equal(motion.getState('left_knee').angles[0], 120);
  assert.equal(motion.getState('left_knee').velocity[0], 0);
});

test('exact spring solver stays finite at extreme permitted stiffness/damping and caps stalled frame time', () => {
  for (const [stiffness, damping] of [[100000, 7], [100000, 100000], [25, 10], [0, 100000], [0, 0]]) {
    const motion = createMotion(document(), manifest);
    motion.setSpring('left_knee', {stiffness, damping}); motion.setTarget('left_knee', [90, 0, 0]); motion.nudge('left_knee');
    for (let i = 0; i < 600; i++) {
      motion.step(i % 2 ? 1 / 60 : .05);
      const state = motion.getState('left_knee');
      assert.ok(state.angles.every(Number.isFinite) && state.velocity.every(Number.isFinite));
      assert.ok(state.angles[0] >= -5 && state.angles[0] <= 120);
    }
  }
  const a = createMotion(document(), manifest), b = createMotion(document(), manifest);
  for (const model of [a, b]) model.setTarget('neck', [20, 10, 0]);
  a.step(600); b.step(.05); assert.deepEqual(a.getState('neck'), b.getState('neck'));
});

test('nudge returns toward the unchanged target and reset holds exact scan rest outside edited limits', () => {
  const doc = document(); doc.joints.find(j => j.id === 'left_knee').limits[0] = [10, 100];
  const motion = createMotion(doc, manifest);
  assert.deepEqual(motion.evaluate().parts.get('left_shin').matrix, identity.matrix);
  motion.step(.05); assert.deepEqual(motion.evaluate().parts.get('left_shin').matrix, identity.matrix);
  motion.setTarget('left_knee', [20, 0, 0]);
  assert.equal(validatePose(motion.exportPose(), doc, manifest).valid, true, 'target commands cannot export an out-of-range transient pose');
  motion.reset(); motion.nudge('left_knee');
  assert.equal(validatePose(motion.exportPose(), doc, manifest).valid, true, 'nudges clamp scan rest into an edited active range');
  motion.reset();
  assert.equal(motion.setTarget('left_knee', [0, 0, 0], {immediate: true}).angles[0], 10);
  motion.reset(); motion.nudge('neck', 0, 35);
  for (let i = 0; i < 360; i++) motion.step(1 / 60);
  close(motion.getState('neck').angles, [0, 0, 0], 1e-6);
  assert.deepEqual(motion.getState('neck').target, [0, 0, 0]);
  const saved = motion.exportPose(); assert.equal(validatePose(saved, doc, manifest).valid, true);
  motion.importPose(saved); motion.step(.05);
  assert.deepEqual(motion.evaluate().parts.get('left_shin').matrix, identity.matrix);
});

test('saved pose reproduces displayed transforms and settings and binds to geometry and source identity', () => {
  const doc = document(), motion = createMotion(doc, manifest);
  motion.setTarget('right_shoulder', [12, -24, 31]); motion.step(.05);
  motion.setTarget('left_knee', [73, 0, 0], {immediate: true}); motion.setSpring('neck', {stiffness: 50, damping: 12});
  const saved = motion.exportPose(), expected = motion.evaluate();
  motion.reset(); motion.importPose(JSON.parse(JSON.stringify(saved)));
  for (const part of motion.partIds) close(motion.evaluate().parts.get(part).matrix, expected.parts.get(part).matrix);
  assert.equal(motion.getState('neck').stiffness, 50);
  assert.deepEqual(motion.getState('right_shoulder').velocity, [0, 0, 0]);
  const renamed = structuredClone(doc); renamed.joints[0].label += ' reviewed'; renamed.joints[0].notes = 'new note'; renamed.joints[0].reviewed = true;
  assert.equal(annotationSignature(doc), annotationSignature(renamed));
  assert.equal(validatePose(saved, renamed, manifest).valid, true);
  const moved = structuredClone(doc); moved.joints[0].pivot[0] += .001;
  assert.equal(validatePose(saved, moved, manifest).valid, false);
  const foreign = structuredClone(saved); foreign.scene.sourceHashes[Object.keys(foreign.scene.sourceHashes)[0]] = 'f'.repeat(64);
  assert.equal(validatePose(foreign, doc, manifest).valid, false);
});

test('invalid runtime commands and pose imports fail before mutating state', () => {
  const motion = createMotion(document(), manifest); motion.setTarget('neck', [10, 20, 30], {immediate: true});
  const state = motion.getState('neck');
  for (const command of [
    () => motion.setTarget('neck', [NaN, 0, 0]), () => motion.setTarget('missing', [0, 0, 0]),
    () => motion.setTarget('neck', [1, 2]), () => motion.setSpring('neck', {stiffness: 40, damping: Infinity}),
    () => motion.setSpring('neck', {stiffness: null}), () => motion.step(-1), () => motion.step(Infinity),
    () => motion.nudge('neck', 3), () => motion.nudge('left_knee', 1), () => motion.nudge('neck', 0, Infinity),
  ]) assert.throws(command);
  assert.deepEqual(motion.getState('neck'), state);
  for (const mutate of [
    pose => { pose.joints[0].angles[0] = Infinity; }, pose => { pose.joints[0].angles[0] = 500; },
    pose => { pose.joints[0].atRest = true; }, pose => { pose.joints[1] = structuredClone(pose.joints[0]); },
    pose => { pose.joints[1].target[1] = 1; }, pose => { pose.joints.pop(); },
    pose => { pose.savedAt = '2026-02-30T00:00:00.000Z'; }, pose => { pose.annotationSignature += 'changed'; },
  ]) {
    const saved = motion.exportPose(); mutate(saved); assert.throws(() => motion.importPose(saved));
    assert.deepEqual(motion.getState('neck'), state);
  }
  const copied = motion.getState('neck'); copied.angles[0] = 999; assert.deepEqual(motion.getState('neck'), state);
  assert.throws(() => { motion.joints[0].pivot[0] = 999; });
});

test('invalid hierarchy and wrist splits are rejected rather than creating floating or double-rotated parts', () => {
  for (const mutate of [
    doc => { doc.joints.find(j => j.id === 'left_arm_swivel').childParts = ['left_hand']; },
    doc => { doc.joints.find(j => j.id === 'left_shoulder').parentPart = 'left_hand'; },
    doc => { doc.joints[0].axis = [0, 0, 0]; },
  ]) { const doc = document(); mutate(doc); assert.throws(() => createMotion(doc, manifest)); }
});
