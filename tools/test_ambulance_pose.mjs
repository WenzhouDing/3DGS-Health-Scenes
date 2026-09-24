import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createMotion, transformPoint} from '../viewers/mannequin-articulation/motion-core.mjs';
import {createContactChecker} from '../viewers/ambulance/contact-check.mjs';
import {moveJointChecked} from '../viewers/ambulance/manikin-pose.mjs';

const json = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const manifest = json('../viewers/mannequin-fusion/fusion.json');
const annotations = json('../viewers/mannequin-articulation/joint-refinement.json').annotations;

test('a swept articulated branch stops at an obstacle even when both endpoints are clear', () => {
    const motion = createMotion(annotations, manifest);
    const pivot = motion.joints.find(j => j.id === 'left_shoulder').pivot;
    const point = pivot.map((n, i) => n + (i === 0 ? .5 : 0));
    const checker = createContactChecker({boxes: [{id: 'rail', center: point, halfExtents: [.025, .025, .025]}]});
    const evaluate = parts => checker.evaluate([{part: 'left_upper_arm', position: transformPoint(parts.get('left_upper_arm'), point), radius: .01}]);
    motion.setTarget('left_shoulder', [0, 90, 0], {immediate: true});
    assert.equal(evaluate(motion.evaluate().parts).blocked, false, 'far endpoint is clear');
    motion.setTarget('left_shoulder', [0, -90, 0], {immediate: true});
    assert.equal(evaluate(motion.evaluate().parts).blocked, false, 'starting endpoint is clear');
    const result = moveJointChecked({motion, jointId: 'left_shoulder', target: [0, 90, 0], evaluate});
    assert.equal(result.accepted, false);
    assert.ok(result.fraction > .35 && result.fraction < .5);
    assert.equal(evaluate(motion.evaluate().parts).blocked, false, 'restored pose is clear');
    assert.ok(motion.getState('left_shoulder').angles[1] < 0);
    assert.ok(result.rejected.violations.some(hit => hit.proxy === 'rail'));
    const unrestricted = moveJointChecked({motion, jointId: 'left_shoulder', target: [0, 200, 0], evaluate, enabled: false});
    assert.equal(unrestricted.accepted, true);
    assert.equal(motion.getState('left_shoulder').angles[1], 90, 'disabling contacts preserves joint travel limits');
});

test('an immediately blocked joint request leaves all rig state at the previous pose', () => {
    const motion = createMotion(annotations, manifest);
    motion.setTarget('left_shoulder', [0, -60, 0], {immediate: true});
    const before = motion.exportPose().joints;
    const result = moveJointChecked({motion, jointId: 'left_shoulder', target: [20, 0, 0], evaluate: () => ({blocked: true})});
    assert.equal(result.fraction, 0);
    assert.deepEqual(motion.exportPose().joints, before);
    assert.throws(() => moveJointChecked({motion, jointId: 'left_shoulder', target: [NaN, 0, 0], evaluate: () => ({blocked: false})}));
    assert.deepEqual(motion.exportPose().joints, before);
});
