import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {createAnnotations, defaultJoint, validateAnnotations, rawToViewer,
  normalizeAxis, rayPlaneIntersection, triangulatePlaneMove} from '../viewers/mannequin-joints/annotation-core.mjs';

const manifest = JSON.parse(fs.readFileSync(new URL('../viewers/mannequin-fusion/fusion.json', import.meta.url)));
const landmarks = JSON.parse(fs.readFileSync(new URL('./fusion/front-landmarks.json', import.meta.url)));
const clone = value => structuredClone(value);
const close = (actual, expected, tolerance = 1e-12) => actual.forEach((value, i) => assert.ok(Math.abs(value - expected[i]) < tolerance, `${actual} differs from ${expected}`));

test('starter scene identity binds annotations to the exact fused scan', () => {
  const document = createAnnotations(manifest, landmarks);
  assert.equal(document.scene.revision, manifest.report.parameters.revision);
  assert.deepEqual(document.scene.sourceHashes, Object.fromEntries(manifest.report.sources.map(source => [source.file, source.sha256])));
  assert.equal(document.scene.coordinateFrame, 'fused-viewer-x-y-z');
  assert.equal(document.scene.units, 'scan units');
  assert.equal(document.joints.length, 12);
  assert.equal(validateAnnotations(document, manifest).valid, true);
  assert.ok(document.joints.every(joint => joint.reviewed === false));
  assert.deepEqual(document.joints.find(joint => joint.id === 'neck').pivot, rawToViewer(landmarks.landmarks.neck));
  assert.equal(validateAnnotations(JSON.parse(JSON.stringify(document)), manifest).valid, true);
});

test('measured arm collars are converted without adding wrist pitch or a wrist joint', () => {
  const document = createAnnotations(manifest, landmarks);
  for (const side of ['left', 'right']) {
    const joint = document.joints.find(joint => joint.id === `${side}_arm_swivel`);
    const measured = manifest.report.parts.find(part => part.id === `${side}_forearm`).mechanicalConstraint;
    assert.equal(joint.type, 'swivel');
    assert.deepEqual(joint.pivot, rawToViewer(measured.pivotFrontRaw));
    close(joint.axis, normalizeAxis(rawToViewer(measured.axisFrontRaw)));
    assert.deepEqual(joint.childParts, [`${side}_forearm`, `${side}_hand`]);
    assert.equal(document.joints.some(other => other.id.includes('wrist')), false);
  }
});

test('edits and default resets never mutate starter inputs or other joints', () => {
  const originalLandmarks = clone(landmarks), originalManifest = clone(manifest);
  const first = createAnnotations(manifest, landmarks), second = createAnnotations(manifest, landmarks);
  first.joints[0].pivot[0] = 100;
  first.joints[0].limits[0][0] = -360;
  assert.notDeepEqual(first.joints[0].pivot, second.joints[0].pivot);
  assert.deepEqual(defaultJoint('neck', manifest, landmarks), second.joints[0]);
  assert.deepEqual(manifest, originalManifest); assert.deepEqual(landmarks, originalLandmarks);
});

test('imports reject foreign scans, missing hashes, coordinate systems and invalid numbers', () => {
  const mutations = [
    doc => { doc.scene.revision += '-other'; },
    doc => { delete doc.scene.sourceHashes[Object.keys(doc.scene.sourceHashes)[0]]; },
    doc => { doc.scene.sourceHashes.extra = '0'.repeat(64); },
    doc => { doc.scene.coordinateFrame = 'front-raw'; },
    doc => { doc.scene.units = 'mm'; },
    doc => { doc.joints[0].pivot[0] = NaN; },
    doc => { doc.joints[0].pivot[0] = 101; },
    doc => { doc.joints[0].axis = [0, 0, 0]; },
    doc => { doc.joints[0].limits[0] = [30, -30]; },
    doc => { doc.joints[0].limits[0] = [-361, 360]; },
    doc => { doc.joints[0].stiffness = Infinity; },
    doc => { doc.joints[0].damping = -1; },
    doc => { doc.joints[0].reviewed = 'yes'; },
    doc => { doc.updatedAt = 'yesterday'; },
    doc => { doc.updatedAt = '2026-02-30T00:00:00Z'; },
    doc => { doc.updatedAt = '2026-09-20T24:00:00Z'; },
  ];
  for (const mutate of mutations) {
    const document = createAnnotations(manifest, landmarks); mutate(document);
    const before = clone(document);
    assert.equal(validateAnnotations(document, manifest).valid, false, String(mutate));
    assert.deepEqual(document, before, 'validation must not alter imported data');
  }
});

test('part ownership rejects duplicate joints, body cycles, unknown parts and wrist splits', () => {
  const mutations = [
    doc => { doc.joints[1].id = doc.joints[0].id; },
    doc => { doc.joints[1].childParts.push('head'); },
    doc => { doc.joints[0].parentPart = 'head'; },
    doc => { doc.joints[0].childParts.push('unknown_part'); },
    doc => { doc.joints.find(joint => joint.id === 'right_arm_swivel').childParts = ['right_hand']; },
    doc => {
      doc.joints.find(joint => joint.id === 'left_shoulder').parentPart = 'left_forearm';
    },
  ];
  for (const mutate of mutations) {
    const document = createAnnotations(manifest, landmarks); mutate(document);
    assert.equal(validateAnnotations(document, manifest).valid, false, String(mutate));
  }
  for (const malformed of [null, [], 3, {}, {schema: 'mannequin-joint-annotations', version: 1, joints: [null]}]) {
    assert.equal(validateAnnotations(malformed, manifest).valid, false);
  }
});

test('orthographic placement preserves the hidden coordinate from each view', () => {
  const old = [.2, .05, -.5];
  close(rayPlaneIntersection([.4, 2, -.6], [0, -1, 0], old, [0, 1, 0]), [.4, .05, -.6]);
  close(rayPlaneIntersection([2, -.07, -.8], [-1, 0, 0], old, [1, 0, 0]), [.2, -.07, -.8]);
  assert.equal(rayPlaneIntersection([0, 0, 0], [1, 0, 0], old, [0, 1, 0]), null);
  assert.equal(rayPlaneIntersection([0, 0, 0], [0, -1, 0], old, [0, 1, 0]), null);
  assert.equal(rayPlaneIntersection([NaN, 0, 0], [0, 1, 0], old, [0, 1, 0]), null);
});

test('perspective placement and plane nudging stay on the current depth plane', () => {
  close(rayPlaneIntersection([0, 3, 0], [1, -2, -1], [0, .1, 0], [0, 1, 0]), [1.45, .1, -1.45]);
  close(triangulatePlaneMove([.2, .05, -.5], {right: [1, 0, 0], up: [0, 0, 1]}, {horizontal: .01, vertical: -.02}), [.21, .05, -.52]);
  assert.throws(() => triangulatePlaneMove([0, 0, 0], {right: [1, 0, 0], up: [1, 0, 0]}, {horizontal: 1}), /perpendicular/);
});
