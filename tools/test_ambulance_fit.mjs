#!/usr/bin/env node
/** Exercises the real capture loader/sample selection and saved articulated pose
 * without a GPU. Only rendering resource classes are replaced; all scan bytes,
 * joint maps, pose transforms and collision calculations are the actual modules.
 */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createAmbulanceManikin } from '../viewers/ambulance/manikin-scan.mjs';
import { createContactChecker } from '../viewers/ambulance/contact-check.mjs';
import { createMotion } from '../viewers/mannequin-articulation/motion-core.mjs';
import { buildPreset } from '../viewers/mannequin-articulation/pose-presets.mjs';

test('saved manikin fits measured stretcher and downward movement is rejected', async t => {
  const previousFetch = globalThis.fetch, previousLocation = globalThis.location;
  globalThis.location = { hostname: '127.0.0.1' };
  globalThis.fetch = async url => {
    try {
      const bytes = await readFile(fileURLToPath(url));
      return { ok: true, status: 200, text: async () => bytes.toString(),
        json: async () => JSON.parse(bytes.toString()),
        arrayBuffer: async () => bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) };
    } catch (error) {
      return { ok: false, status: error.code === 'ENOENT' ? 404 : 500 };
    }
  };
  t.after(() => { globalThis.fetch = previousFetch; globalThis.location = previousLocation; });
  class Entity {
    addChild() {} addComponent() {} setLocalPosition() {} setLocalRotation() {} setLocalScale() {} destroy() {}
  }
  const pc = { Entity, GSplatData: class {}, GSplatResource: class { destroy() {} }, WORKBUFFER_UPDATE_AUTO: 0 };
  const scan = await createAmbulanceManikin({ app: { root: new Entity(), graphicsDevice: {} }, pc });
  t.after(() => scan.dispose());
  const configBytes = await readFile(new URL('../viewers/ambulance/bed-placement.json', import.meta.url));
  const config = JSON.parse(configBytes);
  const mesh = await readFile(new URL('../viewers/ambulance/scene-collision.glb', import.meta.url));
  const meshReport = JSON.parse(await readFile(new URL('../viewers/ambulance/scene-collision.report.json', import.meta.url)));
  assert.equal(meshReport.source_config_sha256, createHash('sha256').update(configBytes).digest('hex'));
  assert.equal(meshReport.output_sha256, createHash('sha256').update(mesh).digest('hex'));
  // Measured bench length runs along +X, with a small negative-Z slope.
  // A degree/radian error previously rotated its long side across the cabin.
  const bench = config.collision.boxes.find(box => box.id === 'right-bench');
  assert.ok(bench);
  const [qx, qy, qz, qw] = bench.rotation;
  const benchLong = [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy + qz * qw), 2 * (qx * qz - qy * qw)];
  const expectedLength = Math.hypot(1, .08432874), expectedLong = [1 / expectedLength, 0, -.08432874 / expectedLength];
  assert.ok(benchLong.reduce((sum, v, i) => sum + v * expectedLong[i], 0) > .999999);
  const motion = createMotion(scan.annotations, scan.manifest), checker = createContactChecker(config);
  const preset = buildPreset('lying', motion, scan.manifest);
  for (const [joint, angles] of Object.entries({ ...preset.targets, ...config.jointOverrides })) {
    motion.setTarget(joint, angles, { immediate: true });
  }
  const parts = motion.evaluate().parts, samples = scan.getCollisionSamples(parts, config.placement);
  assert.equal(samples.length, scan.sampleCount);
  assert.ok(samples.length > 1000, 'The fitted body must be checked with captured surface samples');
  const baseline = checker.evaluate(samples);
  assert.equal(baseline.blocked, false, JSON.stringify(baseline.violations.slice(0, 4)));
  assert.equal(baseline.status, 'contact');
  assert.ok(baseline.contacts.some(contact => contact.proxy.startsWith('mattress-')));

  const down = structuredClone(config.placement);
  down.position[1] -= .03;
  const overlap = checker.evaluate(scan.getCollisionSamples(parts, down));
  assert.equal(overlap.blocked, true, 'Lowering the body into the mattress must be rejected');
  assert.ok(overlap.violations.some(contact => contact.proxy.startsWith('mattress-')));
  // Queries of a proposed placement must not mutate the render/sample state.
  assert.deepEqual(scan.getCollisionSamples(parts, config.placement), samples);

  const floor = config.collision.surfaces.find(surface => surface.id === 'floor');
  assert.ok(floor, 'A measured floor boundary is required');
  const normalLength = Math.hypot(...floor.normal), n = floor.normal.map(v => v / normalLength);
  const offset = floor.offset / normalLength, x = 0, z = .4;
  const surface = [x, (offset - n[0] * x - n[2] * z) / n[1], z];
  const floorOnly = createContactChecker({ surfaces: [floor] });
  const above = surface.map((v, i) => v + .03 * n[i]), below = surface.map((v, i) => v - .03 * n[i]);
  assert.equal(floorOnly.evaluate([{ part: 'probe', position: above, radius: .005 }]).blocked, false);
  const floorHit = floorOnly.evaluate([{ part: 'probe', position: below, radius: .005 }]);
  assert.equal(floorHit.blocked, true);
  assert.equal(floorHit.violations[0].proxy, 'floor');
  assert.ok(Math.abs(floorHit.maxPenetration - .035) < 1e-8);
});
