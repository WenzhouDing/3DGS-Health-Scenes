import assert from 'node:assert/strict';
import { test } from 'node:test';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { createMotion, transformPoint } from '../viewers/mannequin-articulation/motion-core.mjs';
import { createArmGravity } from '../viewers/ambulance/arm-gravity.mjs';
const identity = {position:[0,0,0],rotation:[0,0,0,1],scale:1};
function energy(f, side = 'left', placement = identity) {
  const props = f.c.getMassProperties(), parts = f.motion.evaluate().parts;
  return Object.values(props).filter(p => p.part.startsWith(side)).reduce((sum, p) => {
    const native = transformPoint(parts.get(p.part), p.center).map(v => v * placement.scale);
    const world = transformPoint(placement, native);
    return sum + p.mass * 9.81 * world[1];
  }, 0);
}

test('actual captured arms stay held, then fall safely into modeled scene/body contact', async t => {
  const { createAmbulanceManikin } = await import('../viewers/ambulance/manikin-scan.mjs');
  const { createContactChecker } = await import('../viewers/ambulance/contact-check.mjs');
  const { createBodyContactChecker } = await import('../viewers/ambulance/body-contact.mjs');
  const { buildPreset } = await import('../viewers/mannequin-articulation/pose-presets.mjs');
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
  const scan = await createAmbulanceManikin({ app: { root: new Entity(), graphicsDevice: {} }, pc });
  t.after(() => scan.dispose());
  const config = JSON.parse(await readFile(new URL('../viewers/ambulance/bed-placement.json', import.meta.url)));
  const motion = createMotion(scan.annotations, scan.manifest), restSamples = scan.getCollisionSamples(new Map(), identity);
  const targets = { ...buildPreset('lying', motion, scan.manifest).targets, ...config.jointOverrides };
  const resetPose = () => { for (const [id, angles] of Object.entries(targets)) motion.setTarget(id, angles, { immediate: true }); };
  resetPose();
  const scene = createContactChecker(config), body = createBodyContactChecker({ restSamples, annotations: scan.annotations, referenceParts: motion.evaluate().parts });
  const evaluate = (pose, context) => {
    const a = scene.evaluate(scan.getCollisionSamples(pose.parts, config.placement, { parts: context.movingParts }), { stopOnBlock: true });
    if (a.blocked) return a;
    const b = body.evaluate(pose.parts, config.placement, { stopOnBlock: true, movingParts: context.movingParts });
    return { ...a, blocked: b.blocked, contacts: a.contacts.concat(b.contacts), violations: a.violations.concat(b.violations),
      status: b.blocked ? 'penetrating' : a.contacts.length + b.contacts.length ? 'contact' : 'clear' };
  };
  const baseline = evaluate(motion.evaluate(), {movingParts:null});
  assert.equal(baseline.blocked, false, 'Current fitted baseline must remain clear');
  const results = [];
  for (const side of ['left', 'right']) {
    resetPose(); const updates = [], settled = [];
    const c = createArmGravity({ motion, restSamples, getPlacement: () => config.placement, evaluate,
      onUpdate: s => updates.push(s), onSettle: s => settled.push(s) });
    const id = `${side}_shoulder`, raised = [...targets[id]]; raised[0] = 50;
    c.hold(id); motion.setTarget(id, raised, { immediate: true });
    const context = { movingParts: ['upper_arm', 'forearm', 'hand'].map(p => `${side}_${p}`) };
    assert.equal(evaluate(motion.evaluate(), context).blocked, false, `${side} raised pose must be collision clear`);
    const held = motion.exportPose(); for (let i = 0; i < 30; i++) c.update(1 / 60);
    assert.deepEqual(motion.getState(id).angles, raised);
    const f = { c, motion }, beforeEnergy = energy(f, side, config.placement), times = [];
    c.release(id);
    for (let i = 0; i < 360 && c.getState().active; i++) {
      const start = performance.now(); c.update(1 / 60); times.push(performance.now() - start);
      assert.equal(evaluate(motion.evaluate(), context).blocked, false, 'Each accepted actual arm pose must be safe');
      if(i === 5 && c.getState().active) {
        c.hold(id); const caught=motion.exportPose().joints.map(j=>[j.id,j.angles]);
        for(let heldFrame=0;heldFrame<30;heldFrame++) c.update(1/60);
        assert.deepEqual(motion.exportPose().joints.map(j=>[j.id,j.angles]),caught,'Regrab freezes the falling branch immediately');
        c.release(id);
      }
    }
    const afterEnergy = energy(f, side, config.placement), state = c.getState();
    assert.ok(afterEnergy < beforeEnergy - .01, `${side} arm COM must lower materially after release`);
    assert.equal(state.active, false, `${side} arm must settle within six simulated seconds`);
    assert.equal(settled.length, 1);
    for (const j of held.joints.filter(j => !j.id.startsWith(`${side}_shoulder`) && !j.id.startsWith(`${side}_arm_swivel`))) assert.deepEqual(motion.getState(j.id).angles, j.angles);
    times.sort((a, b) => a - b);
    results.push({ side, sampleCount: scan.sampleCount, potentialChange: afterEnergy - beforeEnergy, comHeightChange: (afterEnergy - beforeEnergy) / (.05 * 9.81),
      reason: state.sides[side].reason, finalAngles: state.sides[side].angles, frames: times.length,
      medianFrameMs: times[Math.floor(times.length / 2)], maxFrameMs: times.at(-1), independentAcceptedPoseChecks: times.length });
  }
  const hashes={};
  for(const path of ['viewers/ambulance/arm-gravity.mjs','viewers/ambulance/body-contact.mjs','viewers/ambulance/bed-placement.json']) hashes[path]=createHash('sha256').update(await readFile(new URL('../'+path,import.meta.url))).digest('hex');
  const audit={status:'pass',sampleCount:scan.sampleCount,sourceHashes:hashes,baseline:{blocked:baseline.blocked,maxPenetration:baseline.maxPenetration},actualArmGravity:results};
  if (process.env.AMBULANCE_ARM_GRAVITY_AUDIT_PATH) {
    await writeFile(process.env.AMBULANCE_ARM_GRAVITY_AUDIT_PATH, JSON.stringify(audit, null, 2) + '\n');
  }
  console.log(JSON.stringify(audit));
});
