/** Gravity-driven arm joints in uncalibrated scene units.
 * Captured surface samples estimate each part's COM and second moment. Each
 * part has a fixed relative mass, independent of its number of scan samples.
 * Joint coordinates use a diagonal inertia approximation, with damping and
 * sampled collision sweeps; this is not a full coupled rigid-body solver.
 * No restoring spring or saved-pose target is used.
 */
import { transformPoint } from '../mannequin-articulation/motion-core.mjs';
import { jointWorldFrame } from './manikin-interaction.mjs';

const DEG = Math.PI / 180, EPS = 1e-10;
const SIDES = ['left', 'right'];
const PARTS = ['upper_arm', 'forearm', 'hand'];
const DEFAULT_MASS = { upper_arm: .028, forearm: .016, hand: .006 };
const clone = value => structuredClone(value);
const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const rotate = (q, p) => transformPoint({ rotation: q, position: [0, 0, 0] }, p);
const inverseRotate = (q, p) => rotate([-q[0], -q[1], -q[2], q[3]], p);
function positive(value, name, zero = false) {
  if (!Number.isFinite(value) || (zero ? value < 0 : value <= 0)) throw new RangeError(`${name} must be ${zero ? 'nonnegative' : 'positive'} and finite`);
}
function placementOf(value) {
  if (!value || !Array.isArray(value.position) || value.position.length !== 3 || !value.position.every(Number.isFinite)
    || !Array.isArray(value.rotation) || value.rotation.length !== 4 || !value.rotation.every(Number.isFinite)) throw new TypeError('A finite position and quaternion are required');
  positive(value.scale, 'placement.scale');
  const norm = Math.hypot(...value.rotation); positive(norm, 'placement quaternion length');
  return { position: [...value.position], rotation: value.rotation.map(v => v / norm), scale: value.scale };
}
function moments(samples, mass, part) {
  if (!samples.length) throw new TypeError(`Missing captured rest samples for ${part}`);
  const center = [0, 0, 0], covariance = Array.from({ length: 3 }, () => [0, 0, 0]);
  for (const s of samples) {
    if (!s.position || s.position.length !== 3 || !s.position.every(Number.isFinite)) throw new TypeError(`Invalid rest sample for ${part}`);
    for (let i = 0; i < 3; i++) center[i] += s.position[i] / samples.length;
  }
  for (const s of samples) for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) covariance[i][j] += (s.position[i] - center[i]) * (s.position[j] - center[j]) / samples.length;
  return { part, mass, sampleCount: samples.length, center, covariance, trace: covariance[0][0] + covariance[1][1] + covariance[2][2] };
}

export function createArmGravity({
  motion, getPlacement, restSamples, evaluate, onUpdate, onSettle,
  gravity = 9.81, damping = 4, fixedStep = 1 / 120, maxFrameDelta = .05,
  maxAngularStep = .75, maxAngularSpeed = 2.5, maxAngularAcceleration = 80,
  refinementSteps = 8, settleTime = .25, massWeights = DEFAULT_MASS
} = {}) {
  if (!motion?.joints || typeof motion.getState !== 'function' || typeof motion.setTarget !== 'function' || typeof motion.evaluate !== 'function') throw new TypeError('A fitted articulation motion controller is required');
  if (typeof getPlacement !== 'function' || typeof evaluate !== 'function') throw new TypeError('getPlacement and collision evaluate callbacks are required');
  if (!Array.isArray(restSamples)) throw new TypeError('Captured native restSamples are required');
  for (const [name, value] of Object.entries({ gravity, damping })) positive(value, name, true);
  for (const [name, value] of Object.entries({ fixedStep, maxFrameDelta, maxAngularStep, maxAngularSpeed, maxAngularAcceleration, settleTime })) positive(value, name);
  if (!Number.isInteger(refinementSteps) || refinementSteps < 1 || refinementSteps > 20) throw new RangeError('refinementSteps must be 1..20');
  for (const [name, fn] of Object.entries({ onUpdate, onSettle })) if (fn !== undefined && typeof fn !== 'function') throw new TypeError(`${name} must be a function`);
  const joints = new Map(motion.joints.map(j => [j.id, j])), properties = new Map(), sides = {};
  for (const side of SIDES) {
    const ids = [`${side}_shoulder`, `${side}_arm_swivel`];
    if (joints.get(ids[0])?.type !== 'ball' || !['swivel', 'hinge'].includes(joints.get(ids[1])?.type)) throw new TypeError(`Expected ${side} shoulder ball joint and arm swivel`);
    for (const suffix of PARTS) {
      const part = `${side}_${suffix}`, mass = massWeights[part] ?? massWeights[suffix];
      positive(mass, `${part} relative mass`);
      properties.set(part, moments(restSamples.filter(s => s.part === part), mass, part));
    }
    sides[side] = { ids, movingParts: PARTS.map(p => `${side}_${p}`), status: 'settled', reason: 'not-released',
      holds: new Set(), velocity: Object.fromEntries(ids.map(id => [id, [0, 0, 0]])), quiet: 0,
      report: null, needsCheck: true, lastDynamics: null };
  }
  let enabled = true, paused = false, accumulator = 0, elapsed = 0, discardedTime = 0, changed = false;
  let events = [];
  const sideFor = id => {
    const side = SIDES.find(s => id === s || sides[s].ids.includes(id));
    if (!side) throw new TypeError(`Not an arm joint: ${String(id)}`);
    return side;
  };
  const zero = s => { for (const id of s.ids) s.velocity[id] = [0, 0, 0]; };
  const anglesOf = s => Object.fromEntries(s.ids.map(id => [id, motion.getState(id).angles]));
  const apply = angles => { for (const [id, value] of Object.entries(angles)) motion.setTarget(id, value, { immediate: true }); };
  const reportSummary = r => r ? { status: r.status, blocked: r.blocked, contactCount: r.contacts?.length ?? 0,
    maxPenetration: r.maxPenetration ?? 0, contact: clone(r.violations?.[0] ?? r.contacts?.[0] ?? null) } : null;
  function getState() {
    return { changed, enabled, paused, active: enabled && !paused && SIDES.some(side => sides[side].status === 'active'), elapsed, discardedTime,
      sides: Object.fromEntries(SIDES.map(side => { const s = sides[side]; return [side, {
        status: s.status, reason: s.reason, heldJoints: [...s.holds], velocityRadPerSecond: clone(s.velocity),
        angles: anglesOf(s), report: clone(s.report), dynamics: clone(s.lastDynamics)
      }]; })) };
  }
  function publish() {
    const state = getState(); onUpdate?.(state);
    const pending = events; events = [];
    for (const e of pending) onSettle?.({ ...e, state: getState() });
    return getState();
  }
  function settle(side, reason, report, blocked = false) {
    const s = sides[side]; s.status = blocked ? 'blocked' : 'settled'; s.reason = reason; zero(s); s.quiet = 0;
    if (report) s.report = reportSummary(report);
    events.push({ side, reason, report: report ? clone(report) : null });
  }
  function check(pose, side, phase) {
    const s = sides[side], report = evaluate(pose, { side, movingParts: [...s.movingParts], jointIds: [...s.ids], phase });
    if (!report || typeof report.blocked !== 'boolean' || report.status === 'unavailable') throw new Error('Arm collision evaluation unavailable');
    return report;
  }
  function dynamics(side, pose, placement, angles) {
    const s = sides[side], world = new Map();
    for (const part of s.movingParts) {
      const p = properties.get(part), transform = pose.parts.get(part);
      if (!transform) throw new Error(`Missing posed arm part ${part}`);
      const center = rotate(placement.rotation, transformPoint(transform, p.center).map(v => v * placement.scale)).map((v, i) => v + placement.position[i]);
      world.set(part, { p, transform, center });
    }
    const result = {};
    for (const id of s.ids) {
      const joint = joints.get(id), count = joint.type === 'ball' ? 3 : 1, descendants = id.endsWith('_shoulder') ? s.movingParts : s.movingParts.slice(1);
      result[id] = [];
      for (let axisIndex = 0; axisIndex < count; axisIndex++) {
        const frame = jointWorldFrame(joint, pose.joints.get(id), angles[id], placement, axisIndex), axis = frame.normal;
        let torque = 0, inertia = 0;
        for (const part of descendants) {
          const { p, transform, center } = world.get(part), r = sub(center, frame.center);
          // dot(axis, cross(r, [0,-m*g,0])). Inertia includes part spread about its COM.
          torque += p.mass * gravity * (axis[0] * r[2] - axis[2] * r[0]);
          const a = inverseRotate(transform.rotation, inverseRotate(placement.rotation, axis));
          const projectedVariance = dot(a, p.covariance.map(row => dot(row, a)));
          inertia += p.mass * (Math.max(0, dot(r, r) - dot(axis, r) ** 2) + placement.scale ** 2 * Math.max(0, p.trace - projectedVariance));
        }
        inertia = Math.max(inertia, 1e-8 * placement.scale ** 2);
        result[id].push({ torque, inertia, acceleration: Math.max(-maxAngularAcceleration, Math.min(maxAngularAcceleration, torque / inertia)), axis: [...axis] });
      }
    }
    return result;
  }
  function stepSide(side, h, placement) {
    const s = sides[side]; if (s.status !== 'active' || s.holds.size) return;
    const original = anglesOf(s), pose = motion.evaluate();
    if (s.needsCheck) {
      const initial = check(pose, side, 'initial'); s.report = reportSummary(initial); s.needsCheck = false;
      if (initial.blocked) { settle(side, 'initial-overlap', initial, true); return; }
    }
    const acceleration = dynamics(side, pose, placement, original); s.lastDynamics = acceleration;
    const target = clone(original); let maxDelta = 0, quiet = true;
    for (const id of s.ids) for (let i = 0; i < acceleration[id].length; i++) {
      const [lo, hi] = joints.get(id).limits[i], a = acceleration[id][i].acceleration;
      let velocity = Math.max(-maxAngularSpeed, Math.min(maxAngularSpeed, (s.velocity[id][i] + a * h) * Math.exp(-damping * h)));
      const desired = original[id][i] + velocity * h / DEG;
      target[id][i] = Math.max(lo, Math.min(hi, desired));
      const atStop = (target[id][i] <= lo + EPS && velocity < 0) || (target[id][i] >= hi - EPS && velocity > 0);
      if (atStop) velocity = 0;
      s.velocity[id][i] = velocity;
      const delta = Math.abs(target[id][i] - original[id][i]); maxDelta = Math.max(maxDelta, delta);
      if (Math.abs(velocity) > .012 || (!atStop && Math.abs(a) > .06) || delta > .002) quiet = false;
    }
    const steps = Math.max(1, Math.ceil(maxDelta / maxAngularStep));
    let safe = 0, safeAngles = original, last = null, blocking = null;
    const at = t => Object.fromEntries(s.ids.map(id => [id, original[id].map((v, i) => v + (target[id][i] - v) * t)]));
    try {
      if (maxDelta > 1e-10) for (let i = 1; i <= steps; i++) {
        let t = i / steps; apply(at(t)); const report = check(motion.evaluate(), side, 'sweep');
        if (report.blocked) {
          blocking = report;
          for (let j = 0; j < refinementSteps; j++) {
            const mid = (safe + t) / 2; apply(at(mid)); const probe = check(motion.evaluate(), side, 'refine');
            if (probe.blocked) t = mid; else { safe = mid; last = probe; }
          }
          safeAngles = at(safe); break;
        }
        safe = t; safeAngles = at(t); last = report;
      }
    } finally {
      // Probes must never leak their tentative pose to the renderer or caller.
      apply(original);
    }
    if (safe > 0 && maxDelta > 1e-10) { apply(safeAngles); changed = true; }
    if (last) s.report = reportSummary(last);
    if (blocking) { settle(side, 'contact', blocking); return; }
    s.quiet = quiet ? s.quiet + h : 0;
    if (s.quiet >= settleTime) settle(side, 'gravity-equilibrium-or-limit', last);
  }
  function update(dt) {
    if (!Number.isFinite(dt) || dt < 0) throw new RangeError('dt must be nonnegative finite seconds');
    changed = false;
    if (!enabled || paused || !SIDES.some(side => sides[side].status === 'active')) return getState();
    const admitted = Math.min(dt, maxFrameDelta); discardedTime += dt - admitted; accumulator += admitted;
    const placement = placementOf(getPlacement());
    while (accumulator + EPS >= fixedStep) {
      accumulator = Math.max(0, accumulator - fixedStep); elapsed += fixedStep;
      for (const side of SIDES) {
        try { stepSide(side, fixedStep, placement); }
        catch (error) { if (sides[side].status === 'active') settle(side, `evaluation-error: ${error.message}`, null, true); }
      }
      if (!SIDES.some(side => sides[side].status === 'active')) { accumulator = 0; break; }
    }
    return publish();
  }
  function hold(id) {
    changed = false; const s = sides[sideFor(id)]; s.holds.add(id); s.status = 'held'; s.reason = 'user-hold'; zero(s); s.quiet = 0;
    apply(anglesOf(s)); accumulator = 0; return publish();
  }
  function wake(id) {
    changed = false; enabled = true;
    for (const side of id === undefined ? SIDES : [sideFor(id)]) {
      const s = sides[side]; zero(s); s.quiet = 0; s.needsCheck = true;
      if (!s.holds.size) { s.status = 'active'; s.reason = 'gravity'; apply(anglesOf(s)); }
    }
    accumulator = 0; return publish();
  }
  function release(id) {
    const s = sides[sideFor(id)]; s.holds.delete(id); return wake(id);
  }
  function pause() { changed = false; paused = true; accumulator = 0; for (const s of Object.values(sides)) zero(s); return publish(); }
  function resume() { changed = false; paused = false; accumulator = 0; for (const s of Object.values(sides)) s.needsCheck = true; return publish(); }
  function stop() {
    changed = false; enabled = false; accumulator = 0;
    for (const s of Object.values(sides)) { zero(s); s.holds.clear(); s.status = 'stopped'; s.reason = 'stopped'; }
    return publish();
  }
  function reset() {
    changed = false; enabled = true; paused = false; accumulator = 0; elapsed = 0; discardedTime = 0; events = [];
    for (const s of Object.values(sides)) { zero(s); s.holds.clear(); s.status = 'settled'; s.reason = 'not-released'; s.quiet = 0; s.needsCheck = true; s.report = null; s.lastDynamics = null; }
    return publish();
  }
  return { hold, release, wake, update, pause, resume, stop, reset, getState,
    getMassProperties: () => clone(Object.fromEntries(properties)) };
}
