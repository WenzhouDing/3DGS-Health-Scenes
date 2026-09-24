/** Pose-held vertical drop in uncalibrated scene units.
 * The caller freezes articulation and supplies body-to-scene contact reports.
 * This samples the swept path; maxTravelStep must resolve the proxy geometry.
 * It is not a rigid-body, self-collision, or ragdoll simulation.
 */
const EPS = 1e-10;
const active = status => status === 'running' || status === 'paused';
const copy = value => structuredClone(value);
function positive(value, name) {
  if (!Number.isFinite(value) || value <= 0) throw new RangeError(`${name} must be positive and finite`);
  return value;
}
function position(value) {
  if (!value || value.length !== 3 || !Array.from(value).every(Number.isFinite)) throw new TypeError('Position requires three finite coordinates');
  return Array.from(value);
}
const equalPosition = (a, b) => a.every((v, i) => Math.abs(v - b[i]) <= EPS);

export function createGravityTest({
  getPosition, setPosition, evaluateAt, onUpdate, onStop,
  gravity = 9.81, fixedStep = 1 / 120, maxTravelStep = .004,
  maxFrameDelta = .1, maxDuration = 5, maxFallDistance = 3, minY = -2,
  maxLift = 1, contactNormalY = .5, contactClearance = 0,
  refinementSteps = 12, maxProbesPerCall = 4096
} = {}) {
  for (const [name, fn] of Object.entries({ getPosition, setPosition, evaluateAt })) {
    if (typeof fn !== 'function') throw new TypeError(`${name} must be a function`);
  }
  for (const [name, value] of Object.entries({ gravity, fixedStep, maxTravelStep, maxFrameDelta, maxDuration, maxFallDistance, maxLift })) positive(value, name);
  if (!Number.isFinite(minY)) throw new TypeError('minY must be finite');
  if (!Number.isFinite(contactNormalY) || contactNormalY <= 0 || contactNormalY > 1) throw new RangeError('contactNormalY must be in (0, 1]');
  if (!Number.isFinite(contactClearance) || contactClearance > 0) throw new RangeError('contactClearance must be finite and <= 0');
  if (!Number.isInteger(refinementSteps) || refinementSteps < 1 || refinementSteps > 24) throw new RangeError('refinementSteps must be 1..24');
  if (!Number.isInteger(maxProbesPerCall) || maxProbesPerCall < 1) throw new RangeError('maxProbesPerCall must be a positive integer');
  for (const [name, fn] of Object.entries({ onUpdate, onStop })) if (fn !== undefined && typeof fn !== 'function') throw new TypeError(`${name} must be a function`);

  let accumulator = 0, probes = 0, pendingStop = false;
  let state = { status: 'idle', reason: null, position: position(getPosition()), initialPosition: null,
    releasePosition: null, velocity: 0, elapsed: 0, fallDistance: 0, lift: 0,
    contact: null, lastReport: null, evaluations: 0, discardedTime: 0, error: null };
  const getState = () => copy(state);
  function summary(report) {
    return { status: report.status, blocked: report.blocked, contactCount: report.contacts.length,
      violationCount: report.violations?.length ?? 0, maxPenetration: report.maxPenetration ?? 0,
      sampleCount: report.sampleCount ?? null };
  }
  function evaluate(p) {
    if (++probes > maxProbesPerCall) throw new Error('Collision probe budget exceeded');
    state.evaluations++;
    const r = evaluateAt([...p]);
    if (!r || typeof r.blocked !== 'boolean' || !Array.isArray(r.contacts) || r.status === 'unavailable' || r.sampleCount === 0) throw new Error('Collision report unavailable or invalid');
    for (const c of r.contacts) {
      if (!Number.isFinite(c.clearance) || !c.normal || c.normal.length !== 3 || !c.normal.every(Number.isFinite)) throw new Error('Invalid collision contact');
    }
    return r;
  }
  function hit(report, direction) {
    if (report.blocked) return report.violations?.[0] ?? report.contacts.find(c => c.blocked) ?? report.contacts[0] ?? { normal: [0, 0, 0] };
    return report.contacts.find(c => c.clearance <= contactClearance + EPS &&
      (direction < 0 ? c.normal[1] >= -EPS : c.normal[1] < -EPS)) ?? null;
  }
  function finish(status, reason, contact = null) {
    state.status = status; state.reason = reason; state.contact = contact ? copy(contact) : null;
    state.velocity = 0; accumulator = 0; pendingStop = true;
  }
  function publish(commit = true) {
    if (commit && !equalPosition(position(getPosition()), state.position)) setPosition([...state.position]);
    onUpdate?.(getState());
    if (pendingStop) { pendingStop = false; onStop?.(getState()); }
    return getState();
  }
  function fail(error, initial = false) {
    state.error = String(error?.message ?? error);
    finish(initial ? 'rejected' : 'error', 'collision-evaluation-failed');
  }
  // Returns the first touching endpoint and a separately verified safe point.
  function sweep(from, to, direction, firstReport) {
    const distance = Math.abs(to[1] - from[1]);
    const count = Math.max(1, Math.ceil(distance / maxTravelStep));
    let safe = [...from], safeReport = firstReport;
    for (let i = 1; i <= count; i++) {
      const next = [from[0], from[1] + (to[1] - from[1]) * i / count, from[2]];
      let nextReport = evaluate(next), contact = hit(nextReport, direction);
      if (contact) {
        let lo = [...safe], hi = next;
        for (let j = 0; j < refinementSteps; j++) {
          const mid = [from[0], (lo[1] + hi[1]) / 2, from[2]], report = evaluate(mid);
          if (hit(report, direction)) { hi = mid; nextReport = report; }
          else { lo = mid; safeReport = report; }
        }
        contact = hit(nextReport, direction);
        return { position: nextReport.blocked ? lo : hi, report: nextReport.blocked ? safeReport : nextReport, contact };
      }
      safe = next; safeReport = nextReport;
    }
    return { position: safe, report: safeReport, contact: null };
  }
  function start({ lift = .25 } = {}) {
    if (active(state.status)) throw new Error('Gravity test is already active');
    if (!Number.isFinite(lift) || lift < 0 || lift > maxLift) throw new RangeError(`lift must be between 0 and ${maxLift}`);
    const initial = position(getPosition());
    state = { status: 'running', reason: null, position: initial, initialPosition: [...initial],
      releasePosition: [...initial], velocity: 0, elapsed: 0, fallDistance: 0, lift,
      contact: null, lastReport: null, evaluations: 0, discardedTime: 0, error: null };
    accumulator = 0; probes = 0; pendingStop = false;
    try {
      const report = evaluate(initial); state.lastReport = summary(report);
      if (report.blocked) finish('rejected', 'initial-overlap', hit(report, -1));
      else if (initial[1] < minY) finish('out-of-bounds', 'initial-position-below-limit');
      else if (lift > 0) {
        const target = [initial[0], initial[1] + lift, initial[2]], lifted = sweep(initial, target, 1, report);
        // Lift is atomic: an obstruction leaves the body at its original pose.
        if (lifted.contact) finish('obstructed', 'lift-obstructed', lifted.contact);
        else { state.position = target; state.releasePosition = [...target]; state.lastReport = summary(lifted.report); }
      } else {
        const contact = hit(report, -1);
        if (contact) finish(contact.normal[1] >= contactNormalY ? 'landed' : 'obstructed', 'initial-contact', contact);
      }
    } catch (error) { fail(error, true); }
    return publish();
  }
  function update(dt) {
    if (!Number.isFinite(dt) || dt < 0) throw new RangeError('dt must be a nonnegative finite number of seconds');
    if (state.status !== 'running') return getState();
    const external = position(getPosition());
    if (!equalPosition(external, state.position)) {
      state.position = external; finish('stopped', 'external-position-change'); return publish(false);
    }
    probes = 0;
    const admitted = Math.min(dt, maxFrameDelta);
    state.discardedTime += dt - admitted; accumulator += admitted;
    try {
      while (accumulator + EPS >= fixedStep && state.status === 'running') {
        const h = Math.min(fixedStep, maxDuration - state.elapsed);
        if (h <= EPS) { finish('timeout', 'maximum-duration'); break; }
        accumulator = Math.max(0, accumulator - h);
        const from = [...state.position], nextY = from[1] + state.velocity * h - .5 * gravity * h * h;
        const lowerBound = Math.max(minY, state.releasePosition[1] - maxFallDistance);
        const target = [from[0], Math.max(lowerBound, nextY), from[2]];
        const report = evaluate(from);
        if (report.blocked) { state.lastReport = summary(report); finish('obstructed', 'current-position-overlap', hit(report, -1)); break; }
        const result = sweep(from, target, -1, report);
        state.position = result.position; state.lastReport = summary(result.report);
        state.elapsed += h; state.fallDistance = Math.max(0, state.releasePosition[1] - state.position[1]);
        state.velocity -= gravity * h;
        if (result.contact) finish(result.contact.normal[1] >= contactNormalY ? 'landed' : 'obstructed', 'surface-contact', result.contact);
        else if (nextY <= lowerBound + EPS) finish('out-of-bounds', lowerBound === minY ? 'minimum-height' : 'maximum-fall-distance');
        else if (state.elapsed >= maxDuration - EPS) finish('timeout', 'maximum-duration');
      }
    } catch (error) { fail(error); }
    return publish();
  }
  function pause() {
    if (state.status !== 'running') return getState();
    state.status = 'paused'; accumulator = 0; return publish();
  }
  function resume() {
    if (state.status !== 'paused') return getState();
    state.status = 'running'; accumulator = 0; return publish();
  }
  function stop(reason = 'cancelled') {
    if (!active(state.status)) return getState();
    finish('stopped', String(reason)); return publish();
  }
  return { start, update, pause, resume, stop, getState };
}
