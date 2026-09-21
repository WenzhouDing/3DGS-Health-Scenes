/** Rigid articulation of the fused parts; scan coordinates never get rewritten.
 * All angles are degrees. Part transforms map REST WORLD coordinates directly
 * to posed world coordinates. Apply each transform once to both captures of its
 * part, including the Gaussian covariance (an entity quaternion does both).
 * Springs are independent, unit-inertia interaction controls, not calibrated
 * material physics or a collision/contact simulation.
 */
import {validateAnnotations, normalizeAxis, sceneSignature} from '../mannequin-joints/annotation-core.mjs';
import {evaluateRig, clampJointAngles, stepSpring} from '../mannequin-rig/rig-core.mjs';

export const POSE_SCHEMA = 'mannequin-articulation-pose';
const DEG = Math.PI / 180;
const clone = value => structuredClone(value);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const finiteVector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const identity = () => ({position: [0, 0, 0], rotation: [0, 0, 0, 1], matrix: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]});

function freeze(value) {
  if (value && typeof value === 'object') {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (object(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  return JSON.stringify(value);
}

/** Exact semantic signature, deliberately not a collision-prone short hash.
 * Labels, review state and notes may change without invalidating a saved pose.
 * Baseline spring preferences do not affect its geometry; a pose saves its own.
 */
export function annotationSignature(annotations) {
  return canonical({version: 1, joints: annotations.joints.map(joint => ({
    id: joint.id, type: joint.type, parentPart: joint.parentPart,
    childParts: [...joint.childParts].sort(), pivot: joint.pivot,
    axis: normalizeAxis(joint.axis), limits: joint.limits,
  })).sort((a, b) => a.id.localeCompare(b.id))});
}

export function dofCount(joint) {
  return joint.type === 'fixed' ? 0 : joint.type === 'ball' ? 3 : 1;
}

function rotate(q, p) {
  const [x, y, z, w] = q, [px, py, pz] = p;
  const tx = 2 * (y * pz - z * py), ty = 2 * (z * px - x * pz), tz = 2 * (x * py - y * px);
  return [px + w * tx + y * tz - z * ty,
    py + w * ty + z * tx - x * tz,
    pz + w * tz + x * ty - y * tx];
}

/** Transform a rest-world point with an evaluate() part/joint transform. */
export function transformPoint(transform, point) {
  if (!finiteVector(point)) throw new TypeError('A point needs three finite coordinates.');
  return rotate(transform.rotation, point).map((value, i) => value + transform.position[i]);
}

function springValue(value, name) {
  if (!Number.isFinite(value) || value < 0 || value > 100000) throw new TypeError(`${name} must be between 0 and 100000.`);
}

/** Validate before changing runtime state: failed imports are atomic. */
export function validatePose(document, annotations, manifest) {
  const errors = [];
  const annotationCheck = validateAnnotations(annotations, manifest);
  if (!annotationCheck.valid) return {valid: false, errors: annotationCheck.errors};
  if (!object(document)) return {valid: false, errors: ['The pose must be a JSON object.']};
  if (document.schema !== POSE_SCHEMA || document.version !== 1) errors.push('Unsupported pose format or version.');
  if (canonical(document.scene) !== canonical(sceneSignature(manifest))) errors.push('This pose belongs to a different fused scan.');
  if (document.annotationSignature !== annotationSignature(annotations)) errors.push('The joint centers, axes, connections or limits have changed since this pose was saved.');
  if (typeof document.savedAt !== 'string' || !/^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}Z$/.test(document.savedAt)
      || !Number.isFinite(Date.parse(document.savedAt)) || new Date(document.savedAt).toISOString() !== document.savedAt) errors.push('The pose save time must be a valid ISO date.');
  if (!Array.isArray(document.joints) || document.joints.length !== annotations.joints.length) {
    errors.push('The pose must include every joint exactly once.');
    return {valid: false, errors};
  }
  const definitions = new Map(annotations.joints.map(joint => [joint.id, joint]));
  const seen = new Set();
  for (const item of document.joints) {
    if (!object(item) || !definitions.has(item.id) || seen.has(item.id)) {
      errors.push('The pose has an unknown or duplicate joint.'); continue;
    }
    seen.add(item.id);
    const joint = definitions.get(item.id), dof = dofCount(joint);
    if (typeof item.atRest !== 'boolean') errors.push(`${item.id}: rest state must be true or false.`);
    for (const field of ['angles', 'target']) {
      if (!finiteVector(item[field])) { errors.push(`${item.id}: ${field} needs three finite angles.`); continue; }
      item[field].forEach((angle, i) => {
        // Exact scan rest is always valid, even if edited limits exclude zero.
        if (item.atRest === true || i >= dof) {
          if (angle !== 0) errors.push(`${item.id}: rest and inactive angles must be zero.`);
        } else if (angle < joint.limits[i][0] || angle > joint.limits[i][1]) {
          errors.push(`${item.id}: ${field} is outside its joint limits.`);
        }
      });
    }
    for (const field of ['stiffness', 'damping']) {
      try { springValue(item[field], `${item.id} ${field}`); } catch (error) { errors.push(error.message); }
    }
  }
  return {valid: errors.length === 0, errors};
}

export function createMotion(annotations, manifest) {
  const validation = validateAnnotations(annotations, manifest);
  if (!validation.valid) throw new TypeError(`Invalid joint map: ${validation.errors.join(' ')}`);
  const partIds = (manifest.parts ?? []).map(part => typeof part === 'string' ? part : part.id);
  if (!partIds.length || new Set(partIds).size !== partIds.length) throw new TypeError('The scan needs a unique list of body parts.');
  const definitions = clone(annotations);
  const joints = freeze(definitions.joints.map(joint => ({...clone(joint), axis: normalizeAxis(joint.axis)})));
  const byId = new Map(joints.map(joint => [joint.id, joint]));
  const owner = new Map(joints.flatMap(joint => joint.childParts.map(part => [part, joint.id])));
  const state = new Map(joints.map(joint => [joint.id, {
    angles: [0, 0, 0], target: [0, 0, 0], velocity: [0, 0, 0],
    stiffness: joint.stiffness, damping: joint.damping, atRest: true,
  }]));
  const rig = {
    version: 1,
    joints: joints.map(joint => ({...joint, type: joint.type === 'swivel' ? 'hinge' : joint.type,
      parent: owner.get(joint.parentPart) ?? null, frame: [0, 0, 0],
      angles: [0, 0, 0], target: [0, 0, 0], inertia: 1})),
    segments: [{id: 'environment', label: 'Fixed root', joint: null, color: [0, 0, 0], shapes: []}],
  };
  const rigById = new Map(rig.joints.map(joint => [joint.id, joint]));
  const signature = annotationSignature(definitions);
  const scene = sceneSignature(manifest);

  function need(id) {
    if (!byId.has(id)) throw new TypeError(`Unknown joint: ${String(id)}.`);
    return state.get(id);
  }

  function getState(id) { return clone(need(id)); }

  function setTarget(id, angles, {immediate = false} = {}) {
    const entry = need(id);
    if (!finiteVector(angles)) throw new TypeError('A target needs three finite angles.');
    if (typeof immediate !== 'boolean') throw new TypeError('Immediate mode must be true or false.');
    const activeJoint = {...rigById.get(id), limits: byId.get(id).limits};
    const target = clampJointAngles(activeJoint, angles);
    entry.target = target;
    entry.angles = clampJointAngles(activeJoint, entry.angles);
    entry.atRest = false;
    if (immediate) { entry.angles = [...target]; entry.velocity = [0, 0, 0]; }
    return getState(id);
  }

  function setSpring(id, {stiffness, damping}) {
    const entry = need(id);
    const nextStiffness = stiffness === undefined ? entry.stiffness : stiffness;
    const nextDamping = damping === undefined ? entry.damping : damping;
    springValue(nextStiffness, 'Stiffness'); springValue(nextDamping, 'Damping');
    entry.stiffness = nextStiffness; entry.damping = nextDamping;
    return getState(id);
  }

  function nudge(id, axisIndex = 0, velocityDegPerSecond = 35) {
    const entry = need(id), joint = byId.get(id);
    if (!Number.isInteger(axisIndex) || axisIndex < 0 || axisIndex >= 3) throw new TypeError('Nudge axis must be 0, 1 or 2.');
    if (!Number.isFinite(velocityDegPerSecond) || Math.abs(velocityDegPerSecond) > 100000) throw new TypeError('Nudge velocity must be finite and within ±100000 degrees per second.');
    if (joint.type === 'fixed') return getState(id);
    if (axisIndex >= dofCount(joint)) throw new TypeError('This rotation axis is inactive.');
    const velocity = entry.velocity[axisIndex] + velocityDegPerSecond;
    if (Math.abs(velocity) > 100000) throw new RangeError('The accumulated nudge velocity is too large.');
    const activeJoint = {...rigById.get(id), limits: joint.limits};
    entry.angles = clampJointAngles(activeJoint, entry.angles);
    entry.target = clampJointAngles(activeJoint, entry.target);
    entry.velocity[axisIndex] = velocity;
    entry.atRest = false;
    return getState(id);
  }

  /** Hidden-tab delays are capped to 50 ms rather than jumping the pose. */
  function step(dt) {
    if (!Number.isFinite(dt) || dt < 0) throw new TypeError('Time step must be finite and nonnegative.');
    dt = Math.min(dt, .05);
    if (dt === 0) return false;
    let changed = false;
    for (const joint of joints) {
      const entry = state.get(joint.id), dof = dofCount(joint);
      if (entry.atRest || dof === 0) continue;
      for (let i = 0; i < dof; i++) {
        const result = stepSpring(entry.angles[i] * DEG, entry.velocity[i] * DEG, entry.target[i] * DEG,
          entry.stiffness, entry.damping, 1, dt, joint.limits[i][0] * DEG, joint.limits[i][1] * DEG);
        let angle = result.angle / DEG, velocity = result.velocity / DEG;
        // Avoid tiny degree conversion errors beyond hard stops.
        angle = Math.max(joint.limits[i][0], Math.min(joint.limits[i][1], angle));
        if (Math.abs(angle - entry.target[i]) < 1e-8 && Math.abs(velocity) < 1e-7) {
          angle = entry.target[i]; velocity = 0;
        }
        if (angle !== entry.angles[i] || velocity !== entry.velocity[i]) changed = true;
        entry.angles[i] = angle; entry.velocity[i] = velocity;
      }
    }
    return changed;
  }

  function reset() {
    for (const entry of state.values()) Object.assign(entry, {
      angles: [0, 0, 0], target: [0, 0, 0], velocity: [0, 0, 0], atRest: true,
    });
    return evaluate();
  }

  function evaluate() {
    const pose = new Map([...state].map(([id, entry]) => [id, entry.angles]));
    // Scan-rest is an explicit exception to active joint limits. A temporary
    // expanded range lets the shared rig evaluator retain exact identity.
    for (const joint of rig.joints) joint.limits = state.get(joint.id).atRest
      ? byId.get(joint.id).limits.map(([lo, hi]) => [Math.min(lo, 0), Math.max(hi, 0)])
      : byId.get(joint.id).limits;
    const evaluated = evaluateRig(rig, pose);
    const result = new Map();
    for (const joint of joints) {
      const transform = evaluated.get(joint.id);
      const parentId = owner.get(joint.parentPart);
      const parentRotation = parentId ? evaluated.get(parentId).rotation : [0, 0, 0, 1];
      result.set(joint.id, {...transform, axis: rotate(parentRotation, joint.axis),
        axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(axis => rotate(parentRotation, axis))});
    }
    const parts = new Map(partIds.map(id => [id, owner.has(id) ? evaluated.get(owner.get(id)) : identity()]));
    return {parts, joints: result};
  }

  function exportPose() {
    return {schema: POSE_SCHEMA, version: 1, scene: clone(scene), annotationSignature: signature,
      savedAt: new Date().toISOString(), joints: joints.map(joint => {
        const {angles, target, stiffness, damping, atRest} = state.get(joint.id);
        return {id: joint.id, angles: [...angles], target: [...target], stiffness, damping, atRest};
      })};
  }

  function importPose(document) {
    const check = validatePose(document, definitions, manifest);
    if (!check.valid) throw new TypeError(`Cannot load pose: ${check.errors.join(' ')}`);
    for (const item of document.joints) Object.assign(state.get(item.id), {
      angles: [...item.angles], target: [...item.target], velocity: [0, 0, 0],
      stiffness: item.stiffness, damping: item.damping, atRest: item.atRest,
    });
    return evaluate();
  }

  return Object.freeze({joints, partIds: Object.freeze(partIds), signature, warnings: Object.freeze([...validation.warnings]),
    getState, setTarget, setSpring, nudge, step, reset, evaluate, exportPose, importPose});
}
