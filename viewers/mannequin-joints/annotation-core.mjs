// Coordinates match the fused Gaussian viewer: [front raw X, -Y, -Z].
// Joint angles are degrees. Spring values are annotation preferences, not SI
// material measurements: the source scans have no calibrated physical scale.
export const JOINT_TYPES = Object.freeze(['ball', 'hinge', 'swivel', 'fixed']);
export const ANNOTATION_SCHEMA = 'mannequin-joint-annotations';

const FALLBACK_LANDMARKS = {
  neck: [.006, -.055, .045], waist: [0, -.078, .575],
  left_shoulder: [.23, -.02, .16], right_shoulder: [-.23, -.056, .16],
  left_elbow: [.3839064154997863, .0011946936093699852, .2685460886187963],
  right_elbow: [-.3558778211475503, -.047657546478842475, .255655440325286],
  left_wrist: [.704, .065, .584], right_wrist: [-.697, .03, .597],
  left_hip: [.128, -.06, .73], right_hip: [-.128, -.064, .73],
  left_knee: [.321, -.021, 1.145], right_knee: [-.324, -.023, 1.145],
  left_ankle: [.456, -.01, 1.535], right_ankle: [-.46, -.009, 1.53],
};
const finiteVector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const dot = (a, b) => a.reduce((sum, value, i) => sum + value * b[i], 0);
const subtract = (a, b) => a.map((value, i) => value - b[i]);
function validTimestamp(value) {
  if (typeof value !== 'string' || value.length > 64) return false;
  const match = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})(?:\.\d+)?(?:Z|([+-])(\d{2}):(\d{2}))$/.exec(value);
  if (!match) return false;
  const [, yearText, monthText, dayText, hourText, minuteText, secondText, zone, zoneHour, zoneMinute] = match;
  const [year, month, day, hour, minute, second] = [yearText, monthText, dayText, hourText, minuteText, secondText].map(Number);
  const leap = year % 4 === 0 && (year % 100 !== 0 || year % 400 === 0);
  const monthDays = [31, leap ? 29 : 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
  return year > 0 && month >= 1 && month <= 12 && day >= 1 && day <= monthDays[month - 1]
    && hour <= 23 && minute <= 59 && second <= 59
    && (!zone || Number(zoneHour) <= 23 && Number(zoneMinute) <= 59) && Number.isFinite(Date.parse(value));
}

export function rawToViewer(point) {
  if (!finiteVector(point)) throw new TypeError('A point must have three finite coordinates.');
  return [point[0], -point[1], -point[2]];
}

export function normalizeAxis(axis) {
  if (!finiteVector(axis)) throw new TypeError('The axis must have three finite coordinates.');
  const length = Math.hypot(...axis);
  if (length <= 1e-8 || !Number.isFinite(length)) throw new RangeError('The axis must have a nonzero direction.');
  return axis.map(value => value / length);
}

export function sceneSignature(manifest) {
  const revision = manifest?.report?.parameters?.revision ?? manifest?.revision;
  const sources = manifest?.report?.sources ?? manifest?.sources;
  if (typeof revision !== 'string' || !revision || !Array.isArray(sources) || !sources.length) {
    throw new TypeError('The scan manifest is missing its revision or source hashes.');
  }
  const sourceHashes = {};
  for (const source of sources) {
    if (typeof source.file !== 'string' || !source.file || !/^[a-f0-9]{64}$/i.test(source.sha256 ?? '')) {
      throw new TypeError('The scan manifest contains an invalid source identity.');
    }
    sourceHashes[source.file] = source.sha256;
  }
  return {revision, sourceHashes, coordinateFrame: 'fused-viewer-x-y-z', units: 'scan units'};
}

function starterJoints(manifest, landmarkDocument) {
  const landmarks = {...FALLBACK_LANDMARKS, ...(landmarkDocument?.landmarks ?? landmarkDocument ?? {})};
  if (landmarkDocument?.frame && landmarkDocument.frame !== 'raw') {
    throw new TypeError('Starter landmarks must be in the front raw coordinate frame.');
  }
  const point = key => rawToViewer(landmarks[key]);
  const make = (id, label, type, parentPart, childParts, pivot, axis, limits, notes) => ({
    id, label, type, parentPart, childParts, pivot, axis: normalizeAxis(axis), limits,
    stiffness: 25, damping: 7, notes, reviewed: false,
  });
  const estimate = 'Starter estimate. Check the joint center from the front and side before marking reviewed.';
  const joints = [
    make('neck', 'Neck', 'ball', 'torso', ['neck', 'head'], point('neck'), [0, 0, 1],
      [[-45, 45], [-45, 45], [-60, 60]], estimate),
    make('waist', 'Waist / pelvis', 'fixed', 'torso', ['pelvis'], point('waist'), [0, 0, 1],
      [[0, 0], [0, 0], [0, 0]], 'Fixed starter connection. Change its type only if this manikin has a movable waist.'),
  ];
  for (const side of ['left', 'right']) {
    const name = side === 'left' ? 'Left' : 'Right';
    const upper = `${side}_upper_arm`, forearm = `${side}_forearm`;
    const mechanical = manifest?.report?.parts?.find(part => part.id === forearm)?.mechanicalConstraint;
    const swivelPivot = finiteVector(mechanical?.pivotFrontRaw) ? rawToViewer(mechanical.pivotFrontRaw) : point(`${side}_elbow`);
    const swivelAxis = finiteVector(mechanical?.axisFrontRaw) ? rawToViewer(mechanical.axisFrontRaw)
      : subtract(point(`${side}_wrist`), point(`${side}_elbow`));
    joints.push(
      make(`${side}_shoulder`, `${name} shoulder`, 'ball', 'torso', [upper], point(`${side}_shoulder`), [0, 0, 1],
        [[-90, 90], [-90, 90], [-90, 90]], estimate),
      make(`${side}_arm_swivel`, `${name} arm swivel`, 'swivel', upper, [forearm, `${side}_hand`], swivelPivot, swivelAxis,
        [[-90, 90], [0, 0], [0, 0]], 'Measured axial collar from the fusion review. Forearm, wrist pad and hand form one rigid body; there is no wrist joint. Confirm the axis and travel limits.'),
      make(`${side}_hip`, `${name} hip`, 'ball', 'pelvis', [`${side}_thigh`], point(`${side}_hip`), [0, 0, 1],
        [[-60, 60], [-45, 45], [-45, 45]], estimate),
      make(`${side}_knee`, `${name} knee`, 'hinge', `${side}_thigh`, [`${side}_shin`], point(`${side}_knee`), [1, 0, 0],
        [[-5, 120], [0, 0], [0, 0]], estimate),
      make(`${side}_ankle`, `${name} ankle`, 'hinge', `${side}_shin`, [`${side}_foot`], point(`${side}_ankle`), [1, 0, 0],
        [[-35, 35], [0, 0], [0, 0]], estimate),
    );
  }
  return joints;
}

export function createAnnotations(manifest, landmarks) {
  const result = {
    schema: ANNOTATION_SCHEMA, version: 1, scene: sceneSignature(manifest),
    joints: starterJoints(manifest, landmarks), updatedAt: new Date().toISOString(),
  };
  const validation = validateAnnotations(result, manifest);
  if (!validation.valid) throw new TypeError(`Cannot seed this scan: ${validation.errors.join(' ')}`);
  return result;
}

export function defaultJoint(id, manifest, landmarks) {
  const existing = starterJoints(manifest, landmarks).find(joint => joint.id === id);
  if (existing) return existing;
  return {
    id, label: 'New joint', type: 'fixed', parentPart: 'torso', childParts: [],
    pivot: rawToViewer(FALLBACK_LANDMARKS.waist), axis: [0, 0, 1],
    limits: [[0, 0], [0, 0], [0, 0]], stiffness: 25, damping: 7,
    notes: '', reviewed: false,
  };
}

export function validateAnnotations(document, manifest) {
  const errors = [], warnings = [];
  if (!object(document)) return {valid: false, errors: ['The annotation file must contain a JSON object.'], warnings};
  if (document.schema !== ANNOTATION_SCHEMA || document.version !== 1) errors.push('Unsupported annotation format or version.');
  let expected;
  try { expected = sceneSignature(manifest); } catch (error) { errors.push(error.message); }
  if (!object(document.scene)) errors.push('The scan identity is missing.');
  else if (expected) {
    if (document.scene.revision !== expected.revision) errors.push('This annotation belongs to a different scan revision.');
    if (document.scene.coordinateFrame !== expected.coordinateFrame) errors.push('This annotation uses a different coordinate frame.');
    if (document.scene.units !== expected.units) errors.push('Positions must be recorded in scan units.');
    const actual = document.scene.sourceHashes;
    if (!object(actual) || Object.keys(actual).length !== Object.keys(expected.sourceHashes).length
      || Object.entries(expected.sourceHashes).some(([file, hash]) => actual[file] !== hash)) {
      errors.push('The annotation source hashes do not match these scans.');
    }
  }
  if (!validTimestamp(document.updatedAt)) errors.push('The annotation timestamp must be a valid ISO date.');
  if (!Array.isArray(document.joints) || document.joints.length < 1 || document.joints.length > 128) {
    errors.push('The annotation must contain between 1 and 128 joints.');
    return {valid: false, errors, warnings};
  }
  const parts = new Set((manifest?.parts ?? []).map(part => typeof part === 'string' ? part : part.id));
  const ids = new Set(), owners = new Map(), edges = new Map();
  document.joints.forEach((joint, index) => {
    const prefix = `Joint ${index + 1}: `;
    if (!object(joint)) { errors.push(`${prefix}expected an object.`); return; }
    if (typeof joint.id !== 'string' || !/^[a-z][a-z0-9_-]{0,63}$/.test(joint.id)) errors.push(`${prefix}invalid joint ID.`);
    if (ids.has(joint.id)) errors.push(`${prefix}duplicate joint ID ${joint.id}.`);
    ids.add(joint.id);
    if (typeof joint.label !== 'string' || joint.label.trim().length === 0 || joint.label.length > 120) errors.push(`${prefix}label must have 1–120 characters.`);
    if (!JOINT_TYPES.includes(joint.type)) errors.push(`${prefix}unknown joint type.`);
    if (!parts.has(joint.parentPart)) errors.push(`${prefix}unknown parent body part.`);
    if (!Array.isArray(joint.childParts) || joint.childParts.length === 0 || joint.childParts.length > parts.size) {
      errors.push(`${prefix}select at least one moving body part.`);
    } else {
      const local = new Set();
      for (const child of joint.childParts) {
        if (!parts.has(child)) errors.push(`${prefix}unknown moving body part ${String(child)}.`);
        if (child === joint.parentPart) errors.push(`${prefix}a body part cannot be its own parent.`);
        if (local.has(child)) errors.push(`${prefix}duplicate moving body part ${child}.`);
        else if (owners.has(child)) errors.push(`${prefix}${child} is already owned by another joint.`);
        local.add(child); owners.set(child, joint.id);
        if (!edges.has(joint.parentPart)) edges.set(joint.parentPart, []);
        edges.get(joint.parentPart).push(child);
      }
      for (const side of ['left', 'right']) {
        if (local.has(`${side}_forearm`) !== local.has(`${side}_hand`)) {
          errors.push(`${prefix}the ${side} forearm and hand must remain one rigid body.`);
        }
      }
    }
    if (!finiteVector(joint.pivot) || joint.pivot.some(value => Math.abs(value) > 100)) errors.push(`${prefix}position must have three finite coordinates within ±100 scan units.`);
    if (!finiteVector(joint.axis) || Math.hypot(...joint.axis) <= 1e-8 || !Number.isFinite(Math.hypot(...joint.axis))) errors.push(`${prefix}axis must have a finite, nonzero direction.`);
    if (!Array.isArray(joint.limits) || joint.limits.length !== 3 || joint.limits.some(pair =>
      !Array.isArray(pair) || pair.length !== 2 || !pair.every(Number.isFinite)
      || pair[0] < -360 || pair[1] > 360 || pair[0] > pair[1])) errors.push(`${prefix}each angle range must be ordered and within −360° to 360°.`);
    for (const property of ['stiffness', 'damping']) {
      if (!Number.isFinite(joint[property]) || joint[property] < 0 || joint[property] > 100000) errors.push(`${prefix}${property} must be between 0 and 100000.`);
    }
    if (typeof joint.notes !== 'string' || joint.notes.length > 10000) errors.push(`${prefix}notes must be text up to 10000 characters.`);
    if (typeof joint.reviewed !== 'boolean') errors.push(`${prefix}reviewed must be true or false.`);
  });
  const visiting = new Set(), visited = new Set();
  function cycle(node) {
    if (visiting.has(node)) return true;
    if (visited.has(node)) return false;
    visiting.add(node);
    if ((edges.get(node) ?? []).some(cycle)) return true;
    visiting.delete(node); visited.add(node); return false;
  }
  if ([...edges.keys()].some(cycle)) errors.push('Joint connections contain a cycle. Every moving body part needs an acyclic parent chain.');
  const unchecked = document.joints.filter(joint => object(joint) && joint.reviewed === false).length;
  if (unchecked) warnings.push(`${unchecked} joint${unchecked === 1 ? '' : 's'} still need review.`);
  const unowned = [...parts].filter(part => part !== 'torso' && !owners.has(part));
  if (unowned.length) warnings.push(`Unassigned parts: ${unowned.join(', ')}.`);
  return {valid: errors.length === 0, errors, warnings};
}

// Returns the point on the forward ray, or null for parallel/behind-camera
// intersections. This preserves pivot depth while the UI moves it in a view.
export function rayPlaneIntersection(origin, direction, planePoint, normal) {
  if (![origin, direction, planePoint, normal].every(finiteVector)) return null;
  const directionLength = Math.hypot(...direction), normalLength = Math.hypot(...normal);
  if (directionLength <= 1e-12 || normalLength <= 1e-12) return null;
  const denominator = dot(direction, normal);
  if (Math.abs(denominator) <= 1e-10 * directionLength * normalLength) return null;
  const distance = dot(subtract(planePoint, origin), normal) / denominator;
  if (!Number.isFinite(distance) || distance < 0) return null;
  const point = origin.map((value, i) => value + direction[i] * distance);
  return point.every(Number.isFinite) ? point : null;
}

// A signed scan-unit move along a camera's right/up vectors. The two vectors
// must span its image plane, so movement cannot accidentally pitch the joint.
export function triangulatePlaneMove(currentPoint, view, {horizontal = 0, vertical = 0} = {}) {
  if (!finiteVector(currentPoint) || !Number.isFinite(horizontal) || !Number.isFinite(vertical)) {
    throw new TypeError('Plane movement requires a finite point and displacement.');
  }
  const right = normalizeAxis(view.right), up = normalizeAxis(view.up);
  if (Math.abs(dot(right, up)) > 1e-6) throw new RangeError('View right and up directions must be perpendicular.');
  return currentPoint.map((value, i) => value + horizontal * right[i] + vertical * up[i]);
}
