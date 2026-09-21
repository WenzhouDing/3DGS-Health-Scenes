/**
 * Dependency-free articulation math shared by the editor and its Node checks.
 *
 * Coordinates and pivots are in the prepared scan's REST WORLD space. A joint's
 * matrix maps an original splat center directly into posed world space; its
 * rotation also rotates that splat's covariance. Do not apply ancestor matrices
 * again: evaluateRig already accumulates the complete hierarchy.
 *
 * Angles, frames, and limits use degrees. Euler XYZ means rotate X, then Y,
 * then Z (quaternion qZ * qY * qX). A frame establishes the joint's local axes.
 * A ball joint rotates about those axes; a hinge uses angles[0] about its `axis`
 * expressed in that frame. Fixed joints and inactive hinge axes remain zero.
 * Spring arguments alone use radians and seconds, with inertia > 0.
 */

const DEG = Math.PI / 180;
const IDENTITY = [0, 0, 0, 1];

function fail(path, message) {
  throw new TypeError(`${path}: ${message}`);
}

function finite(value, path) {
  if (!Number.isFinite(value)) fail(path, 'must be a finite number');
}

function vector(value, length, path) {
  if ((!Array.isArray(value) && !ArrayBuffer.isView(value)) || value.length !== length) {
    fail(path, `must contain ${length} numbers`);
  }
  for (let i = 0; i < length; i++) finite(value[i], `${path}[${i}]`);
}

function identifier(value, path) {
  if (typeof value !== 'string' || !value.trim()) fail(path, 'must be a nonempty string');
}

function limits(value, path) {
  if (!Array.isArray(value) || value.length !== 3) fail(path, 'must contain three [min, max] pairs');
  value.forEach((pair, i) => {
    vector(pair, 2, `${path}[${i}]`);
    if (pair[0] > pair[1]) fail(`${path}[${i}]`, 'minimum must not exceed maximum');
  });
}

/** Validate an imported rig, throwing an error with the offending field path. */
export function validateRig(rig) {
  if (!rig || typeof rig !== 'object' || Array.isArray(rig)) fail('rig', 'must be an object');
  if (rig.version !== 1) fail('rig.version', 'unsupported version; expected 1');
  if (!Array.isArray(rig.joints)) fail('rig.joints', 'must be an array');
  if (!Array.isArray(rig.segments) || !rig.segments.length) fail('rig.segments', 'must be a nonempty array');

  const joints = new Map();
  rig.joints.forEach((joint, i) => {
    const path = `rig.joints[${i}]`;
    if (!joint || typeof joint !== 'object') fail(path, 'must be an object');
    identifier(joint.id, `${path}.id`);
    if (joints.has(joint.id)) fail(`${path}.id`, `duplicate joint '${joint.id}'`);
    joints.set(joint.id, joint);
    identifier(joint.label, `${path}.label`);
    if (joint.parent !== null) identifier(joint.parent, `${path}.parent`);
    if (!['ball', 'hinge', 'fixed'].includes(joint.type)) fail(`${path}.type`, 'expected ball, hinge, or fixed');
    vector(joint.pivot, 3, `${path}.pivot`);
    vector(joint.axis, 3, `${path}.axis`);
    if (joint.type === 'hinge' && Math.abs(Math.hypot(...joint.axis) - 1) > 1e-4) {
      fail(`${path}.axis`, 'hinge axis must have unit length');
    }
    vector(joint.frame, 3, `${path}.frame`);
    vector(joint.angles, 3, `${path}.angles`);
    vector(joint.target, 3, `${path}.target`);
    limits(joint.limits, `${path}.limits`);
    for (const field of ['stiffness', 'damping', 'inertia']) {
      finite(joint[field], `${path}.${field}`);
      if (joint[field] < 0 || (field === 'inertia' && joint[field] === 0)) {
        fail(`${path}.${field}`, field === 'inertia' ? 'must be greater than zero' : 'must be nonnegative');
      }
    }
  });

  const complete = new Set();
  const visiting = new Set();
  function visit(joint) {
    if (complete.has(joint.id)) return;
    if (visiting.has(joint.id)) fail('rig.joints', `parent cycle involving '${joint.id}'`);
    visiting.add(joint.id);
    if (joint.parent !== null) {
      const parent = joints.get(joint.parent);
      if (!parent) fail(`joint '${joint.id}'.parent`, `unknown joint '${joint.parent}'`);
      visit(parent);
    }
    visiting.delete(joint.id);
    complete.add(joint.id);
  }
  rig.joints.forEach(visit);

  const segmentIds = new Set();
  rig.segments.forEach((segment, i) => {
    const path = `rig.segments[${i}]`;
    if (!segment || typeof segment !== 'object') fail(path, 'must be an object');
    identifier(segment.id, `${path}.id`);
    identifier(segment.label, `${path}.label`);
    if (segmentIds.has(segment.id)) fail(`${path}.id`, `duplicate segment '${segment.id}'`);
    segmentIds.add(segment.id);
    if (segment.joint !== null && !joints.has(segment.joint)) fail(`${path}.joint`, `unknown joint '${segment.joint}'`);
    vector(segment.color, 3, `${path}.color`);
    if (segment.color.some(value => value < 0 || value > 1)) fail(`${path}.color`, 'RGB components must be between 0 and 1');
    if (!Array.isArray(segment.shapes)) fail(`${path}.shapes`, 'must be an array');
    if (i === 0 && (segment.id !== 'environment' || segment.joint !== null || segment.shapes.length)) {
      fail(path, "first segment must be 'environment', with joint: null and no shapes");
    }
    segment.shapes.forEach((shape, shapeIndex) => {
      const shapePath = `${path}.shapes[${shapeIndex}]`;
      if (!shape || typeof shape !== 'object') fail(shapePath, 'must be an object');
      if (shape.type === 'capsule') {
        vector(shape.a, 3, `${shapePath}.a`);
        vector(shape.b, 3, `${shapePath}.b`);
        finite(shape.radius, `${shapePath}.radius`);
        if (shape.radius <= 0) fail(`${shapePath}.radius`, 'must be greater than zero');
      } else if (shape.type === 'box') {
        vector(shape.min, 3, `${shapePath}.min`);
        vector(shape.max, 3, `${shapePath}.max`);
        if (shape.min.some((value, axis) => value > shape.max[axis])) fail(shapePath, 'box minimum must not exceed maximum');
      } else {
        fail(`${shapePath}.type`, 'expected capsule or box');
      }
    });
  });
  return rig;
}

function classifyXYZ(x, y, z, segments) {
  let selected = 0;
  let best = Infinity;
  for (let i = 1; i < segments.length; i++) {
    for (const shape of segments[i].shapes) {
      let score;
      if (shape.type === 'capsule') {
        const ax = shape.a[0], ay = shape.a[1], az = shape.a[2];
        const dx = shape.b[0] - ax, dy = shape.b[1] - ay, dz = shape.b[2] - az;
        const lengthSquared = dx * dx + dy * dy + dz * dz;
        const t = lengthSquared ? Math.max(0, Math.min(1, ((x - ax) * dx + (y - ay) * dy + (z - az) * dz) / lengthSquared)) : 0;
        const px = x - ax - t * dx, py = y - ay - t * dy, pz = z - az - t * dz;
        score = Math.sqrt(px * px + py * py + pz * pz) / shape.radius;
        if (score > 1) continue;
      } else {
        if (x < shape.min[0] || x > shape.max[0] || y < shape.min[1] || y > shape.max[1] || z < shape.min[2] || z > shape.max[2]) continue;
        score = 0.95;
      }
      if (score < best) {
        best = score;
        selected = i;
      }
    }
  }
  return selected;
}

/**
 * Select a rigid segment using rest-space shapes. Inside capsules compete by
 * distance to their centerline / radius; boxes score 0.95. Ties keep the first
 * segment. A capsule with identical endpoints is a sphere. Outside all masks
 * returns index 0 (environment). Validate the rig before classifying a scan.
 */
export function classifyPoint(point, segments) {
  vector(point, 3, 'point');
  return classifyXYZ(point[0], point[1], point[2], segments);
}

/** Classify packed/interleaved XYZ data; count defaults to all available points. */
export function classifyPoints(positions, segments, { stride = 3, offset = 0, count } = {}) {
  if ((!Array.isArray(positions) && !ArrayBuffer.isView(positions)) || !Number.isInteger(positions.length)) {
    fail('positions', 'must be an array or typed array');
  }
  if (!Number.isInteger(stride) || stride < 3) fail('stride', 'must be an integer of at least 3');
  if (!Number.isInteger(offset) || offset < 0 || offset > positions.length) fail('offset', 'must be an integer within positions');
  const available = positions.length - offset < 3 ? 0 : Math.floor((positions.length - offset - 3) / stride) + 1;
  if (count === undefined) count = available;
  if (!Number.isInteger(count) || count < 0 || count > available) fail('count', 'exceeds available XYZ points');
  const result = segments.length <= 65536 ? new Uint16Array(count) : new Uint32Array(count);
  for (let i = 0; i < count; i++) {
    const start = offset + i * stride;
    const x = positions[start], y = positions[start + 1], z = positions[start + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) fail(`positions[${start}]`, 'XYZ components must be finite');
    result[i] = classifyXYZ(x, y, z, segments);
  }
  return result;
}

export function clampJointAngles(joint, angles = joint.angles) {
  vector(angles, 3, `joint '${joint.id}'.angles`);
  limits(joint.limits, `joint '${joint.id}'.limits`);
  if (joint.type === 'fixed') return [0, 0, 0];
  if (!['ball', 'hinge'].includes(joint.type)) fail(`joint '${joint.id}'.type`, 'expected ball, hinge, or fixed');
  return Array.from(angles, (angle, i) => joint.type === 'hinge' && i !== 0 ? 0 : Math.max(joint.limits[i][0], Math.min(joint.limits[i][1], angle)));
}

function multiply(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [aw * bx + ax * bw + ay * bz - az * by,
    aw * by - ax * bz + ay * bw + az * bx,
    aw * bz + ax * by - ay * bx + az * bw,
    aw * bw - ax * bx - ay * by - az * bz];
}

function normalize(q) {
  const length = Math.hypot(...q);
  return q.map(value => value / length);
}

function rotate(q, point) {
  const [x, y, z, w] = q, [px, py, pz] = point;
  const tx = 2 * (y * pz - z * py), ty = 2 * (z * px - x * pz), tz = 2 * (x * py - y * px);
  return [px + w * tx + y * tz - z * ty,
    py + w * ty + z * tx - x * tz,
    pz + w * tz + x * ty - y * tx];
}

function eulerXYZ(angles) {
  const [x, y, z] = angles.map(angle => angle * DEG / 2);
  const sx = Math.sin(x), cx = Math.cos(x), sy = Math.sin(y), cy = Math.cos(y), sz = Math.sin(z), cz = Math.cos(z);
  return [sx * cy * cz - cx * sy * sz, cx * sy * cz + sx * cy * sz,
    cx * cy * sz - sx * sy * cz, cx * cy * cz + sx * sy * sz];
}

/** Joint rotation relative to rest-world axes, after frame and limit handling. */
export function getJointQuaternion(joint, angles = joint.angles) {
  const limited = clampJointAngles(joint, angles);
  if (joint.type === 'fixed') return [...IDENTITY];
  vector(joint.frame, 3, `joint '${joint.id}'.frame`);
  const frame = eulerXYZ(joint.frame);
  let local;
  if (joint.type === 'hinge') {
    vector(joint.axis, 3, `joint '${joint.id}'.axis`);
    const length = Math.hypot(...joint.axis);
    if (Math.abs(length - 1) > 1e-4) fail(`joint '${joint.id}'.axis`, 'hinge axis must have unit length');
    const half = limited[0] * DEG / 2;
    const sine = Math.sin(half) / length;
    local = [...joint.axis.map(value => value * sine), Math.cos(half)];
  } else {
    local = eulerXYZ(limited);
  }
  return normalize(multiply(multiply(frame, local), [-frame[0], -frame[1], -frame[2], frame[3]]));
}

function matrix(q, t) {
  const [x, y, z, w] = q;
  const xx = x * x, xy = x * y, xz = x * z, yy = y * y, yz = y * z, zz = z * z;
  const wx = w * x, wy = w * y, wz = w * z;
  return [1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0,
    2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0,
    2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0,
    t[0], t[1], t[2], 1];
}

/**
 * Return Map<jointId, { rotation, position, matrix, pivot }>.
 * `position` is affine translation, NOT the posed joint center; `pivot` is the
 * posed joint center. Matrices are column-major, quaternions [x, y, z, w].
 * poseById can be a Map or an object of [x,y,z] angle arrays, in degrees.
 * Omitted entries use joint.angles. Input joint order need not be topological.
 */
export function evaluateRig(rig, poseById) {
  validateRig(rig);
  if (poseById !== undefined && (!(poseById instanceof Map) && (!poseById || typeof poseById !== 'object' || Array.isArray(poseById)))) {
    fail('poseById', 'must be a Map or object of angle arrays');
  }
  const joints = new Map(rig.joints.map(joint => [joint.id, joint]));
  const result = new Map();
  function evaluate(joint) {
    if (result.has(joint.id)) return result.get(joint.id);
    const parent = joint.parent === null ? null : evaluate(joints.get(joint.parent));
    const supplied = poseById instanceof Map ? poseById.has(joint.id) : poseById && Object.hasOwn(poseById, joint.id);
    const angles = supplied ? (poseById instanceof Map ? poseById.get(joint.id) : poseById[joint.id]) : joint.angles;
    const local = getJointQuaternion(joint, angles);
    const turnedPivot = rotate(local, joint.pivot);
    const translation = joint.pivot.map((value, i) => value - turnedPivot[i]);
    let rotation = local, position = translation, pivot = [...joint.pivot];
    if (parent) {
      rotation = normalize(multiply(parent.rotation, local));
      position = rotate(parent.rotation, translation).map((value, i) => value + parent.position[i]);
      pivot = rotate(parent.rotation, joint.pivot).map((value, i) => value + parent.position[i]);
    }
    const transform = { rotation, position, matrix: matrix(rotation, position), pivot };
    result.set(joint.id, transform);
    return transform;
  }
  rig.joints.forEach(evaluate);
  return result;
}

/**
 * Exact single-step solution of I*angle'' + damping*angle' +
 * stiffness*(angle-target) = 0 for a target constant during dt.
 * Returns {angle, velocity}. Limits clamp the target and project the result
 * onto hard stops, removing outward velocity. Run at a fixed small dt for
 * responsive changing targets and hard stops; this is independent joint
 * spring dynamics, not a coupled collision/contact rigid-body simulation.
 */
export function stepSpring(angle, velocity, target, stiffness, damping, inertia, dt, min, max) {
  const argumentsByName = { angle, velocity, target, stiffness, damping, inertia, dt, min, max };
  for (const [name, value] of Object.entries(argumentsByName)) finite(value, `spring.${name}`);
  if (stiffness < 0 || damping < 0) fail('spring', 'stiffness and damping must be nonnegative');
  if (inertia <= 0) fail('spring.inertia', 'must be greater than zero');
  if (dt < 0) fail('spring.dt', 'must be nonnegative');
  if (min > max) fail('spring.limits', 'minimum must not exceed maximum');
  target = Math.max(min, Math.min(max, target));
  if (angle < min) { angle = min; if (velocity < 0) velocity = 0; }
  if (angle > max) { angle = max; if (velocity > 0) velocity = 0; }
  if (min === max) return { angle: min, velocity: 0 };
  if (dt === 0) return { angle, velocity };

  let nextAngle, nextVelocity;
  const gamma = damping / inertia;
  const omegaSquared = stiffness / inertia;
  if (!Number.isFinite(gamma) || !Number.isFinite(omegaSquared)) fail('spring', 'coefficient/inertia ratio is too large');
  if (stiffness === 0) {
    const decay = Math.exp(-gamma * dt);
    nextVelocity = velocity * decay;
    nextAngle = angle + velocity * (gamma === 0 ? dt : -Math.expm1(-gamma * dt) / gamma);
  } else {
    const displacement = angle - target;
    const alpha = gamma / 2;
    const discriminant = alpha * alpha - omegaSquared;
    if (!Number.isFinite(discriminant)) fail('spring', 'damping/inertia ratio is too large');
    const tolerance = 1e-10 * Math.max(omegaSquared, alpha * alpha);
    let nextDisplacement;
    if (Math.abs(discriminant) <= tolerance) {
      const decay = Math.exp(-alpha * dt);
      const coefficient = velocity + alpha * displacement;
      nextDisplacement = decay * (displacement + coefficient * dt);
      nextVelocity = decay * (velocity - alpha * coefficient * dt);
    } else if (discriminant < 0) {
      const frequency = Math.sqrt(-discriminant);
      const phase = frequency * dt;
      const cosine = Math.cos(phase), sine = Math.sin(phase), decay = Math.exp(-alpha * dt);
      nextDisplacement = decay * (displacement * cosine + (velocity + alpha * displacement) * sine / frequency);
      nextVelocity = decay * (velocity * cosine - (alpha * velocity + omegaSquared * displacement) * sine / frequency);
    } else {
      const root = Math.sqrt(discriminant);
      // This form of the slow root avoids cancellation for strong damping.
      const slow = -omegaSquared / (alpha + root), fast = -(alpha + root);
      const a = (velocity - fast * displacement) / (slow - fast), b = displacement - a;
      const slowDecay = Math.exp(slow * dt), fastDecay = Math.exp(fast * dt);
      nextDisplacement = a * slowDecay + b * fastDecay;
      nextVelocity = slow * a * slowDecay + fast * b * fastDecay;
    }
    nextAngle = target + nextDisplacement;
  }
  if (!Number.isFinite(nextAngle) || !Number.isFinite(nextVelocity)) fail('spring', 'step exceeds finite numeric range');
  if (nextAngle <= min) { nextAngle = min; if (nextVelocity < 0) nextVelocity = 0; }
  if (nextAngle >= max) { nextAngle = max; if (nextVelocity > 0) nextVelocity = 0; }
  return { angle: nextAngle, velocity: nextVelocity };
}
