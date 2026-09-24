/** Sampled body-to-scene contacts against measured world-space proxies.
 * Units are uncalibrated ambulance scene units. This is not self-collision or
 * continuous solid-body simulation. Call at intermediate joint poses to avoid
 * stepping through obstacles. Geometry is independent of Gaussian haze.
 */
const EPS = 1e-10;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const add = (a, b) => a.map((v, i) => v + b[i]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const mul = (a, s) => a.map(v => v * s);
const norm = a => Math.hypot(...a);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

function finite(value, label) {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new TypeError(`${label} must be finite`);
  return value;
}
function vector(value, label, length = 3) {
  if (!value || value.length !== length) throw new TypeError(`${label} requires ${length} coordinates`);
  return Array.from(value, (v, i) => finite(v, `${label}[${i}]`));
}
function nonnegative(value, label) {
  finite(value, label);
  if (value < 0) throw new RangeError(`${label} must be nonnegative`);
  return value;
}
function rotate(q, p) {
  const u = q.slice(0, 3), t = mul(cross(u, p), 2);
  return add(p, add(mul(t, q[3]), cross(u, t)));
}

/** Convex intersection of a plane with an axis-aligned finite bounds box. */
export function planePolygon(normal, offset, bounds) {
  const [lo, hi] = bounds;
  const vertices = [];
  const keep = p => {
    if (!vertices.some(v => norm(sub(v, p)) < 1e-8)) vertices.push(p);
  };
  const corners = Array.from({ length: 8 }, (_, bits) => [0, 1, 2].map(i => (bits & (1 << i)) ? hi[i] : lo[i]));
  for (let a = 0; a < 8; a++) for (let axis = 0; axis < 3; axis++) {
    const b = a ^ (1 << axis);
    if (b < a) continue;
    const da = dot(normal, corners[a]) - offset, db = dot(normal, corners[b]) - offset;
    if (Math.abs(da) < EPS) keep(corners[a]);
    if (Math.abs(db) < EPS) keep(corners[b]);
    if (da * db < 0) keep(add(corners[a], mul(sub(corners[b], corners[a]), da / (da - db))));
  }
  if (vertices.length < 3) throw new RangeError('Surface plane must intersect bounds in a polygon');
  const center = mul(vertices.reduce((sum, p) => add(sum, p), [0, 0, 0]), 1 / vertices.length);
  const seed = Math.abs(normal[0]) < .8 ? [1, 0, 0] : [0, 1, 0];
  const a = cross(normal, seed), u = mul(a, 1 / norm(a)), v = cross(normal, u);
  return vertices.sort((p, q) => Math.atan2(dot(sub(p, center), v), dot(sub(p, center), u)) - Math.atan2(dot(sub(q, center), v), dot(sub(q, center), u)));
}

// Scratch hit is reused for every broad-phase query. Only actual contacts
// allocate output objects, so 10k+ surface samples remain practical in a sweep.
function surfaceContact(p, surface, limit, hit) {
  const n = surface.normal, bounds = surface.bounds;
  const d = n[0] * p[0] + n[1] * p[1] + n[2] * p[2] - surface.offset;
  if (d > limit) return false;
  const x = p[0] - n[0] * d, y = p[1] - n[1] * d, z = p[2] - n[2] * d;
  if (x >= bounds[0][0] - EPS && x <= bounds[1][0] + EPS &&
      y >= bounds[0][1] - EPS && y <= bounds[1][1] + EPS &&
      z >= bounds[0][2] - EPS && z <= bounds[1][2] + EPS) {
    hit.distance = d; hit.nx = n[0]; hit.ny = n[1]; hit.nz = n[2];
    hit.x = x; hit.y = y; hit.z = z; hit.support = surface.kind === 'support';
    return true;
  }
  // A finite edge can touch a sphere outside the face footprint, but it cannot
  // touch one outside its padded bounding box. Negative half-space extension
  // is deliberately restricted to the actual face footprint above.
  if (p[0] < bounds[0][0] - limit || p[0] > bounds[1][0] + limit ||
      p[1] < bounds[0][1] - limit || p[1] > bounds[1][1] + limit ||
      p[2] < bounds[0][2] - limit || p[2] > bounds[1][2] + limit) return false;
  let best = limit * limit + EPS, found = false;
  const vertices = surface.polygon;
  for (let i = 0; i < vertices.length; i++) {
    const a = vertices[i], b = vertices[(i + 1) % vertices.length];
    const ex = b[0] - a[0], ey = b[1] - a[1], ez = b[2] - a[2];
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * ex + (p[1] - a[1]) * ey + (p[2] - a[2]) * ez) / (ex * ex + ey * ey + ez * ez)));
    const qx = a[0] + t * ex, qy = a[1] + t * ey, qz = a[2] + t * ez;
    const dx = p[0] - qx, dy = p[1] - qy, dz = p[2] - qz, squared = dx * dx + dy * dy + dz * dz;
    if (squared < best) {
      best = squared; found = true;
      hit.x = qx; hit.y = qy; hit.z = qz;
      const length = Math.sqrt(squared);
      hit.nx = length > EPS ? dx / length : n[0];
      hit.ny = length > EPS ? dy / length : n[1];
      hit.nz = length > EPS ? dz / length : n[2];
    }
  }
  hit.distance = Math.sqrt(best); hit.support = false;
  return found && hit.distance <= limit + EPS;
}

function boxContact(p, box, limit, hit) {
  const bounds = box.bounds;
  if (p[0] < bounds[0][0] - limit || p[0] > bounds[1][0] + limit ||
      p[1] < bounds[0][1] - limit || p[1] > bounds[1][1] + limit ||
      p[2] < bounds[0][2] - limit || p[2] > bounds[1][2] + limit) return false;
  const axes = box.axes, h = box.halfExtents;
  const px = p[0] - box.center[0], py = p[1] - box.center[1], pz = p[2] - box.center[2];
  const x = axes[0][0] * px + axes[0][1] * py + axes[0][2] * pz;
  const y = axes[1][0] * px + axes[1][1] * py + axes[1][2] * pz;
  const z = axes[2][0] * px + axes[2][1] * py + axes[2][2] * pz;
  let qx = Math.max(-h[0], Math.min(h[0], x)), qy = Math.max(-h[1], Math.min(h[1], y)), qz = Math.max(-h[2], Math.min(h[2], z));
  const dx = x - qx, dy = y - qy, dz = z - qz, outside = Math.hypot(dx, dy, dz);
  if (outside > limit) return false;
  let nx, ny, nz, distance;
  if (outside > EPS) { nx = dx / outside; ny = dy / outside; nz = dz / outside; distance = outside; }
  else {
    const gx = h[0] - Math.abs(x), gy = h[1] - Math.abs(y), gz = h[2] - Math.abs(z);
    nx = ny = nz = 0;
    if (gx <= gy && gx <= gz) { nx = x >= 0 ? 1 : -1; qx = nx * h[0]; distance = -gx; }
    else if (gy <= gz) { ny = y >= 0 ? 1 : -1; qy = ny * h[1]; distance = -gy; }
    else { nz = z >= 0 ? 1 : -1; qz = nz * h[2]; distance = -gz; }
  }
  hit.distance = distance; hit.support = box.kind === 'mattress' && ny > .999 && y >= 0;
  hit.nx = axes[0][0] * nx + axes[1][0] * ny + axes[2][0] * nz;
  hit.ny = axes[0][1] * nx + axes[1][1] * ny + axes[2][1] * nz;
  hit.nz = axes[0][2] * nx + axes[1][2] * ny + axes[2][2] * nz;
  hit.x = box.center[0] + axes[0][0] * qx + axes[1][0] * qy + axes[2][0] * qz;
  hit.y = box.center[1] + axes[0][1] * qx + axes[1][1] * qy + axes[2][1] * qz;
  hit.z = box.center[2] + axes[0][2] * qx + axes[1][2] * qy + axes[2][2] * qz;
  return true;
}

/**
 * config: {surfaces:[{id,normal,offset,bounds,kind:'boundary'|'support'}],
 * boxes:[{id,center,halfExtents,rotation?:xyzw,kind:'solid'|'mattress'}],
 * tolerances:{penetration,contact}}. Plane allowedPenetration overrides its
 * default tolerance. For mattress boxes it applies only to the top support;
 * side/bottom contacts retain the ordinary solid tolerance.
 */
export function createContactChecker(input) {
  const config = input?.collision ?? input;
  if (!config || typeof config !== 'object') throw new TypeError('Collision config is required');
  const tolerance = nonnegative(config.tolerances?.penetration ?? .003, 'penetration tolerance');
  const contactDistance = nonnegative(config.tolerances?.contact ?? .008, 'contact distance');
  const used = new Set();
  const common = (p, type) => {
    if (typeof p.id !== 'string' || !p.id || used.has(p.id)) throw new TypeError('Proxy IDs must be unique nonempty strings');
    used.add(p.id);
    return { ...p, type, label: p.label ?? p.id,
      allowedPenetration: nonnegative(p.allowedPenetration ?? tolerance, `${p.id} support allowance`),
      contactDistance: nonnegative(p.contactDistance ?? contactDistance, `${p.id} contact distance`) };
  };
  const surfaces = (config.surfaces ?? []).map(p => {
    const s = common(p, 'surface'), n = vector(p.normal, `${p.id} normal`), length = norm(n);
    if (length < EPS) throw new RangeError('Surface normal cannot be zero');
    s.normal = mul(n, 1 / length); s.offset = finite(p.offset, `${p.id} offset`) / length;
    if (!Array.isArray(p.bounds) || p.bounds.length !== 2) throw new TypeError('Surface bounds require min and max');
    s.bounds = p.bounds.map((b, i) => vector(b, `${p.id} bounds ${i}`));
    if (s.bounds[0].some((v, i) => v > s.bounds[1][i])) throw new RangeError('Reversed surface bounds');
    s.kind = p.kind ?? 'boundary';
    if (!['boundary', 'support'].includes(s.kind)) throw new RangeError('Unknown surface kind');
    s.polygon = planePolygon(s.normal, s.offset, s.bounds);
    return s;
  });
  const boxes = (config.boxes ?? []).map(p => {
    const b = common(p, 'box');
    b.center = vector(p.center, `${p.id} center`); b.halfExtents = vector(p.halfExtents, `${p.id} halfExtents`);
    if (b.halfExtents.some(v => v <= 0)) throw new RangeError('Box halfExtents must be positive');
    const q = vector(p.rotation ?? [0, 0, 0, 1], `${p.id} rotation`, 4), length = norm(q);
    if (length < EPS) throw new RangeError('Box quaternion cannot be zero');
    b.rotation = mul(q, 1 / length); b.kind = p.kind ?? 'solid';
    b.axes = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map(axis => rotate(b.rotation, axis));
    const extent = [0, 1, 2].map(i => b.axes.reduce((sum, axis, j) => sum + Math.abs(axis[i]) * b.halfExtents[j], 0));
    b.bounds = [sub(b.center, extent), add(b.center, extent)];
    if (!['solid', 'mattress'].includes(b.kind)) throw new RangeError('Unknown box kind');
    return b;
  });
  const proxies = [...surfaces, ...boxes];
  if (!proxies.length) throw new RangeError('At least one collision proxy is required');

  function evaluate(samples, { stopOnBlock = false } = {}) {
    if (!samples || typeof samples[Symbol.iterator] !== 'function') throw new TypeError('Collision samples must be iterable');
    const contacts = [], violations = [], byPart = Object.create(null), hit = {};
    let count = 0, maxPenetration = 0, minimumContactClearance = Infinity, complete = true;
    outer: for (const sample of samples) {
      const p = sample.position;
      if (!p || p.length !== 3 || !Number.isFinite(p[0]) || !Number.isFinite(p[1]) || !Number.isFinite(p[2])) throw new TypeError('sample position must have three finite coordinates');
      const radius = nonnegative(sample.radius ?? 0, 'sample radius');
      const part = String(sample.part ?? 'body'), id = sample.id ?? count;
      count++;
      byPart[part] ??= { contacts: 0, violations: 0, maxPenetration: 0 };
      for (const proxy of proxies) {
        const limit = radius + proxy.contactDistance;
        if (!(proxy.type === 'surface' ? surfaceContact(p, proxy, limit, hit) : boxContact(p, proxy, limit, hit))) continue;
        const clearance = hit.distance - radius, penetration = Math.max(0, -clearance);
        minimumContactClearance = Math.min(minimumContactClearance, clearance);
        const allowance = hit.support || proxy.type === 'surface' ? proxy.allowedPenetration : tolerance;
        const blocked = penetration > allowance + EPS;
        const contact = { sampleId: id, part, proxy: proxy.id, label: proxy.label, support: hit.support,
          penetration, clearance, allowedPenetration: allowance,
          normal: [hit.nx, hit.ny, hit.nz], point: [hit.x, hit.y, hit.z], blocked };
        contacts.push(contact);
        maxPenetration = Math.max(maxPenetration, penetration);
        byPart[part].contacts++;
        byPart[part].maxPenetration = Math.max(byPart[part].maxPenetration, penetration);
        if (blocked) {
          violations.push(contact); byPart[part].violations++;
          if (stopOnBlock) { complete = false; break outer; }
        }
      }
    }
    return { status: !count ? 'unavailable' : violations.length ? 'penetrating' : contacts.length ? 'contact' : 'clear',
      blocked: !count || violations.length > 0, sampleCount: count, maxPenetration, complete,
      minimumContactClearance: Number.isFinite(minimumContactClearance) ? minimumContactClearance : null,
      contacts, violations, byPart, scope: 'sampled body versus scene proxies; no self-collision' };
  }

  /** Pure callback, sampled at all steps. Caller supplies actual intermediate
   * articulated poses, not linear interpolation of the endpoint sample clouds.
   * This is sampled motion checking, not a continuous collision guarantee.
   */
  function evaluatePath(sampleAt, { steps = 16, refinementSteps = 8 } = {}) {
    if (typeof sampleAt !== 'function') throw new TypeError('sampleAt must be a function');
    if (!Number.isInteger(steps) || steps < 1 || steps > 4096) throw new RangeError('steps must be 1..4096');
    if (!Number.isInteger(refinementSteps) || refinementSteps < 0 || refinementSteps > 24) throw new RangeError('refinementSteps must be 0..24');
    let last = evaluate(sampleAt(0)), safe = 0;
    if (last.blocked) return { acceptedFraction: 0, blocked: true, report: last, blockingReport: last };
    for (let i = 1; i <= steps; i++) {
      let t = i / steps, next = evaluate(sampleAt(t));
      if (next.blocked) {
        const blockingReport = next;
        for (let j = 0; j < refinementSteps; j++) {
          const mid = (safe + t) / 2, probe = evaluate(sampleAt(mid));
          if (probe.blocked) t = mid; else { safe = mid; last = probe; }
        }
        return { acceptedFraction: safe, blocked: true, report: last, blockingReport };
      }
      safe = t; last = next;
    }
    return { acceptedFraction: 1, blocked: false, report: last, blockingReport: null };
  }
  return { evaluate, evaluatePath, proxyCount: proxies.length };
}
