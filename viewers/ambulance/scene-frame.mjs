/** One rigid runtime frame for the scan, placements, cameras and contacts.
 * Saved assets/layouts stay in their original source frame. Gravity stays -Y
 * in the leveled runtime frame. Quaternion convention is [x,y,z,w].
 */
export const AMBULANCE_SOURCE_UP = Object.freeze([.0078022315341343415, .9997975329996311, .018547727379366337]);
export const AMBULANCE_LEVEL_ROTATION = Object.freeze([-.009274333138167175, 0, .00390131324386261, .9999493819688152]);
export const AMBULANCE_LEVEL_EVIDENCE = Object.freeze({
  tiltDegrees: 1.152980818840982,
  source: 'Equal-weight normalized mean of measured reference-floor and front-roof normals; their disagreement is 0.347134 degrees.',
  floor: 'raw/ambulance-cleanup/pass6/wall-diagnostics/floor-independent-fit.json',
  roof: 'raw/ambulance-cleanup/pass7/diagnostics/front-roof-near-sheet-evidence.json',
  units: 'uncalibrated scene units'
});
const vector = (v, size, label) => {
  if ((!Array.isArray(v) && !ArrayBuffer.isView(v)) || v.length !== size || !Array.from(v).every(Number.isFinite)) throw new TypeError(`${label} needs ${size} finite values`);
  return Array.from(v);
};
function quaternion(v) {
  const q = vector(v, 4, 'Quaternion'), n = Math.hypot(...q);
  if (n < 1e-12) throw new RangeError('Quaternion cannot have zero length');
  return q.map(x => x / n);
}
function rotate(q, v) {
  const [x, y, z, w] = q, [a, b, c] = v;
  const tx = 2 * (y * c - z * b), ty = 2 * (z * a - x * c), tz = 2 * (x * b - y * a);
  return [a + w * tx + y * tz - z * ty, b + w * ty + z * tx - x * tz, c + w * tz + x * ty - y * tx];
}
function multiply(a, b) {
  const [x, y, z, w] = a, [X, Y, Z, W] = b;
  return [w * X + x * W + y * Z - z * Y, w * Y - x * Z + y * W + z * X,
    w * Z + x * Y - y * X + z * W, w * W - x * X - y * Y - z * Z];
}

export function createSceneFrame(rotation = [0, 0, 0, 1]) {
  const q = quaternion(rotation ?? [0, 0, 0, 1]), inverse = [-q[0], -q[1], -q[2], q[3]];
  const identity = Math.hypot(q[0], q[1], q[2]) < 1e-14;
  const point = p => rotate(q, vector(p, 3, 'Point'));
  const inversePoint = p => rotate(inverse, vector(p, 3, 'Point'));
  function transformPlacement(value, reverse = false) {
    if (!value || typeof value !== 'object') throw new TypeError('Placement is required');
    if (value.scale !== undefined && (!Number.isFinite(value.scale) || value.scale <= 0)) throw new RangeError('Placement scale must be positive and finite');
    const r = reverse ? inverse : q;
    return { ...value, position: rotate(r, vector(value.position, 3, 'Placement position')),
      rotation: quaternion(multiply(r, quaternion(value.rotation))) };
  }
  const camera = value => ({ ...value, position: point(value.position), target: point(value.target) });
  function settings(input) {
    const out = structuredClone(input);
    if (identity) return out;
    if (out.camera?.position && out.camera?.target) out.camera = camera(out.camera);
    for (const entry of out.cameras ?? []) if (entry.initial?.position && entry.initial?.target) entry.initial = camera(entry.initial);
    for (const track of out.animTracks ?? []) {
      for (const key of ['position', 'target']) {
        const values = track.keyframes?.values?.[key];
        if (!values) continue;
        if (values.length % 3 !== 0) throw new TypeError('Camera animation XYZ values must have a multiple of three entries');
        for (let i = 0; i < values.length; i += 3) values.splice(i, 3, ...point(values.slice(i, i + 3)));
      }
    }
    for (const annotation of out.annotations ?? []) {
      if (annotation.position) annotation.position = point(annotation.position);
      if (annotation.camera?.position && annotation.camera?.target) annotation.camera = camera(annotation.camera);
    }
    return out;
  }
  function worldReport(report) {
    if (!report || typeof report !== 'object') return report;
    const contact = c => ({ ...c, ...(c.point ? { point: point(c.point) } : {}), ...(c.normal ? { normal: point(c.normal) } : {}) });
    return { ...report,
      ...(report.contacts ? { contacts: report.contacts.map(contact) } : {}),
      ...(report.violations ? { violations: report.violations.map(contact) } : {}),
      ...(report.report ? { report: worldReport(report.report) } : {}),
      ...(report.blockingReport ? { blockingReport: worldReport(report.blockingReport) } : {}) };
  }
  function wrapContactChecker(checker) {
    if (typeof checker?.evaluate !== 'function') throw new TypeError('A contact checker is required');
    const sourceSamples = samples => Array.from(samples, sample => ({ ...sample, position: inversePoint(sample.position) }));
    const wrapper = { ...checker, evaluate: (samples, ...args) => worldReport(checker.evaluate(sourceSamples(samples), ...args)) };
    if (typeof checker.evaluatePath === 'function') wrapper.evaluatePath = (sampleAt, ...args) => worldReport(checker.evaluatePath(t => sourceSamples(sampleAt(t)), ...args));
    return wrapper;
  }
  return Object.freeze({ rotation: Object.freeze([...q]), sourceUp: Object.freeze(inversePoint([0, 1, 0])), identity,
    point, inversePoint, direction: point, inverseDirection: inversePoint,
    placement: value => transformPlacement(value), inversePlacement: value => transformPlacement(value, true),
    camera, settings, wrapContactChecker });
}
