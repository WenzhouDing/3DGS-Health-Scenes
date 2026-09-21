/** Prepared demonstrations, in scan units. These are posed rigid parts, not a
 * gravity/contact simulation. The support is a simple bed, never a scan asset.
 * Existing annotation limits are always retained, including rigid wrists.
 */
import {dofCount, transformPoint} from './motion-core.mjs';

const DEG = Math.PI / 180;
const IDENTITY = [0, 0, 0, 1];
export const PRESET_REVISION = 2;
export const PRESETS = Object.freeze([
  Object.freeze({id: 'scan', label: 'Scanned pose', description: 'The original fused scan.'}),
  Object.freeze({id: 'lying', label: 'Lie flat on bed', description: 'Face up, with arms and legs brought alongside the body.'}),
  Object.freeze({id: 'sitting', label: 'Sit on bed', description: 'Upper body raised on the backrest, with both legs extended along the bed.'}),
]);
const sub = (a, b) => a.map((v, i) => v - b[i]);
const add = (a, b) => a.map((v, i) => v + b[i]);
const unit = v => { const d = Math.hypot(...v); if (d < 1e-10) throw new Error('The preset needs distinct joint centers.'); return v.map(x => x / d); };
const dot = (a, b) => a.reduce((n, x, i) => n + x * b[i], 0);
const clamp = (x, range) => Math.max(range[0], Math.min(range[1], x));
function multiply(a, b) {
  const [ax, ay, az, aw] = a, [bx, by, bz, bw] = b;
  return [aw*bx+ax*bw+ay*bz-az*by, aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw, aw*bw-ax*bx-ay*by-az*bz];
}
function quaternion(axis, degrees) { const s = Math.sin(degrees * DEG / 2); return [...unit(axis).map(x => x * s), Math.cos(degrees * DEG / 2)]; }
function rotate(q, v) { return transformPoint({position: [0,0,0], rotation: q}, v); }
function euler(a) { return multiply(quaternion([0,0,1], a[2]), multiply(quaternion([0,1,0], a[1]), quaternion([1,0,0], a[0]))); }
function matrix(q, p) {
  const [x,y,z] = [[1,0,0],[0,1,0],[0,0,1]].map(v => rotate(q,v));
  return [x[0],x[1],x[2],0,y[0],y[1],y[2],0,z[0],z[1],z[2],0,...p,1];
}

/** Apply a display pose after the anatomical hierarchy, exactly once. Shared
 * part transforms remain shared (the forearm and hand cannot split apart).
 */
export function composePresentation(evaluated, presentation) {
  const p = presentation?.position ?? [0,0,0], q = presentation?.rotation ?? IDENTITY;
  if (!Array.isArray(p) || p.length !== 3 || !p.every(Number.isFinite)
      || !Array.isArray(q) || q.length !== 4 || !q.every(Number.isFinite) || Math.abs(Math.hypot(...q)-1) > 1e-6) {
    throw new TypeError('The presentation needs a finite position and unit quaternion.');
  }
  const cache = new Map();
  function compose(value) {
    if (cache.has(value)) return cache.get(value);
    const rotation = multiply(q, value.rotation), position = add(rotate(q, value.position), p);
    const result = {...value, position, rotation, matrix: matrix(rotation, position)};
    if (value.pivot) result.pivot = add(rotate(q, value.pivot), p);
    if (value.axis) result.axis = rotate(q, value.axis);
    if (value.axes) result.axes = value.axes.map(axis => rotate(q, axis));
    cache.set(value, result); return result;
  }
  return {...evaluated, parts: new Map([...evaluated.parts].map(([id,v]) => [id,compose(v)])),
    joints: new Map([...evaluated.joints].map(([id,v]) => [id,compose(v)]))};
}

// A small deterministic constrained fit uses actual annotated centerlines. It
// adjusts only requested angles and never widens a limit or moves a pivot.
function fitDirections(starts, limits, loss) {
  let best = null;
  for (const start of starts) {
    let a = start.map((v,i) => clamp(v,limits[i]));
    for (const step of [16,8,4,2,1,.5,.2,.1,.05,.02,.01]) {
      for (let iteration = 0; iteration < 70; iteration++) {
        let changed = false;
        for (let i = 0; i < a.length; i++) {
          let score = loss(a), chosen = a;
          for (const delta of [-step,step]) {
            const candidate = a.slice(); candidate[i] = clamp(candidate[i]+delta,limits[i]);
            const next = loss(candidate);
            if (next < score-1e-13) { score=next; chosen=candidate; changed=true; }
          }
          a=chosen;
        }
        if (!changed) break;
      }
    }
    const score=loss(a); if (!best || score<best.score) best={a,score};
  }
  return best.a.map(v => Number(v.toFixed(2)));
}

/** Build independent targets from either the annotation document or createMotion
 * result. No mutation of its state or definitions occurs. Call motion.reset()
 * before applying a preset; scan should use reset alone for exact identity.
 */
export function buildPreset(id, annotationsOrMotion, manifest) {
  const preset = PRESETS.find(p => p.id === id);
  if (!preset) throw new TypeError(`Unknown prepared pose: ${String(id)}.`);
  const joints = annotationsOrMotion?.joints;
  if (!Array.isArray(joints) || !joints.length) throw new TypeError('A prepared pose needs a joint map.');
  const byId = new Map(joints.map(j => [j.id,j]));
  const targets = Object.fromEntries(joints.map(j => [j.id,[0,0,0]]));
  const presentation = {position:[0,0,0],rotation:[...IDENTITY],bed:null,camera:{yaw:55,pitch:12}};
  const result = {...preset,targets,presentation,note:''};
  if (id === 'scan') return result;
  const need = name => { const j=byId.get(name); if (!j) throw new Error(`This pose needs the ${name.replaceAll('_',' ')} joint.`); return j; };
  const set = (name, angles) => { const j=need(name), dof=dofCount(j); targets[name]=angles.map((v,i)=>i<dof?clamp(v,j.limits[i]):0); };
  // Keep an inactive, edited angle within the current map's travel range too.
  for (const j of joints) set(j.id,[0,0,0]);
  const surfaceZ=.66, thickness=.12;
  if (id === 'lying') {
    for (const side of ['left','right']) {
      const sign=side==='left'?1:-1, shoulder=need(side+'_shoulder'), swivel=need(side+'_arm_swivel');
      const arm=sub(swivel.pivot,shoulder.pivot);
      set(shoulder.id,[0,Math.atan2(arm[0],-arm[2])/DEG-sign*12,0]);
      const hip=need(side+'_hip'), knee=need(side+'_knee'), thigh=sub(knee.pivot,hip.pivot);
      set(hip.id,[0,Math.atan2(thigh[0],-thigh[2])/DEG,0]);
    }
    const bounds=manifest?.bounds;
    if (!bounds || !bounds.min?.every(Number.isFinite) || !bounds.max?.every(Number.isFinite)) throw new Error('The lying pose needs the scan bounds.');
    // Y-only arm/hip adduction preserves every splat center's rest Y. Under
    // +90° X rotation, rest Y is height, so the global minimum gives clearance.
    presentation.rotation=quaternion([1,0,0],90);
    presentation.position=[0,(bounds.min[2]+bounds.max[2])/2,surfaceZ-bounds.min[1]+.008];
    presentation.bed={center:[0,0,surfaceZ-thickness/2],size:[1.08,bounds.max[2]-bounds.min[2]+.30,thickness],frameHeight:.48};
    presentation.camera={yaw:38,pitch:34};
    result.note='Lying flat · Reference bed';
    return result;
  }

  // Raise the upper body 45° above the mattress while the hips compensate to
  // leave both legs extended along it. Only a small correction at each measured
  // knee axis is allowed, retaining the original scan's nearly straight legs.
  const recline=45, root=quaternion([1,0,0],recline);
  presentation.rotation=root;
  let maxMisalignment=0;
  for (const side of ['left','right']) {
    const sign=side==='left'?1:-1, hip=need(side+'_hip'), knee=need(side+'_knee'), ankle=need(side+'_ankle');
    const thigh=unit(sub(knee.pivot,hip.pivot)), shin=unit(sub(ankle.pivot,knee.pivot));
    const directions=a=>[rotate(root,rotate(euler(a),thigh)),rotate(root,rotate(euler(a),rotate(quaternion(knee.axis,a[3]),shin)))];
    const loss=a=>{const[t,s]=directions(a);return 1-t[1]+1-s[1];};
    const straightKneeRange=[Math.max(-5,knee.limits[0][0]),Math.min(10,knee.limits[0][1])];
    const kneeRange=straightKneeRange[0]<=straightKneeRange[1]?straightKneeRange:knee.limits[0];
    const angles=fitDirections([[45,0,0,0],[45,sign*25,0,0],[45,0,sign*25,0]], [...hip.limits,kneeRange],loss);
    set(hip.id,angles.slice(0,3)); set(knee.id,[angles[3],0,0]);
    const [t,s]=directions(angles); maxMisalignment=Math.max(maxMisalignment,Math.acos(clamp(t[1],[-1,1]))/DEG,Math.acos(clamp(s[1],[-1,1]))/DEG);
    // Straight rigid arms reach forward beside the upper thighs; no invented
    // elbow flexion or wrist pitch is used to pose the hands.
    const shoulder=need(side+'_shoulder'), swivel=need(side+'_arm_swivel');
    const arm=unit(sub(swivel.pivot,shoulder.pivot)), desired=unit([sign*.25,.82,-.54]);
    const armAngles=fitDirections([[0,sign*40,0],[-20,sign*40,0]],shoulder.limits,a=>1-dot(rotate(root,rotate(euler(a),arm)),desired));
    set(shoulder.id,armAngles);
  }
  const hips=['left','right'].map(side=>need(side+'_hip').pivot), hipCenter=hips[0].map((v,i)=>(v+hips[1][i])/2);
  // Calibrated against the opaque thigh points in both captures. Hip centers
  // sit 0.142 scan units above the mattress, clearing the rigid legs. The long
  // loose-pants shell still owned by the fixed pelvis cannot follow this bend;
  // its known overlap is quantified in the full-capture test, not hidden by
  // raising the entire body and leaving the legs suspended above the bed.
  const seatHeight=surfaceZ+.142;
  presentation.position=sub([0,0,seatHeight],rotate(root,hipCenter));
  // A full-length mattress continues beneath the straight lower legs and feet.
  const bedLength=2.32;
  presentation.bed={center:[0,.04,surfaceZ-thickness/2],size:[1.08,bedLength,thickness],frameHeight:.48};
  // Backrest normal matches the torso's front normal; its top face is just
  // behind the posterior torso. A world-space slab rotates -45° about X.
  const backrestLength=1.04, backrestThickness=.10;
  const normal=rotate(root,[0,1,0]), headward=rotate(root,[0,0,1]);
  const lower=add(rotate(root,[hipCenter[0],manifest.bounds.min[1]-.008,hipCenter[2]-.015]),presentation.position);
  const backCenter=add(add(lower,headward.map(v=>v*backrestLength/2)),normal.map(v=>-v*backrestThickness/2));
  presentation.bed.backrest={center:backCenter,size:[.80,backrestLength,backrestThickness],rotation:quaternion([1,0,0],recline-90)};
  presentation.camera={yaw:38,pitch:28};
  result.note='Raised back · Legs extended · Reference bed';
  if (maxMisalignment>3) result.note+=' Edited joint limits restrict this pose; adjust it in the joint controls.';
  result.analysis={maximumLegDirectionErrorDegrees:Number(maxMisalignment.toFixed(3)),reclineDegrees:recline,backrestElevationDegrees:90-recline};
  return result;
}
