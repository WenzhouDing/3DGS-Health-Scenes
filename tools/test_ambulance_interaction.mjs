import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {createMotion, transformPoint, dofCount} from '../viewers/mannequin-articulation/motion-core.mjs';
import {jointWorldFrame, rotationIncrement, createManikinInteraction} from '../viewers/ambulance/manikin-interaction.mjs';

const json = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const manifest = json('../viewers/mannequin-fusion/fusion.json');
const annotations = json('../viewers/mannequin-articulation/joint-refinement.json').annotations;
const motion = createMotion(annotations, manifest);
const dot = (a, b) => a.reduce((sum, n, i) => sum + n * b[i], 0);
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const near = (a, b, tolerance = 1e-7) => a.forEach((n, i) => assert.ok(Math.abs(n - b[i]) < tolerance, `${a} != ${b}`));
const placement = {position: [1.2, -.4, .7], rotation: [0, Math.sin(.37), 0, Math.cos(.37)], scale: .66};
const place = p => transformPoint(placement, p.map(n => n * placement.scale));
for (const joint of motion.joints) {
    if (dofCount(joint)) motion.setTarget(joint.id, joint.limits.map(([lo, hi]) => .5 * (lo + hi)), {immediate: true});
}
let checkedAxes = 0;
for (const joint of motion.joints) for (let axis = 0; axis < dofCount(joint); axis++) {
    const before = motion.evaluate(), angles = motion.getState(joint.id).angles;
    const frame = jointWorldFrame(joint, before.joints.get(joint.id), angles, placement, axis);
    near(frame.center, place(before.joints.get(joint.id).pivot));
    near([dot(frame.normal, frame.u), dot(frame.normal, frame.v), dot(frame.u, frame.v)], [0, 0, 0]);
    near(cross(frame.u, frame.v), frame.normal);
    const rest = joint.pivot.map((n, i) => n + [.11, -.07, .05][i]), child = joint.childParts[0];
    const p0 = place(transformPoint(before.parts.get(child), rest));
    const deltaDegrees = .0001, next = angles.slice(); next[axis] += deltaDegrees;
    motion.setTarget(joint.id, next, {immediate: true});
    const p1 = place(transformPoint(motion.evaluate().parts.get(child), rest));
    motion.setTarget(joint.id, angles, {immediate: true});
    const numericalVelocity = p1.map((n, i) => (n - p0[i]) / (deltaDegrees * Math.PI / 180));
    const predictedVelocity = cross(frame.normal, p0.map((n, i) => n - frame.center[i]));
    near(numericalVelocity, predictedVelocity, 1e-5); checkedAxes++;
}
near([rotationIncrement(179 * Math.PI / 180, -179 * Math.PI / 180)], [2 * Math.PI / 180]);
near([rotationIncrement(-179 * Math.PI / 180, 179 * Math.PI / 180)], [-2 * Math.PI / 180]);

// Minimal DOM/engine doubles exercise interaction ownership and event lifecycle,
// while projection/rotation equations above use the actual fitted rig.
class Element {
    constructor(name) {
        this.name = name; this.style = {}; this.dataset = {}; this.attrs = {}; this.children = []; this.listeners = new Map(); this.captures = new Set();
        const classes = new Set(); this.classList = {add: name => classes.add(name), remove: name => classes.delete(name),
            toggle: (name, yes = !classes.has(name)) => yes ? classes.add(name) : classes.delete(name)};
    }
    setAttribute(key, value) {this.attrs[key] = value;}
    append(...children) {for (const child of children) {this.children.push(child); child.parent = this;}}
    remove() {this.parent?.children.splice(this.parent.children.indexOf(this), 1);}
    addEventListener(name, callback) {if (!this.listeners.has(name)) this.listeners.set(name, new Set()); this.listeners.get(name).add(callback);}
    removeEventListener(name, callback) {this.listeners.get(name)?.delete(callback);}
    fire(name, props = {}) {
        const event = {type: name, currentTarget: this, preventDefault() {this.prevented = true;}, stopPropagation() {this.stopped = true;}, stopImmediatePropagation() {this.immediateStopped = true;}, ...props};
        for (const callback of this.listeners.get(name) ?? []) callback(event);
        return event;
    }
    setPointerCapture(id) {this.captures.add(id);}
    hasPointerCapture(id) {return this.captures.has(id);}
    releasePointerCapture(id) {this.captures.delete(id); this.fire('lostpointercapture', {pointerId: id});}
    focus() {}
    getBoundingClientRect() {return {left: 15, top: 25, width: 400, height: 300};}
}
class Vec3 {
    constructor(x = 0, y = 0, z = 0) {this.set(x, y, z);}
    set(x, y, z) {this.x = x; this.y = y; this.z = z; return this;}
}
const oldDocument = globalThis.document, oldWindow = globalThis.window;
const body = new Element('body'), window = new Element('window'), canvas = new Element('canvas'), appEvents = new Map();
globalThis.document = {body, createElement: name => new Element(name), createElementNS: (namespace, name) => new Element(name)};
globalThis.window = window;
try {
    const camera = {nearClip: .01, farClip: 100,
        worldToScreen(p, out) {return out.set(400 + p.x * 100, 300 - p.y * 100, p.z);},
        screenToWorld(x, y, depth, out) {return out.set((x - 400) / 100, (300 - y) / 100, 4 - depth);}};
    const cameraEntity = {camera, getPosition: () => new Vec3(0, 0, 4), forward: new Vec3(0, 0, -1)};
    const app = {graphicsDevice: {canvas, clientRect: {width: 800, height: 600}},
        on(name, callback) {appEvents.set(name, callback);}, off(name, callback) {assert.equal(appEvents.get(name), callback); appEvents.delete(name);}};
    let selected = 'left_shoulder', at = {position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: 1};
    const requests = [], starts = [], ends = [], cancelledTravel = [];
    let heldJoint = null, falling = false, fallSteps = 0;
    const frame = () => {if (falling && !heldJoint) fallSteps++; appEvents.get('postrender')();};
    const controls = createManikinInteraction({viewer: {global: {app, camera: cameraEntity, events: {fire: name => cancelledTravel.push(name)}}, cameraManager: {snap() {}}},
        pc: {Vec3}, motion, getPlacement: () => at, getSelected: () => selected,
        onSelect: id => {selected = id;}, onAngle: (...args) => requests.push(args),
        onGestureStart: (...args) => {starts.push(args); heldJoint = args[0]; falling = false;},
        onGestureEnd: result => {ends.push(result); heldJoint = null; falling = !result.cancelled;}});
    const overlay = body.children[0], button = overlay.children.find(child => child.dataset.joint === selected);
    assert.equal(overlay.hidden, true, 'Explore mode is noninteractive by default');
    controls.setEnabled(true);
    const pivot = motion.evaluate().joints.get(selected).pivot;
    near([parseFloat(button.style.left), parseFloat(button.style.top)], [(400 + pivot[0] * 100) / 2, (300 - pivot[1] * 100) / 2]);
    const stateBefore = motion.exportPose().joints, angle = motion.getState(selected).angles[0];
    assert.equal(button.fire('pointerdown', {button: 0, pointerId: 1, clientX: 100, clientY: 100}).stopped, true);
    button.fire('pointermove', {pointerId: 1, clientX: 125, clientY: 100});
    assert.deepEqual(requests.at(-1), [selected, 0, angle + 10]);
    assert.deepEqual(motion.exportPose().joints, stateBefore, 'the overlay only requests motion through its callback');
    assert.equal(window.fire('pointerdown', {pointerId: 9}).immediateStopped, true, 'a second finger cannot start camera navigation');
    assert.equal(window.fire('pointermove', {pointerId: 9}).immediateStopped, true);
    button.fire('pointerup', {pointerId: 1});
    assert.equal(window.fire('pointerup', {pointerId: 9}).immediateStopped, true, 'suppressed finger release cannot leak to camera');
    assert.equal(ends.length, 1, 'release plus lostpointercapture ends only once');
    assert.equal(ends[0].cancelled, false);
    assert.ok(cancelledTravel.includes('navigateCancel'));
    frame(); assert.equal(fallSteps, 1, 'normal release permits the caller to start gravity');
    const beforeHold = {starts: starts.length, ends: ends.length, requests: requests.length};
    button.fire('pointerdown', {button: 0, pointerId: 10, clientX: 100, clientY: 100});
    assert.equal(heldJoint, selected, 'grabbing a falling arm holds immediately before any drag');
    for (let i = 0; i < 20; i++) frame();
    assert.equal(fallSteps, 1, 'twenty rendered frames do not release a held pointer');
    assert.equal(starts.length, beforeHold.starts + 1); assert.equal(ends.length, beforeHold.ends);
    assert.equal(requests.length, beforeHold.requests, 'holding needs no artificial angle request');
    // Pointer capture delivers release even when its coordinates leave the canvas.
    assert.equal(button.hasPointerCapture(10), true);
    button.fire('pointerup', {pointerId: 10, clientX: -800, clientY: 2000});
    assert.equal(ends.at(-1).changed, false); assert.equal(ends.at(-1).cancelled, false);
    frame(); assert.equal(fallSteps, 2, 'unchanged click-grab still releases gravity');
    button.fire('pointerdown', {button: 0, pointerId: 11, clientX: 100, clientY: 100});
    frame(); assert.equal(fallSteps, 2, 'regrabbing holds again without a timing workaround');
    button.fire('keydown', {key: 'Escape'});
    const afterEscape = ends.length;
    button.fire('pointerup', {pointerId: 11}); frame();
    assert.equal(ends.length, afterEscape); assert.equal(ends.at(-1).cancelled, true); assert.equal(falling, false);
    button.fire('pointerdown', {button: 0, pointerId: 12, clientX: 100, clientY: 100});
    button.fire('pointercancel', {pointerId: 12}); frame();
    assert.equal(ends.at(-1).cancelled, true); assert.equal(fallSteps, 2, 'pointer cancellation does not release a restored pose');
    button.fire('pointerdown', {button: 0, pointerId: 2, clientX: 100, clientY: 100});
    controls.setEnabled(false); assert.equal(ends.at(-1).cancelled, true); assert.equal(overlay.hidden, true);
    controls.setEnabled(true); controls.setAxis(2);
    const key = button.fire('keydown', {key: 'ArrowUp', shiftKey: true});
    assert.equal(key.stopped, true); assert.equal(key.prevented, true);
    assert.deepEqual(requests.at(-1), [selected, 2, motion.getState(selected).angles[2] + 10]);
    const keyup = button.fire('keyup', {key: 'ArrowUp'}); assert.equal(keyup.stopped, undefined, 'keyup clears camera key state');
    const beforeKeys = {starts: starts.length, ends: ends.length};
    button.fire('keydown', {key: 'ArrowUp'});
    button.fire('keydown', {key: 'ArrowUp', repeat: true});
    button.fire('keydown', {key: 'ArrowRight'});
    for (let i = 0; i < 10; i++) frame();
    assert.equal(starts.length, beforeKeys.starts + 1, 'key repeat/multiple arrows share one hold');
    button.fire('keyup', {key: 'ArrowUp'});
    assert.equal(ends.length, beforeKeys.ends); assert.equal(heldJoint, selected, 'remaining arrow keeps the arm held');
    button.fire('keyup', {key: 'ArrowRight'}); assert.equal(ends.length, beforeKeys.ends + 1); assert.equal(falling, true);
    button.fire('keydown', {key: 'ArrowDown'}); button.fire('blur');
    assert.equal(ends.at(-1).cancelled, true); assert.equal(falling, false, 'lost keyboard focus cancels instead of releasing gravity');
    const svg = overlay.children[0], ringHit = svg.children.find(child => child.attrs.class === 'am-joint-ring-hit');
    const pathPoints = ringHit.attrs.d.match(/[ML](-?[\d.]+),(-?[\d.]+)/g).map(value => value.slice(1).split(',').map(Number));
    const pointerAt = (point, pointerId = 3) => ({button: 0, pointerId, clientX: point[0] + 15, clientY: point[1] + 25});
    ringHit.fire('pointerdown', pointerAt(pathPoints[0]));
    ringHit.fire('pointermove', pointerAt(pathPoints[16]));
    assert.equal(requests.at(-1)[1], 2);
    assert.ok(Math.abs(requests.at(-1)[2] - motion.getState(selected).angles[2] - 90) < .05, 'projected quarter-ring requests an axis-plane quarter-turn');
    ringHit.fire('pointerup', {pointerId: 3});
    selected = 'left_knee'; controls.update();
    const knee = overlay.children.find(child => child.dataset.joint === selected);
    knee.fire('keydown', {key: 'ArrowLeft', shiftKey: false});
    assert.deepEqual(requests.at(-1), [selected, 0, motion.getState(selected).angles[0] - 2], 'hinge axis remains zero');
    knee.fire('keyup', {key: 'ArrowLeft'});
    at = {...at, position: [0, 0, 10]}; controls.update();
    assert.equal(knee.hidden, true, 'behind-camera joints are hidden');
    controls.dispose(); controls.dispose(); assert.equal(appEvents.size, 0); assert.equal(body.children.length, 0);
    assert.equal(starts.length, ends.length);
} finally {
    if (oldDocument === undefined) delete globalThis.document; else globalThis.document = oldDocument;
    if (oldWindow === undefined) delete globalThis.window; else globalThis.window = oldWindow;
}
console.log(`Manikin interaction checks passed: ${checkedAxes} real joint axes, projection scaling, drag/keyboard callbacks, cancellation and disposal.`);
