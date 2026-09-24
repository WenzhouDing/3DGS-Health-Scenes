import {dofCount} from '../mannequin-articulation/motion-core.mjs';

const SVG = 'http://www.w3.org/2000/svg', DEG = Math.PI / 180;
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const unit = a => { const length = Math.hypot(...a); return a.map(n => n / length); };
const rotate = (q, p) => {
    const t = cross(q, p).map(n => 2 * n), c = cross(q, t);
    return p.map((n, i) => n + q[3] * t[i] + c[i]);
};

/** The actual instantaneous Euler axis is used, including preceding Euler
 * rotations, parent articulation and the uniform scene placement. A joint's
 * evaluated pivot, not its affine translation, is its visible center.
 */
export function jointWorldFrame(joint, evaluated, angles, placement, axis = 0) {
    const activeAxis = dofCount(joint) === 3 ? axis : 0;
    if (!Number.isInteger(activeAxis) || activeAxis < 0 || activeAxis > 2) throw new TypeError('Invalid joint axis.');
    const qLength = Math.hypot(...placement.rotation), q = placement.rotation.map(n => n / qLength);
    const center = rotate(q, evaluated.pivot.map(n => n * placement.scale)).map((n, i) => n + placement.position[i]);
    let nativeAxis;
    if (dofCount(joint) !== 3) nativeAxis = evaluated.axis;
    else {
        // The rig uses Rz * Ry * Rx. Increasing X turns around Rz*Ry*X,
        // increasing Y around Rz*Y, and increasing Z around parent Z.
        const y = angles[1] * DEG, z = angles[2] * DEG;
        const local = activeAxis === 0 ? [Math.cos(z) * Math.cos(y), Math.sin(z) * Math.cos(y), -Math.sin(y)]
            : activeAxis === 1 ? [-Math.sin(z), Math.cos(z), 0] : [0, 0, 1];
        nativeAxis = [0, 1, 2].map(i => evaluated.axes.reduce((sum, basis, k) => sum + basis[i] * local[k], 0));
    }
    const normal = unit(rotate(q, nativeAxis));
    const u = unit(cross(Math.abs(normal[1]) < .9 ? [0, 1, 0] : [1, 0, 0], normal));
    return {center, normal, u, v: cross(normal, u), axis: activeAxis};
}

/** Continuous signed increment through the atan2 branch cut, in radians. */
export function rotationIncrement(previous, current) {
    return Math.atan2(Math.sin(current - previous), Math.cos(current - previous));
}

/** Overlay-only manipulation. onAngle owns limit/contact checking and applies
 * accepted poses; this module never changes motion state itself.
 *
 * Callbacks: onGestureStart(jointId, axis),
 * onGestureEnd({jointId, axis, cancelled, changed}). A cancelled gesture lets
 * the caller restore its undo snapshot. Mode/visibility changes cancel safely.
 * Pointerdown starts a hold immediately, including a grab without movement.
 * Normal pointerup releases that hold even when changed=false; changed only
 * describes an angle request, not whether the caller should release physics.
 * Repeated/multiple arrow keys share a hold until its last arrow key is released.
 */
export function createManikinInteraction({viewer, pc, motion, getPlacement, getSelected,
    onSelect, onAngle, onGestureStart = () => {}, onGestureEnd = () => {}, onMessage = () => {}}) {
    const {app, events} = viewer.global, cameraEntity = viewer.global.camera;
    const camera = cameraEntity.camera, canvas = app.graphicsDevice.canvas;
    if (!canvas || !camera?.worldToScreen || typeof onAngle !== 'function' || typeof onSelect !== 'function') {
        throw new TypeError('Manikin interaction needs the live camera, canvas and pose callbacks.');
    }
    const overlay = document.createElement('div');
    overlay.className = 'am-joint-overlay';
    overlay.setAttribute('aria-label', 'Manikin joint handles');
    const svg = document.createElementNS(SVG, 'svg');
    svg.classList.add('am-joint-ring'); svg.setAttribute('aria-hidden', 'true');
    const element = (name, className) => {
        const node = document.createElementNS(SVG, name); node.setAttribute('class', className); svg.append(node); return node;
    };
    const ring = element('path', 'am-joint-ring-line'), axisLine = element('path', 'am-joint-axis-line');
    const handle = element('circle', 'am-joint-ring-handle'); handle.setAttribute('r', '5');
    const ringHit = element('path', 'am-joint-ring-hit');
    const caption = document.createElement('div'); caption.className = 'am-joint-caption';
    overlay.append(svg, caption); document.body.append(overlay);
    const point = new pc.Vec3(), screen = new pc.Vec3(), rayNear = new pc.Vec3(), rayFar = new pc.Vec3();
    const markers = new Map(), joints = new Map(motion.joints.filter(joint => dofCount(joint)).map(joint => [joint.id, joint]));
    const listeners = [], pointersDown = new Set(), blockedPointers = new Set();
    let enabled = false, visible = true, disposed = false, axis = 0, gesture = null, ringFrame = null;
    let rect = canvas.getBoundingClientRect(), projectionWidth = rect.width, projectionHeight = rect.height;
    const listen = (node, name, fn, options) => { node.addEventListener(name, fn, options); listeners.push(() => node.removeEventListener(name, fn, options)); };
    const active = () => enabled && visible && !disposed;
    const stopPointer = event => { event.preventDefault(); event.stopPropagation(); };
    const axisFor = joint => dofCount(joint) === 3 ? axis : 0;
    const ringPoint = (frame, angle, radius) => frame.center.map((n, i) => n + radius * (Math.cos(angle) * frame.u[i] + Math.sin(angle) * frame.v[i]));

    function project(world) {
        const origin = cameraEntity.getPosition(), forward = cameraEntity.forward;
        const depth = (world[0] - origin.x) * forward.x + (world[1] - origin.y) * forward.y + (world[2] - origin.z) * forward.z;
        camera.worldToScreen(point.set(...world), screen);
        // This bundled CameraComponent projects in graphicsDevice.clientRect
        // units (CSS pixels), not drawing-buffer pixels. Keep the conversion
        // explicit for DPR, resized canvases and a non-fullscreen viewport.
        const x = screen.x * rect.width / projectionWidth, y = screen.y * rect.height / projectionHeight;
        const inFront = Number.isFinite(x) && Number.isFinite(y) && depth > camera.nearClip && depth < camera.farClip;
        return {x, y, inFront, visible: inFront && x >= 0 && x <= rect.width && y >= 0 && y <= rect.height};
    }

    function pointerAngle(event, frame) {
        const x = (event.clientX - rect.left) * projectionWidth / rect.width;
        const y = (event.clientY - rect.top) * projectionHeight / rect.height;
        camera.screenToWorld(x, y, Math.max(camera.nearClip, .0001), rayNear);
        camera.screenToWorld(x, y, Math.max(camera.nearClip, .0001) + 1, rayFar);
        const origin = [rayNear.x, rayNear.y, rayNear.z], direction = unit([rayFar.x - rayNear.x, rayFar.y - rayNear.y, rayFar.z - rayNear.z]);
        const denominator = dot(direction, frame.normal);
        if (Math.abs(denominator) < .10) return null; // Edge-on rings cannot give stable angular input.
        const t = dot(frame.center.map((n, i) => n - origin[i]), frame.normal) / denominator;
        if (t < 0 || !Number.isFinite(t)) return null;
        const radial = origin.map((n, i) => n + t * direction[i] - frame.center[i]);
        if (Math.hypot(...radial) < frame.radius * .15) return null;
        return Math.atan2(dot(radial, frame.v), dot(radial, frame.u));
    }

    function endGesture(cancelled = false) {
        if (!gesture) return;
        const ended = gesture; gesture = null;
        overlay.classList.remove('am-joint-dragging');
        if (ended.pointerId !== undefined && ended.target.hasPointerCapture?.(ended.pointerId)) ended.target.releasePointerCapture(ended.pointerId);
        onGestureEnd({jointId: ended.jointId, axis: ended.axis, cancelled, changed: ended.changed});
        update();
    }

    function prepareGesture(jointId) {
        endGesture(false);
        onSelect(jointId);
        events?.fire('navigateCancel');
        viewer.cameraManager?.snap();
        // The root Pose/Explore controller owns broader navigation policy.
        update();
        return {jointId, axis: axisFor(joints.get(jointId)), changed: false};
    }

    function beginPointer(event, jointId, useRing) {
        if (!active() || event.button !== 0 || gesture) return;
        stopPointer(event);
        // Do not mix a still-held background navigation finger with a joint
        // gesture. Extra fingers arriving during a joint drag are consumed by
        // the capture guard below until their own release.
        if ([...pointersDown].some(id => id !== event.pointerId)) {
            blockedPointers.add(event.pointerId);
            onMessage('Lift the other finger before dragging a joint.');
            return;
        }
        const joint = joints.get(jointId);
        if (!joint) return;
        const next = prepareGesture(jointId), target = event.currentTarget;
        const frame = ringFrame?.jointId === jointId ? ringFrame : null;
        const startPointerAngle = useRing && frame ? pointerAngle(event, frame) : null;
        markers.get(jointId).focus({preventScroll: true});
        gesture = {...next, target, pointerId: event.pointerId, startX: event.clientX,
            startAngle: motion.getState(jointId).angles[next.axis], frame,
            mode: startPointerAngle === null ? 'horizontal' : 'ring', lastPointerAngle: startPointerAngle, delta: 0};
        target.setPointerCapture(event.pointerId);
        overlay.classList.add('am-joint-dragging');
        onGestureStart(jointId, next.axis);
        onMessage(startPointerAngle === null ? 'Drag left or right to rotate. Arrow keys move 2°; Shift moves 10°.' : 'Drag around the ring to rotate this axis.');
    }

    function movePointer(event) {
        if (!gesture || event.pointerId !== gesture.pointerId) return;
        stopPointer(event);
        const current = gesture;
        let delta;
        if (current.mode === 'ring') {
            const angle = pointerAngle(event, current.frame);
            if (angle === null) return;
            current.delta += rotationIncrement(current.lastPointerAngle, angle) / DEG;
            current.lastPointerAngle = angle; delta = current.delta;
        } else delta = (event.clientX - current.startX) * .4;
        if (Math.abs(delta) < .05 && !current.changed) return;
        current.changed = true;
        onAngle(current.jointId, current.axis, current.startAngle + delta);
        update();
    }

    function bindPointer(target, jointId) {
        listen(target, 'pointerdown', event => beginPointer(event, typeof jointId === 'function' ? jointId() : jointId, target === ringHit));
        listen(target, 'pointermove', movePointer);
        listen(target, 'pointerup', event => { if (event.pointerId === gesture?.pointerId) {stopPointer(event); endGesture(false);} });
        listen(target, 'pointercancel', event => { if (event.pointerId === gesture?.pointerId) {stopPointer(event); endGesture(true);} });
        listen(target, 'lostpointercapture', event => { if (event.pointerId === gesture?.pointerId) endGesture(true); });
        listen(target, 'contextmenu', stopPointer);
    }

    for (const joint of joints.values()) {
        const button = document.createElement('button');
        button.type = 'button'; button.className = 'am-joint-marker'; button.dataset.joint = joint.id;
        button.setAttribute('aria-label', `${joint.label}. Select and drag to rotate; arrow keys adjust the selected axis.`);
        button.title = joint.label; button.innerHTML = '<span aria-hidden="true"></span>';
        overlay.append(button); markers.set(joint.id, button); bindPointer(button, joint.id);
        listen(button, 'click', event => { event.stopPropagation(); if (active()) {onSelect(joint.id); update();} });
        listen(button, 'keydown', event => {
            event.stopPropagation();
            if (!active()) return;
            if (event.key === 'Escape') {event.preventDefault(); endGesture(true); return;}
            const direction = ['ArrowRight', 'ArrowUp'].includes(event.key) ? 1 : ['ArrowLeft', 'ArrowDown'].includes(event.key) ? -1 : 0;
            if (!direction) return;
            event.preventDefault();
            if (gesture?.pointerId !== undefined) return;
            if (!gesture || gesture.jointId !== joint.id) {
                gesture = {...prepareGesture(joint.id), mode: 'keyboard', target: button, keys: new Set()};
                onGestureStart(gesture.jointId, gesture.axis);
            }
            gesture.keys.add(event.code || event.key);
            gesture.changed = true;
            onAngle(joint.id, gesture.axis, motion.getState(joint.id).angles[gesture.axis] + direction * (event.shiftKey ? 10 : 2));
            update();
        });
        // Let keyup reach the camera source so previously held navigation keys clear.
        listen(button, 'keyup', event => {
            const key = event.code || event.key;
            if (gesture?.mode === 'keyboard' && gesture.keys.has(key)) {
                gesture.keys.delete(key);
                if (!gesture.keys.size) endGesture(false);
            }
        });
        // Losing focus is a cancellation, not an intentional release into
        // gravity. The caller restores the captured pose and keeps it stopped.
        listen(button, 'blur', () => { if (gesture?.mode === 'keyboard') endGesture(true); });
    }
    bindPointer(ringHit, getSelected);
    const pointerGuard = event => {
        if (event.type === 'pointerdown') {
            pointersDown.add(event.pointerId);
            if (gesture?.pointerId !== undefined && event.pointerId !== gesture.pointerId) blockedPointers.add(event.pointerId);
        }
        if (blockedPointers.has(event.pointerId)) {
            event.preventDefault(); event.stopImmediatePropagation();
            if (event.type === 'pointerup' || event.type === 'pointercancel') blockedPointers.delete(event.pointerId);
        }
        if (event.type === 'pointerup' || event.type === 'pointercancel') pointersDown.delete(event.pointerId);
    };
    for (const event of ['pointerdown', 'pointermove', 'pointerup', 'pointercancel']) {
        listen(window, event, pointerGuard, {capture: true, passive: false});
    }
    listen(window, 'blur', () => {pointersDown.clear(); blockedPointers.clear(); endGesture(true);});

    function update() {
        if (disposed) return;
        overlay.hidden = !active();
        if (!active()) return;
        rect = canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) {overlay.hidden = true; return;}
        projectionWidth = app.graphicsDevice.clientRect?.width || rect.width;
        projectionHeight = app.graphicsDevice.clientRect?.height || rect.height;
        Object.assign(overlay.style, {left: `${rect.left}px`, top: `${rect.top}px`, width: `${rect.width}px`, height: `${rect.height}px`});
        svg.setAttribute('viewBox', `0 0 ${rect.width} ${rect.height}`);
        const evaluated = motion.evaluate(), placement = getPlacement(), selected = getSelected();
        let selectedProjection = null;
        ringFrame = null; svg.style.display = 'none'; caption.hidden = true;
        for (const [id, joint] of joints) {
            const button = markers.get(id), pose = evaluated.joints.get(id);
            const frame = jointWorldFrame(joint, pose, motion.getState(id).angles, placement, axisFor(joint));
            const projected = project(frame.center), isSelected = id === selected;
            button.hidden = !projected.visible;
            button.classList.toggle('am-joint-selected', isSelected);
            button.setAttribute('aria-pressed', String(isSelected));
            button.style.left = `${projected.x}px`; button.style.top = `${projected.y}px`;
            if (isSelected && projected.visible) {ringFrame = {...frame, jointId: id}; selectedProjection = projected;}
        }
        if (!ringFrame) return;
        const frame = ringFrame, center = selectedProjection;
        let radius = .10 * placement.scale;
        const u = project(ringPoint(frame, 0, radius)), v = project(ringPoint(frame, Math.PI / 2, radius));
        const pixelRadius = Math.max(Math.hypot(u.x - center.x, u.y - center.y), Math.hypot(v.x - center.x, v.y - center.y));
        if (pixelRadius > .001) radius *= Math.max(.5, Math.min(3, 54 / pixelRadius));
        frame.radius = radius;
        const points = Array.from({length: 65}, (_, i) => project(ringPoint(frame, i * Math.PI / 32, radius)));
        if (points.every(p => p.inFront)) {
            const path = points.map((p, i) => `${i ? 'L' : 'M'}${p.x.toFixed(2)},${p.y.toFixed(2)}`).join(' ') + ' Z';
            ring.setAttribute('d', path); ringHit.setAttribute('d', path);
            const a = project(frame.center.map((n, i) => n - frame.normal[i] * radius * .6));
            const b = project(frame.center.map((n, i) => n + frame.normal[i] * radius * .8));
            axisLine.setAttribute('d', a.inFront && b.inFront ? `M${a.x},${a.y} L${b.x},${b.y}` : '');
            const angle = motion.getState(selected).angles[frame.axis], hp = project(ringPoint(frame, angle * DEG, radius));
            handle.setAttribute('cx', hp.x); handle.setAttribute('cy', hp.y);
            svg.dataset.axis = String(frame.axis); svg.style.display = '';
        }
        const joint = joints.get(selected), axisName = dofCount(joint) === 3 ? ['X', 'Y', 'Z'][frame.axis] : joint.type === 'swivel' ? 'Twist' : 'Bend';
        caption.textContent = `${joint.label} · ${axisName} ${motion.getState(selected).angles[frame.axis].toFixed(1)}°`;
        caption.style.left = `${Math.max(120, Math.min(rect.width - 120, center.x))}px`;
        caption.style.top = `${Math.max(12, center.y - 43)}px`; caption.hidden = false;
    }

    app.on('postrender', update);
    update();
    return {update,
        setEnabled(value) {if (!value) endGesture(true); enabled = Boolean(value); update();},
        setVisible(value) {if (!value) endGesture(true); visible = Boolean(value); update();},
        setAxis(value) {
            if (!Number.isInteger(value) || value < 0 || value > 2) throw new TypeError('Joint axis must be 0, 1 or 2.');
            if (axis !== value) endGesture(true); axis = value; update();
        },
        dispose() {
            if (disposed) return;
            endGesture(true); disposed = true; app.off('postrender', update);
            for (const remove of listeners) remove(); pointersDown.clear(); blockedPointers.clear(); overlay.remove();
        }};
}
