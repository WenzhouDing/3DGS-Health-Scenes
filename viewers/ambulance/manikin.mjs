import {createAmbulanceManikin} from './manikin-scan.mjs';
import {createContactChecker} from './contact-check.mjs';
import {createMotion, dofCount} from '../mannequin-articulation/motion-core.mjs';
import {buildPreset} from '../mannequin-articulation/pose-presets.mjs';
import {moveJointChecked} from './manikin-pose.mjs';
import {createManikinInteraction} from './manikin-interaction.mjs';
import {createEditHistory} from './edit-history.mjs';
import {createGravityTest} from './gravity-test.mjs';
import {createArmGravity} from './arm-gravity.mjs';
import {createBodyContactChecker} from './body-contact.mjs';

const panelMarkup = `
    <h1>Manikin on stretcher</h1>
    <button class="am-collapse" aria-label="Collapse manikin controls" aria-expanded="true">−</button>
    <p class="am-subtitle">Select a joint on the body to pose it</p>
    <div class="am-content">
        <div id="am-status" role="status">Loading ambulance…</div>
        <p id="am-message" aria-live="polite">Preparing the fitted manikin.</p>
        <fieldset id="am-controls" disabled style="border:0">
            <div class="am-mode" role="group" aria-label="Interaction mode"><button id="am-pose-mode" aria-pressed="true">Pose manikin</button><button id="am-explore-mode" aria-pressed="false">Explore cabin</button></div>
            <p class="am-note am-help" id="am-interaction-help">Drag a joint handle or its ring to rotate. Drag the background to orbit; scroll to zoom.</p>
            <label><input id="am-collisions" type="checkbox" checked>Prevent surface collisions</label>
            <label class="am-arm-toggle"><input id="am-arm-gravity" type="checkbox" checked>Arms fall on release</label>
            <p id="am-arm-status" class="am-note" role="status">Hold an arm handle to support it. Release to let it fall.</p>
            <div class="am-row"><button id="am-bed-view">Bed view</button><button id="am-cabin-view">Cabin view</button></div>
            <details class="am-gravity"><summary>Whole-body drop test</summary>
                <p class="am-note">Current joint pose held. Gravity moves the whole body vertically.</p>
                <div class="am-gravity-options"><label for="am-lift">Lift (scene units)</label><input id="am-lift" type="number" min="0.02" max="0.5" step="0.05" value="0.25">
                    <label for="am-speed">Playback</label><select id="am-speed"><option value="0.25">¼ speed</option><option value="0.5">½ speed</option><option value="1">Normal speed</option></select></div>
                <div class="am-row"><button id="am-drop-lift">Lift & drop</button><button id="am-drop">Drop from here</button></div>
                <div id="am-gravity-actions" class="am-row" hidden><button id="am-gravity-pause">Pause</button><button id="am-gravity-cancel">Cancel</button></div>
                <p id="am-gravity-status" class="am-note" role="status">Ready · Surface checks stay on during a drop.</p>
            </details>
            <label for="am-joint">Selected joint</label><select id="am-joint"></select>
            <div id="am-axis" class="am-axis" role="group" aria-label="Rotation axis"></div>
            <div class="am-row"><button id="am-focus">Focus joint</button><button id="am-reset-joint">Reset joint</button></div>
            <details id="am-precise"><summary>Precise angles</summary><div id="am-angles"></div></details>
            <div class="am-row"><button id="am-undo" disabled>Undo</button><button id="am-redo" disabled>Redo</button></div>
            <div class="am-row"><button id="am-reset">Reset onto bed</button></div>
            <details><summary>Scene & placement</summary>
                <label class="am-row"><input id="am-visible" type="checkbox" checked>Show manikin</label>
                <label class="am-row"><input id="am-overlay" type="checkbox">Show collision surfaces</label>
                <p class="am-note" id="am-camera-contact">Camera collision loading…</p>
                <p class="am-note" id="am-contact-detail"></p>
                <div class="am-placement"><label for="am-offset-x">Along cabin</label><input id="am-offset-x" type="number" min="-1" max="1" step="0.01" value="0">
                <label for="am-offset-y">Height offset</label><input id="am-offset-y" type="number" min="-0.6" max="0.8" step="0.01" value="0">
                <label for="am-offset-z">Across cabin</label><input id="am-offset-z" type="number" min="-0.8" max="0.8" step="0.01" value="0"></div>
                <p class="am-note">Arm gravity uses approximate body and cabin contacts. Torso position, neck and legs stay posed. Mattress deformation is not simulated; scene units are uncalibrated.</p>
                <p class="am-note" id="am-scan-detail"></p>
            </details>
        </fieldset>
    </div>`;

function waitForScene(viewer) {
    return new Promise((resolve, reject) => {
        const app = viewer.global.app;
        const finish = () => {clearTimeout(timer); app.off('update', check);};
        const check = () => {if (viewer.cameraManager && viewer.inputController) {finish(); resolve();}};
        const timer = setTimeout(() => {finish(); reject(new Error('The ambulance did not finish loading. Reload to try again.'));}, 120000);
        app.on('update', check); check();
    });
}

export async function initManikin({viewer, pc}) {
    const panel = document.createElement('section');
    panel.className = 'am-panel'; panel.setAttribute('aria-label', 'Ambulance manikin controls');
    panel.innerHTML = panelMarkup; document.body.append(panel);
    const $ = id => panel.querySelector('#am-' + id);
    const message = text => {if ($('message').textContent !== text) $('message').textContent = text;};
    const collapse = panel.querySelector('.am-collapse');
    collapse.addEventListener('click', () => {
        const collapsed = panel.classList.toggle('am-collapsed');
        collapse.textContent = collapsed ? '+' : '−';
        collapse.setAttribute('aria-expanded', String(!collapsed));
        collapse.setAttribute('aria-label', `${collapsed ? 'Expand' : 'Collapse'} manikin controls`);
    });
    // Editing a pose must not also feed WASD/arrow navigation to the camera.
    // Let keyup reach navigation so a key held before focusing the panel clears.
    for (const type of ['keydown', 'pointerdown', 'wheel']) panel.addEventListener(type, event => event.stopPropagation());
    let scan, interaction, gravity, armGravity;
    try {
        const configResponse = await fetch(new URL('./bed-placement.json', import.meta.url), {cache: 'no-store'});
        if (!configResponse.ok) throw new Error('The stretcher placement could not be loaded.');
        const config = await configResponse.json();
        if (config.schema !== 'ambulance-manikin-placement' || config.version !== 1 || config.manikin?.applyPresentation !== false) {
            throw new Error('The stretcher placement has an unsupported coordinate frame or version.');
        }
        await waitForScene(viewer);
        const {app, state, events} = viewer.global;
        scan = await createAmbulanceManikin({app, pc, onStatus: text => {$('status').textContent = text;}});
        if (config.manikin.revision !== scan.manifest.report?.parameters?.revision) {
            throw new Error('The fused scan changed; refit its placement before using the stretcher.');
        }
        const motion = createMotion(scan.annotations, scan.manifest);
        const checker = createContactChecker(config.collision);
        const restSamples = scan.getCollisionSamples(new Map(), {position:[0,0,0],rotation:[0,0,0,1],scale:1});
        const fittedTargets = {...buildPreset('lying', motion, scan.manifest).targets, ...config.jointOverrides};
        for (const [id,angles] of Object.entries(fittedTargets)) motion.setTarget(id,angles,{immediate:true});
        const bodyChecker = createBodyContactChecker({restSamples, annotations:scan.annotations,referenceParts:motion.evaluate().parts});
        let placement = structuredClone(config.placement), offsets = [0, 0, 0], selected = 'left_shoulder', activeAxis = 0, mode = 'pose';
        let history, sliderGesture = false, sliderJoint = null, gravityBusy = false, gravityBefore = null;
        let gravityOrigin, gravitySourceSamples, gravityQuerySamples;
        const isArmJoint = id => /^(left|right)_(shoulder|arm_swivel)$/.test(id);
        const holdArm = id => {if (isArmJoint(id)) armGravity?.hold(id);};
        const enabled = () => $('collisions').checked;
        const check = (parts = motion.evaluate().parts, at = placement, stopOnBlock = false, movingParts = null) => {
            const scene = checker.evaluate(scan.getCollisionSamples(parts, at, {parts:movingParts}), {stopOnBlock});
            if (stopOnBlock && scene.blocked) return scene;
            const body = bodyChecker.evaluate(parts, at, {stopOnBlock,movingParts:movingParts ?? undefined});
            const contacts = scene.contacts.concat(body.contacts), violations = scene.violations.concat(body.violations);
            const blocked = scene.blocked || body.blocked;
            return {...scene, contacts, violations, blocked, status:blocked ? 'penetrating' : contacts.length ? 'contact' : 'clear',
                maxPenetration:Math.max(scene.maxPenetration,body.maxPenetration ?? 0)};
        };
        const surfaceNames = report => [...new Set((report?.violations ?? []).map(item => item.label || item.surfaceLabel || item.surface || item.id || 'surface'))].join(', ');
        function updateStatus(report) {
            $('status').dataset.blocked = String(report.blocked);
            $('status').dataset.off = String(!enabled());
            $('status').textContent = !enabled() ? (report.blocked ? 'Checks off · Surface overlap' : 'Collision prevention off')
                : report.blocked ? 'Surface overlap detected' : report.contacts?.length ? 'Collision checks on · In contact' : 'Collision checks on · Clear';
            const names = [...new Set((report.blocked ? report.violations : report.contacts ?? []).map(hit => hit.label))];
            $('contact-detail').textContent = names.length ? `${report.blocked ? 'Overlap' : 'Contact'}: ${names.join(', ')}.` : 'No modeled surface overlaps.';
        }
        function render(report = check()) {
            scan.setPlacement(placement); scan.setPartTransforms(motion.evaluate().parts);
            updateStatus(report); interaction?.update(); app.renderNextFrame = true;
        }
        const snapshot = () => ({pose:motion.exportPose(), placement:structuredClone(placement), offsets:offsets.slice()});
        function restore(saved) {
            armGravity?.reset();
            motion.importPose(saved.pose); placement = structuredClone(saved.placement); offsets = saved.offsets.slice();
            ['x','y','z'].forEach((name,i) => {$('offset-'+name).value = offsets[i];});
            render(); syncAngles();
        }
        function syncAngles(preserveInput = false) {
            const angles = motion.getState(selected).angles;
            panel.querySelectorAll('[data-axis]').forEach(input => {if (!preserveInput || input.type !== 'number' || input !== document.activeElement) input.value = String(Number(angles[Number(input.dataset.axis)].toFixed(2)));});
        }
        function setAngle(axis, value, jointId = selected, grouped = false) {
            if (gravityBusy) return;
            if (!Number.isFinite(value)) {syncAngles(); return;}
            const before = grouped || sliderGesture ? null : snapshot();
            if (before) holdArm(jointId);
            const target = motion.getState(jointId).angles; target[axis] = value;
            const result = moveJointChecked({motion, jointId, target, evaluate: parts => check(parts, placement, true), enabled: enabled()});
            render(result.report); syncAngles();
            message(result.rejected ? `Stopped at ${surfaceNames(result.rejected)}. The last clear pose is retained.` : 'Joint pose updated.');
            if (before) {history.record(before); releaseArm(jointId);}
        }
        function selectAxis(value) {
            activeAxis = value; interaction?.setAxis(activeAxis);
            $('axis').querySelectorAll('button').forEach((button,i) => button.setAttribute('aria-pressed', String(i === activeAxis)));
        }
        function selectJoint(id) {
            if (gravityBusy) return;
            if (!motion.joints.some(j => j.id === id && dofCount(j))) return;
            selected = id; $('joint').value = id;
            drawJoint(); interaction?.update();
        }
        function cancelSlider() {
            if (!sliderGesture) return;
            finishSlider(true);
        }
        function finishSlider(cancelled = false) {
            if (!sliderGesture) return;
            const jointId = sliderJoint; sliderGesture = false; sliderJoint = null;
            history.end({cancelled});
            if (cancelled) {armGravity?.reset(); message('Adjustment cancelled.');}
            else releaseArm(jointId);
        }
        function releaseArm(jointId) {
            if (!isArmJoint(jointId)) return;
            if ($('arm-gravity').checked && !gravityBusy) {armGravity?.resume(); armGravity?.release(jointId);}
            else armGravity?.reset();
        }
        function drawJoint() {
            const joint = motion.joints.find(j => j.id === selected), angles = motion.getState(selected).angles;
            if (activeAxis >= dofCount(joint)) activeAxis = 0;
            $('axis').replaceChildren();
            for (let axis = 0; axis < dofCount(joint); axis++) {
                const button = document.createElement('button');
                button.textContent = dofCount(joint) === 3 ? ['X','Y','Z'][axis] : joint.type === 'swivel' ? 'Twist' : 'Bend';
                button.dataset.rotationAxis = axis; button.setAttribute('aria-label', `Drag around ${button.textContent} axis`);
                button.addEventListener('click', () => selectAxis(axis)); $('axis').append(button);
            }
            selectAxis(activeAxis);
            $('angles').replaceChildren();
            for (let axis = 0; axis < dofCount(joint); axis++) {
                const name = dofCount(joint) === 3 ? ['Rotation X', 'Rotation Y', 'Rotation Z'][axis] : joint.type === 'swivel' ? 'Twist' : 'Bend';
                const row = document.createElement('div'), heading = document.createElement('div'), label = document.createElement('label');
                heading.className = 'am-angle-heading'; label.textContent = name; label.htmlFor = `am-angle-${axis}`;
                const number = document.createElement('input'); Object.assign(number, {type: 'number', min: joint.limits[axis][0], max: joint.limits[axis][1], step: .5, value: angles[axis]});
                number.dataset.axis = axis; number.setAttribute('aria-label', name + ' degrees');
                number.addEventListener('change', () => setAngle(axis, number.valueAsNumber));
                number.addEventListener('blur', () => syncAngles());
                const slider = document.createElement('input'); Object.assign(slider, {id: `am-angle-${axis}`, type: 'range', min: joint.limits[axis][0], max: joint.limits[axis][1], step: .5, value: angles[axis]});
                slider.dataset.axis = axis; slider.setAttribute('aria-label', name);
                slider.addEventListener('pointerdown', () => {sliderGesture = true; sliderJoint = selected; holdArm(selected); history.begin();});
                slider.addEventListener('pointerup', () => finishSlider());
                slider.addEventListener('pointercancel', () => finishSlider(true));
                slider.addEventListener('lostpointercapture', cancelSlider);
                slider.addEventListener('keydown', event => {
                    if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','PageUp','PageDown'].includes(event.key) && !sliderGesture) {
                        sliderGesture = true; sliderJoint = selected; holdArm(selected); history.begin();
                    }
                });
                slider.addEventListener('keyup', event => {if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Home','End','PageUp','PageDown'].includes(event.key)) finishSlider();});
                slider.addEventListener('blur', cancelSlider);
                slider.addEventListener('input', () => setAngle(axis, Number(slider.value)));
                const limits = document.createElement('div'); limits.className = 'am-limit';
                for (const value of joint.limits[axis]) {const span = document.createElement('span'); span.textContent = value + '°'; limits.append(span);}
                heading.append(label, number); row.append(heading, slider, limits); $('angles').append(row);
            }
        }
        function reset({record = true} = {}) {
            if (gravityBusy) return;
            armGravity?.reset();
            const before = record ? snapshot() : null;
            motion.reset();
            for (const [id, angles] of Object.entries(fittedTargets)) motion.setTarget(id, angles, {immediate: true});
            placement = structuredClone(config.placement); offsets = [0, 0, 0];
            ['x', 'y', 'z'].forEach(axis => {$('offset-' + axis).value = '0';});
            render(); drawJoint(); message('Face up on the scanned stretcher.');
            if (before) history.record(before);
        }
        let navAttached = true;
        function setMode(value) {
            cancelSlider(); mode = value; interaction?.setEnabled(mode === 'pose' && !gravityBusy); history?.end();
            document.body.dataset.manikinMode = mode;
            const navigation = viewer.inputController._navInteraction;
            if (navAttached) {navigation.detach(); navAttached = false;}
            navigation._targetPickRequest++; navigation._mouseClickTracking = false; navigation._suppressClick = false; navigation._lastTap.time = 0;
            if (mode === 'explore') {navigation.attach(app.graphicsDevice.canvas, viewer.global); navAttached = true;}
            viewer.inputController._keyboardMouse.source._keyNow.fill(0);
            events.fire('navigateCancel'); events.fire('orbitTarget:clear'); app.graphicsDevice.canvas.style.cursor = '';
            state.animationPaused = true; state.cameraMode = mode === 'pose' ? 'orbit' : 'fly'; viewer.cameraManager.snap();
            $('pose-mode').setAttribute('aria-pressed',String(mode === 'pose')); $('explore-mode').setAttribute('aria-pressed',String(mode === 'explore'));
            $('interaction-help').textContent = mode === 'pose' ? 'Drag a handle or ring to pose. Hold to support an arm; release to let it fall. Drag the background to orbit.' : 'Drag to look around; use WASD to move. Choose Pose manikin to edit joints.';
            $('camera-contact').textContent = mode === 'pose' ? 'Orbit inspection · Camera collision applies in Explore cabin.' : state.hasCollision ? 'Camera collisions enabled in fly mode.' : 'Camera collider unavailable.';
        }
        function viewCamera(name) {
            const camera = config.cameras?.[name] ?? (name === 'bed'
                ? {position: [-.7, .49, .42], target: [-.25, -.18, -.27], fov: 75}
                : {position: [-1.1, .25, .45], target: [.3, -.05, .1], fov: 70});
            setMode(name === 'cabin' ? 'explore' : 'pose');
            viewer.cameraManager.camera.look(new pc.Vec3(...camera.position), new pc.Vec3(...camera.target));
            viewer.cameraManager.camera.fov = camera.fov; viewer.cameraManager.snap();
        }
        history = createEditHistory({capture:snapshot, restore,
            canRestore: candidate => {
                if (!enabled()) return true;
                const before = motion.exportPose();
                try {motion.importPose(candidate.pose); return !check(motion.evaluate().parts, candidate.placement, true).blocked;}
                finally {motion.importPose(before);}
            }, onChange: () => {if (history) {$('undo').disabled = gravityBusy || !history.canUndo; $('redo').disabled = gravityBusy || !history.canRedo;}}
        });
        armGravity = createArmGravity({motion,getPlacement:()=>placement,restSamples,
            evaluate:(pose,context) => check(pose.parts,placement,true,context.movingParts),
            onUpdate: armState => {
                const held = Object.entries(armState.sides).filter(([,side]) => side.status === 'held').map(([name]) => name);
                const active = Object.entries(armState.sides).filter(([,side]) => side.status === 'active').map(([name]) => name);
                const description = !$('arm-gravity').checked ? 'Arm gravity off · Poses stay where you set them.'
                    : held.length ? `Holding ${held.join(' and ')} arm. Release to let it fall.`
                    : armState.paused && active.length ? 'Arm gravity paused. Grab and release an arm to continue.'
                    : active.length ? `${active.map(name => name[0].toUpperCase()+name.slice(1)).join(' and ')} arm falling · Grab a handle to catch it.`
                    : 'Arms at rest · Hold a handle to lift and support an arm.';
                if ($('arm-status').textContent !== description) $('arm-status').textContent = description;
            },
            onSettle: ({side,reason,report}) => {
                const label = report?.violations?.[0]?.label ?? report?.contacts?.[0]?.label;
                const action = reason === 'contact' ? 'stopped at contact' : reason === 'initial-overlap' ? 'stopped: move it out of the overlap first' : reason.startsWith('evaluation-error') ? 'paused because its contacts could not be checked' : 'came to rest';
                message(`${side === 'left' ? 'Left' : 'Right'} arm ${action}${label ? ' · '+label : ''}.`);
            }
        });
        $('arm-gravity').addEventListener('change', () => {
            if (gravityBusy) return;
            if ($('arm-gravity').checked) {armGravity.resume(); armGravity.wake();}
            else armGravity.reset();
        });
        function syncGravityControls() {
            for (const name of ['joint','collisions','arm-gravity','reset','reset-joint','visible','offset-x','offset-y','offset-z','drop','drop-lift','lift','speed']) $(name).disabled = gravityBusy;
            panel.querySelectorAll('#am-axis button, #am-angles input').forEach(input => {input.disabled = gravityBusy;});
            $('undo').disabled = gravityBusy || !history.canUndo; $('redo').disabled = gravityBusy || !history.canRedo;
            $('gravity-actions').hidden = !gravityBusy;
            interaction?.setEnabled(mode === 'pose' && !gravityBusy);
        }
        function gravityState(state) {
            if (state.status === 'running' || state.status === 'paused') {
                $('gravity-pause').textContent = state.status === 'paused' ? 'Resume' : 'Pause';
                $('status').textContent = state.status === 'paused' ? 'Gravity test · Paused' : 'Gravity test · Falling';
                $('status').dataset.blocked = $('status').dataset.off = 'false';
                message(state.status === 'paused' ? 'Gravity test paused. Resume to continue, or Cancel to restore.' : 'Gravity test running. Current joint pose is held.');
                $('gravity-status').textContent = `${state.status === 'paused' ? 'Paused' : 'Falling'} · ${state.fallDistance.toFixed(3)} units · ${Math.abs(state.velocity).toFixed(2)} units/s`;
            }
        }
        gravity = createGravityTest({
            getPosition: () => placement.position,
            setPosition: position => {
                placement.position = position.slice(); offsets = position.map((v,i) => v - config.placement.position[i]);
                scan.setPlacement(placement); app.renderNextFrame = true;
            },
            evaluateAt: position => {
                const delta = position.map((value,i) => value - gravityOrigin[i]);
                for (let i = 0; i < gravityQuerySamples.length; i++) {
                    const p = gravityQuerySamples[i].position, source = gravitySourceSamples[i].position;
                    p[0] = source[0]+delta[0]; p[1] = source[1]+delta[1]; p[2] = source[2]+delta[2];
                }
                return checker.evaluate(gravityQuerySamples, {stopOnBlock:true});
            },
            onUpdate: gravityState,
            onStop: state => {
                const before = gravityBefore; gravityBefore = null; gravityBusy = false;
                if (state.reason === 'cancelled' && before) restore(before);
                else {
                    ['x','y','z'].forEach((name,i) => {$('offset-'+name).value = Number(offsets[i].toFixed(4));});
                    render(); if (before) history.record(before);
                }
                syncGravityControls();
                const label = state.contact?.label ?? state.contact?.proxy ?? 'a modeled surface';
                const result = state.status === 'landed' ? `Landed on ${label} · ${state.fallDistance.toFixed(3)} units in ${state.elapsed.toFixed(2)} s.`
                    : state.reason === 'cancelled' ? 'Drop cancelled · Starting pose restored.'
                    : state.status === 'rejected' ? 'Drop could not start. Check clearance and try a smaller lift.'
                    : state.status === 'obstructed' ? `Stopped at ${label}.`
                    : `Drop stopped (${state.reason || state.status}). Last clear placement retained.`;
                $('gravity-status').textContent = result; message(result);
                gravitySourceSamples = gravityQuerySamples = null;
                if (state.reason !== 'cancelled' && $('arm-gravity').checked) armGravity.resume();
            }
        });
        function startGravity(lift) {
            if (gravityBusy) return;
            if (!Number.isFinite(lift) || lift < 0 || lift > .5) {message('Choose a lift between 0.02 and 0.5 scene units.'); return;}
            cancelSlider(); interaction?.setEnabled(false); history.end();
            armGravity.pause();
            state.animationPaused = true; if (state.cameraMode === 'anim') setMode(mode);
            events.fire('navigateCancel'); viewer.cameraManager.snap();
            gravityBefore = snapshot(); gravityOrigin = placement.position.slice();
            gravitySourceSamples = scan.getCollisionSamples(motion.evaluate().parts, placement);
            gravityQuerySamples = gravitySourceSamples.map(sample => ({...sample,position:sample.position.slice()}));
            $('collisions').checked = true; $('visible').checked = true; scan.setVisible(true); interaction?.setVisible(true);
            gravityBusy = true; syncGravityControls(); message('Gravity test running. Current joint pose is held.');
            gravity.start({lift});
        }
        $('drop-lift').addEventListener('click', () => startGravity($('lift').valueAsNumber));
        $('drop').addEventListener('click', () => startGravity(0));
        $('gravity-pause').addEventListener('click', () => gravity.getState().status === 'paused' ? gravity.resume() : gravity.pause());
        $('gravity-cancel').addEventListener('click', () => gravity.stop('cancelled'));
        const gravityUpdate = dt => {
            if (gravityBusy) gravity.update(dt * Number($('speed').value));
            else if ($('arm-gravity').checked) {const result = armGravity.update(dt); if (result.changed) {render(); syncAngles(true);}}
        };
        const pauseGravity = () => {if (gravityBusy) gravity.pause(); armGravity.pause();};
        const visibilityGravity = () => {if (document.hidden) pauseGravity();};
        app.on('update', gravityUpdate); window.addEventListener('blur',pauseGravity); document.addEventListener('visibilitychange',visibilityGravity);
        $('joint').replaceChildren(...motion.joints.filter(j => dofCount(j)).map(j => new Option(j.label, j.id)));
        if (!motion.joints.some(j => j.id === selected)) selected = $('joint').value;
        $('joint').value = selected;
        $('joint').addEventListener('change', () => selectJoint($('joint').value));
        $('collisions').addEventListener('change', () => {if (gravityBusy) {$('collisions').checked = true; return;} render(); message(enabled() ? 'Surface collision prevention enabled.' : 'Movement is unrestricted; overlaps are still reported.');});
        $('visible').addEventListener('change', () => {if (gravityBusy) {$('visible').checked = true; return;} scan.setVisible($('visible').checked); interaction?.setVisible($('visible').checked);});
        $('overlay').addEventListener('change', () => {state.collisionOverlayEnabled = $('overlay').checked;});
        events.on('collisionOverlayEnabled:changed', value => {$('overlay').checked = value;});
        $('bed-view').addEventListener('click', () => viewCamera('bed'));
        $('cabin-view').addEventListener('click', () => viewCamera('cabin'));
        $('reset').addEventListener('click', () => reset());
        $('pose-mode').addEventListener('click', () => viewCamera('bed'));
        $('explore-mode').addEventListener('click', () => setMode('explore'));
        $('undo').addEventListener('click', () => {if (!gravityBusy) message(history.undo() ? 'Undid the last adjustment.' : 'Cannot restore that pose while surface prevention is on.');});
        $('redo').addEventListener('click', () => {if (!gravityBusy) message(history.redo() ? 'Redid the adjustment.' : 'Cannot restore that pose while surface prevention is on.');});
        $('reset-joint').addEventListener('click', () => {
            if (gravityBusy) return;
            holdArm(selected);
            const before = snapshot(), result = moveJointChecked({motion, jointId:selected, target:fittedTargets[selected], evaluate:parts => check(parts,placement,true), enabled:enabled()});
            render(result.report); syncAngles(); history.record(before); message(result.rejected ? 'Stopped at '+surfaceNames(result.rejected)+'.' : 'Selected joint returned to its bed pose.');
            releaseArm(selected);
        });
        $('focus').addEventListener('click', () => {
            setMode('pose');
            const native = motion.evaluate().joints.get(selected).pivot;
            const target = new pc.Quat(...placement.rotation).transformVector(new pc.Vec3(...native)).mulScalar(placement.scale).add(new pc.Vec3(...placement.position));
            const camera = viewer.cameraManager.camera, direction = camera.position.clone().sub(target).normalize();
            camera.look(target.clone().add(direction.mulScalar(.65)), target); camera.fov = 65; viewer.cameraManager.snap();
        });
        ['x', 'y', 'z'].forEach((axis, index) => $('offset-' + axis).addEventListener('change', () => {
            if (gravityBusy) return;
            const input = $('offset-' + axis), request = Math.max(Number(input.min), Math.min(Number(input.max), input.valueAsNumber));
            if (!Number.isFinite(request)) {input.value = offsets[index]; return;}
            armGravity.reset();
            const before = snapshot();
            const start = offsets[index], steps = Math.max(1, Math.ceil(Math.abs(request - start) / .004));
            let rejected = null;
            const parts = motion.evaluate().parts;
            for (let i = 1; i <= steps; i++) {
                const offset = start + (request - start) * i / steps, candidate = structuredClone(placement);
                candidate.position[index] = config.placement.position[index] + offset;
                const report = check(parts, candidate, true);
                if (enabled() && report.blocked) {rejected = report; break;}
                placement = candidate; offsets[index] = offset;
            }
            input.value = Number(offsets[index].toFixed(4)); render();
            message(rejected ? `Stopped at ${surfaceNames(rejected)}.` : 'Placement updated.');
            history.record(before);
            if ($('arm-gravity').checked) armGravity.wake();
        }));
        $('scan-detail').textContent = `${scan.manifest.captures.reduce((sum, capture) => sum + capture.count, 0).toLocaleString()} Gaussians · Both captures · ${scan.manifest.parts.length} body regions. Scale fitted to the stretcher.`;
        $('camera-contact').textContent = state.hasCollision ? 'Camera collisions enabled in fly mode.' : 'Camera collider unavailable; manikin surface checks remain active.';
        $('overlay').disabled = !state.hasCollisionOverlay;
        $('controls').disabled = false;
        interaction = createManikinInteraction({viewer, pc, motion, getPlacement: () => placement, getSelected: () => selected,
            onSelect: selectJoint, onAngle: (jointId, axis, value) => setAngle(axis, value, jointId, true),
            onGestureStart: jointId => {holdArm(jointId); history.begin();},
            onGestureEnd: data => {history.end(data); if (data.cancelled) {armGravity.reset(); message('Adjustment cancelled.');} else releaseArm(data.jointId);}, onMessage: message});
        const poseKeys = event => {
            if (event.code === 'Escape' && gravityBusy) {gravity.stop('cancelled'); event.preventDefault(); event.stopImmediatePropagation(); return;}
            if (event.code === 'Escape' && sliderGesture) {cancelSlider(); event.preventDefault(); event.stopImmediatePropagation(); return;}
            if ((event.metaKey || event.ctrlKey) && !event.altKey && !event.target.closest?.('input, select, textarea, [contenteditable]')) {
                const redo = event.code === 'KeyY' || (event.code === 'KeyZ' && event.shiftKey);
                if (redo || event.code === 'KeyZ') {
                    event.preventDefault(); event.stopImmediatePropagation();
                    if (!gravityBusy && !history.inGesture && (redo ? history.canRedo : history.canUndo)) {
                        const restored = redo ? history.redo() : history.undo();
                        message(restored ? (redo ? 'Redid the adjustment.' : 'Undid the last adjustment.') : 'Cannot restore that pose while surface prevention is on.');
                    }
                    return;
                }
            }
            if (mode !== 'pose' || event.target.closest?.('.am-panel, .am-joint-overlay') || event.metaKey || event.ctrlKey || event.altKey) return;
            if (['KeyW','KeyA','KeyS','KeyD','KeyQ','KeyE','KeyG','KeyR','KeyF','Digit1','Digit2','Digit3','ArrowLeft','ArrowRight','ArrowUp','ArrowDown','Space'].includes(event.code)) {event.preventDefault(); event.stopImmediatePropagation();}
        };
        const endSlider = () => finishSlider();
        window.addEventListener('keydown', poseKeys, true); window.addEventListener('pointerup',endSlider); window.addEventListener('blur',cancelSlider);
        reset({record:false});
        if (viewer.global.config.noanim) viewCamera('bed');
        else {setMode('explore'); if (state.hasAnimation) {state.cameraMode = 'anim'; state.animationPaused = false;}}
        panel.dataset.ready = 'true';
        window.addEventListener('pagehide', () => {app.off('update',gravityUpdate); window.removeEventListener('blur',pauseGravity); document.removeEventListener('visibilitychange',visibilityGravity); gravity.stop('cancelled'); armGravity.reset(); cancelSlider(); window.removeEventListener('keydown',poseKeys,true); window.removeEventListener('pointerup',endSlider); window.removeEventListener('blur',cancelSlider); interaction.dispose(); scan.dispose();}, {once: true});
        window.addEventListener('pageshow', event => {if (event.persisted) location.reload();});
    } catch (error) {
        $('status').textContent = 'Manikin could not be loaded';
        $('status').dataset.blocked = 'true'; message(error.message);
        interaction?.dispose(); scan?.dispose(); console.error('Ambulance manikin:', error);
    }
}
