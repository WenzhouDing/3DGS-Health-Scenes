#!/usr/bin/env node
/** Focused Props Lab bootstrap/collider regressions and real-engine affine
 * round trips. No GPU, browser, server, or source asset mutation is required.
 */
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFile} from 'node:fs/promises';
import {ambulanceScene} from '../viewers/ambulance/scene-config.mjs';
import {createSceneFrame} from '../viewers/ambulance/scene-frame.mjs';
import {sceneTools} from '../viewers/ambulance/index.js?v=surface-frame-v2';

const file = path => new URL(`../${path}`, import.meta.url);
const html = await readFile(file('viewers/props-lab/index.html'), 'utf8');
const js = await readFile(file('viewers/props-lab/props.js'), 'utf8');
// Exercise the shipped bootstrap, supplying its imported shared asset config.
const bootstrap = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1]
    .replace(/^\s*import .*;$/m, '');
const run = query => {
    const context = {URL, Image: class {}, location: {href: `https://example.test/project/viewers/props-lab/${query}`},
        window: {}, ambulanceScene, fetch: () => Promise.resolve({ok: true, json: async () => ({})})};
    vm.runInNewContext(bootstrap, context);
    return context.window.sse.config;
};

assert.equal(run('').contentUrl, ambulanceScene.contentUrl);
assert.equal(run('').collisionUrl, ambulanceScene.collisionUrl);
assert.match(ambulanceScene.contentUrl, /\/ambulance\/index\.sog\?v=cleanup-pass7-32e0ec80$/);
assert.match(ambulanceScene.collisionUrl, /\/ambulance\/scene-collision\.glb\?v=surface-frame-v2$/);
assert.equal(run('?scene=ambulance-insta360').contentUrl, '../ambulance-insta360/index.sog');
assert.equal(run('?scene=ambulance-insta360').collisionUrl, '../ambulance-insta360/index.voxel.json');
assert.deepEqual(run('').sceneRotation, ambulanceScene.rotation);
assert.equal(run('?scene=other').sceneRotation, undefined);
assert.equal(run('?content=custom.sog').sceneRotation, undefined);
assert.equal(run('?content=custom.sog').contentUrl, 'custom.sog');
assert.equal(run('?content=custom.sog').collisionUrl, null);
assert.equal(run('?content=custom.sog&collision=own.glb').collisionUrl, 'own.glb');
assert.equal(run('?voxel=custom.voxel.json').collisionUrl, 'custom.voxel.json');
assert.equal(run('?collision=&voxel=ignored.json').collisionUrl, '');
assert.equal(run('?scene=other&content=mine.sog').collisionUrl, '../other/index.voxel.json');

// Run the actual support helper against the different collision interfaces.
const helper = js.match(/    const sceneSurfaceBelow = ([\s\S]*?^    };)/m)[1];
const below = vm.runInNewContext(`(${helper.replace(/;$/, '')})`);
let rays = 0;
const first = {x: 0, y: -.3, z: 0}, lower = {x: 0, y: -.8, z: 0};
assert.equal(below({triangles: {count: 120}, queryRay: () => {rays++; return first;},
    isFreeAt: () => {throw Error('Mesh must not use voxel occupancy heuristic');}}, 0, 0, 0, 2), first);
assert.equal(rays, 1);
rays = 0;
assert.equal(below({queryRay: () => ++rays === 1 ? first : lower, isFreeAt: () => true}, 0, 0, 0, 2), lower);
assert.equal(rays, 2);
assert.equal(below({queryRay: () => first, isFreeAt: () => false}, 0, 0, 0, 2), first);
assert.equal(below({queryRay: () => null}, 0, 0, 0, 2), null);
const calibrate = js.match(/    const calibrateDropBias = ([\s\S]*?^    };)/m)[1];
const calibrateMesh = vm.runInNewContext(`(${calibrate.replace(/;$/, '')})`, {
    getCollision: () => ({triangles: {count: 120}}),
    viewer: {picker: {pickSurface: () => {throw Error('Measured mesh must not use Gaussian calibration');}}}
});
assert.equal(await calibrateMesh({}), 0);

// Use the shipped PlayCanvas Entity/Quat implementations, not transform mocks.
const frame = createSceneFrame(ambulanceScene.rotation);
const layout = JSON.parse(await readFile(file('viewers/props-lab/layout-ambulance.json')));
const engineRoot = new sceneTools.Entity('root'), group = new sceneTools.Entity('source');
engineRoot.addChild(group);
group.setLocalRotation(...ambulanceScene.rotation);
const close = (a, b) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-6, `${a} != ${b}`));
const vector = v => [v.x, v.y, v.z], quaternion = q => [q.x, q.y, q.z, q.w];
for (const saved of layout.props) {
    const entity = new sceneTools.Entity(saved.id);
    group.addChild(entity);
    entity.setLocalPosition(...saved.position);
    entity.setLocalEulerAngles(...saved.eulerAngles);
    entity.setLocalScale(saved.scale, saved.scale, saved.scale);
    close(vector(entity.getPosition()), frame.point(saved.position));
    close(vector(entity.getLocalPosition()), saved.position);
    const q = new sceneTools.Quat().setFromEulerAngles(...saved.eulerAngles);
    const expected = frame.placement({position: saved.position, rotation: quaternion(q), scale: saved.scale});
    assert.ok(Math.abs(quaternion(entity.getRotation()).reduce((sum, n, i) => sum + n * expected.rotation[i], 0)) > 1 - 1e-6);
    const moved = vector(entity.getPosition());
    moved[1] -= .13;
    entity.setPosition(...moved);
    close(frame.point(vector(entity.getLocalPosition())), moved);
    close(vector(entity.getLocalScale()), [saved.scale, saved.scale, saved.scale]);
}
const spawned = new sceneTools.Entity('new');
group.addChild(spawned);
spawned.setPosition(.2, .1, .3);
spawned.setEulerAngles(0, 0, 0);
const stored = {position: vector(spawned.getLocalPosition()), euler: vector(spawned.getLocalEulerAngles())};
const restored = new sceneTools.Entity('restored');
group.addChild(restored);
restored.setLocalPosition(...stored.position);
restored.setLocalEulerAngles(...stored.euler);
close(vector(restored.getPosition()), [.2, .1, .3]);
assert.ok(Math.abs(restored.getRotation().w) > 1 - 1e-6);
engineRoot.destroy();

assert.match(js, /if \(config\?\.noui\) panel\.style\.display = 'none';/);
assert.match(html, /index\.js\?v=surface-frame-v2/);
assert.match(js, /index\.js\?v=surface-frame-v2/);
assert.match(html, /props\.js\?v=surface-frame-v2/);
console.log('Props Lab checks passed: shared assets and overrides, mesh/voxel support, saved source transforms, world-Y motion, upright new props, noui and module revisions.');
