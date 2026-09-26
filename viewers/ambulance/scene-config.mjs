/** Shared accepted ambulance assets. Both scene viewers reuse these files.
 * Change the revision when replacing the corresponding accepted asset/frame.
 */
import { AMBULANCE_LEVEL_ROTATION } from './scene-frame.mjs';

export const AMBULANCE_SCENE_REVISION = 'cleanup-pass7-32e0ec80';
export const AMBULANCE_SURFACE_REVISION = 'surface-frame-v2';

const asset = (path, revision) => {
    const url = new URL(path, import.meta.url);
    url.searchParams.set('v', revision);
    return url.href;
};

export const ambulanceScene = Object.freeze({
    revision: AMBULANCE_SCENE_REVISION,
    contentUrl: asset('./index.sog', AMBULANCE_SCENE_REVISION),
    settingsUrl: asset('./settings.json', AMBULANCE_SURFACE_REVISION),
    collisionUrl: asset('./scene-collision.glb', AMBULANCE_SURFACE_REVISION),
    rotation: AMBULANCE_LEVEL_ROTATION,
});
