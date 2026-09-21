"""Preserve a continuous native wrist pad and the reviewed distal finger pose.

Input/output are original back-scan coordinates. A smooth knuckle transition
joins the forearm's rigid pose to the prior finger pose. Keeping this field in
source coordinates makes subsequent forearm adjustments carry the whole hand.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def settings(parameters):
    c = dict(parameters)
    for key in ['referenceForearm', 'referenceHand']:
        tr = c[key]
        r = np.asarray(tr['rotation'], float); t = np.asarray(tr['translation'], float)
        if (r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all()
                or not np.isfinite(t).all() or not np.allclose(r @ r.T, np.eye(3), atol=1e-8)
                or not np.isclose(np.linalg.det(r), 1, atol=1e-8)):
            raise ValueError('Invalid wrist reference transform')
        c[key] = {'rotation': r, 'translation': t, 'scale': float(tr['scale'])}
    scale = c['referenceForearm']['scale']
    if not np.isfinite(scale) or scale <= 0 or not np.isclose(scale, c['referenceHand']['scale'], atol=1e-12):
        raise ValueError('Wrist reference poses must share the same positive scale')
    for key in ['origin', 'longitudinalAxis']:
        c[key] = np.array(c[key], dtype=float, copy=True)
        if c[key].shape != (3,) or not np.isfinite(c[key]).all():
            raise ValueError('Invalid wrist support frame')
    length = np.linalg.norm(c['longitudinalAxis'])
    if length < 1e-9:
        raise ValueError('Zero wrist longitudinal axis')
    c['longitudinalAxis'] /= length
    c['longitudinalFade'] = np.asarray(c['longitudinalFade'], float)
    interval = c['longitudinalFade']
    if interval.shape != (2,) or not np.isfinite(interval).all() or interval[1] - interval[0] < .025:
        raise ValueError('Wrist transition must span at least .025 scan units')
    if 'thumbLongitudinalFade' in c:
        c['lateralAxis'] = np.array(c['lateralAxis'], dtype=float, copy=True)
        lateral = c['lateralAxis']
        if lateral.shape != (3,) or not np.isfinite(lateral).all() or np.linalg.norm(lateral) < 1e-9:
            raise ValueError('Invalid thumb support axis')
        lateral /= np.linalg.norm(lateral)
        if abs(lateral @ c['longitudinalAxis']) > 1e-6:
            raise ValueError('Thumb and finger axes must be perpendicular')
        for name in ['thumbLongitudinalFade', 'thumbLateralFade']:
            c[name] = np.asarray(c[name], float)
            if c[name].shape != (2,) or not np.isfinite(c[name]).all() or abs(c[name][1]-c[name][0]) < .01:
                raise ValueError('Invalid thumb fade interval')
    return c


def displacement_and_jacobian(points, parameters):
    c = settings(parameters); f = c['referenceForearm']; h = c['referenceHand']
    p = np.asarray(points, float)
    reference = f['scale'] * np.einsum('ni,ji->nj', p, f['rotation']) + f['translation']
    station = np.einsum('ni,i->n', reference - c['origin'], c['longitudinalAxis'])
    start, end = c['longitudinalFade']
    s = np.clip((station - start) / (end - start), 0, 1)
    weight = s * s * (3 - 2 * s)
    station_gradient = f['scale'] * f['rotation'].T @ c['longitudinalAxis']
    gradient = (6 * s * (1 - s) / (end - start))[:, None] * station_gradient
    if 'thumbLongitudinalFade' in c:
        lateral = np.einsum('ni,i->n', reference-c['origin'], c['lateralAxis'])
        a,b = c['thumbLongitudinalFade']; lateral_start,lateral_end = c['thumbLateralFade']
        tu = np.clip((station-a)/(b-a),0,1); tv = np.clip((lateral-lateral_start)/(lateral_end-lateral_start),0,1)
        us = tu*tu*(3-2*tu); vs = tv*tv*(3-2*tv); tw = us*vs
        lateral_gradient = f['scale'] * f['rotation'].T @ c['lateralAxis']
        tg = (6*tu*(1-tu)/(b-a)*vs)[:,None]*station_gradient
        tg += (us*6*tv*(1-tv)/(lateral_end-lateral_start))[:,None]*lateral_gradient
        gradient = gradient*(1-tw[:,None]) + tg*(1-weight[:,None])
        weight = 1-(1-weight)*(1-tw)
    rotation = f['rotation'].T @ h['rotation']
    translation = f['rotation'].T @ (h['translation'] - f['translation']) / f['scale']
    full_delta = np.einsum('ni,ji->nj', p, rotation - np.eye(3)) + translation
    delta = weight[:, None] * full_delta
    jacobian = (np.eye(3)[None, :, :] + weight[:, None, None] * (rotation - np.eye(3))
                + full_delta[:, :, None] * gradient[:, None, :])
    return delta, jacobian, weight


def preserve_pad_align_fingers(data, parameters):
    c = settings(parameters)
    delta, jacobian, weight = displacement_and_jacobian(data[:, :3], c)
    changed = weight > 0
    determinants = np.linalg.det(jacobian)
    if np.any(determinants <= 0):
        raise ValueError('Wrist transition folds the source; widen or reposition the transition')
    out = data.copy()
    out[changed, :3] += delta[changed]
    if changed.any():
        selected = data[changed]
        q = selected[:, 3:7].astype(float)
        r = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_matrix()
        factors = np.einsum('nij,njk->nik', jacobian[changed], r) * np.exp(selected[:, None, 7:10])
        vectors, sigma, _ = np.linalg.svd(factors, full_matrices=False)
        vectors[np.linalg.det(vectors) < 0, :, 0] *= -1
        quat = Rotation.from_matrix(vectors).as_quat()
        out[changed, 3:7] = np.c_[quat[:, 3], quat[:, :3]]
        out[changed, 7:10] = np.log(np.maximum(sigma, np.finfo(float).tiny))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    return out, {
        'method': 'continuous rigid wrist pad with smooth transition to the previous distal finger pose',
        'sourceCapture': 'back', 'frame': 'original back source; support measured in reference forearm pose',
        'affected': int(changed.sum()), 'rigidDistalRows': int((weight == 1).sum()),
        'transitionRows': int(((weight > 0) & (weight < 1)).sum()),
        'longitudinalFade': c['longitudinalFade'].tolist(),
        'minJacobianDeterminant': float(determinants.min(initial=1)),
        'minimumJacobianStretch': float(singular.min(initial=1)),
        'maximumJacobianStretch': float(singular.max(initial=1)),
        'maximumNativeDisplacement': float(np.linalg.norm(delta, axis=1).max(initial=0)),
        'originalColorAndOpacityUnchanged': True,
    }
