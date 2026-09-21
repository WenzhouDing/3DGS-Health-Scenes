"""Small, editable left-finger registration correction in aligned front space.

Apply to the back capture's left_hand only, before fusion weights. The distal
fingers receive a lengthwise translation; a smooth transition leaves the wrist
and thumb fixed. It never moves one skin surface toward the opposing surface.
"""
import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_ALIGNMENT = {
    'amount': .0033,
    'origin': [.704, .065, .584],
    'longitudinalAxis': [.68, 0, .7332],
    'lateralAxis': [.7332, 0, -.68],
    'longitudinalFade': [.080, .125],
    'thumbFade': [.043, .055],
}


def settings(parameters):
    c = {**DEFAULT_ALIGNMENT, **parameters}
    c['amount'] = float(c['amount'])
    if not np.isfinite(c['amount']) or abs(c['amount']) > .006:
        raise ValueError('Finger correction must be within .006 scan units')
    for key in ['origin', 'longitudinalAxis', 'lateralAxis']:
        c[key] = np.asarray(c[key], float)
        if c[key].shape != (3,) or not np.isfinite(c[key]).all():
            raise ValueError('Invalid finger frame: ' + key)
    for key in ['longitudinalAxis', 'lateralAxis']:
        length = np.linalg.norm(c[key])
        if length < 1e-9:
            raise ValueError('Zero finger axis')
        c[key] = c[key] / length
    if abs(np.dot(c['longitudinalAxis'], c['lateralAxis'])) > 1e-6:
        raise ValueError('Finger axes must be perpendicular')
    for key in ['longitudinalFade', 'thumbFade']:
        c[key] = np.asarray(c[key], float)
        if (c[key].shape != (2,) or not np.isfinite(c[key]).all()
                or c[key][1] - c[key][0] < .008):
            raise ValueError('Finger transition must span at least .008 scan units')
    return c


def displacement_and_jacobian(points, parameters):
    c = settings(parameters)
    relative = np.asarray(points, float) - c['origin']
    longitudinal = np.einsum('ni,i->n', relative, c['longitudinalAxis'])
    lateral = np.einsum('ni,i->n', relative, c['lateralAxis'])
    a, b = c['longitudinalFade']; p, q = c['thumbFade']
    s = np.clip((longitudinal - a) / (b - a), 0, 1)
    t = np.clip((lateral - p) / (q - p), 0, 1)
    ss = s * s * (3 - 2 * s); ts = t * t * (3 - 2 * t)
    gradient = (6 * s * (1 - s) / (b - a) * (1 - ts))[:, None] * c['longitudinalAxis']
    gradient -= (ss * 6 * t * (1 - t) / (q - p))[:, None] * c['lateralAxis']
    vector = c['amount'] * c['longitudinalAxis']
    delta = (ss * (1 - ts))[:, None] * vector
    jacobian = np.eye(3)[None, :, :] + vector[None, :, None] * gradient[:, None, :]
    return delta, jacobian


def align_left_digits(data, parameters):
    """Transport both means and covariance; keep other source fields intact."""
    c = settings(parameters)
    delta, jacobian = displacement_and_jacobian(data[:, :3], c)
    moved = np.any(delta != 0, axis=1)
    deformed = np.any(jacobian != np.eye(3), axis=(1, 2))
    if np.any(np.linalg.det(jacobian) <= 0):
        raise ValueError('Finger correction folds the transition; reduce amount or widen the fade')
    out = data.copy()
    out[moved, :3] += delta[moved]
    if deformed.any():
        selected = data[deformed]
        q = selected[:, 3:7].astype(float)
        rotations = Rotation.from_quat(np.c_[q[:, 1:], q[:, 0]]).as_matrix()
        factors = rotations * np.exp(selected[:, None, 7:10])
        factors = np.einsum('nij,njk->nik', jacobian[deformed], factors)
        vectors, sigma, _ = np.linalg.svd(factors, full_matrices=False)
        vectors[np.linalg.det(vectors) < 0, :, 0] *= -1
        quaternion = Rotation.from_matrix(vectors).as_quat()
        out[deformed, 3:7] = np.c_[quaternion[:, 3], quaternion[:, :3]]
        out[deformed, 7:10] = np.log(np.maximum(sigma, np.finfo(float).tiny))
    singular = np.linalg.svd(jacobian, compute_uv=False)
    audit = {
        'method': 'lengthwise back-finger correction with smooth knuckle/thenar transition',
        'frame': 'aligned front raw', 'sourceCapture': 'back',
        'parameters': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in c.items()},
        'affected': int(moved.sum()), 'covariancesTransported': int(deformed.sum()),
        'maximumDisplacement': float(np.linalg.norm(delta, axis=1).max(initial=0)),
        'minJacobianDeterminant': float(np.linalg.det(jacobian).min(initial=1)),
        'maxJacobianStretch': float(singular.max(initial=1)),
        'minJacobianStretch': float(singular.min(initial=1)),
        'maximumDepthDisplacement': float(np.abs(delta[:, 1]).max(initial=0)),
        'colorsAndOriginalOpacityUnchanged': True,
    }
    return out, audit
