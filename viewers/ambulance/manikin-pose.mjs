/** Check the whole path, including intermediate articulated poses. A rejected
 * request restores the last clear pose, not the colliding endpoint. */
export function moveJointChecked({motion, jointId, target, evaluate, enabled = true, maxStep = 2}) {
    if (!Array.isArray(target) || target.length !== 3 || !target.every(Number.isFinite)) throw new TypeError('Invalid joint target.');
    if (!(maxStep > 0) || !Number.isFinite(maxStep)) throw new TypeError('Invalid sweep step.');
    const from = motion.getState(jointId).angles;
    const baseline = motion.exportPose();
    // Clamp the requested endpoint using the actual fitted rig's travel limits.
    motion.setTarget(jointId, target, {immediate: true});
    const to = motion.getState(jointId).angles;
    motion.importPose(baseline);
    const steps = Math.max(1, Math.ceil(Math.max(...to.map((n, i) => Math.abs(n - from[i]))) / maxStep));
    let accepted = baseline, report, rejected = null, fraction = 0;
    for (let i = 1; i <= steps; i++) {
        const t = i / steps;
        motion.setTarget(jointId, from.map((n, k) => n + (to[k] - n) * t), {immediate: true});
        report = evaluate(motion.evaluate().parts);
        if (enabled && report.blocked) { rejected = report; motion.importPose(accepted); break; }
        accepted = motion.exportPose(); fraction = t;
    }
    return {accepted: !rejected, fraction, rejected, report: rejected ? evaluate(motion.evaluate().parts) : report};
}
