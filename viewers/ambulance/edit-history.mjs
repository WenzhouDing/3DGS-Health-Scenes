/** Bounded, synchronous pose history. A drag is one transaction, not one entry
 * per pointer move. Capture timestamps and derived offsets do not create edits.
 */
const clone = value => structuredClone(value);

function canonical(value) {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonical(value[key])}`).join(',')}}`;
  }
  if (typeof value === 'number' && !Number.isFinite(value)) throw new TypeError('History values must be finite');
  return JSON.stringify(value);
}

function signature(snapshot) {
  if (!snapshot || !Array.isArray(snapshot.pose?.joints) || !snapshot.placement || typeof snapshot.placement !== 'object') {
    throw new TypeError('History snapshot requires pose.joints and placement');
  }
  const joints = [...snapshot.pose.joints];
  if (joints.every(joint => typeof joint?.id === 'string')) joints.sort((a, b) => a.id.localeCompare(b.id));
  return canonical({ joints, placement: snapshot.placement });
}

export function createEditHistory({ capture, restore, canRestore = () => true, onChange = () => {}, limit = 40 }) {
  if ([capture, restore, canRestore, onChange].some(fn => typeof fn !== 'function')) throw new TypeError('History callbacks must be functions');
  if (!Number.isInteger(limit) || limit < 1) throw new RangeError('History limit must be a positive integer');
  const undoStack = [], redoStack = [];
  let gesture = null;
  const snapshot = value => {
    const result = clone(value);
    signature(result);
    return result;
  };
  const current = () => snapshot(capture());
  const notify = () => onChange({ canUndo: undoStack.length > 0, canRedo: redoStack.length > 0,
    undoCount: undoStack.length, redoCount: redoStack.length });
  function commit(before, after) {
    if (signature(before) === signature(after)) return false;
    undoStack.push(snapshot(before));
    if (undoStack.length > limit) undoStack.shift();
    redoStack.length = 0;
    return true;
  }
  function begin() {
    if (gesture) return false;
    gesture = current();
    return true;
  }
  function end({ cancelled = false } = {}) {
    if (!gesture) return false;
    const before = gesture, after = current();
    if (cancelled) {
      // Immediate rollback to the user's pre-gesture state. It must not be
      // rejected merely because collision-prevention was toggled mid-drag.
      if (signature(before) !== signature(after)) restore(snapshot(before));
      gesture = null;
      return false;
    }
    const changed = commit(before, after);
    gesture = null;
    if (changed) notify();
    return changed;
  }
  function record(before) {
    const previous = snapshot(before), after = current();
    let changed = false;
    // If an instant edit arrives during a gesture, its explicit "before"
    // snapshot is also the endpoint of that gesture; do not merge the actions.
    if (gesture) { changed = commit(gesture, previous); gesture = null; }
    changed = commit(previous, after) || changed;
    if (changed) notify();
    return changed;
  }
  function travel(from, to) {
    end();
    if (!from.length) return false;
    const destination = from[from.length - 1], previous = current();
    // Callbacks receive copies so rejection or mutation cannot damage history.
    if (!canRestore(snapshot(destination))) return false;
    restore(snapshot(destination));
    from.pop();
    to.push(previous);
    if (to.length > limit) to.shift();
    notify();
    return true;
  }
  return {
    begin, end, record,
    undo: () => travel(undoStack, redoStack),
    redo: () => travel(redoStack, undoStack),
    get canUndo() { return undoStack.length > 0; },
    get canRedo() { return redoStack.length > 0; },
    get inGesture() { return gesture !== null; }
  };
}
