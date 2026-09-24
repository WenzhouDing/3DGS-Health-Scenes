#!/usr/bin/env node
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createEditHistory } from '../viewers/ambulance/edit-history.mjs';

function fixture(options = {}) {
  let state = { pose: { savedAt: 'initial', joints: [{ id: 'shoulder', angles: [0, 0, 0] }] },
    placement: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: 1 }, offsets: [0, 0, 0] };
  const updates = [], restored = [];
  let allowed = true, throwRestore = false, serial = 0;
  const capture = () => ({ ...structuredClone(state), pose: { ...structuredClone(state.pose), savedAt: `capture-${serial++}` } });
  const history = createEditHistory({ capture, restore: value => {
    if (throwRestore) throw new Error('Restore failed');
    restored.push(value); state = structuredClone(value);
  }, canRestore: value => { if (options.mutateGate) value.placement.scale = 999; return allowed; },
  onChange: value => updates.push(value), ...options });
  return { history, capture, updates, restored,
    get state() { return state; }, set state(value) { state = value; },
    allow(value) { allowed = value; }, failRestore(value) { throwRestore = value; },
    angle(value) { state.pose.joints[0].angles[0] = value; },
    instant(value) { const before = capture(); state.pose.joints[0].angles[0] = value; history.record(before); } };
}

test('a full gesture is one reversible edit, with placement and offsets restored', () => {
  const f = fixture(), h = f.history;
  assert.equal(h.begin(), true); assert.equal(h.begin(), false);
  f.angle(10); f.angle(20); f.state.placement.position[1] = .1; f.state.offsets[1] = .1;
  assert.equal(h.end(), true); assert.equal(f.updates.at(-1).undoCount, 1);
  assert.equal(h.undo(), true); assert.equal(f.state.pose.joints[0].angles[0], 0);
  assert.equal(f.state.placement.position[1], 0); assert.equal(f.state.offsets[1], 0);
  assert.equal(h.redo(), true); assert.equal(f.state.pose.joints[0].angles[0], 20);
  assert.equal(f.state.placement.position[1], .1); assert.equal(f.state.offsets[1], .1);
});

test('no-op timestamps and blocked drags preserve redo without false entries', () => {
  const f = fixture(), h = f.history;
  f.instant(30); h.undo(); const updates = f.updates.length;
  h.begin(); assert.equal(h.end(), false);
  assert.equal(h.record(f.capture()), false);
  f.state.offsets[0] = 1; // Derived UI metadata alone is not a pose edit.
  h.begin(); f.state.offsets[0] = 0; assert.equal(h.end(), false);
  assert.equal(f.updates.length, updates); assert.equal(h.canRedo, true);
  h.redo(); assert.equal(f.state.pose.joints[0].angles[0], 30);
});

test('pointer cancellation restores baseline and preserves both history stacks', () => {
  const f = fixture(), h = f.history;
  f.instant(10); f.instant(20); h.undo();
  const updates = f.updates.length;
  h.begin(); f.angle(45); f.allow(false); h.end({ cancelled: true });
  assert.equal(f.state.pose.joints[0].angles[0], 10);
  assert.equal(h.inGesture, false); assert.equal(h.canUndo, true); assert.equal(h.canRedo, true);
  assert.equal(f.updates.length, updates);
  f.allow(true); h.redo(); assert.equal(f.state.pose.joints[0].angles[0], 20);
});

test('undo during a gesture commits it once then restores its baseline', () => {
  const f = fixture(), h = f.history;
  f.instant(10); h.begin(); f.angle(25);
  assert.equal(h.undo(), true); assert.equal(f.state.pose.joints[0].angles[0], 10);
  assert.equal(h.inGesture, false); assert.equal(h.canRedo, true);
  h.redo(); assert.equal(f.state.pose.joints[0].angles[0], 25);
  h.undo(); h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 0);
});

test('only an actual new edit discards the redo branch', () => {
  const f = fixture(), h = f.history;
  f.instant(10); f.instant(20); h.undo();
  h.begin(); f.angle(15); h.end();
  assert.equal(h.canRedo, false); assert.equal(h.redo(), false);
  h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 10);
});

test('canRestore rejects undo and redo atomically and can later permit them', () => {
  const f = fixture(), h = f.history;
  f.instant(10); const updates = f.updates.length;
  f.allow(false); assert.equal(h.undo(), false);
  assert.equal(f.state.pose.joints[0].angles[0], 10); assert.equal(h.canUndo, true); assert.equal(h.canRedo, false);
  assert.equal(f.updates.length, updates);
  f.allow(true); h.undo(); f.allow(false); assert.equal(h.redo(), false);
  assert.equal(f.state.pose.joints[0].angles[0], 0); assert.equal(h.canRedo, true);
  f.allow(true); h.redo(); assert.equal(f.state.pose.joints[0].angles[0], 10);
});

test('history snapshots cannot be corrupted by live state or gate mutations', () => {
  const f = fixture({ mutateGate: true }), h = f.history;
  const before = f.capture(); f.angle(12); h.record(before);
  before.pose.joints[0].angles[0] = 99;
  h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 0); assert.equal(f.state.placement.scale, 1);
  f.restored[0].placement.scale = 888;
  h.redo(); assert.equal(f.state.placement.scale, 1);
});

test('bounded history evicts only the oldest entries', () => {
  const f = fixture({ limit: 2 }), h = f.history;
  f.instant(10); f.instant(20); f.instant(30);
  h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 20);
  h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 10);
  assert.equal(h.undo(), false);
  h.redo(); h.redo(); assert.equal(f.state.pose.joints[0].angles[0], 30);
});

test('instant edit arriving during a gesture retains separate reversible actions', () => {
  const f = fixture(), h = f.history;
  h.begin(); f.angle(10); const beforeInstant = f.capture(); f.angle(40); h.record(beforeInstant);
  assert.equal(h.inGesture, false); assert.equal(f.updates.at(-1).undoCount, 2);
  h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 10);
  h.undo(); assert.equal(f.state.pose.joints[0].angles[0], 0);
});

test('failed restore keeps the undo entry and state can be restored later', () => {
  const f = fixture(), h = f.history;
  f.instant(10); f.failRestore(true); assert.throws(() => h.undo(), /Restore failed/);
  assert.equal(h.canUndo, true); assert.equal(h.canRedo, false);
  f.failRestore(false); assert.equal(h.undo(), true);
});

test('invalid configuration or numeric snapshots fail explicitly', () => {
  assert.throws(() => createEditHistory({ capture: () => {}, restore: () => {}, limit: 0 }), /limit/);
  const f = fixture(); f.state.placement.scale = NaN;
  assert.throws(() => f.history.begin(), /finite/);
});
