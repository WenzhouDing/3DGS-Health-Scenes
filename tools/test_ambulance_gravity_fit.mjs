#!/usr/bin/env node
/** Actual captured-body gravity audit. Rendering resources alone are mocked;
 * scan bytes, articulated pose, sample selection and scene contacts are real.
 * Optional AMBULANCE_GRAVITY_AUDIT_PATH writes the measured audit as JSON.
 */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile, writeFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {fileURLToPath} from 'node:url';
import {performance} from 'node:perf_hooks';
import {createAmbulanceManikin} from '../viewers/ambulance/manikin-scan.mjs';
import {createContactChecker} from '../viewers/ambulance/contact-check.mjs';
import {createGravityTest} from '../viewers/ambulance/gravity-test.mjs';
import {createMotion} from '../viewers/mannequin-articulation/motion-core.mjs';
import {buildPreset} from '../viewers/mannequin-articulation/pose-presets.mjs';

const sha256 = bytes => createHash('sha256').update(bytes).digest('hex');
const percentile = (values, q) => [...values].sort((a,b) => a-b)[Math.min(values.length-1, Math.floor(q*values.length))];

test('actual captured manikin falls vertically and stops on modeled surfaces', async t => {
  const previousFetch = globalThis.fetch, previousLocation = globalThis.location;
  globalThis.location = {hostname:'127.0.0.1'};
  globalThis.fetch = async url => {
    try {
      const bytes = await readFile(fileURLToPath(url));
      return {ok:true, status:200, text:async()=>bytes.toString(), json:async()=>JSON.parse(bytes),
        arrayBuffer:async()=>bytes.buffer.slice(bytes.byteOffset,bytes.byteOffset+bytes.byteLength)};
    } catch (error) {return {ok:false,status:error.code==='ENOENT'?404:500};}
  };
  t.after(()=>{globalThis.fetch=previousFetch;globalThis.location=previousLocation;});
  class Entity {addChild(){} addComponent(){} setLocalPosition(){} setLocalRotation(){} setLocalScale(){} destroy(){}}
  const pc={Entity,GSplatData:class{},GSplatResource:class{destroy(){}},WORKBUFFER_UPDATE_AUTO:0};
  const scan=await createAmbulanceManikin({app:{root:new Entity(),graphicsDevice:{}},pc});
  t.after(()=>scan.dispose());
  const configBytes=await readFile(new URL('../viewers/ambulance/bed-placement.json',import.meta.url));
  const config=JSON.parse(configBytes), motion=createMotion(scan.annotations,scan.manifest);
  for(const [id,angles] of Object.entries({...buildPreset('lying',motion,scan.manifest).targets,...config.jointOverrides}))
    motion.setTarget(id,angles,{immediate:true});
  const parts=motion.evaluate().parts, origin=config.placement.position.slice();
  const fixedJoints=motion.exportPose().joints;
  const initialSamples=scan.getCollisionSamples(parts,config.placement);
  assert.equal(initialSamples.length,11513,'Audit the current complete collision-sample policy');
  assert.equal(new Set(initialSamples.map(s=>s.id)).size,initialSamples.length);
  assert.ok(initialSamples.every(s=>s.radius>=0 && s.radius<=.01*config.placement.scale+1e-12));
  const checker=createContactChecker(config), baseline=checker.evaluate(initialSamples);
  assert.equal(baseline.blocked,false);

  // A held pose can reuse world samples with pure translation. Verify this
  // optimization independently against the complete adapter transformation.
  const shifted=structuredClone(config.placement);shifted.position[1]+=.173;
  const adapterShifted=scan.getCollisionSamples(parts,shifted);
  adapterShifted.forEach((s,i)=>{
    assert.equal(s.id,initialSamples[i].id);assert.equal(s.radius,initialSamples[i].radius);
    s.position.forEach((v,k)=>assert.ok(Math.abs(v-initialSamples[i].position[k]-(k===1?.173:0))<1e-12));
  });

  const audit={schema:'ambulance-gravity-actual-sample-audit',config_sha256:sha256(configBytes),
    controller_sha256:sha256(await readFile(new URL('../viewers/ambulance/gravity-test.mjs',import.meta.url))),
    checker_sha256:sha256(await readFile(new URL('../viewers/ambulance/contact-check.mjs',import.meta.url))),
    sampleInfo:scan.collisionSampleInfo,scope:'Held articulated pose, vertical translation only; uncalibrated scene units; no ragdoll, body self-collision or deformable mattress.',
    baseline:{blocked:baseline.blocked,contacts:baseline.contacts.length,maxPenetration:baseline.maxPenetration},cases:[]};

  function harness(collision=config,options={}) {
    const {auditCommits=true,stopOnBlock=false,...controllerOptions}=options;
    const localChecker=createContactChecker(collision), samples=initialSamples.map(s=>({...s,position:s.position.slice()}));
    let position=origin.slice(), evaluations=0;
    const committed=[],durations=[];
    function evaluateAt(at) {
      for(let i=0;i<samples.length;i++) for(let k=0;k<3;k++) samples[i].position[k]=initialSamples[i].position[k]+at[k]-origin[k];
      evaluations++;
      return localChecker.evaluate(samples,{stopOnBlock});
    }
    const gravity=createGravityTest({getPosition:()=>position.slice(),setPosition:at=>{
      const report=auditCommits?evaluateAt(at):null;
      if(report)assert.equal(report.blocked,false,`Gravity committed overlap: ${JSON.stringify(report.violations.slice(0,2))}`);
      assert.equal(at[0],origin[0]);assert.equal(at[2],origin[2]);
      position=at.slice();committed.push({position:position.slice(),contacts:report?.contacts.length,maxPenetration:report?.maxPenetration});
    },evaluateAt,...controllerOptions});
    return {gravity,committed,durations,evaluateAt,get position(){return position.slice();},get evaluations(){return evaluations;},
      update(dt){const before=performance.now();const result=gravity.update(dt);durations.push(performance.now()-before);return result;}};
  }
  function finish(h,dt=1/60,maxFrames=1000) {
    let frames=0;
    while(h.gravity.getState().status==='running' && frames++<maxFrames)h.update(dt);
    assert.ok(frames<maxFrames,'Gravity must terminate within the simulation budget');
    const state=h.gravity.getState(),report=h.evaluateAt(h.position);
    assert.equal(state.status,'landed',JSON.stringify(state));
    assert.equal(state.velocity,0);
    assert.equal(report.blocked,false);
    assert.ok(report.contacts.some(c=>c.normal[1]>=.5),'Landing requires upward-facing contact');
    return {state,report,frames};
  }
  function record(name,h,result) {
    audit.cases.push({name,frames:result.frames,evaluations:h.evaluations,position:h.position,state:result.state,
      violations:result.report.violations.length,maxPenetration:result.report.maxPenetration,
      supportProxies:[...new Set(result.report.contacts.filter(c=>c.normal[1]>=.5).map(c=>c.proxy))],
      updateMilliseconds:h.durations.length?{median:percentile(h.durations,.5),p95:percentile(h.durations,.95),max:Math.max(...h.durations)}:null});
  }

  await t.test('current near-supported pose settles without a lift or blocking overlap',()=>{
    const h=harness();h.gravity.start({lift:0});const result=finish(h);
    assert.ok(h.position[1]<=origin[1]+1e-12 && h.position[1]>origin[1]-.015);
    assert.ok(result.report.contacts.some(c=>c.proxy.startsWith('mattress-')));
    record('current fitted near-support pose',h,result);
  });
  await t.test('raised fitted pose accelerates then lands on the mattress without changing pose',()=>{
    const h=harness();h.gravity.start({lift:.25});
    assert.equal(h.gravity.getState().status,'running');
    const liftedY=h.position[1];assert.ok(Math.abs(liftedY-origin[1]-.25)<1e-10);
    h.update(1/60);const first=h.gravity.getState();h.update(1/60);const second=h.gravity.getState();
    assert.ok(first.velocity<0 && second.velocity<first.velocity,'Downward speed must increase in free fall');
    assert.ok(h.position[1]<liftedY);
    const result=finish(h);assert.ok(result.report.contacts.some(c=>c.proxy.startsWith('mattress-')));
    assert.ok(h.position[1]>origin[1]-.02,'The body must not pass through the thin mattress');
    assert.deepEqual(motion.exportPose().joints,fixedJoints);
    record('actual mattress at 60 Hz',h,result);
  });
  await t.test('large delayed frames cannot tunnel through the actual thin mattress',()=>{
    const h=harness();h.gravity.start({lift:.25});const result=finish(h,2);
    assert.ok(result.report.contacts.some(c=>c.proxy.startsWith('mattress-')));
    assert.ok(h.position[1]>origin[1]-.02);
    record('actual mattress with 2-second frame input',h,result);
  });
  await t.test('actual-body floor-only drop respects the measured tilted floor',()=>{
    const floor=config.collision.surfaces.find(s=>s.id==='floor');
    const h=harness({surfaces:[floor],tolerances:config.collision.tolerances});
    h.gravity.start({lift:.1});const result=finish(h,.25);
    assert.ok(result.report.contacts.some(c=>c.proxy==='floor'));
    assert.ok(h.position[1]<origin[1]-.1,'With the bed removed, the same body should reach the lower floor');
    const deeper=h.position;deeper[1]-=.03;
    assert.equal(h.evaluateAt(deeper).blocked,true,'Floor landing must be adjacent to the true collision boundary');
    record('actual-body measured floor, mattress intentionally omitted',h,result);
  });
  await t.test('roof-obstructed lift is rejected atomically without moving the body',()=>{
    const h=harness(config,{maxLift:3});h.gravity.start({lift:2});
    assert.equal(h.gravity.getState().status,'obstructed');assert.equal(h.gravity.getState().reason,'lift-obstructed');assert.deepEqual(h.position,origin);
    assert.equal(h.committed.length,0);
    audit.cases.push({name:'roof-obstructed lift',state:h.gravity.getState(),position:h.position,committed:0,evaluations:h.evaluations});
  });
  await t.test('pause and resume preserve the falling body state',()=>{
    const h=harness();h.gravity.start({lift:.25});h.update(1/60);h.gravity.pause();
    const position=h.position,velocity=h.gravity.getState().velocity;h.update(2);
    assert.deepEqual(h.position,position);assert.equal(h.gravity.getState().velocity,velocity);
    h.gravity.resume();const result=finish(h);record('pause/resume',h,result);
  });
  await t.test('production-equivalent probes preserve safe landings with checks outside the timing window',()=>{
    const durations=[],starts=[],finals=[];
    for(let trial=0;trial<5;trial++){
      const h=harness(config,{auditCommits:false,stopOnBlock:true}),before=performance.now();
      h.gravity.start({lift:.25});starts.push(performance.now()-before);
      assert.equal(h.evaluateAt(h.position).blocked,false);
      let frames=0;
      while(h.gravity.getState().status==='running' && frames++<100){
        h.update(1/60);
        assert.equal(h.evaluateAt(h.position).blocked,false,'Validate each production-equivalent commit outside its timer');
      }
      assert.equal(h.gravity.getState().status,'landed');
      durations.push(...h.durations);finals.push(h.durations.at(-1));
    }
    audit.productionTiming={note:'Node CPU only; GPU render/upload excluded. Checks after each committed frame occur outside measured update.',
      trials:5,frames:durations.length,updateMilliseconds:{median:percentile(durations,.5),p95:percentile(durations,.95),max:Math.max(...durations)},
      finalContactFrameMilliseconds:{median:percentile(finals,.5),max:Math.max(...finals)},
      atomicLiftMilliseconds:{median:percentile(starts,.5),max:Math.max(...starts)}};
  });
  assert.deepEqual(scan.getCollisionSamples(parts,config.placement),initialSamples,'Drop probes must not mutate capture/render placement');
  assert.equal(audit.cases.length,6,'Every gravity scenario must complete before publishing a passing audit');
  assert.ok(audit.productionTiming?.frames>0,'Production-equivalent checks must also pass before publishing the audit');
  audit.result='passed';
  if(process.env.AMBULANCE_GRAVITY_AUDIT_PATH)await writeFile(process.env.AMBULANCE_GRAVITY_AUDIT_PATH,JSON.stringify(audit,null,2)+'\n');
  console.log(`Actual gravity audit: ${initialSamples.length} samples; ${audit.cases.length} scenarios; no committed blocking overlaps.`);
});
