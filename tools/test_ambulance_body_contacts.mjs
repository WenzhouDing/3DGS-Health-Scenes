#!/usr/bin/env node
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile,writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import {performance} from 'node:perf_hooks';
import {createAmbulanceManikin} from '../viewers/ambulance/manikin-scan.mjs';
import {createBodyContactChecker} from '../viewers/ambulance/body-contact.mjs';
import {createContactChecker} from '../viewers/ambulance/contact-check.mjs';
import {createMotion,transformPoint} from '../viewers/mannequin-articulation/motion-core.mjs';
import {buildPreset} from '../viewers/mannequin-articulation/pose-presets.mjs';
import {moveJointChecked} from '../viewers/ambulance/manikin-pose.mjs';
const IDENTITY={position:[0,0,0],rotation:[0,0,0,1],scale:1};
const armParts=side=>[`${side}_upper_arm`,`${side}_forearm`,`${side}_hand`];
const sha=b=>createHash('sha256').update(b).digest('hex');
const median=values=>[...values].sort((a,b)=>a-b)[Math.floor(values.length/2)];

test('captured body envelopes protect arm motion without blocking fitted attachments',async t=>{
  const oldFetch=globalThis.fetch,oldLocation=globalThis.location;
  globalThis.location={hostname:'127.0.0.1'};
  globalThis.fetch=async url=>{try{const b=await readFile(fileURLToPath(url));return {ok:true,status:200,text:async()=>b.toString(),json:async()=>JSON.parse(b),arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)};}catch{return {ok:false,status:404};}};
  t.after(()=>{globalThis.fetch=oldFetch;globalThis.location=oldLocation;});
  class Entity{addChild(){}addComponent(){}setLocalPosition(){}setLocalRotation(){}setLocalScale(){}destroy(){}}
  const pc={Entity,GSplatData:class{},GSplatResource:class{destroy(){}},WORKBUFFER_UPDATE_AUTO:0};
  const scan=await createAmbulanceManikin({app:{root:new Entity(),graphicsDevice:{}},pc});t.after(()=>scan.dispose());
  const bytes=await readFile(new URL('../viewers/ambulance/bed-placement.json',import.meta.url)),config=JSON.parse(bytes);
  const motion=createMotion(scan.annotations,scan.manifest),targets={...buildPreset('lying',motion,scan.manifest).targets,...config.jointOverrides};
  for(const[id,angles]of Object.entries(targets))motion.setTarget(id,angles,{immediate:true});
  const baselinePose=motion.exportPose(),baselineParts=motion.evaluate().parts;
  const restSamples=scan.getCollisionSamples(new Map(),IDENTITY),samplesBefore=structuredClone(restSamples);
  const checker=createBodyContactChecker({restSamples,annotations:scan.annotations,referenceParts:baselineParts});
  const scene=createContactChecker(config),baseline=checker.evaluate(baselineParts,config.placement);
  const audit={schema:'ambulance-arm-body-contact-audit',config_sha256:sha(bytes),checker_sha256:sha(await readFile(new URL('../viewers/ambulance/body-contact.mjs',import.meta.url))),
    capturedSamples:restSamples.length,armSamples:checker.sampleCount,proxyCount:checker.proxies.length,referenceExceptions:checker.referenceExceptions,
    baseline:{blocked:baseline.blocked,maxPenetration:baseline.maxPenetration,contacts:baseline.contacts.length},policy:checker.policy,cases:[]};
  function translatedPart(part,destination,sourceParts=baselineParts){
    const poses=new Map(sourceParts),existing=sourceParts.get(part),points=restSamples.filter(s=>s.part===part).map(s=>transformPoint(existing,s.position));
    const center=[0,1,2].map(k=>points.reduce((sum,p)=>sum+p[k],0)/points.length);
    poses.set(part,{...existing,position:existing.position.map((v,k)=>v+destination[k]-center[k])});return poses;
  }
  await t.test('accepted pose is clear with default, null, and explicit moving-part filters',()=>{
    assert.equal(restSamples.length,11513);assert.equal(checker.sampleCount,1888);
    assert.equal(baseline.blocked,false);assert.equal(baseline.maxPenetration,0);
    assert.equal(checker.referenceExceptions.length,0,'Current fit should not need overlap allowances');
    assert.equal(checker.evaluate(baselineParts,config.placement,{movingParts:null}).sampleCount,1888);
    assert.equal(checker.evaluate(baselineParts,config.placement,{movingParts:[]}).sampleCount,0);
    const left=checker.evaluate(baselineParts,config.placement,{movingParts:armParts('left')});
    assert.equal(left.sampleCount,963);assert.ok(Object.keys(left.byPart).every(p=>p.startsWith('left_')));
    audit.cases.push({name:'accepted fitted pose and filters',passed:true});
  });
  await t.test('each arm can raise to 50 degrees through the actual checked manual path',()=>{
    for(const side of ['left','right']){
      motion.importPose(baselinePose);const id=`${side}_shoulder`,start=motion.getState(id).angles;
      const target=[50,start[1],start[2]];
      const result=moveJointChecked({motion,jointId:id,target,maxStep:1,evaluate:parts=>{
        const self=checker.evaluate(parts,config.placement,{stopOnBlock:true});
        return self.blocked?self:scene.evaluate(scan.getCollisionSamples(parts,config.placement),{stopOnBlock:true});
      }});
      assert.equal(result.accepted,true,JSON.stringify(result.rejected?.violations.slice(0,2)));
      assert.equal(motion.getState(id).angles[0],50);
    }
    motion.importPose(baselinePose);audit.cases.push({name:'left and right shoulder X to50 at1degree steps',passed:true});
  });
  await t.test('moving a captured forearm into the central torso is blocked',()=>{
    const parts=translatedPart('left_forearm',[0,.1,-.36]),report=checker.evaluate(parts,config.placement,{movingParts:['left_forearm']});
    assert.equal(report.blocked,true);assert.ok(report.violations.some(c=>c.targetPart==='torso'));
    const fast=checker.evaluate(parts,config.placement,{movingParts:['left_forearm'],stopOnBlock:true});
    assert.equal(fast.blocked,true);assert.equal(fast.complete,false);
    audit.cases.push({name:'captured forearm inside torso',violations:report.violations.length,maxPenetration:report.maxPenetration});
  });
  await t.test('the same samples cannot enter the posed head or opposite forearm',()=>{
    const headCenter=transformPoint(baselineParts.get('head'),[.01,.015,.18]);
    const head=checker.evaluate(translatedPart('left_forearm',headCenter),config.placement,{movingParts:['left_forearm']});
    assert.ok(head.violations.some(c=>c.targetPart==='head'));
    const proxy=checker.proxies.find(p=>p.part==='right_forearm'),center=transformPoint(baselineParts.get('right_forearm'),proxy.center);
    const other=checker.evaluate(translatedPart('left_forearm',center),config.placement,{movingParts:['left_forearm']});
    assert.ok(other.violations.some(c=>c.targetPart==='right_forearm'));
    assert.ok(other.contacts.every(c=>!['left_upper_arm','left_forearm','left_hand'].includes(c.targetPart)));
    audit.cases.push({name:'head and opposite arm',headViolations:head.violations.filter(c=>c.targetPart==='head').length,oppositeArmViolations:other.violations.filter(c=>c.targetPart==='right_forearm').length});
  });
  await t.test('world rigid placement preserves contact geometry and uniform scale rescales distances',()=>{
    const parts=translatedPart('left_forearm',[0,.1,-.36]);
    const native=checker.evaluate(parts,IDENTITY,{movingParts:['left_forearm']});
    const at={position:[2,-3,1],rotation:[0,Math.sin(.3),0,Math.cos(.3)],scale:.75};
    const posed=checker.evaluate(parts,at,{movingParts:['left_forearm']});
    const nativeHits=new Map(native.contacts.map(c=>[`${c.sampleId}|${c.proxy}`,c]));
    for(const c of posed.violations){const n=nativeHits.get(`${c.sampleId}|${c.proxy}`);assert.ok(n);assert.ok(Math.abs(c.clearance-.75*n.clearance)<1e-10);assert.ok(Math.abs(Math.hypot(...c.normal)-1)<1e-9);assert.ok(c.point.every(Number.isFinite));}
    assert.deepEqual(restSamples,samplesBefore);
    audit.cases.push({name:'placement/scale and source preservation',passed:true});
  });
  await t.test('metadata cannot mutate fitted proxies and invalid geometry fails explicitly',()=>{
    const before=checker.evaluate(baselineParts,config.placement);checker.proxies[0].radii[0]=100;
    assert.deepEqual(checker.evaluate(baselineParts,config.placement),before);
    assert.throws(()=>createBodyContactChecker({restSamples:[],annotations:scan.annotations}),/samples/);
    assert.throws(()=>checker.evaluate(baselineParts,{...IDENTITY,scale:0}),/scale/);
    const invalid=new Map(baselineParts);invalid.set('head',{position:[NaN,0,0],rotation:[0,0,0,1]});
    assert.throws(()=>checker.evaluate(invalid,IDENTITY),/finite/);
    audit.cases.push({name:'defensive copies and validation',passed:true});
  });
  const timings=[];for(let i=0;i<40;i++){const before=performance.now();checker.evaluate(baselineParts,config.placement,{movingParts:armParts('left'),stopOnBlock:true});timings.push(performance.now()-before);}
  audit.performance={scope:'Node CPU, moving left arm only, 963 samples',medianMilliseconds:median(timings),p95Milliseconds:[...timings].sort((a,b)=>a-b)[38]};
  assert.equal(audit.cases.length,6,'All cases must pass before publishing audit');audit.result='passed';
  if(process.env.AMBULANCE_BODY_CONTACT_AUDIT_PATH)await writeFile(process.env.AMBULANCE_BODY_CONTACT_AUDIT_PATH,JSON.stringify(audit,null,2)+'\n');
  console.log(`Body contact audit: ${checker.sampleCount} captured arm samples, ${checker.proxies.length} fitted envelopes, ${checker.referenceExceptions.length} reference exceptions; median moving-arm ${audit.performance.medianMilliseconds.toFixed(2)}ms.`);
});
