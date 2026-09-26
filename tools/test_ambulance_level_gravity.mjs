/** Integration check with actual captured samples: one runtime scene rotation
 * preserves the fitted contacts, while gravity remains vertical in display space.
 * Only the rendering resources are mocked; no output files or assets are changed.
 */
import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {createAmbulanceManikin} from '../viewers/ambulance/manikin-scan.mjs';
import {createContactChecker} from '../viewers/ambulance/contact-check.mjs';
import {createBodyContactChecker} from '../viewers/ambulance/body-contact.mjs';
import {createArmGravity} from '../viewers/ambulance/arm-gravity.mjs';
import {createGravityTest} from '../viewers/ambulance/gravity-test.mjs';
import {createSceneFrame, AMBULANCE_SOURCE_UP} from '../viewers/ambulance/scene-frame.mjs';
import {ambulanceScene} from '../viewers/ambulance/scene-config.mjs';
import {createMotion, transformPoint} from '../viewers/mannequin-articulation/motion-core.mjs';
import {buildPreset} from '../viewers/mannequin-articulation/pose-presets.mjs';

const identity = {position:[0,0,0], rotation:[0,0,0,1], scale:1};
const near = (a, b, message) => assert.ok(Math.abs(a-b)<1e-10, `${message}: ${a} versus ${b}`);
const vectorNear = (a, b, message) => a.forEach((v,i)=>near(v,b[i],message));

test('leveled ambulance preserves captured fit and safe world-vertical gravity', async t => {
  const oldFetch=globalThis.fetch, oldLocation=globalThis.location;
  globalThis.location={hostname:'published.example'};
  globalThis.fetch=async url=>{
    try {
      const path=fileURLToPath(url);
      assert.ok(!path.includes('/raw/'), 'Published integration must use tracked annotation assets');
      return new Response(await readFile(path));
    } catch(error) {
      if(error.code==='ENOENT') return new Response('',{status:404});
      throw error;
    }
  };
  t.after(()=>{globalThis.fetch=oldFetch;globalThis.location=oldLocation;});
  class Entity {addChild(){} addComponent(){} setLocalPosition(){} setLocalRotation(){} setLocalScale(){} destroy(){}}
  const pc={Entity, GSplatData:class{}, GSplatResource:class{destroy(){}}, WORKBUFFER_UPDATE_AUTO:0};
  const scan=await createAmbulanceManikin({app:{root:new Entity(),graphicsDevice:{}},pc});
  t.after(()=>scan.dispose());
  const source=JSON.parse(await readFile(new URL('../viewers/ambulance/bed-placement.json',import.meta.url)));
  const savedConfig=structuredClone(source);
  assert.equal(scan.annotationSource,'published');
  assert.ok(scan.refinement, 'Use the published fitted articulation');
  assert.equal(scan.sampleCount,11513);
  assert.ok(ambulanceScene.rotation?.length===4, 'Default scene must provide its leveling quaternion');
  const frame=createSceneFrame(ambulanceScene.rotation), placement=frame.placement(source.placement);
  const sourceChecker=createContactChecker(source.collision), scene=frame.wrapContactChecker(sourceChecker);
  const motion=createMotion(scan.annotations,scan.manifest);
  const targets={...buildPreset('lying',motion,scan.manifest).targets,...source.jointOverrides};
  const resetPose=()=>{for(const [id,angles]of Object.entries(targets))motion.setTarget(id,angles,{immediate:true});};
  resetPose();
  const restSamples=scan.getCollisionSamples(new Map(),identity);
  const body=createBodyContactChecker({restSamples,annotations:scan.annotations,referenceParts:motion.evaluate().parts});
  const evaluate=(pose,{movingParts=null}={})=>{
    const a=scene.evaluate(scan.getCollisionSamples(pose.parts,placement,{parts:movingParts}),{stopOnBlock:true});
    if(a.blocked)return a;
    const b=body.evaluate(pose.parts,placement,{stopOnBlock:true,movingParts:movingParts??undefined});
    return {...a,blocked:b.blocked,contacts:a.contacts.concat(b.contacts),violations:a.violations.concat(b.violations)};
  };

  await t.test('fit, contact normals and retained cameras transform once',()=>{
    const parts=motion.evaluate().parts, originalSamples=scan.getCollisionSamples(parts,source.placement);
    const runtimeSamples=scan.getCollisionSamples(parts,placement);
    const rigid={...identity,rotation:ambulanceScene.rotation};
    runtimeSamples.forEach((sample,i)=>{
      assert.equal(sample.id,originalSamples[i].id);assert.equal(sample.radius,originalSamples[i].radius);
      vectorNear(sample.position,transformPoint(rigid,originalSamples[i].position),'single rigid sample transform');
    });
    const a=sourceChecker.evaluate(originalSamples), b=scene.evaluate(runtimeSamples);
    assert.equal(a.blocked,false);assert.equal(b.blocked,false);assert.equal(b.contacts.length,a.contacts.length);
    assert.ok(b.contacts.length>0,'Fitted mattress contacts must remain present');
    for(let i=0;i<a.contacts.length;i++){
      const before=a.contacts[i],after=b.contacts[i];
      assert.equal(after.sampleId,before.sampleId);assert.equal(after.proxy,before.proxy);
      near(after.penetration,before.penetration,'contact penetration');
      vectorNear(after.point,transformPoint(rigid,before.point),'contact point');
      vectorNear(after.normal,transformPoint(rigid,before.normal),'contact normal');
    }
    assert.equal(evaluate(motion.evaluate()).blocked,false,'Body contacts remain safe too');
    const floor=source.collision.surfaces.find(s=>s.id==='floor');
    const normal=transformPoint(rigid,floor.normal), length=Math.hypot(...normal);
    vectorNear(transformPoint(rigid,AMBULANCE_SOURCE_UP),[0,1,0],'fitted source up aligns with runtime +Y');
    // The leveling fit averages independently measured floor and ceiling; the
    // existing finite floor proxy retains its own slight measured slope.
    assert.ok(normal[1]/length>Math.cos(.6*Math.PI/180),'Measured floor proxy stays within 0.6° of level');
    for(const [name,sourceCamera] of Object.entries(source.cameras)){
      const camera=frame.camera(sourceCamera);
      vectorNear(camera.position,transformPoint(rigid,sourceCamera.position),`${name} camera position`);
      vectorNear(camera.target,transformPoint(rigid,sourceCamera.target),`${name} camera target`);
      assert.equal(camera.fov,sourceCamera.fov);
    }
    assert.deepEqual(source,savedConfig,'Leveling must not mutate the measured source configuration');
  });

  for(const side of ['left','right'])await t.test(`${side} shoulder stays held and settles safely after release in leveled space`,()=>{
    resetPose();
    const c=createArmGravity({motion,restSamples,getPlacement:()=>placement,evaluate});
    const id=`${side}_shoulder`,raised=[...targets[id]];raised[0]=50;
    c.hold(id);motion.setTarget(id,raised,{immediate:true});
    const context={movingParts:['upper_arm','forearm','hand'].map(part=>`${side}_${part}`)};
    assert.equal(evaluate(motion.evaluate(),context).blocked,false);
    const poseSignature=()=>motion.exportPose().joints.map(j=>({id:j.id,angles:j.angles}));
    const held=poseSignature();for(let i=0;i<30;i++)c.update(1/60);
    assert.deepEqual(poseSignature(),held,'Holding must prevent all motion on this resting scene');
    const potential=()=>Object.values(c.getMassProperties()).filter(p=>p.part.startsWith(`${side}_`)).reduce((sum,p)=>{
      const native=transformPoint(motion.evaluate().parts.get(p.part),p.center).map(v=>v*placement.scale);
      return sum+p.mass*9.81*transformPoint(placement,native)[1];
    },0);
    const before=potential();c.release(id);let frames=0;
    while(c.getState().active&&frames++<360){
      c.update(1/60);
      assert.equal(evaluate(motion.evaluate(),context).blocked,false,'Every committed arm pose remains clear');
      if(frames===5&&c.getState().active){
        c.hold(id);const caught=poseSignature();for(let i=0;i<20;i++)c.update(1/60);
        assert.deepEqual(poseSignature(),caught,'Regrab holds the falling arm without an extra step');c.release(id);
      }
    }
    assert.ok(frames<360,'Released arm must settle within six simulated seconds');
    assert.ok(potential()<before-.01,'Gravity must lower actual arm COM along runtime Y');
    for(const joint of held.filter(j=>![`${side}_shoulder`,`${side}_arm_swivel`].includes(j.id)))
      assert.deepEqual(motion.getState(joint.id).angles,joint.angles,'Other joints remain posed');
  });

  await t.test('whole-body drop follows runtime vertical and lands on the rotated mattress',()=>{
    resetPose();const parts=motion.evaluate().parts, origin=placement.position.slice();
    const initial=scan.getCollisionSamples(parts,placement), fixedPose=motion.exportPose().joints;
    let position=origin.slice(),commits=0;
    const evaluateAt=at=>scene.evaluate(initial.map(sample=>({...sample,position:sample.position.map((v,k)=>v+at[k]-origin[k])})),{stopOnBlock:true});
    const gravity=createGravityTest({getPosition:()=>position,setPosition:at=>{
      assert.equal(at[0],origin[0]);assert.equal(at[2],origin[2]);
      assert.equal(evaluateAt(at).blocked,false,'Leveled body drop cannot cross a collision boundary');
      position=at.slice();commits++;
    },evaluateAt});
    gravity.start({lift:.12});assert.equal(gravity.getState().status,'running');
    let frames=0;while(gravity.getState().status==='running'&&frames++<360)gravity.update(1/60);
    assert.equal(gravity.getState().status,'landed');assert.ok(commits>2);
    assert.ok(Math.abs(position[1]-origin[1])<.02,`Landing remains near the fitted mattress: deltaY=${position[1]-origin[1]}`);
    const contact=evaluateAt(position);assert.equal(contact.blocked,false);
    assert.ok(contact.contacts.some(c=>c.proxy.startsWith('mattress-')&&c.normal[1]>.5));
    assert.deepEqual(motion.exportPose().joints,fixedPose);
  });
});
