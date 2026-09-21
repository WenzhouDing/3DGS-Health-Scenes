import assert from 'node:assert/strict';
import test from 'node:test';
import fs from 'node:fs';
import {createMotion, transformPoint, dofCount} from '../viewers/mannequin-articulation/motion-core.mjs';
import {PRESET_REVISION, PRESETS, buildPreset, composePresentation} from '../viewers/mannequin-articulation/pose-presets.mjs';
const manifest = JSON.parse(fs.readFileSync(new URL('../viewers/mannequin-fusion/fusion.json', import.meta.url)));
const annotations = JSON.parse(fs.readFileSync(new URL('../raw/mannequin-fused/joint-refinement.json', import.meta.url))).annotations;
const close = (a,b,e=1e-9) => a.forEach((v,i) => assert.ok(Math.abs(v-b[i])<e, `${a} != ${b}`));
const subtract = (a,b) => a.map((v,i)=>v-b[i]);
const unit = a => a.map(v=>v/Math.hypot(...a));
function apply(preset,motion) {
  motion.reset();
  if (preset.id!=='scan') for (const [id,target] of Object.entries(preset.targets)) motion.setTarget(id,target,{immediate:true});
  return composePresentation(motion.evaluate(),preset.presentation);
}

test('prepared poses honor existing joint travel and do not mutate annotations or running motion',()=>{
  const motion=createMotion(annotations,manifest), before=structuredClone(annotations);
  motion.setTarget('neck',[12,-3,17],{immediate:true}); const state=motion.getState('neck');
  for(const entry of PRESETS){
    const preset=buildPreset(entry.id,motion,manifest);
    assert.equal(Object.keys(preset.targets).length,annotations.joints.length);
    for(const j of annotations.joints)for(let i=0;i<3;i++){
      const a=preset.targets[j.id][i];assert.ok(Number.isFinite(a));
      if(i>=dofCount(j))assert.equal(a,0);
      else assert.ok(a>=j.limits[i][0]&&a<=j.limits[i][1]);
    }
  }
  assert.deepEqual(annotations,before);assert.deepEqual(motion.getState('neck'),state);
});

test('flat pose faces up and brings the legs parallel along the bed without bending knees or wrists',()=>{
  const motion=createMotion(annotations,manifest),preset=buildPreset('lying',motion,manifest),pose=apply(preset,motion);
  const root=pose.parts.get('torso');close(subtract(transformPoint(root,[0,1,0]),root.position),[0,0,1]);
  for(const side of ['left','right']){
    const hip=pose.joints.get(side+'_hip').pivot,knee=pose.joints.get(side+'_knee').pivot;
    const along=unit(subtract(knee,hip));assert.ok(Math.abs(along[0])<1e-8);assert.ok(along[1]>.99);
    assert.deepEqual(preset.targets[side+'_knee'],[0,0,0]);
    assert.equal(pose.parts.get(side+'_hand'),pose.parts.get(side+'_forearm'));
  }
});

test('raised-back pose keeps the thighs and shins extended over the full mattress',()=>{
  const motion=createMotion(annotations,manifest),preset=buildPreset('sitting',motion,manifest),pose=apply(preset,motion);
  assert.equal(PRESET_REVISION,2);
  assert.equal(preset.analysis.backrestElevationDegrees,45); assert.ok(preset.analysis.maximumLegDirectionErrorDegrees<2.1);
  const torso=pose.parts.get('torso'), up=unit(subtract(transformPoint(torso,[0,0,1]),torso.position));
  close(up,[0,-Math.SQRT1_2,Math.SQRT1_2]);
  assert.ok(preset.presentation.bed.size[1]>2.2);
  assert.deepEqual(preset.targets.waist,[0,0,0]);
  for(const side of ['left','right']){
    const hip=pose.joints.get(side+'_hip').pivot,knee=pose.joints.get(side+'_knee').pivot,ankle=pose.joints.get(side+'_ankle').pivot;
    const thigh=unit(subtract(knee,hip)),shin=unit(subtract(ankle,knee));
    assert.ok(thigh[1]>.9993);assert.ok(shin[1]>.9993);
    assert.ok(Math.abs(preset.targets[side+'_knee'][0])<10);
    const bed=preset.presentation.bed;
    assert.ok(Math.abs(ankle[1]-bed.center[1])<bed.size[1]/2-.15);
    assert.ok(preset.targets[side+'_hip'][0]<=60);
    assert.equal(pose.parts.get(side+'_hand'),pose.parts.get(side+'_forearm'));
  }
});

test('presentation composition agrees with independently applied transforms, including joint pivots and axes',()=>{
  const motion=createMotion(annotations,manifest);motion.setTarget('left_shoulder',[26,-17,8],{immediate:true});
  motion.setTarget('left_arm_swivel',[39,0,0],{immediate:true});
  const raw=motion.evaluate(),p=buildPreset('lying',motion,manifest).presentation,posed=composePresentation(raw,p);
  // This presentation is exactly +90 degrees about X, followed by translation.
  const expected = a=>[a[0]+p.position[0],-a[2]+p.position[1],a[1]+p.position[2]];
  const point=[.59,-.07,-.63];close(transformPoint(posed.parts.get('left_hand'),point),expected(transformPoint(raw.parts.get('left_hand'),point)));
  const joint=raw.joints.get('left_arm_swivel'),result=posed.joints.get('left_arm_swivel');
  close(result.pivot,expected(joint.pivot));close(result.axis,[joint.axis[0],-joint.axis[2],joint.axis[1]]);
  const t=posed.parts.get('left_hand'),m=t.matrix;
  close([m[0]*point[0]+m[4]*point[1]+m[8]*point[2]+m[12],m[1]*point[0]+m[5]*point[1]+m[9]*point[2]+m[13],m[2]*point[0]+m[6]*point[1]+m[10]*point[2]+m[14]],transformPoint(t,point));
});

test('returning from a bed preset restores the exact fused scan and removes the support',()=>{
  const motion=createMotion(annotations,manifest);apply(buildPreset('sitting',motion,manifest),motion);
  const scan=buildPreset('scan',motion,manifest),posed=apply(scan,motion);
  assert.equal(scan.presentation.bed,null);
  for(const part of posed.parts.values()){
    close(part.position,[0,0,0]);close(part.rotation,[0,0,0,1]);
    close(part.matrix,[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]);
  }
});

test('edited tighter limits are retained and produce an explicit constrained-pose note',()=>{
  const edited=structuredClone(annotations);edited.joints.find(j=>j.id==='right_hip').limits[0]=[-20,20];
  const preset=buildPreset('sitting',edited,manifest);
  assert.ok(preset.targets.right_hip[0]<=20);assert.match(preset.note,/Edited joint limits restrict/);
  assert.throws(()=>buildPreset('unknown',annotations,manifest));
  assert.throws(()=>composePresentation(createMotion(annotations,manifest).evaluate(),{position:[0,0,0],rotation:[0,0,0,2]}));
});

test('full-capture audit clears rigid limbs and the backrest while quantifying the fixed pelvis garment overlap',()=>{
  // Check all 2.7M centers from both captures, not joint proxies or an overhead
  // view. Gaussian extents and arbitrary edited poses are outside this check.
  const motion=createMotion(annotations,manifest);
  const captures=manifest.captures.map(c=>{
    const data=fs.readFileSync(new URL('../viewers/mannequin-fusion/'+c.url,import.meta.url));
    return {...c,packed:new Float32Array(data.buffer,data.byteOffset,data.byteLength/4),labels:fs.readFileSync(new URL('../viewers/mannequin-fusion/'+c.labelsUrl,import.meta.url))};
  });
  for(const id of ['lying','sitting']){
    const preset=buildPreset(id,motion,manifest),pose=apply(preset,motion),bed=preset.presentation.bed;
    let tested=0,inside=0,pelvisCount=0,pelvisBelow=0;
    const transforms=manifest.parts.map(p=>pose.parts.get(p.id));
    for(const capture of captures)for(let i=0;i<capture.count;i++){
      const at=i*14,p=capture.packed;if(p[at+10]<-1.38)continue;tested++;
      const part=manifest.parts[capture.labels[i]].id;
      const world=transformPoint(transforms[capture.labels[i]],[p[at],p[at+1],p[at+2]]);
      if(part==='pelvis')pelvisCount++;
      const d=subtract(world,bed.center);
      const insideXY=Math.abs(d[0])<bed.size[0]/2&&Math.abs(d[1])<bed.size[1]/2;
      if(insideXY&&d[2]<bed.size[2]/2){
        if(id==='sitting'&&part==='pelvis')pelvisBelow++;
        else inside++;
      }
      if(['left_foot','right_foot','left_shin','right_shin'].includes(part))assert.ok(insideXY,'extended legs must remain over the full mattress');
      if(bed.backrest){
        const b=bed.backrest,inverse={position:[0,0,0],rotation:b.rotation.map((v,k)=>k===3?v:-v)};
        const local=transformPoint(inverse,subtract(world,b.center));
        if(local.every((v,k)=>Math.abs(v)<b.size[k]/2))inside++;
      }
    }
    assert.ok(tested>250000);assert.equal(inside,0,`${id} rigid limbs or torso intersect the support`);
    // The unchanged fixed pelvis segment contains loose pants below the hip,
    // which cannot rotate with the legs. Record the known <8% garment overlap
    // explicitly instead of lifting the whole body until the legs float.
    if(id==='sitting'){assert.ok(pelvisBelow>0);assert.ok(pelvisBelow/pelvisCount<.08);}
    else assert.equal(pelvisBelow,0);
  }
});
