import {createArticulationView} from './scan-view.mjs';
import {createMotion, dofCount, validatePose} from './motion-core.mjs';
import {validateAnnotations} from '../mannequin-joints/annotation-core.mjs';
import {buildPreset, composePresentation, PRESET_REVISION} from './pose-presets.mjs';

const $ = id => document.getElementById(id);
const svgNS = 'http://www.w3.org/2000/svg';
const localHost=['localhost','127.0.0.1','[::1]'].includes(location.hostname);
const labels = {perspective:'Perspective',front:'Front · orthographic',back:'Back · orthographic',left:'Left · orthographic',right:'Right · orthographic',top:'Above · orthographic',bottom:'Below · orthographic',orbit:'Orbit view'};
const hints = {ball:'Rotates in three directions around one center. Parent motion carries the whole connected branch.',hinge:'Bends around the yellow axis. The connected body parts follow as a rigid group.',swivel:'Twists around the arm’s length. Forearm, wrist pad and hand stay together; the collar does not bend.',fixed:'This is a fixed connection. Its attached regions follow the parent without separate rotation.'};
let view, motion, annotations, report, selected, evaluated, ready=false, frame=0, lastTime=0, storageKey, saveTimer, raf;
let presentation={position:[0,0,0],rotation:[0,0,0,1],bed:null},presetId='scan',display='both',demo=null;
let localJointMap=false;
const markers = new Map(), axisLines=[];
let caption;
const joint = () => motion?.joints.find(j=>j.id===selected);
const status = message => {$('status').textContent=message;};
const fmt = (n,d=4) => String(Number(n.toFixed(d)));
function svg(tag, attrs={}) {const e=document.createElementNS(svgNS,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));return e;}
function errorMessage(error) {status(error.message || String(error));}
function branchParts() {
  const j=joint(), parts=new Set(j.childParts);let changed=true;
  while(changed){changed=false;for(const other of motion.joints)if(parts.has(other.parentPart))for(const id of other.childParts)if(!parts.has(id)){parts.add(id);changed=true;}}
  return [...parts];
}
function updateVisibility() {view.setVisibleParts($('isolate').checked?[joint().parentPart,...branchParts()]:null);renderMarkers();}
function selectJoint(id) {if(!motion.joints.some(j=>j.id===id))return;stopDemo();selected=id;drawPanel();updateVisibility();}
function poseDocument({forDraft=false}={}) {return {...(forDraft&&demo?structuredClone(demo.before):motion.exportPose()),presentationPreset:presetId,presentationRevision:PRESET_REVISION};}
function saveDraft() {
  clearTimeout(saveTimer);if(!motion||!storageKey)return;
  try {localStorage.setItem(storageKey,JSON.stringify({pose:poseDocument({forDraft:true}),spring:$('spring').checked}));$('draft-status').textContent='Your pose is saved in this browser.';}
  catch {$('draft-status').textContent='Browser storage is unavailable. Export a pose to keep your changes.';}
}
function queueSave() {clearTimeout(saveTimer);saveTimer=setTimeout(saveDraft,500);}
function setTarget(axis,value) {
  if(!Number.isFinite(value))return;stopDemo();const target=motion.getState(selected).target;target[axis]=value;
  motion.setTarget(selected,target,{immediate:!$('spring').checked});syncAngles();queueSave();status(joint().label+' target updated.');
}
function syncAngles() {
  if(!ready)return;const s=motion.getState(selected),n=dofCount(joint());
  for(let i=0;i<n;i++){
    const slider=$('angle-'+i),number=$('angle-number-'+i);slider.value=s.target[i];
    if(document.activeElement!==number)number.value=fmt(s.target[i],1);
    $('actual-'+i).textContent='Now '+fmt(s.angles[i],1)+'°';
  }
}
function drawPanel() {
  const j=joint(),s=motion.getState(selected),n=dofCount(j);$('joint-select').value=selected;
  $('joint-type').textContent={ball:'Ball joint',hinge:'Hinge',swivel:'Axial swivel',fixed:'Fixed'}[j.type];$('type-help').textContent=hints[j.type];
  $('fixed-help').hidden=n>0;$('target-rest').disabled=!n;$('nudge').disabled=!n||!$('spring').checked;$('demo').disabled=!n;
  $('stiffness').disabled=!n;$('damping').disabled=!n;$('stiffness').value=s.stiffness;$('damping').value=s.damping;
  $('angles').replaceChildren();
  for(let i=0;i<n;i++){
    const name=n===3?['Rotation X','Rotation Y','Rotation Z'][i]:j.type==='swivel'?'Twist':'Bend';
    const row=document.createElement('div');row.className='angle-control';const title=document.createElement('div');title.className='angle-title';
    const label=document.createElement('label');label.htmlFor='angle-'+i;label.textContent=name;
    const number=document.createElement('input');Object.assign(number,{id:'angle-number-'+i,type:'number',min:j.limits[i][0],max:j.limits[i][1],step:.5,value:fmt(s.target[i],1)});number.setAttribute('aria-label',name+' target degrees');
    number.addEventListener('input',()=>{if(number.value.trim()&&Number.isFinite(number.valueAsNumber))setTarget(i,number.valueAsNumber);});
    number.addEventListener('change',()=>{if(number.value.trim()&&Number.isFinite(number.valueAsNumber)){setTarget(i,number.valueAsNumber);number.value=fmt(motion.getState(selected).target[i],1);}else{number.value=fmt(motion.getState(selected).target[i],1);status('Enter an angle in degrees.');}});
    title.append(label,number);const slider=document.createElement('input');Object.assign(slider,{id:'angle-'+i,type:'range',min:j.limits[i][0],max:j.limits[i][1],step:.5,value:s.target[i]});slider.setAttribute('aria-label',name);
    slider.addEventListener('input',()=>setTarget(i,Number(slider.value)));
    const foot=document.createElement('div');foot.className='angle-foot';const min=document.createElement('span'),actual=document.createElement('span'),max=document.createElement('span');min.textContent=j.limits[i][0]+'°';max.textContent=j.limits[i][1]+'°';actual.className='actual';actual.id='actual-'+i;actual.textContent='Now '+fmt(s.angles[i],1)+'°';foot.append(min,actual,max);row.append(title,slider,foot);$('angles').append(row);
  }
  const evidence=report?.joints.find(item=>item.id===j.id);
  $('fit-note').textContent=evidence?`${evidence.status} · ${evidence.method}`:'Uses your saved annotation. No matching geometry refinement is available for this map.';
  const plot=evidence?.gaussianPlot||evidence?.plot;
  $('projection-link').hidden=!(localJointMap&&typeof plot==='string'&&/^raw\/fusion-work\/joint-analysis\/[a-z_]+(?:-gaussians)?\.png$/.test(plot));
  if(!$('projection-link').hidden)$('projection-link').href='../../'+plot;
  $('geometry').replaceChildren();
  for(const [name,value] of [['Rest center',j.pivot.map(x=>fmt(x)).join(', ')],['Rest axis',j.axis.map(x=>fmt(x)).join(', ')],['Moving parts',branchParts().map(id=>view.manifest.parts.find(p=>p.id===id)?.label||id).join(', ')]] ){
    const dt=document.createElement('dt'),dd=document.createElement('dd');dt.textContent=name;dd.textContent=value;$('geometry').append(dt,dd);
  }
}
function rotateVector(q,v) {
  const [x,y,z,w]=q,[a,b,c]=v,tx=2*(y*c-z*b),ty=2*(z*a-x*c),tz=2*(x*b-y*a);
  return [a+w*tx+y*tz-z*ty,b+w*ty+z*tx-x*tz,c+w*tz+x*ty-y*tx];
}
function createMarkers() {
  for(const [i,j]of motion.joints.entries()){
    const group=svg('g',{class:'joint-marker',role:'button',tabindex:0,'aria-label':j.label,'data-joint':j.id});
    group.append(svg('circle',{r:17,class:'hit'}),svg('circle',{r:9,class:'ring'}));const text=svg('text');text.textContent=i+1;group.append(text);
    group.addEventListener('click',()=>selectJoint(j.id));
    group.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectJoint(j.id);}});
    $('overlay').append(group);markers.set(j.id,group);
  }
  for(let i=0;i<3;i++){const line=svg('line',{class:'axis-line'});$('overlay').append(line);axisLines.push(line);}
  caption=svg('text',{class:'marker-caption'});$('overlay').append(caption);
}
function renderMarkers() {
  if(!view||!evaluated||!motion)return;const rect=$('viewport').getBoundingClientRect();$('overlay').setAttribute('viewBox',`0 0 ${rect.width} ${rect.height}`);
  const show=$('markers').checked||!!demo,isolated=$('isolate').checked,visibleParts=new Set(branchParts());
  for(const j of motion.joints){const g=markers.get(j.id),p=view.project(evaluated.joints.get(j.id).pivot),visible=show&&p.visible&&($('markers').checked||j.id===selected)&&(!isolated||j.id===selected||visibleParts.has(j.parentPart));g.style.display=visible?'':'none';g.setAttribute('transform',`translate(${p.x} ${p.y})`);g.classList.toggle('selected',j.id===selected);g.classList.toggle('fixed',j.type==='fixed');}
  const j=joint(),state=evaluated.joints.get(selected),p=view.project(state.pivot);
  caption.style.display=show&&p.visible?'':'none';caption.textContent=j.label;caption.setAttribute('x',p.x+14);caption.setAttribute('y',p.y-12);
  // The axes belong to the parent frame at rest; the hinge axis itself is invariant under its own rotation.
  const parentTransform=evaluated.parts.get(j.parentPart);
  const axes=j.type==='ball'?[[1,0,0],[0,1,0],[0,0,1]]:j.type==='fixed'?[]:[j.axis];
  axisLines.forEach((line,i)=>{
    line.style.display=($('axis-visible').checked||!!demo)&&p.visible&&i<axes.length?'':'none';if(i>=axes.length)return;
    const axis=rotateVector(parentTransform.rotation,axes[i]);const length=j.type==='ball'?.09:.14;
    const a=view.project(state.pivot.map((x,k)=>x-axis[k]*length)),b=view.project(state.pivot.map((x,k)=>x+axis[k]*length));
    if(!Number.isFinite(a.x+b.x+a.y+b.y)){line.style.display='none';return;}
    line.setAttribute('x1',a.x);line.setAttribute('y1',a.y);line.setAttribute('x2',b.x);line.setAttribute('y2',b.y);line.style.stroke=j.type==='ball'?['#ec927b','#a8cf89','#85bce0'][i]:'#f3c56b';
  });
}
function stopDemo() {
  if(!demo)return;const before=demo.before;demo=null;motion.importPose(before);$('demo').textContent='Play joint demo';$('demo').setAttribute('aria-pressed','false');syncAngles();
}
function toggleDemo() {
  if(demo){stopDemo();status('Joint demo stopped.');return;}
  if(!dofCount(joint()))return;demo={before:motion.exportPose(),base:motion.getState(selected).angles,elapsed:0};$('demo').textContent='Stop demo';$('demo').setAttribute('aria-pressed','true');status('Showing '+joint().label.toLowerCase()+' rotation.');
}
function stepDemo(dt) {
  if(!demo)return;demo.elapsed+=Math.min(dt,.05);const n=dofCount(joint()),axis=Math.floor(demo.elapsed/5)%n,t=(demo.elapsed%5)/5;
  const [lo,hi]=joint().limits[axis],base=demo.base[axis],travel=joint().type==='swivel'?45:30;
  const delta=hi-base>=Math.min(travel,base-lo)?Math.min(travel,hi-base):-Math.min(travel,base-lo);
  const target=demo.base.slice();target[axis]=base+delta*(.5-.5*Math.cos(2*Math.PI*t));motion.setTarget(selected,target,{immediate:true});
}
function setDisplay(value) {
  if(!['both','front','back','segments'].includes(value))return;display=value;
  view.setSource(value==='segments'?'both':value);view.setColorMode(value==='segments'?'segments':'natural');$('capture').value=value==='segments'?'both':value;
  document.querySelectorAll('[data-display]').forEach(button=>{const active=button.dataset.display===value;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  $('mode-label').textContent={both:'Fused scan',front:'Front scan',back:'Back scan',segments:'Body segmentation'}[value];$('segment-legend').hidden=value!=='segments';
}
function updatePresetUI(note='') {
  document.querySelectorAll('[data-preset]').forEach(button=>{const active=button.dataset.preset===presetId;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});
  $('pose-note').textContent=note;$('pose-note').hidden=!note;
}
function updateScene() {evaluated=composePresentation(motion.evaluate(),presentation);view.setPartTransforms(evaluated.parts);renderMarkers();}
function applyPreset(id,{save=true}={}) {
  const preset=buildPreset(id,motion,view.manifest);stopDemo();motion.reset();
  if(id!=='scan')for(const [jointId,target] of Object.entries(preset.targets))motion.setTarget(jointId,target,{immediate:true});
  presentation=preset.presentation;presetId=id;$('isolate').checked=false;view.setVisibleParts(null);updateScene();view.setBed(presentation.bed);view.fit();
  view.setCamera(preset.camera||presentation.camera||{yaw:55,pitch:id==='scan'?12:27});view.fit();updatePresetUI(preset.note);drawPanel();if(save)queueSave();status(preset.label+' ready.');
}
function loadPose(document) {
  const check=validatePose(document,annotations,view.manifest);if(!check.valid)throw new Error(check.errors.join(' '));
  const id=document.presentationPreset??'scan';if(!['scan','lying','sitting'].includes(id))throw new Error('This pose has an unknown presentation.');
  if(id==='sitting'&&document.presentationRevision!==PRESET_REVISION)throw new Error('This pose uses the earlier sitting layout. Choose Upper body up to use the revised bed pose.');
  const preset=buildPreset(id,motion,view.manifest);stopDemo();motion.importPose(document);presetId=id;presentation=preset.presentation;
  view.setBed(presentation.bed);updateScene();view.setCamera(preset.camera||presentation.camera||{yaw:55,pitch:id==='scan'?12:27});view.fit();updatePresetUI(preset.note);
}
function animate(time) {
  if(!ready)return;const dt=lastTime?(time-lastTime)/1000:0;lastTime=time;
  if($('spring').checked)motion.step(dt);stepDemo(dt);updateScene();
  if(frame++%6===0){syncAngles();let maxAngle=0,maxVelocity=0;for(const j of motion.joints){const s=motion.getState(j.id);maxAngle=Math.max(maxAngle,...s.angles.map(Math.abs));maxVelocity=Math.max(maxVelocity,...s.velocity.map(Math.abs));}$('motion-state').textContent=maxAngle<.001&&maxVelocity<.001?'Rest pose':maxVelocity>.15?'Moving · spring response':'Posed';}
  raf=requestAnimationFrame(animate);
}
function bindUI() {
  $('joint-select').onchange=()=>selectJoint($('joint-select').value);
  for(const [id,delta]of [['previous',-1],['next',1]])$(id).onclick=()=>selectJoint(motion.joints[(motion.joints.findIndex(j=>j.id===selected)+delta+motion.joints.length)%motion.joints.length].id);
  $('demo').onclick=toggleDemo;$('reset').onclick=()=>applyPreset('scan');
  $('target-rest').onclick=()=>{stopDemo();motion.setTarget(selected,[0,0,0],{immediate:!$('spring').checked});syncAngles();queueSave();};
  $('nudge').onclick=()=>{stopDemo();motion.nudge(selected,0,35);queueSave();status('A small rotational impulse was applied to '+joint().label+'.');};
  $('spring').onchange=()=>{stopDemo();if(!$('spring').checked)for(const j of motion.joints)motion.setTarget(j.id,motion.getState(j.id).target,{immediate:true});drawPanel();queueSave();};
  for(const id of ['stiffness','damping']){
    const update=()=>{const value=$(id).valueAsNumber;if(!$(id).value.trim()||!Number.isFinite(value)||!$(id).checkValidity())return false;stopDemo();motion.setSpring(selected,{[id]:value});queueSave();return true;};
    $(id).oninput=update;$(id).onchange=()=>{if(!update()){status('Enter a resistance value between 0 and 100000.');drawPanel();}};
  }
  $('markers').onchange=renderMarkers;$('axis-visible').onchange=renderMarkers;$('isolate').onchange=updateVisibility;$('capture').onchange=()=>setDisplay($('capture').value);
  document.querySelectorAll('[data-display]').forEach(button=>button.onclick=()=>setDisplay(button.dataset.display));
  document.querySelectorAll('[data-preset]').forEach(button=>button.onclick=()=>applyPreset(button.dataset.preset));
  $('pro-toggle').onclick=()=>{
    const open=$('pro-toggle').getAttribute('aria-expanded')!=='true';$('pro-toggle').setAttribute('aria-expanded',String(open));$('pro-panel').hidden=!open;document.body.classList.toggle('pro',open);
    if(!open){const isolated=$('isolate').checked;$('isolate').checked=false;$('markers').checked=false;$('axis-visible').checked=false;updateVisibility();if(isolated)view.fit();}
  };
  $('focus').onclick=()=>{const id=joint().id;const span=id.includes('shoulder')?.48:id.includes('hip')?.5:id.includes('knee')?.4:id.includes('ankle')?.34:.3;view.focusPoint(evaluated.joints.get(selected).pivot,{span});};
  $('fit-all').onclick=()=>{$('isolate').checked=false;view.setVisibleParts(null);view.fit();};
  document.querySelectorAll('[data-view]').forEach(button=>button.onclick=()=>view.setView(button.dataset.view));
  $('export').onclick=()=>{const blob=new Blob([JSON.stringify(poseDocument(),null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='mannequin-fused-pose.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status('Pose exported.');};
  $('export-map').onclick=()=>{const map=structuredClone(annotations);for(const j of map.joints){const s=motion.getState(j.id);j.stiffness=s.stiffness;j.damping=s.damping;}map.updatedAt=new Date().toISOString();const blob=new Blob([JSON.stringify(map,null,2)+'\n'],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='mannequin-refined-joints.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status('Current joint map exported. Import it in Edit joint map to refine these centers.');};
  $('import').onclick=()=>$('import-file').click();$('import-file').onchange=async event=>{const file=event.target.files[0];event.target.value='';if(!file)return;try{if(file.size>512*1024)throw new Error('The pose file is too large.');loadPose(JSON.parse(await file.text()));drawPanel();queueSave();status('Pose loaded.');}catch(error){errorMessage(error);}};
  window.addEventListener('pagehide',()=>{saveDraft();ready=false;cancelAnimationFrame(raf);view.dispose();},{once:true});
  window.addEventListener('pageshow',event=>{if(event.persisted)window.location.reload();});
}
async function readJointMap() {
  // Hash the saved bytes, matching the geometry audit rather than the API's JSON formatting.
  let response;
  localJointMap=false;
  if(localHost){
    response=await fetch('../../raw/mannequin-fused/joint-annotations.json',{cache:'no-store'});
    if(response.ok)localJointMap=true;else if(response.status!==404)throw new Error('The local joint map could not be read.');
  }
  if(!localJointMap)response=await fetch('./joint-annotations.json',{cache:'no-store'});
  if(!response.ok)throw new Error('The joint map could not be loaded. Reload the page to try again.');
  const sourceText=await response.text(),saved=JSON.parse(sourceText);const validation=validateAnnotations(saved,view.manifest);if(!validation.valid)throw new Error(validation.errors.join(' '));
  annotations=saved;report=null;
  const digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(sourceText)),hash=Array.from(new Uint8Array(digest),n=>n.toString(16).padStart(2,'0')).join('');
  try {
    const fitted=await fetch(localJointMap?'../../raw/mannequin-fused/joint-refinement.json':'./joint-refinement.json',{cache:'no-store'});
    if(fitted.ok){const candidate=await fitted.json();if(candidate.schema==='mannequin-joint-refinement'&&candidate.version===1&&Array.isArray(candidate.joints)&&candidate.joints.every(item=>item&&typeof item.id==='string')&&candidate.inputAnnotationSha256===hash&&candidate.inputSceneRevision===saved.scene.revision&&validateAnnotations(candidate.annotations,view.manifest).valid){report=candidate;annotations=candidate.annotations;}}
  } catch { /* An optional audit must never prevent a valid saved map from opening. */ }
  $('map-status').textContent=(localJointMap?'Your joint map':'Published joint map')+(report?' · geometry checked':'');
}
async function start() {
  try {
    view=await createArticulationView({canvas:$('canvas'),container:$('viewport'),onStatus:message=>{$('scene-info').textContent=message;$('load-detail').textContent=message;},onCameraChange:camera=>{$('view-name').textContent=labels[camera.view]||'Orbit view';document.querySelectorAll('[data-view]').forEach(button=>{const active=button.dataset.view===camera.view;button.classList.toggle('active',active);button.setAttribute('aria-pressed',String(active));});renderMarkers();}});
    await readJointMap();motion=createMotion(annotations,view.manifest);selected=motion.joints.find(j=>j.id==='left_shoulder')?.id||motion.joints[0].id;
    // Presentation-aware drafts keep the previous version's saved poses intact.
    storageKey='mannequin-fused-pose-v3:'+motion.signature;let restored=false;
    const theme=matchMedia('(prefers-color-scheme: dark)'),updateTheme=()=>view.setTheme(theme.matches?'dark':'light');updateTheme();theme.addEventListener('change',updateTheme);
    window.addEventListener('pagehide',()=>theme.removeEventListener('change',updateTheme),{once:true});
    $('joint-select').replaceChildren(...motion.joints.map(j=>new Option(j.label,j.id)));$('joint-count').textContent=motion.joints.filter(j=>dofCount(j)>0).length+' movable joints';
    $('segment-legend').replaceChildren(...view.manifest.parts.map(part=>{const item=document.createElement('span');item.className='segment-item';const swatch=document.createElement('i');swatch.className='segment-swatch';swatch.style.background=`rgb(${part.color.map(n=>Math.round(n*255)).join(' ')})`;item.append(swatch,document.createTextNode(part.label));return item;}));
    document.querySelectorAll('button,select,input,fieldset').forEach(element=>element.disabled=false);bindUI();createMarkers();ready=true;applyPreset('scan',{save:false});
    try {
      const saved=JSON.parse(localStorage.getItem(storageKey)||localStorage.getItem('mannequin-fused-pose-v2:'+motion.signature)||'null');
      if(saved){
        // Keep the old draft untouched, but never revive its dangling-leg pose
        // on the revised full bed. Other poses retain their saved joint angles.
        if(saved.pose.presentationPreset==='sitting'&&saved.pose.presentationRevision!==PRESET_REVISION)applyPreset('sitting',{save:false});else loadPose(saved.pose);
        $('spring').checked=saved.spring!==false;restored=true;drawPanel();
      }
    }
    catch {$('draft-status').textContent='A previous browser pose could not be loaded. Starting from the saved joint map.';}
    setDisplay('both');$('loading').hidden=true;
    status(restored?'Your saved browser pose is restored.':'Ready. Compare the scans, choose a pose, or play a joint demo.');raf=requestAnimationFrame(animate);
  } catch(error){$('load-title').textContent='Could not open articulation';$('load-detail').textContent=error.message;document.querySelector('.loader').hidden=true;status(error.message);console.error(error);view?.dispose();}
}
start();
