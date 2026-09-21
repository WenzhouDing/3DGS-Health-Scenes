import {createAnnotations, validateAnnotations, normalizeAxis, rayPlaneIntersection} from './annotation-core.mjs';
import {createScanView} from './scan-view.mjs';

const $ = id => document.getElementById(id);
const clone = value => structuredClone(value);
const svgNS = 'http://www.w3.org/2000/svg';
const localServer = location.protocol==='http:' && (location.hostname==='localhost' || /^127(?:\.\d{1,3}){3}$/.test(location.hostname));
const saveDirection = localServer?'Save for articulation to update the project copy.':'Export JSON to keep or share your changes.';
const viewLabels = {perspective:'Perspective',front:'Front · orthographic',back:'Back · orthographic',left:'Left · orthographic',right:'Right · orthographic',top:'Above · orthographic',bottom:'Below · orthographic',orbit:'Orbit view'};
const typeHints = {ball:'A ball joint rotates in three directions. Check the center inside the shoulder or socket.',swivel:'A swivel twists along its axis. The forearm and hand stay together as one rigid piece.',hinge:'A hinge bends around one axis. Align the yellow line with the pin or axle.',fixed:'This connection stays rigid. Its body parts follow the parent without a separate rotation.'};
let view, doc, defaults, selected, storageKey, mode='navigate', drag=null, ready=false;
let generation=0, saveTimer, projectSaved=false, undoStack=[], redoStack=[], markers=new Map();
let axisLine, axisHandle, caption, lastSavedText='';
const axisLength = .13;
const joint = () => doc?.joints.find(item => item.id===selected);
const status = text => { $('status').textContent=text; };
const add = (a,b) => a.map((x,i)=>x+b[i]);
const subtract = (a,b) => a.map((x,i)=>x-b[i]);
const scale = (a,n) => a.map(x=>x*n);
const fmt = value => String(Number(value.toFixed(5)));
function svg(tag,attrs={}) { const el=document.createElementNS(svgNS,tag);for(const [key,value] of Object.entries(attrs))el.setAttribute(key,String(value));return el; }
function validate(value) { const result=validateAnnotations(value,view.manifest);if(!result.valid)throw new Error(result.errors.join(' '));return result; }
function stamp() { doc.updatedAt=new Date().toISOString();generation++; }
function remember() { undoStack.push(clone(doc));if(undoStack.length>60)undoStack.shift();redoStack=[]; }
function historyButtons() { $('undo').disabled=!undoStack.length;$('redo').disabled=!redoStack.length; }
function draftChanged() {
  stamp();projectSaved=false;$('save-state').textContent=localServer?'Draft · not yet saved to project':'Editable browser draft';
  $('save-detail').textContent=saveDirection;
  clearTimeout(saveTimer);saveTimer=setTimeout(()=>{
    try { localStorage.setItem(storageKey,JSON.stringify(doc));if(!projectSaved)$('save-detail').textContent='Draft also kept in this browser. '+saveDirection; }
    catch { $('save-detail').textContent='Browser storage is unavailable. Export JSON to keep your work.'; }
  },250);historyButtons();
}
function change(mutator,message,{keepReviewed=false}={}) {
  const next=clone(doc);const item=next.joints.find(j=>j.id===selected);
  try { mutator(item,next);if(!keepReviewed)item.reviewed=false;validate(next); }
  catch(error) { status(error.message);drawPanel();return false; }
  if(JSON.stringify(next)===JSON.stringify(doc))return true;
  remember();doc=next;draftChanged();drawPanel();updateVisibility();renderMarkers();status(message);return true;
}
function undo(redo=false) {
  const from=redo?redoStack:undoStack,to=redo?undoStack:redoStack;if(!from.length)return;
  to.push(clone(doc));doc=from.pop();if(!joint())selected=doc.joints[0].id;
  draftChanged();drawPanel();updateVisibility();renderMarkers();status(redo?'Change restored.':'Change undone.');
}
function selectJoint(id) {
  if(!doc.joints.some(j=>j.id===id))return;selected=id;setMode('navigate');drawPanel();updateVisibility();renderMarkers();
}
function connectedParts() { const j=joint();return [j.parentPart,...j.childParts]; }
function updateVisibility() { if(!view||!doc)return;view.setVisibleParts($('isolate').checked?connectedParts():null); }
function drawPanel() {
  if(!doc)return;const j=joint();
  $('joint-select').replaceChildren(...doc.joints.map(item=>new Option((item.reviewed?'✓ ':'')+item.label,item.id)));$('joint-select').value=selected;
  const checked=doc.joints.filter(j=>j.reviewed).length;$('progress').textContent=checked+' / '+doc.joints.length+' checked';
  $('joint-label').value=j.label;$('joint-type').value=j.type;$('type-help').textContent=typeHints[j.type];
  for(let i=0;i<3;i++){ $('pivot-'+i).value=fmt(j.pivot[i]);$('axis-'+i).value=fmt(j.axis[i]); }
  const axial=['hinge','swivel'].includes(j.type);$('axis-section').hidden=!axial;$('set-axis').disabled=!axial;
  $('limits-section').hidden=j.type==='fixed';$('dynamics').hidden=j.type==='fixed';$('limits').replaceChildren();
  const axes=j.type==='ball'?['Across · X','Depth · Y','Height · Z']:j.type==='fixed'?[]:[j.type==='swivel'?'Twist':'Bend'];
  axes.forEach((label,i)=>{
    const row=document.createElement('div');row.className='limit-row';const title=document.createElement('span');title.textContent=label;row.append(title);
    for(let bound=0;bound<2;bound++){
      const l=document.createElement('label');l.textContent=bound?'Maximum':'Minimum';const input=document.createElement('input');input.type='number';input.step='1';input.min='-360';input.max='360';input.value=j.limits[i][bound];input.setAttribute('aria-label',label+' '+(bound?'maximum':'minimum')+' angle');
      input.onchange=()=>{const value=readNumber(input);if(value===null)return;change(item=>{item.limits[i][bound]=value;},'Movement limit updated.');};l.append(input);row.append(l);
    }$('limits').append(row);
  });
  $('stiffness').value=j.stiffness;$('damping-value').value=j.damping;$('notes').value=j.notes;$('reviewed').checked=j.reviewed;
  $('reset-joint').disabled=!defaults.joints.some(item=>item.id===j.id);
  $('parent-part').replaceChildren(...view.manifest.parts.map(p=>new Option(p.label,p.id)));$('parent-part').value=j.parentPart;
  $('child-parts').replaceChildren(...view.manifest.parts.map(p=>{
    const label=document.createElement('label');label.className='check';const input=document.createElement('input');input.type='checkbox';input.checked=j.childParts.includes(p.id);
    const owner=doc.joints.find(other=>other.id!==selected&&other.childParts.includes(p.id));input.disabled=!!owner||p.id===j.parentPart;
    if(owner)label.title='Assigned to '+owner.label;
    input.onchange=()=>change(item=>{
      // A wrist is not a joint: selecting either member toggles the rigid pair.
      const side=['left','right'].find(side=>[side+'_forearm',side+'_hand'].includes(p.id));const ids=side?[side+'_forearm',side+'_hand']:[p.id];
      item.childParts= input.checked ? [...new Set([...item.childParts,...ids])] : item.childParts.filter(id=>!ids.includes(id));
    },'Connected body parts updated.');label.append(input,document.createTextNode(p.label));return label;
  }));historyButtons();
}
function readNumber(input) { const value=Number(input.value);if(!input.value.trim()||!Number.isFinite(value)||!input.checkValidity()){status('Enter a finite number within the shown range.');drawPanel();return null;}return value; }
function setMode(value) {
  mode=value;$('viewport').classList.toggle('placing',value!=='navigate');$('place-center').classList.toggle('active',value==='center');$('set-axis').classList.toggle('active',value==='axis');
  $('place-center').setAttribute('aria-pressed',String(value==='center'));$('set-axis').setAttribute('aria-pressed',String(value==='axis'));
  $('mode-name').textContent=value==='center'?'Click to place center · depth stays fixed':value==='axis'?'Place axis tip · current tip depth stays fixed':'Drag a joint marker to move it';
  view?.setNavigationEnabled(value==='navigate');
}
function positionOnPlane(event,point,normal) {
  const rect=$('viewport').getBoundingClientRect();const ray=view.ray(event.clientX-rect.left,event.clientY-rect.top);
  return rayPlaneIntersection(ray.origin,ray.direction,point,normal);
}
function beginDrag(event,id,kind) {
  if(!ready||event.button!==0)return;event.preventDefault();event.stopPropagation();
  if(selected!==id)selectJoint(id);
  const j=joint(),point=kind==='axis'?add(j.pivot,scale(j.axis,axisLength)):j.pivot.slice();
  const normal=view.cameraDirection(),hit=positionOnPlane(event,point,normal);if(!hit)return;
  drag={id,kind,startDoc:clone(doc),planePoint:point,normal,offset:subtract(point,hit),changed:false,pointerId:event.pointerId,target:event.currentTarget};
  event.currentTarget.setPointerCapture(event.pointerId);view.setNavigationEnabled(false);
}
function moveDrag(event) {
  if(!drag||event.pointerId!==drag.pointerId)return;const hit=positionOnPlane(event,drag.planePoint,drag.normal);if(!hit)return;
  const point=add(hit,drag.offset);let candidate;
  try { candidate=drag.kind==='axis'?normalizeAxis(subtract(point,joint().pivot)):point; } catch { return; }
  if(candidate.some(x=>!Number.isFinite(x)||Math.abs(x)>100))return;
  const previous=drag.kind==='axis'?joint().axis:joint().pivot;if(Math.hypot(...subtract(candidate,previous))<1e-8)return;
  if(!drag.changed){undoStack.push(drag.startDoc);if(undoStack.length>60)undoStack.shift();redoStack=[];drag.changed=true;}
  if(drag.kind==='axis')joint().axis=candidate;else joint().pivot=candidate;
  joint().reviewed=false;renderMarkers();for(let i=0;i<3;i++)$(drag.kind==='axis'?'axis-'+i:'pivot-'+i).value=fmt(candidate[i]);
}
function endDrag(event) {
  if(!drag||event.pointerId!==drag.pointerId)return;
  const changed=drag.changed,kind=drag.kind;try{drag.target.releasePointerCapture(event.pointerId);}catch{}drag=null;view.setNavigationEnabled(mode==='navigate');
  if(changed){draftChanged();drawPanel();renderMarkers();status(kind==='axis'?'Axis updated. Check it from another direction.':'Joint center moved in the view plane. Check depth from another view.');}
}
function renderMarkers() {
  if(!view||!doc)return;const overlay=$('overlay'),width=$('viewport').clientWidth,height=$('viewport').clientHeight;
  overlay.setAttribute('viewBox','0 0 '+width+' '+height);
  const ids=new Set(doc.joints.map(j=>j.id));for(const [id,el] of markers)if(!ids.has(id)){el.remove();markers.delete(id);}
  if(!axisLine){axisLine=svg('line',{class:'axis-line'});overlay.prepend(axisLine);caption=svg('text',{class:'marker-caption'});overlay.append(caption);axisHandle=svg('circle',{r:8,class:'axis-handle',tabindex:0,role:'button','aria-label':'Drag rotation axis handle'});overlay.append(axisHandle);axisHandle.addEventListener('pointerdown',event=>beginDrag(event,selected,'axis'));}
  doc.joints.forEach((j,index)=>{
    let g=markers.get(j.id);if(!g){g=svg('g',{class:'joint-marker',tabindex:0,role:'button'});g.append(svg('circle',{class:'hit',r:19}),svg('circle',{class:'ring',r:10}),svg('text'));
      g.addEventListener('pointerdown',event=>beginDrag(event,j.id,'pivot'));
      g.addEventListener('keydown',event=>{if(['Enter',' '].includes(event.key)){event.preventDefault();selectJoint(j.id);}});
      overlay.append(g);markers.set(j.id,g);
    }
    const p=view.project(j.pivot),visible=p.visible&&(!$('isolate').checked||j.id===selected);
    g.style.display=visible?'':'none';g.setAttribute('transform','translate('+p.x+','+p.y+')');g.setAttribute('class','joint-marker'+(j.id===selected?' selected':'')+(j.reviewed?' checked':''));g.setAttribute('aria-label',j.label+' joint marker'+(j.reviewed?', checked':''));g.setAttribute('aria-pressed',String(j.id===selected));g.querySelector('text').textContent=String(index+1);
  });
  const j=joint(),p=view.project(j.pivot);caption.textContent=j.label;caption.style.display=p.visible?'':'none';caption.setAttribute('x',Math.min(width-120,Math.max(12,p.x+17)));caption.setAttribute('y',Math.max(18,p.y-17));
  const axial=['hinge','swivel'].includes(j.type),tip=view.project(add(j.pivot,scale(j.axis,axisLength))),tail=view.project(add(j.pivot,scale(j.axis,-axisLength)));
  const show=axial&&p.visible&&tip.visible&&tail.visible;
  axisLine.style.display=show?'':'none';axisHandle.style.display=show&&Math.hypot(tip.x-p.x,tip.y-p.y)>15?'':'none';
  if(show){axisLine.setAttribute('x1',tail.x);axisLine.setAttribute('y1',tail.y);axisLine.setAttribute('x2',tip.x);axisLine.setAttribute('y2',tip.y);axisHandle.setAttribute('cx',tip.x);axisHandle.setAttribute('cy',tip.y);}
}
function cameraChanged(cameraState={}) {
  if(cameraState.view){$('view-name').textContent=viewLabels[cameraState.view]||'Orbit view';document.querySelectorAll('[data-view]').forEach(b=>{const active=b.dataset.view===cameraState.view;b.classList.toggle('active',active);b.setAttribute('aria-pressed',String(active));});}
  renderMarkers();
}
async function saveProject() {
  if(!localServer){status(saveDirection);return;}
  try {
    const document=clone(doc);document.updatedAt=new Date().toISOString();validate(document);const token=generation;$('save').disabled=true;$('save').textContent='Saving…';
    const response=await fetch('/api/mannequin-joints',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(document)});
    const result=await response.json().catch(()=>({error:'The running local server does not support saving yet.'}));if(!response.ok||!result.ok)throw new Error(result.error||'Unable to save annotations.');
    lastSavedText='Saved to the local project';
    if(token===generation){doc.updatedAt=document.updatedAt;projectSaved=true;clearTimeout(saveTimer);try{localStorage.setItem(storageKey,JSON.stringify(doc));}catch{}$('save-state').textContent=lastSavedText;$('save-detail').textContent='Your annotations are ready for the articulation step. You can keep editing and save again.';}
    else{$('save-state').textContent='Saved · newer edits remain in the draft';}
    status('Saved locally: '+result.path+'. Tell me when your joint map is ready.');
  } catch(error) { $('save-state').textContent='Project save did not complete';status(error.message+' Your draft is still open; Export JSON also keeps a copy.'); }
  finally{$('save').disabled=false;$('save').textContent='Save for articulation';}
}
function exportFile() {
  try{validate(doc);const url=URL.createObjectURL(new Blob([JSON.stringify(doc,null,2)+'\n'],{type:'application/json'}));const a=document.createElement('a');a.href=url;a.download='mannequin-joint-annotations.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status(localServer?'Annotation JSON exported. The local project save is a separate copy.':'Annotation JSON exported. Import it in the local joint editor to use it for articulation.');}catch(error){status(error.message);}
}
async function importFile(file) {
  if(!file)return;
  try{if(file.size>512*1024)throw new Error('Choose an annotation JSON smaller than 512 KB.');const candidate=JSON.parse(await file.text());validate(candidate);candidate.joints.forEach(j=>{j.axis=normalizeAxis(j.axis);});remember();doc=candidate;if(!joint())selected=doc.joints[0].id;draftChanged();drawPanel();updateVisibility();renderMarkers();status('Annotations imported. '+saveDirection);}
  catch(error){status('Import left your current draft unchanged: '+error.message);}
  finally{$('import-file').value='';}
}
function events() {
  $('joint-select').onchange=e=>selectJoint(e.target.value);
  for(const [id,step] of [['previous',-1],['next',1]])$(id).onclick=()=>{const i=doc.joints.findIndex(j=>j.id===selected);selectJoint(doc.joints[(i+step+doc.joints.length)%doc.joints.length].id);};
  $('focus').onclick=()=>{view.fit(connectedParts());};
  $('isolate').onchange=()=>{updateVisibility();view.fit($('isolate').checked?connectedParts():null);renderMarkers();};
  $('fit-all').onclick=()=>{$('isolate').checked=false;updateVisibility();view.fit(view.manifest.parts.map(p=>p.id));};
  $('capture').onchange=e=>view.setSource(e.target.value);
  document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>{setMode('navigate');view.setView(b.dataset.view);});
  $('place-center').onclick=()=>setMode(mode==='center'?'navigate':'center');$('set-axis').onclick=()=>setMode(mode==='axis'?'navigate':'axis');
  $('joint-label').onchange=e=>change(j=>{j.label=e.target.value.trim();},'Joint name updated.');
  $('joint-type').onchange=e=>change(j=>{j.type=e.target.value;j.limits=j.type==='fixed'?[[0,0],[0,0],[0,0]]:j.type==='ball'?[[-45,45],[-45,45],[-45,45]]:[[-90,90],[0,0],[0,0]];},'Motion type updated. Check the axis and limits.');
  for(let i=0;i<3;i++){
    $('pivot-'+i).onchange=e=>{const n=readNumber(e.target);if(n!==null)change(j=>{j.pivot[i]=n;},'Joint center updated.');};
    $('axis-'+i).onchange=e=>{const n=readNumber(e.target);if(n!==null)change(j=>{j.axis[i]=n;j.axis=normalizeAxis(j.axis);},'Axis direction updated.');};
  }
  document.querySelectorAll('[data-axis]').forEach(b=>b.onclick=()=>change(j=>{j.axis=[0,0,0];j.axis[Number(b.dataset.axis)]=1;},'Rotation axis set.'));
  $('flip-axis').onclick=()=>change(j=>{j.axis=j.axis.map(x=>-x);const [a,b]=j.limits[0];j.limits[0]=[-b,-a];},'Axis reversed; limits reversed to preserve the same travel.');
  for(const [id,key] of [['stiffness','stiffness'],['damping-value','damping']])$(id).onchange=e=>{const n=readNumber(e.target);if(n!==null)change(j=>{j[key]=n;},'Resistance setting updated.');};
  $('parent-part').onchange=e=>change(j=>{j.parentPart=e.target.value;},'Parent connection updated.');
  $('notes').onchange=e=>change(j=>{j.notes=e.target.value;},'Joint note saved in the draft.',{keepReviewed:true});
  $('reviewed').onchange=e=>change(j=>{j.reviewed=e.target.checked;},e.target.checked?'Joint marked checked.':'Joint marked for another check.',{keepReviewed:true});
  $('reset-joint').onclick=()=>change(j=>{Object.assign(j,clone(defaults.joints.find(d=>d.id===j.id)));},'This joint returned to its starter settings. Undo restores your edit.');
  $('next-unchecked').onclick=()=>{const i=doc.joints.findIndex(j=>j.id===selected);for(let step=1;step<=doc.joints.length;step++){const next=doc.joints[(i+step)%doc.joints.length];if(!next.reviewed){selectJoint(next.id);view.fit(connectedParts());return;}}status('Every joint is checked. '+saveDirection);};
  $('undo').onclick=()=>undo();$('redo').onclick=()=>undo(true);$('save').onclick=saveProject;$('export').onclick=exportFile;$('import').onclick=()=>$('import-file').click();$('import-file').onchange=e=>importFile(e.target.files[0]);
  window.addEventListener('pointermove',moveDrag);window.addEventListener('pointerup',endDrag);window.addEventListener('pointercancel',endDrag);
  $('canvas').addEventListener('pointerdown',event=>{
    if(!ready||mode==='navigate'||event.button!==0)return;event.preventDefault();const j=joint();
    const planePoint=mode==='axis'?add(j.pivot,scale(j.axis,axisLength)):j.pivot;
    const point=positionOnPlane(event,planePoint,view.cameraDirection());if(!point)return;
    const kind=mode;const changed=change(item=>{if(kind==='center')item.pivot=point;else item.axis=normalizeAxis(subtract(point,item.pivot));},kind==='center'?'Center placed at the current depth. Check it from a side view.':'Axis direction updated with its previous tip depth. Check it from a perpendicular view.');if(changed)setMode('navigate');
  });
  window.addEventListener('keydown',e=>{
    if(!ready||['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName))return;
    if(e.key==='Escape'){setMode('navigate');return;}
    if((e.metaKey||e.ctrlKey)&&e.key.toLowerCase()==='z'){e.preventDefault();undo(e.shiftKey);}
  });
}
async function readPublishedMap() {
  const response=await fetch('../mannequin-articulation/joint-annotations.json');
  if(!response.ok)throw new Error('The published joint map could not be loaded.');
  const sourceText=await response.text(),saved=JSON.parse(sourceText);validate(saved);
  // Use the same fitted map as articulation only when its exact source bytes match.
  try {
    const response=await fetch('../mannequin-articulation/joint-refinement.json');
    if(response.ok){
      const fitted=await response.json(),digest=await crypto.subtle.digest('SHA-256',new TextEncoder().encode(sourceText));
      const hash=Array.from(new Uint8Array(digest),n=>n.toString(16).padStart(2,'0')).join('');
      if(fitted.schema==='mannequin-joint-refinement'&&fitted.version===1&&fitted.inputAnnotationSha256===hash&&fitted.inputSceneRevision===saved.scene.revision){validate(fitted.annotations);return fitted.annotations;}
    }
  } catch { /* A valid published map can still open without its optional refinement. */ }
  return saved;
}
async function start() {
  $('save').hidden=!localServer;
  if(!localServer)$('save-detail').textContent='Edits stay in this browser. Export JSON to keep or share them; the published manikin is unchanged.';
  const landmarksPromise=fetch('../../tools/fusion/front-landmarks.json').then(r=>{if(!r.ok)throw new Error('Starter landmark file unavailable.');return r.json();});
  view=await createScanView({canvas:$('canvas'),container:$('viewport'),onStatus:text=>{$('load-detail').textContent=text;},onCameraChange:cameraChanged});
  defaults=createAnnotations(view.manifest,await landmarksPromise);doc=clone(defaults);storageKey='mannequin-joints:v1:'+doc.scene.revision+':'+Object.values(doc.scene.sourceHashes).join(':');
  let notice='Starter joint positions · nothing checked yet.',serverDoc=null;
  if(localServer)try{const response=await fetch('/api/mannequin-joints');if(response.ok){const value=await response.json();const candidate=value.annotations||value;validate(candidate);serverDoc=candidate;doc=candidate;projectSaved=true;notice='Loaded the project joint annotations.';}else if(response.status!==404){notice='Project annotations could not be read. You can use a browser draft or export JSON.';}}
  catch(error){notice='Project annotations not loaded: '+error.message;}
  if(!serverDoc)try{serverDoc=await readPublishedMap();doc=clone(serverDoc);notice='Loaded the published joint map. '+saveDirection;}catch(error){notice+=' '+error.message+' Using starter positions.';}
  try{const saved=localStorage.getItem(storageKey);if(saved){const candidate=JSON.parse(saved);validate(candidate);if(!serverDoc||Date.parse(candidate.updatedAt)>Date.parse(serverDoc.updatedAt)){doc=candidate;projectSaved=false;notice='Restored your browser draft. '+saveDirection;}}}catch{notice+=' An incompatible browser draft was left untouched.';}
  doc.joints.forEach(j=>{j.axis=normalizeAxis(j.axis);});
  selected=doc.joints[0].id;
  for(const button of document.querySelectorAll('button'))button.disabled=false;$('joint-select').disabled=false;$('isolate').disabled=false;$('capture').disabled=false;$('editor').disabled=false;
  $('save').disabled=!localServer;
  const count=view.manifest.captures.reduce((n,c)=>n+c.count,0);$('scene-info').textContent=count.toLocaleString()+' Gaussians · '+doc.joints.length+' editable joint centers';
  $('save-state').textContent=projectSaved?'Loaded from the local project':localServer?'Editable draft · save when ready':'Editable browser draft · export to keep';
  events();ready=true;drawPanel();updateVisibility();renderMarkers();$('loading').hidden=true;status(notice);
  window.mannequinAnnotations={get annotations(){return clone(doc);},get state(){return {selected,mode,generation,projectSaved,ready};}};
}
start().catch(error=>{console.error(error);$('load-title').textContent='Unable to open the annotation tool';$('load-detail').textContent=error.message+' Open this page through the local server.';$('loading').querySelector('.loader').hidden=true;status(error.message);});
