import * as pc from './vendor/playcanvas.mjs';
import {validateRig, classifyPoint, evaluateRig, stepSpring, clampJointAngles} from './rig-core.mjs';
const $ = id => document.getElementById(id);
const clone = obj => structuredClone(obj);
const rad = Math.PI / 180;
const nextFrame = () => new Promise(resolve => requestAnimationFrame(resolve));
const storageKey = 'mannequin-rig:v1';
let rig, defaults, scan, packed, labels, app, camera, selected, segmentIndex=1, shapeIndex=0, tab='pose';
let entities=[], markers=new Map(), pose={}, velocity={}, transforms=new Map(), busy=false, dirty=false;
let spring=true, brushRadius=.03, drag=null, frameCount=0, lastFps=performance.now(), saveTimer;
const orbit={target:new pc.Vec3(0,0,-.75),yaw:35,pitch:48,distance:3.1};
const v = a => new pc.Vec3(...a);
const status = text => { $('status').textContent=text; };
function save() {
 clearTimeout(saveTimer); saveTimer=setTimeout(()=>{
  try { localStorage.setItem(storageKey,JSON.stringify(rig)); $('save-state').textContent='Autosaved in this browser'; }
  catch { $('save-state').textContent='Browser storage unavailable · export your rig'; }
 },250);
}
function loading(title,detail='') { $('loading').hidden=false;$('load-title').textContent=title;$('load-detail').textContent=detail; }
function resetMotion() { for(const j of rig.joints){pose[j.id]=[0,0,0];velocity[j.id]=[0,0,0];j.target=[0,0,0];j.angles=[0,0,0];} }
function num(value,step,onChange,min,max) {
 const input=document.createElement('input'); input.type='number';input.value=value;input.step=step;
 if(min!==undefined)input.min=min;if(max!==undefined)input.max=max;
 input.addEventListener('change',()=>{const n=Number(input.value);if(input.value===''||!Number.isFinite(n)||(min!==undefined&&n<min)||(max!==undefined&&n>max)){input.value=value;return;}value=n;onChange(n);save();});return input;
}
function vector(container,label,value,onChange,step=.005) {
 const wrap=document.createElement('div');wrap.className='vector-label';wrap.textContent=label;
 const row=document.createElement('div');row.className='vector';
 value.forEach((val,i)=>{const l=document.createElement('label');l.textContent='XYZ'[i];l.append(num(Number(val.toFixed(4)),step,n=>{value[i]=n;onChange();}));row.append(l);});container.append(wrap,row);
}
function selectOptions(select,entries,value) { select.replaceChildren(...entries.map(([id,label])=>new Option(label,id)));select.value=value; }
function joint(){return rig.joints.find(j=>j.id===selected);}
function markDirty(){dirty=true;$('apply').textContent='Apply segmentation · pending';status('Segmentation edited. Apply to update the Gaussian assignments.');save();}
function drawJointPanel() {
 const j=joint(); if(!j)return;
 $('joint-meta').textContent=`${j.type==='ball'?'Ball joint · 3 DOF':j.type==='hinge'?(j.motion==='axial'?'Axial swivel · 1 DOF':'Hinge · 1 DOF'):'Fixed joint'} · ${j.parent?rig.joints.find(p=>p.id===j.parent)?.label:'root'}`;
 $('angles').replaceChildren();
 const axes=j.type==='hinge'?[j.motion==='axial'?'Twist':'Rotation']:j.type==='fixed'?[]:['Rotate X','Rotate Y','Rotate Z'];
 axes.forEach((name,i)=>{
  const field=document.createElement('div');field.className='field';const head=document.createElement('div');head.className='field-head';const label=document.createElement('label');label.textContent=name;label.htmlFor=`angle-${i}`;
  const output=document.createElement('output');output.id=`angle-value-${i}`;output.textContent=`${j.target[i].toFixed(1)}°`;
  const range=document.createElement('input');range.type='range';range.id=`angle-${i}`;range.min=j.limits[i][0];range.max=j.limits[i][1];range.step=.5;range.value=j.target[i];
  range.oninput=()=>{j.target[i]=+range.value;output.textContent=`${j.target[i].toFixed(1)}°`;save();};head.append(label,output);field.append(head,range);$('angles').append(field);
 });
 $('dynamics').replaceChildren();
 for(const [key,label,step,min,max] of [['stiffness','Stiffness · k',1,0,1000],['damping','Damping · c',.2,0,100],['inertia','Inertia · I',.1,.01,100]]){
  const field=document.createElement('div');field.className='field-head field';const l=document.createElement('label');l.textContent=label;field.append(l,num(j[key],step,n=>j[key]=n,min,max));$('dynamics').append(field);
 }
 $('pivot').replaceChildren();vector($('pivot'),'Pivot in scan coordinates',j.pivot,()=>{status('Pivot updated in the rest pose.');});
 $('joint-type').value=j.type;
 $('axis').replaceChildren();if(j.type==='hinge')vector($('axis'),'Rotation axis (nonzero vector)',j.axis,()=>{
  const n=Math.hypot(...j.axis);if(n<1e-6)j.axis=[1,0,0];else j.axis=j.axis.map(x=>x/n);drawJointPanel();
 },.1);
 $('frame').replaceChildren();vector($('frame'),'Local frame orientation · degrees',j.frame,()=>{},1);
 $('limits').replaceChildren();axes.forEach((name,i)=>{const row=document.createElement('div');row.className='limit-row';row.append(document.createTextNode(name),num(j.limits[i][0],1,n=>{
  if(n>j.limits[i][1]){drawJointPanel();return;}j.limits[i][0]=n;j.target=clampJointAngles(j,j.target);drawJointPanel();
 }),num(j.limits[i][1],1,n=>{if(n<j.limits[i][0]){drawJointPanel();return;}j.limits[i][1]=n;j.target=clampJointAngles(j,j.target);drawJointPanel();}));$('limits').append(row);});
 $('kick').disabled=j.type==='fixed';
}
function drawSegmentPanel() {
 const s=rig.segments[segmentIndex];
 selectOptions($('segment-select'),rig.segments.map((s,i)=>[i,s.label]),segmentIndex);
 selectOptions($('attachment'),[['','Fixed / environment'],...rig.joints.map(j=>[j.id,j.label])],s.joint??'');
 selectOptions($('shape-select'),s.shapes.map((shape,i)=>[i,`${i+1} · ${shape.type}`]),shapeIndex);
 $('shape-fields').replaceChildren();const shape=s.shapes[shapeIndex];
 if(shape?.type==='capsule'){
  vector($('shape-fields'),'Start point',shape.a,markDirty);vector($('shape-fields'),'End point',shape.b,markDirty);
  const row=document.createElement('div');row.className='field-head field';row.append(document.createTextNode('Radius'),num(shape.radius,.005,n=>{shape.radius=n;markDirty();},.001,2));$('shape-fields').append(row);
 } else if(shape?.type==='box'){
  vector($('shape-fields'),'Minimum corner',shape.min,()=>{shape.min=shape.min.map((x,i)=>Math.min(x,shape.max[i]));markDirty();drawSegmentPanel();});vector($('shape-fields'),'Maximum corner',shape.max,()=>{shape.max=shape.max.map((x,i)=>Math.max(x,shape.min[i]));markDirty();drawSegmentPanel();});
 }
 $('remove-shape').disabled=!shape;
 $('add-capsule').disabled=segmentIndex===0;$('add-box').disabled=segmentIndex===0;$('attachment').disabled=segmentIndex===0;
 $('segment-count').textContent=`${(entities[segmentIndex]?.count??0).toLocaleString()} assigned splats`;
 $('paint-count').textContent=`${rig.paint?.length??0} saved brush strokes · ${dirty?'pending changes':'assignments applied'}`;
 updateVisibility();
}
function selectJoint(id){selected=id;$('joint-select').value=id;drawJointPanel();const si=rig.segments.findIndex(s=>s.joint===id);if(si>=0){segmentIndex=si;shapeIndex=0;drawSegmentPanel();}}
function setTab(name){tab=name;document.querySelectorAll('[data-tab]').forEach(b=>b.classList.toggle('active',b.dataset.tab===name));for(const id of ['pose','joints','segments'])$(`${id}-panel`).hidden=id!==name;
 $('common').hidden=name==='segments';$('paint').checked=false;$('canvas').style.cursor='';
 $('gesture').textContent=name==='joints'?'Drag the selected pivot · use another view to set depth':name==='pose'?'Drag to orbit · drag a joint to rotate · scroll to zoom':'Adjust volumes or paint · Apply segmentation to preview';
 if(name!=='pose'){for(const j of rig.joints){pose[j.id]=[0,0,0];velocity[j.id]=[0,0,0];}drawJointPanel();}save();
}
function updateCamera(){const yaw=orbit.yaw*rad,pitch=orbit.pitch*rad;camera.setPosition(orbit.target.x+orbit.distance*Math.sin(yaw)*Math.cos(pitch),orbit.target.y+orbit.distance*Math.sin(pitch),orbit.target.z+orbit.distance*Math.cos(yaw)*Math.cos(pitch));camera.lookAt(orbit.target);}
function setView(name){const presets={perspective:[35,48],top:[0,89.8],front:[90,7],end:[0,12]};[orbit.yaw,orbit.pitch]=presets[name];orbit.distance=name==='top'?2.9:3.1;orbit.target.set(0,0,-.75);document.querySelectorAll('[data-view]').forEach(b=>b.classList.toggle('active',b.dataset.view===name));updateCamera();}
function updateVisibility(){entities.forEach((e,i)=>{if(e)e.entity.enabled=(!$('isolate').checked||i===segmentIndex)&&(i!==0||$('show-bed').checked);});}
function tintSegments(){entities.forEach((e,i)=>{if(!e)return;const color=rig.segments[i].color; e.entity.gsplat.setWorkBufferModifier($('colors').checked?{glsl:`void modifySplatCenter(inout vec3 center) {}\nvoid modifySplatRotationScale(vec3 originalCenter, vec3 modifiedCenter, inout vec4 rotation, inout vec3 scale) {}\nvoid modifySplatColor(vec3 center, inout vec4 color) { color.rgb=vec3(${color.map(x=>Number(x).toFixed(4)).join(',')}); }`}:null);});}
function rebuildMarkers(){for(const m of markers.values())m.remove();markers.clear();for(const j of rig.joints){const b=document.createElement('button');b.className='marker';b.title=j.label;b.setAttribute('aria-label',j.label+' joint');b.dataset.label=j.label;
 b.onpointerdown=e=>{if(busy)return;e.stopPropagation();selectJoint(j.id);b.setPointerCapture(e.pointerId);drag={type:tab==='joints'?'pivot':tab==='pose'?'joint':'select',x:e.clientX,y:e.clientY,start:[...j.pivot],target:[...j.target]};};
 b.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.type==='pivot'){
  const scale=2*orbit.distance*Math.tan(camera.camera.fov*rad/2)/$('canvas').clientHeight;const right=camera.right,up=camera.up;
  j.pivot=drag.start.map((p,i)=>p+scale*(dx*[right.x,right.y,right.z][i]-dy*[up.x,up.y,up.z][i]));
 }else if(drag.type==='joint'&&j.type!=='fixed'){const a=[...drag.target];a[0]-=dy*.45;if(j.type==='ball')a[1]+=dx*.45;j.target=clampJointAngles(j,a);}drawJointPanel();};
 const stop=()=>{if(drag){drag=null;save();}};b.onpointerup=stop;b.onpointercancel=stop;$('markers').append(b);markers.set(j.id,b);}}
function drawGuides(){
 const color=new pc.Color(.62,.82,.52,.8);
 if($('show-joints').checked)for(const j of rig.joints){const t=transforms.get(j.id),parent=transforms.get(j.parent);if(t&&parent)app.drawLine(v(t.pivot),v(parent.pivot),color,false);}
 if(tab!=='segments')return;
 const shape=rig.segments[segmentIndex].shapes[shapeIndex];if(!shape)return;const c=new pc.Color(.8,1,.5);
 if(shape.type==='box'){const corners=[];for(let i=0;i<8;i++)corners.push([0,1,2].map(k=>(i&(1<<k))?shape.max[k]:shape.min[k]));for(let i=0;i<8;i++)for(let k=0;k<3;k++)if(!(i&(1<<k)))app.drawLine(v(corners[i]),v(corners[i|(1<<k)]),c,false);}
 else {app.drawLine(v(shape.a),v(shape.b),c,false);for(const p of [shape.a,shape.b])for(let axis=0;axis<3;axis++){let prev;for(let n=0;n<=32;n++){const point=[...p];point[(axis+1)%3]+=shape.radius*Math.cos(n*Math.PI/16);point[(axis+2)%3]+=shape.radius*Math.sin(n*Math.PI/16);if(prev)app.drawLine(v(prev),v(point),c,false);prev=point;}}}
}
function animate(dt){if(!rig||busy)return;dt=Math.min(dt,.05);
 for(const j of rig.joints){if(!pose[j.id]){pose[j.id]=[...j.angles];velocity[j.id]=[0,0,0];}if(tab!=='pose'){pose[j.id]=[0,0,0];continue;}
  if(!spring){pose[j.id]=clampJointAngles(j,j.target);velocity[j.id]=[0,0,0];continue;}
  const dims=j.type==='ball'?3:j.type==='hinge'?1:0;
  for(let i=0;i<dims;i++){const result=stepSpring(pose[j.id][i]*rad,velocity[j.id][i],j.target[i]*rad,j.stiffness,j.damping,j.inertia,dt,j.limits[i][0]*rad,j.limits[i][1]*rad);pose[j.id][i]=result.angle/rad;velocity[j.id][i]=result.velocity;}
 }
 transforms=evaluateRig(rig,pose);
 entities.forEach((entry,i)=>{if(!entry)return;const transform=transforms.get(rig.segments[i].joint);if(transform){entry.entity.setPosition(...transform.position);entry.entity.setRotation(new pc.Quat(...transform.rotation));}else{entry.entity.setPosition(0,0,0);entry.entity.setRotation(pc.Quat.IDENTITY);}});
 for(const j of rig.joints){const t=transforms.get(j.id);const m=markers.get(j.id);if(!m||!t)continue;const p=camera.camera.worldToScreen(v(t.pivot));m.hidden=!$('show-joints').checked||p.z<=0||p.x<0||p.y<0||p.x>$('canvas').clientWidth||p.y>$('canvas').clientHeight;m.style.left=p.x+'px';m.style.top=p.y+'px';m.classList.toggle('selected',selected===j.id);}
 drawGuides();frameCount++;const now=performance.now();if(now-lastFps>1000){$('fps').textContent=`${Math.round(frameCount*1000/(now-lastFps))} fps`;frameCount=0;lastFps=now;}
}
function applyPaintToLabels(){const byId=new Map(rig.segments.map((s,i)=>[s.id,i]));for(const stroke of rig.paint??[]){const si=byId.get(stroke.segment);if(si===undefined)continue;const [x,y,z]=stroke.center,r2=stroke.radius**2;for(let i=0;i<scan.count;i++){const k=i*14;if((packed[k]-x)**2+(packed[k+1]-y)**2+(packed[k+2]-z)**2<=r2)labels[i]=si;}}}
async function rebuild(){if(busy)return;try{validateRig(rig);}catch(err){status(`Cannot apply: ${err.message}`);return;}busy=true;document.querySelector('.panel-scroll').inert=true;for(const el of document.querySelectorAll('header button,.tabs button'))el.disabled=true;loading('Assigning the Gaussians','Fitting body parts and rebuilding the local scene…');await nextFrame();await nextFrame();
 try{
  labels=new Uint16Array(scan.count);const point=[0,0,0];for(let i=0;i<scan.count;i++){const k=i*14;point[0]=packed[k];point[1]=packed[k+1];point[2]=packed[k+2];labels[i]=classifyPoint(point,rig.segments);}
  applyPaintToLabels();const counts=new Uint32Array(rig.segments.length);for(const l of labels)counts[l]++;
  for(const entry of entities){if(entry){entry.entity.destroy();entry.resource.destroy();}}entities=[];
  for(let si=0;si<rig.segments.length;si++){
   if(!counts[si]){entities.push(null);continue;}
   const arrays=scan.fields.map(()=>new Float32Array(counts[si]));let at=0;
   for(let i=0;i<scan.count;i++)if(labels[i]===si){const k=i*14;for(let f=0;f<14;f++)arrays[f][at]=packed[k+f];at++;}
   const data=new pc.GSplatData([{name:'vertex',count:counts[si],properties:scan.fields.map((name,f)=>({name,type:'float',byteSize:4,storage:arrays[f]}))}]);
   const resource=new pc.GSplatResource(app.graphicsDevice,data);
   const entity=new pc.Entity(rig.segments[si].id);entity.addComponent('gsplat',{resource});app.root.addChild(entity);entities.push({entity,resource,count:counts[si]});
   $('load-detail').textContent=`${rig.segments[si].label} · ${counts[si].toLocaleString()} splats`;await nextFrame();
  }
  dirty=false;$('apply').textContent='Apply segmentation';tintSegments();updateVisibility();drawSegmentPanel();save();status(`${scan.count.toLocaleString()} Gaussians · ${rig.joints.filter(j=>j.type!=='fixed').length} movable joints · local editing copy`);
 }catch(err){status('Rebuild failed: '+err.message);console.error(err);}finally{busy=false;document.querySelector('.panel-scroll').inert=false;for(const el of document.querySelectorAll('header button,.tabs button'))el.disabled=false;$('loading').hidden=true;}
}
function pickScan(clientX,clientY){
 const rect=$('canvas').getBoundingClientRect(),x=clientX-rect.left,y=clientY-rect.top;
 const origin=camera.getPosition().clone();const far=camera.camera.screenToWorld(x,y,10);const dir=far.sub(origin).normalize();
 let best=-1,bestScore=Infinity;
 // Small angular cone; prefer centers near the cursor, then the front visible layer.
 const cone=5/rect.height*2*Math.tan(camera.camera.fov*rad/2);
 for(let i=0;i<scan.count;i++){
  if((!$('show-bed').checked&&labels[i]===0)||($('isolate').checked&&labels[i]!==segmentIndex))continue;
  const k=i*14,dx=packed[k]-origin.x,dy=packed[k+1]-origin.y,dz=packed[k+2]-origin.z;
  const t=dx*dir.x+dy*dir.y+dz*dir.z;if(t<=0)continue;
  const d2=dx*dx+dy*dy+dz*dz-t*t;
  if(d2<=(cone*t)**2){const score=t+Math.sqrt(Math.max(0,d2))*4;if(score<bestScore){best=i;bestScore=score;}}
 }
 return best<0?null:[packed[best*14],packed[best*14+1],packed[best*14+2]];
}
function setupEvents(){
 $('joint-select').onchange=e=>selectJoint(e.target.value);
 document.querySelectorAll('[data-tab]').forEach(b=>b.onclick=()=>setTab(b.dataset.tab));
 document.querySelectorAll('[data-view]').forEach(b=>b.onclick=()=>setView(b.dataset.view));
 $('spring').onchange=e=>{spring=e.target.checked;$('motion-state').textContent=spring?'Live':'Direct';};
 $('rest').onclick=()=>{resetMotion();drawJointPanel();save();};
 $('release').onclick=()=>{joint().target=[0,0,0];drawJointPanel();save();};
 $('kick').onclick=()=>{const j=joint();spring=true;$('spring').checked=true;$('motion-state').textContent='Live';velocity[j.id][0]+=1.8;};
 $('demo').onclick=()=>{resetMotion();for(const j of rig.joints){if(/elbow/.test(j.id))j.target[0]=j.motion==='axial'?30:60;if(/shoulder/.test(j.id))j.target[2]=j.pivot[0]>0?20:-20;if(/neck/.test(j.id))j.target[1]=20;if(/knee/.test(j.id))j.target[0]=30;j.target=clampJointAngles(j,j.target);}drawJointPanel();save();};
 $('joint-type').onchange=e=>{const j=joint();j.type=e.target.value;const n=Math.hypot(...j.axis);j.axis=n>1e-6?j.axis.map(x=>x/n):[1,0,0];resetMotion();drawJointPanel();save();};
 $('segment-select').onchange=e=>{segmentIndex=+e.target.value;shapeIndex=0;drawSegmentPanel();};
 $('attachment').onchange=e=>{rig.segments[segmentIndex].joint=e.target.value||null;save();};
 $('shape-select').onchange=e=>{shapeIndex=+e.target.value;drawSegmentPanel();};
 $('add-capsule').onclick=()=>{const s=rig.segments[segmentIndex],p=rig.joints.find(j=>j.id===s.joint)?.pivot??[0,.1,-.5];s.shapes.push({type:'capsule',a:[...p],b:[p[0],p[1],p[2]-.15],radius:.06});shapeIndex=s.shapes.length-1;markDirty();drawSegmentPanel();};
 $('add-box').onclick=()=>{const s=rig.segments[segmentIndex],p=rig.joints.find(j=>j.id===s.joint)?.pivot??[0,.1,-.5];s.shapes.push({type:'box',min:p.map(x=>x-.05),max:p.map(x=>x+.05)});shapeIndex=s.shapes.length-1;markDirty();drawSegmentPanel();};
 $('remove-shape').onclick=()=>{rig.segments[segmentIndex].shapes.splice(shapeIndex,1);shapeIndex=0;markDirty();drawSegmentPanel();};
 $('apply').onclick=rebuild;
 $('brush-radius').oninput=e=>{brushRadius=+e.target.value;$('brush-value').textContent=brushRadius.toFixed(3);};
 $('paint').onchange=e=>{$('canvas').style.cursor=e.target.checked?'crosshair':'';};
 $('undo-paint').onclick=()=>{rig.paint?.pop();markDirty();drawSegmentPanel();};
 $('clear-paint').onclick=()=>{rig.paint=[];markDirty();drawSegmentPanel();};
 $('colors').onchange=tintSegments;for(const id of ['show-bed','isolate'])$(id).onchange=updateVisibility;
 $('export').onclick=()=>{for(const j of rig.joints)j.angles=[...pose[j.id]];const blob=new Blob([JSON.stringify(rig,null,2)+'\n'],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='mannequin-rig.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);status('Rig exported: pivots, constraints, target pose, volumes, and paint corrections.');};
 $('import').onclick=()=>$('file').click();
 $('file').onchange=async e=>{try{const file=e.target.files[0];if(!file)return;const imported=JSON.parse(await file.text());validateImport(imported);rig=imported;initializeRig();await rebuild();status('Imported rig and rebuilt segmentation.');}catch(err){status('Import rejected: '+err.message);}e.target.value='';};
 $('defaults').onclick=async()=>{rig=clone(defaults);initializeRig();await rebuild();status('Starter rig restored.');};
 const canvas=$('canvas');canvas.oncontextmenu=e=>e.preventDefault();
 canvas.onpointerdown=e=>{if(busy)return;canvas.setPointerCapture(e.pointerId);
  if(tab==='segments'&&$('paint').checked&&e.button===0){const p=pickScan(e.clientX,e.clientY);if(p){rig.paint??=[];rig.paint.push({segment:rig.segments[segmentIndex].id,center:p,radius:brushRadius});markDirty();drawSegmentPanel();status(`Painted ${rig.segments[segmentIndex].label}. Apply segmentation to see the change.`);}else status('No splat under the cursor. Try a closer view.');return;}
  drag={type:e.button===2||e.shiftKey?'pan':'orbit',x:e.clientX,y:e.clientY,yaw:orbit.yaw,pitch:orbit.pitch,target:orbit.target.clone()};};
 canvas.onpointermove=e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
  if(drag.type==='orbit'){orbit.yaw=drag.yaw-dx*.3;orbit.pitch=Math.max(-85,Math.min(89.8,drag.pitch+dy*.3));}
  else if(drag.type==='pan'){const scale=orbit.distance*.0015;orbit.target.copy(drag.target).add(camera.right.clone().mulScalar(-dx*scale)).add(camera.up.clone().mulScalar(dy*scale));}updateCamera();};
 canvas.onpointerup=canvas.onpointercancel=()=>drag=null;
 canvas.addEventListener('wheel',e=>{e.preventDefault();orbit.distance=Math.max(.3,Math.min(10,orbit.distance*Math.exp(e.deltaY*.001)));updateCamera();},{passive:false});
}
function validateImport(candidate){validateRig(candidate);if(!candidate.joints.length)throw new Error('At least one joint is required.');if(candidate.scan?.sourceSha256&&candidate.scan.sourceSha256!==scan.sourceSha256)throw new Error('Rig belongs to a different source scan.');if(candidate.paint!==undefined&&!Array.isArray(candidate.paint))throw new Error('paint must be an array.');if((candidate.paint?.length??0)>10000)throw new Error('Too many brush strokes (maximum 10,000).');for(const stroke of candidate.paint??[]){if(!rigSegmentIds(candidate).has(stroke.segment)||!Array.isArray(stroke.center)||stroke.center.length!==3||!stroke.center.every(Number.isFinite)||!Number.isFinite(stroke.radius)||stroke.radius<=0||stroke.radius>2)throw new Error('Invalid paint stroke.');}}
function rigSegmentIds(r){return new Set(r.segments.map(s=>s.id));}
function initializeRig(){rig.paint??=[];rig.scan={sourceSha256:scan.sourceSha256,coordinateTransform:'[x,-y,-z]'};pose={};velocity={};for(const j of rig.joints){pose[j.id]=[...j.angles];velocity[j.id]=[0,0,0];}
 selected=rig.joints.find(j=>j.type!=='fixed')?.id??rig.joints[0].id;segmentIndex=Math.max(0,rig.segments.findIndex(s=>s.joint===selected));shapeIndex=0;
 selectOptions($('joint-select'),rig.joints.map(j=>[j.id,j.label]),selected);drawJointPanel();drawSegmentPanel();rebuildMarkers();save();}
async function fetchJson(url){const r=await fetch(url);if(!r.ok)throw new Error(`${url}: HTTP ${r.status}`);return r.json();}
async function start(){
 [scan,defaults]=await Promise.all([fetchJson('./scan.json'),fetchJson('./default-rig.json')]);
 const response=await fetch('./'+scan.file);if(!response.ok)throw new Error('Editing copy missing. Run python3 tools/prepare_mannequin.py.');packed=new Float32Array(await response.arrayBuffer());if(packed.length!==scan.count*14)throw new Error('scan.bin does not match scan.json; regenerate the editing copy.');
 rig=clone(defaults);validateImport(rig);let restoreWarning='';try{const saved=localStorage.getItem(storageKey);if(saved){const candidate=JSON.parse(saved);validateImport(candidate);rig=candidate;}}catch(err){restoreWarning='Saved rig not loaded: '+err.message;}
 app=new pc.Application($('canvas'),{graphicsDeviceOptions:{deviceTypes:['webgl2'],antialias:false,alpha:true,preserveDrawingBuffer:true}});
 app.graphicsDevice.maxPixelRatio=Math.min(devicePixelRatio,1.5);app.scene.toneMapping=pc.TONEMAP_LINEAR;
 camera=new pc.Entity('Camera');camera.addComponent('camera',{clearColor:new pc.Color(.055,.085,.084,1),fov:45,nearClip:.01,farClip:100});app.root.addChild(camera);
 app.setCanvasFillMode(pc.FILLMODE_NONE);app.setCanvasResolution(pc.RESOLUTION_AUTO);
 const resize=()=>{app.resizeCanvas($('viewport').clientWidth,$('viewport').clientHeight);};new ResizeObserver(resize).observe($('viewport'));resize();updateCamera();app.on('update',animate);app.start();
 initializeRig();setupEvents();$('scan-info').textContent=`${(scan.count/1e6).toFixed(2)}M Gaussians · ${rig.joints.filter(j=>j.type!=='fixed').length} movable joints · editable starter segmentation`;
 await rebuild();if(restoreWarning)status(restoreWarning);
 // Read-only diagnostics make validation possible without mutating the renderer.
 window.mannequinLab={get rig(){return clone(rig);},get stats(){return{count:scan.count,parts:entities.map((e,i)=>({id:rig.segments[i].id,count:e?.count??0,visible:e?.entity.enabled??false})),pose:clone(pose),dirty,tab};}};
}
start().catch(err=>{console.error(err);loading('Unable to load the articulation lab',err.message+' Serve this folder over HTTP and run python3 tools/prepare_mannequin.py if scan.bin is missing.');$('loading').querySelector('.loader').hidden=true;status(err.message);});
