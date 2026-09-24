/** Conservative sampled arm-to-body contacts in the fused scan frame.
 * Captured cross sections form the trunk/head envelope; captured PCA ellipsoids
 * approximate opposite-arm regions. This is not exhaustive body self-collision.
 * Same-side connected arm regions and proximal shoulder attachments are exempt.
 */
const IDENTITY={position:[0,0,0],rotation:[0,0,0,1]};
const ARMS=['left_upper_arm','left_forearm','left_hand','right_upper_arm','right_forearm','right_hand'];
const CORE=['torso','pelvis','head','neck'];
const dot=(a,b)=>a[0]*b[0]+a[1]*b[1]+a[2]*b[2];
const sub=(a,b)=>a.map((v,i)=>v-b[i]);
const length=a=>Math.hypot(...a);
const quantile=(v,q)=>{const a=[...v].sort((x,y)=>x-y),p=(a.length-1)*q,i=Math.floor(p);return a[i]+(a[Math.min(i+1,a.length-1)]-a[i])*(p-i);};
function rotate(q,p){const [x,y,z,w]=q,[a,b,c]=p,t=[2*(y*c-z*b),2*(z*a-x*c),2*(x*b-y*a)];return [a+w*t[0]+y*t[2]-z*t[1],b+w*t[1]+z*t[0]-x*t[2],c+w*t[2]+x*t[1]-y*t[0]];}
function transform(pose,p){const r=rotate(pose.rotation,p);return r.map((v,i)=>v+pose.position[i]);}
function inverse(pose,p){const q=pose.rotation;return rotate([-q[0],-q[1],-q[2],q[3]],sub(p,pose.position));}
function pose(value=IDENTITY){
  if(!value.position||value.position.length!==3||!value.position.every(Number.isFinite)||!value.rotation||value.rotation.length!==4||!value.rotation.every(Number.isFinite))throw new TypeError('Body contact transforms require finite position and XYZW rotation');
  const n=length(value.rotation);if(n<1e-12)throw new RangeError('Body contact quaternion cannot be zero');
  return {position:[...value.position],rotation:value.rotation.map(v=>v/n)};
}
function eigenAxes(points){
  const center=[0,1,2].map(k=>points.reduce((s,p)=>s+p[k],0)/points.length);
  const a=Array.from({length:3},()=>[0,0,0]),v=[[1,0,0],[0,1,0],[0,0,1]];
  for(const p of points){const d=sub(p,center);for(let i=0;i<3;i++)for(let j=0;j<3;j++)a[i][j]+=d[i]*d[j];}
  for(let iter=0;iter<30;iter++){
    let i=0,j=1;for(const [x,y]of [[0,2],[1,2]])if(Math.abs(a[x][y])>Math.abs(a[i][j])){i=x;j=y;}
    if(Math.abs(a[i][j])<1e-12)break;
    const angle=.5*Math.atan2(2*a[i][j],a[j][j]-a[i][i]),c=Math.cos(angle),s=Math.sin(angle);
    const ai=a.map(row=>row[i]),aj=a.map(row=>row[j]);
    for(let k=0;k<3;k++){a[k][i]=c*ai[k]-s*aj[k];a[k][j]=s*ai[k]+c*aj[k];}
    const ri=[...a[i]],rj=[...a[j]];
    for(let k=0;k<3;k++){a[i][k]=c*ri[k]-s*rj[k];a[j][k]=s*ri[k]+c*rj[k];const vi=v[k][i],vj=v[k][j];v[k][i]=c*vi-s*vj;v[k][j]=s*vi+c*vj;}
  }
  return {center,axes:[0,1,2].sort((i,j)=>a[j][j]-a[i][i]).map(k=>v.map(row=>row[k]))};
}
function armEllipsoid(part,points){
  const {center,axes}=eigenAxes(points),projected=points.map(p=>axes.map(axis=>dot(sub(p,center),axis)));
  const ranges=[0,1,2].map(k=>[quantile(projected.map(p=>p[k]),.025),quantile(projected.map(p=>p[k]),.975)]);
  const localCenter=ranges.map(r=>(r[0]+r[1])/2);
  return {part,center:center.map((x,k)=>x+axes.reduce((s,axis,i)=>s+axis[k]*localCenter[i],0)),axes,
    radii:ranges.map(r=>Math.max(.012,(r[1]-r[0])/2)),count:points.length,method:'captured PCA 2.5–97.5% ellipsoid'};
}
function coreSections(part,points){
  const zlo=quantile(points.map(p=>p[2]),.015),zhi=quantile(points.map(p=>p[2]),.985),span=zhi-zlo;
  const steps=Math.max(2,Math.ceil(span/.065)),spacing=span/steps,halfLength=Math.max(.045,spacing*1.65);
  const out=[];
  for(let i=0;i<=steps;i++){
    const z=zlo+i*spacing,section=points.filter(p=>Math.abs(p[2]-z)<=halfLength*.65);
    if(section.length<12)continue;
    // Torso/pelvis lateral samples include loose clothing flaps already beside
    // the accepted tucked arms. Use the captured central 60% lateral envelope
    // as an inner-body guard, retaining the measured front/back skin extent.
    const central=['torso','pelvis'].includes(part);
    const lateral=part==='pelvis'?.30:.20;
    const ranges=[0,1].map(k=>[quantile(section.map(p=>p[k]),central&&k===0?lateral:.025),quantile(section.map(p=>p[k]),central&&k===0?1-lateral:.975)]);
    // End sections round off over half a section, rather than extrapolating an
    // entire additional limb-length beyond the captured longitudinal envelope.
    const rz=Math.min(halfLength,Math.max(spacing*.55,Math.min(z-zlo,zhi-z)+spacing*.55));
    out.push({part,center:[(ranges[0][0]+ranges[0][1])/2,(ranges[1][0]+ranges[1][1])/2,z],
      axes:[[1,0,0],[0,1,0],[0,0,1]],radii:[Math.max(.012,(ranges[0][1]-ranges[0][0])/2),Math.max(.012,(ranges[1][1]-ranges[1][0])/2),rz],
      count:section.length,method:central?`captured central ${100*lateral}–${100*(1-lateral)}% lateral / 2.5–97.5% front-back sections`:'captured longitudinal 2.5–97.5% ellipse sections'});
  }
  return out;
}

/** restSamples are scan.getCollisionSamples(new Map(), identity placement).
 * evaluate takes actual motion.evaluate().parts and the uniform parent placement.
 * Report positions, normals, distances and tolerances use ambulance world units.
 */
export function createBodyContactChecker({restSamples,annotations,referenceParts,penetrationTolerance=.003,contactDistance=.006,shoulderAttachmentRadius=.105}={}){
  if(!Array.isArray(restSamples)||!restSamples.length)throw new TypeError('Captured rest samples are required');
  for(const [key,value]of Object.entries({penetrationTolerance,contactDistance,shoulderAttachmentRadius}))if(!Number.isFinite(value)||value<0)throw new RangeError(`${key} must be nonnegative`);
  const samples=restSamples.map(s=>{
    if(!s.position||s.position.length!==3||!s.position.every(Number.isFinite)||!Number.isFinite(s.radius)||s.radius<0)throw new TypeError('Invalid captured rest sample');
    return {...s,position:[...s.position]};
  });
  const byPart=new Map([...CORE,...ARMS].map(part=>[part,samples.filter(s=>s.part===part)]));
  for(const part of ['torso','pelvis','head',...ARMS])if(byPart.get(part).length<12)throw new RangeError(`Insufficient captured samples for ${part}`);
  const proxies=[...CORE.flatMap(part=>byPart.get(part).length>=12?coreSections(part,byPart.get(part).map(s=>s.position)):[]),
    ...ARMS.map(part=>armEllipsoid(part,byPart.get(part).map(s=>s.position)))];
  proxies.forEach((p,i)=>{p.id=`body-${p.part}-${i}`;p.bound=Math.max(...p.radii);});
  const shoulders=new Map(['left','right'].map(side=>{
    const joint=annotations?.joints?.find(j=>j.id===`${side}_shoulder`);
    if(!joint?.pivot?.every(Number.isFinite)||joint.pivot.length!==3)throw new TypeError('Actual shoulder annotations are required');
    return [side,[...joint.pivot]];
  }));
  const armSamples=samples.filter(s=>ARMS.includes(s.part));
  const groups=[...CORE,...ARMS].map(part=>({part,proxies:proxies.filter(p=>p.part===part)})).filter(g=>g.proxies.length);
  const exceptions=new Map();
  const exceptionKey=(sampleId,proxy)=>`${sampleId}|${proxy}`;
  function evaluate(partTransforms,placement={...IDENTITY,scale:1},{stopOnBlock=false,movingParts=ARMS}={}){
    const parent=pose(placement),scale=placement.scale;
    if(!Number.isFinite(scale)||scale<=0)throw new RangeError('Body contact placement needs positive scale');
    const transforms=new Map();for(const part of [...CORE,...ARMS])transforms.set(part,pose(partTransforms instanceof Map?partTransforms.get(part):partTransforms?.[part]));
    const moving=new Set(movingParts??ARMS),contacts=[],violations=[],byPart=Object.create(null);let count=0,maxPenetration=0,minimumContactClearance=Infinity,complete=true;
    outer:for(const sample of armSamples){
      if(!moving.has(sample.part))continue;
      count++;const side=sample.part.startsWith('left_')?'left':'right',p=transform(transforms.get(sample.part),sample.position);
      byPart[sample.part]??={contacts:0,violations:0,maxPenetration:0};
      const proximal=sample.part===`${side}_upper_arm`&&length(sub(sample.position,shoulders.get(side)))<shoulderAttachmentRadius;
      for(const group of groups){
        if(group.part.startsWith(`${side}_`)||(proximal&&['torso','neck'].includes(group.part)))continue;
        const targetPose=transforms.get(group.part),local=inverse(targetPose,p),padding=sample.radius+contactDistance/scale;
      for(const proxy of group.proxies){
        const dx=local[0]-proxy.center[0],dy=local[1]-proxy.center[1],dz=local[2]-proxy.center[2],bound=proxy.bound+padding;
        if(dx*dx+dy*dy+dz*dz>bound*bound)continue;
        const a=proxy.axes,r=proxy.radii;
        const x=dx*a[0][0]+dy*a[0][1]+dz*a[0][2],y=dx*a[1][0]+dy*a[1][1]+dz*a[1][2],z=dx*a[2][0]+dy*a[2][1]+dz*a[2][2];
        if(Math.abs(x)>r[0]+padding||Math.abs(y)>r[1]+padding||Math.abs(z)>r[2]+padding)continue;
        const q=Math.hypot(x/r[0],y/r[1],z/r[2]);
        // Implicit-distance lower estimate outside the ellipsoid deliberately
        // waits until a sampled sphere approaches the fitted captured envelope.
        const gradient=[x/(r[0]*r[0]),y/(r[1]*r[1]),z/(r[2]*r[2])];
        const g=length(gradient),surfaceDistance=q>1e-12?(q-1)*q/Math.max(g,1e-12):-Math.min(...proxy.radii);
        const clearance=(surfaceDistance-sample.radius)*scale;
        if(clearance>contactDistance)continue;
        let localNormal=q>1e-12?proxy.axes[0].map((_,k)=>proxy.axes.reduce((sum,axis,i)=>sum+axis[k]*gradient[i]/g,0)):[0,1,0];
        const normal=rotate(parent.rotation,rotate(targetPose.rotation,localNormal));
        const world=transform(parent,p.map(v=>v*scale)),penetration=Math.max(0,-clearance);
        const exception=exceptions.get(exceptionKey(sample.id,proxy.id));
        const baselineOverlap=exception&&length(sub(local,exception.localPoint))<=exception.radius;
        const allowance=baselineOverlap?Math.max(penetrationTolerance,exception.penetration*scale+1e-8):penetrationTolerance;
        const blocked=penetration>allowance+1e-10;
        const contact={sampleId:sample.id,part:sample.part,proxy:proxy.id,targetPart:proxy.part,label:`Body: ${proxy.part.replaceAll('_',' ')}`,
          support:false,clearance,penetration,allowedPenetration:allowance,normal,localPoint:[...local],baselineOverlap:Boolean(baselineOverlap),
          point:world.map((v,i)=>v-normal[i]*(sample.radius*scale+clearance)),blocked};
        contacts.push(contact);minimumContactClearance=Math.min(minimumContactClearance,clearance);maxPenetration=Math.max(maxPenetration,penetration);
        const stat=byPart[sample.part];stat.contacts++;stat.maxPenetration=Math.max(stat.maxPenetration,penetration);
        if(blocked){violations.push(contact);stat.violations++;if(stopOnBlock){complete=false;break outer;}}
      }
      }
    }
    return {status:violations.length?'penetrating':contacts.length?'contact':'clear',blocked:violations.length>0,sampleCount:count,
      contacts,violations,byPart,complete,maxPenetration,minimumContactClearance:Number.isFinite(minimumContactClearance)?minimumContactClearance:null,
      scope:'sampled arms versus fitted trunk/head/opposite-arm envelopes; connected attachments excluded'};
  }
  if(referenceParts){
    // Existing accepted clothing/scan overlap is not a new allowance elsewhere.
    // It is tied to an exact sample and exact fitted surface near the original
    // overlap. The neighborhood covers its depth so the arm can move outward.
    const baseline=evaluate(referenceParts,{...IDENTITY,scale:1});
    for(const c of baseline.contacts)if(c.penetration>0)exceptions.set(exceptionKey(c.sampleId,c.proxy),{
      sampleId:c.sampleId,proxy:c.proxy,part:c.part,targetPart:c.targetPart,localPoint:c.localPoint,
      penetration:c.penetration,radius:Math.max(.035,c.penetration+.015)});
  }
  return {evaluate,proxies:structuredClone(proxies),sampleCount:armSamples.length,
    referenceExceptions:structuredClone([...exceptions.values()]),
    policy:Object.freeze({penetrationTolerance,contactDistance,shoulderAttachmentRadius,units:'world tolerances; attachment radius in captured scan units',
      limitations:'Approximate captured envelopes, not a watertight body mesh. Does not test legs, same-side connected arm pairs or proximal shoulder attachments.'})};
}
