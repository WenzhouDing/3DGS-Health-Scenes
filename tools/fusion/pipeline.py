#!/usr/bin/env python3
"""Partwise Gaussian fusion. Preserve source PLYs; all fitting uses proxy points."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
from prepare_mannequin import read_ply, FIELDS, sha256_file
from cleanup_masks import load_cleanup_masks, keep_source_rows, validate_restorations, restore_surface_weights
from appearance_colors import load_color_overrides, apply_color_values

BODY_PARTS = [
 ('torso','Chest / abdomen','neck','waist',.235,.145),
 ('pelvis','Pelvis','waist','pelvis',.205,.135),
 ('head','Skull / face','skull_base','crown',.130,.145),
 ('neck','Neck','neck','skull_base',.075,.080),
]
for side in ('left','right'):
 for suffix,title,a,b,rx,ry in [
  ('upper_arm','upper arm','shoulder','elbow',.076,.070),
  ('forearm','forearm','elbow','wrist',.057,.056),
  ('hand','hand','wrist','hand_tip',.055,.034),
  ('thigh','thigh','hip','knee',.125,.100),
  ('shin','lower leg','knee','ankle',.069,.065),
  ('foot','foot','ankle','toe',.075,.063)]:
  BODY_PARTS.append((side+'_'+suffix,side.title()+' '+title,side+'_'+a,side+'_'+b,rx,ry))
PALETTE=[[.2,.76,.82],[.44,.47,.86],[.98,.72,.28],[.84,.36,.50],[.94,.51,.48],[.94,.70,.61],[.44,.71,.41],[.29,.77,.59],[.45,.85,.77],[.48,.63,.94],[.61,.50,.88],[.79,.57,.89],[.81,.67,.30],[.84,.82,.40],[.83,.90,.61],[.72,.87,.95]]


def dot(points,axis):
 # Explicit contraction avoids macOS Accelerate's spurious FP status flags.
 return np.einsum('...i,i->...',points,axis)

def rotate(points,R):
 return np.einsum('ni,ji->nj',points,R)

def apply_transform(points,R,t,scale=1):
 return scale * rotate(points,R) + t


def fit_rigid(source,target,weights=None):
 weights=np.ones(len(source)) if weights is None else np.asarray(weights)
 weights=weights/weights.sum();a=np.sum(source*weights[:,None],axis=0);b=np.sum(target*weights[:,None],axis=0)
 u,_,vt=np.linalg.svd((source-a).T@((target-b)*weights[:,None]))
 correction=np.eye(3);correction[-1,-1]=np.linalg.det(vt.T@u.T)
 R=vt.T@correction@u.T
 return R,b-R@a


def align_vectors(a,b):
 a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);cross=np.cross(a,b);c=np.clip(a@b,-1,1)
 if np.linalg.norm(cross)<1e-9:
  if c>0:return np.eye(3)
  axis=np.cross(a,np.eye(3)[np.argmin(abs(a))]);return Rotation.from_rotvec(np.pi*axis/np.linalg.norm(axis)).as_matrix()
 return Rotation.from_rotvec(cross/np.linalg.norm(cross)*np.arccos(c)).as_matrix()


def part_frame(landmarks,part):
 a=np.array(landmarks[part[2]],float);b=np.array(landmarks[part[3]],float)
 length=np.linalg.norm(b-a)
 if not np.isfinite(length) or length<1e-8:raise ValueError('Zero-length or nonfinite part: '+part[0])
 z=(b-a)/length
 y=np.array([0.,1.,0.]);y-=z*dot(y,z)
 if np.linalg.norm(y)<1e-8:
  y=np.eye(3)[np.argmin(abs(z))];y-=z*dot(y,z)
 y/=np.linalg.norm(y)
 x=np.cross(y,z);x/=np.linalg.norm(x)
 return a,b,x,y,z


def local_coordinates(points,landmarks,part):
 a,b,x,y,z=part_frame(landmarks,part)
 t=np.clip(dot(points-a,z)/np.linalg.norm(b-a),0,1)
 center=a+t[:,None]*(b-a)
 return np.c_[dot(points-center,x),dot(points-center,y),dot(points-center,z)],t


def cervical_labels(points,labels,specification,parts=BODY_PARTS):
 """Use the inspected skull/cuff seam planes, in a fixed reference frame."""
 ids={part[0]:i for i,part in enumerate(parts)};out=labels.copy();tr=specification['referenceTransform']
 p=apply_transform(points,np.asarray(tr['rotation']),np.asarray(tr['translation']),tr['scale'])
 skull=specification['skullPlane'];cuff=specification['cuffPlane']
 above=dot(p-np.asarray(skull['point']),np.asarray(skull['normal']))<0
 below=dot(p-np.asarray(cuff['point']),np.asarray(cuff['normal']))>0
 radial=np.sum(((p[:,:2]-specification['cuffRadialCenter'])/specification['cuffRadialRadii'])**2,axis=1)<1
 out[(out==ids['head'])&~above]=ids['neck']
 out[radial&above&np.isin(out,[ids['head'],ids['neck']])]=ids['head']
 out[radial&~above&~below]=ids['neck']
 out[(out==ids['neck'])&below]=ids['torso']
 return out


def segment_points(points,landmarks,parts=BODY_PARTS,overrides=(),cervical_spec=None):
 """3D elliptical capsule competition, including longitudinal joint boundaries."""
 best=np.full(len(points),np.inf);labels=np.zeros(len(points),np.uint8)
 for index,part in enumerate(parts):
  xyz,t=local_coordinates(points,landmarks,part)
  radius_x,radius_y=part[4:6]
  # Round caps are deliberately shorter than the width of broad torso/shorts.
  cap=min(radius_x,.055 if part[0] in ('torso','pelvis') else .035)
  longitudinal=xyz[:,2]/cap
  if part[0]=='head':
   longitudinal=np.where(xyz[:,2]>0,xyz[:,2]/.045,xyz[:,2]/.020)
  score=(xyz[:,0]/radius_x)**2+(xyz[:,1]/radius_y)**2+longitudinal**2
  if part[0]=='pelvis':
   a,b,x,y,z=part_frame(landmarks,part)
   distal=(np.array(landmarks['left_knee'])+landmarks['right_knee'])/2
   p=points-a
   crotch=(abs(dot(p,x))<.100)&(abs(dot(p,y))<.155)&(dot(p,z)>=0)&(dot(p,z)<=dot(distal-a,z))
   score[crotch]=np.minimum(score[crotch],.85)
  replace=score<best;best[replace]=score[replace];labels[replace]=index
 # Inspected rigid hardware can cross the generic capsule boundary. Keep its
 # complete local surface on one rigid part using a source-frame support volume.
 for rule in overrides:
  names=[part[0] for part in parts]
  if rule['part'] not in names:raise ValueError('Unknown segmentation override part: '+rule['part'])
  if rule.get('type')=='orientedEllipseSlab':
   center=np.asarray(rule['center'],float);axes=np.asarray(rule['axes'],float);radii=np.asarray(rule['radii'],float);half=float(rule['halfDepth'])
   if center.shape!=(3,) or axes.shape!=(3,3) or radii.shape!=(2,) or not np.isfinite(center).all() or not np.isfinite(axes).all() or not np.isfinite(radii).all() or not np.allclose(axes@axes.T,np.eye(3),atol=1e-5) or np.any(radii<=0) or not np.isfinite(half) or half<=0:raise ValueError('Invalid elliptical segmentation support volume')
   q=rotate(points-center,axes);selected=(np.sum((q[:,:2]/radii)**2,axis=1)<=1)&(abs(q[:,2])<=half)
   labels[selected]=names.index(rule['part']);best[selected]=np.minimum(best[selected],1.)
   continue
  center=np.asarray(rule['center'],float);axis=np.asarray(rule['axis'],float)
  radius=float(rule['radius']);half=float(rule['halfLength'])
  if center.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(center).all() or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8 or radius<=0 or half<=0:raise ValueError('Invalid segmentation support volume')
  axis/=np.linalg.norm(axis);relative=points-center;depth=dot(relative,axis)
  radial2=np.sum(relative*relative,axis=1)-depth**2
  selected=(abs(depth)<=half)&(radial2<=radius**2)
  labels[selected]=names.index(rule['part']);best[selected]=np.minimum(best[selected],1.)
 if cervical_spec:
  new_labels=cervical_labels(points,labels,cervical_spec,parts)
  changed=new_labels!=labels;best[changed]=np.minimum(best[changed],1.);labels=new_labels
 return labels,np.sqrt(best)


def apply_joint_boundaries(points,labels,landmarks,rules,parts=BODY_PARTS):
 """Split cleaned neighboring rigid pieces at measured native collar planes."""
 names=[part[0] for part in parts];out=labels.copy();updated=dict(landmarks)
 for rule in rules:
  proximal=names.index(rule['proximalPart']);distal=names.index(rule['distalPart'])
  pivot=np.asarray(rule['pivot'],float);axis=np.asarray(rule['axis'],float)
  half=float(rule['maximumAbsStation']);radius=float(rule['maximumRadius'])
  if proximal==distal or pivot.shape!=(3,) or axis.shape!=(3,) or not np.isfinite(pivot).all() or not np.isfinite(axis).all() or abs(np.linalg.norm(axis)-1)>1e-6 or not np.isfinite([half,radius]).all() or min(half,radius)<=0:
   raise ValueError('Invalid measured joint boundary')
  relative=points-pivot;station=dot(relative,axis)
  radial=np.linalg.norm(relative-station[:,None]*axis,axis=1)
  selected=np.isin(out,[proximal,distal])&(abs(station)<half)&(radial<radius)
  out[selected]=np.where(station[selected]<=0,proximal,distal)
  if rule['landmark'] not in updated:raise ValueError('Unknown boundary landmark')
  updated[rule['landmark']]=pivot.tolist()
 return out,updated


def assign_source_parts(labels,indices,selections,rules,parts=BODY_PARTS):
 """Keep inspected crossing hardware on its owning rigid piece by source row."""
 names=[part[0] for part in parts];out=labels.copy();claimed=np.zeros(len(labels),bool)
 for selected,rule in zip(selections,rules):
  part=names.index(rule['part']);allowed=[names.index(name) for name in rule['allowedParts']]
  hit=np.isin(indices,selected)
  if np.any(hit&claimed):raise ValueError('Source part assignments overlap')
  if np.any(hit&~np.isin(labels,allowed)):raise ValueError('Source part assignment exceeds its allowed body parts')
  out[hit]=part;claimed|=hit
 return out


def voxel_indices(points,size):
 key=np.floor(points/size).astype(np.int32)
 _,indices=np.unique(key,axis=0,return_index=True)
 return indices


def read_scan(path):
 columns,count,fields=read_ply(path)
 if any(field.startswith('f_rest_') for field in fields):
  raise ValueError('This pipeline supports DC-only SH; higher SH must be rotated separately, not silently discarded.')
 data=np.column_stack([columns[name] for name in FIELDS]).astype(np.float32)
 return data,{'file':str(path.relative_to(ROOT)),'sha256':sha256_file(path),'count':count}


def clean_scan(data,landmarks,settings,parts,exclusions=(),segmentation_overrides=(),part_filters=None,cervical_spec=None):
 finite=np.isfinite(data).all(axis=1)
 alpha=1/(1+np.exp(-np.clip(data[:,10],-40,40)))
 maxscale=np.exp(np.clip(data[:,7:10].max(axis=1),-40,40))
 base=finite&(alpha>=settings['minOpacity'])&(maxscale<=settings['maxScale'])&(maxscale>=settings['minScale'])
 excluded=np.zeros(len(data),bool)
 for box in exclusions:
  excluded |= np.all((data[:,:3]>=box['min'])&(data[:,:3]<=box['max']),axis=1)
 base &= ~excluded
 source_indices=np.flatnonzero(base).astype(np.uint32)
 data=data[base].copy();labels,distance=segment_points(data[:,:3],landmarks,parts,segmentation_overrides,cervical_spec)
 anatomy=distance<=settings['anatomyRadiusMultiplier'];data=data[anatomy];labels=labels[anatomy];source_indices=source_indices[anatomy]
 stats={'input':int(len(finite)),'invalidOrOpacityOrScale':int((~base).sum()),'outsideAnatomy':int((~anatomy).sum()),'explicitExclusions':int(excluded.sum())}
 quality=np.ones(len(data),bool)
 for i,part in enumerate(parts):
  rule=(part_filters or {}).get(part[0],{})
  if not rule:continue
  selected=labels==i;largest=np.exp(np.clip(data[selected,7:10].max(axis=1),-40,40));opacity=1/(1+np.exp(-np.clip(data[selected,10],-40,40)))
  quality[selected]=(largest<=rule.get('maxScale',settings['maxScale']))&(largest>=rule.get('minScale',settings['minScale']))&(opacity>=rule.get('minOpacity',settings['minOpacity']))
 stats['partQualityRejected']=int((~quality).sum());data=data[quality];labels=labels[quality];source_indices=source_indices[quality]
 # Uniform spatial support avoids conflating training duplicates with surface coverage.
 proxy=data[voxel_indices(data[:,:3],settings['densityVoxel']),:3]
 tree=cKDTree(proxy);spacing=tree.query(data[:,:3],k=6,workers=-1)[0][:,-1]
 depth=np.zeros(len(data))
 for i,part in enumerate(parts):
  idx=labels==i;xyz,_=local_coordinates(data[idx,:3],landmarks,part);depth[idx]=xyz[:,1]
 isolated=spacing>settings['isolatedSpacing']
 contact=(depth>settings['contactDepth'])&(spacing>settings['contactSpacing'])
 keep=~(isolated|contact)
 stats.update({'isolated':int(isolated.sum()),'sparseContactOnly':int((contact&~isolated).sum()),'retained':int(keep.sum())})
 return data[keep],labels[keep],stats,source_indices[keep]


def normal_proxy(data,landmarks,part,voxel):
 candidates=data[(data[:,10]>-2)&(np.exp(data[:,7:10].max(axis=1))<.025)]
 if len(candidates)<50:return None
 candidates=candidates[voxel_indices(candidates[:,:3],voxel)]
 xyz=candidates[:,:3].astype(float)
 if len(xyz)>24000:xyz=xyz[np.linspace(0,len(xyz)-1,24000,dtype=int)]
 k=min(20,len(xyz));dist,neighbor=cKDTree(xyz).query(xyz,k=k,workers=-1)
 near=xyz[neighbor];center=near.mean(axis=1);centered=near-center[:,None,:]
 covariance=np.einsum('nki,nkj->nij',centered,centered)/k
 eigenvalues,eigenvectors=np.linalg.eigh(covariance);normals=eigenvectors[:,:,0]
 a,b,_,_,z=part_frame(landmarks,part);t=np.clip(dot(xyz-a,z)/np.linalg.norm(b-a),0,1)
 radial=xyz-(a+t[:,None]*(b-a));flip=np.sum(normals*radial,axis=1)<0;normals[flip]*=-1
 planar=eigenvalues[:,0]/np.maximum(eigenvalues.sum(axis=1),1e-12)<.12
 return xyz[planar],normals[planar]


def match_overlap(source,normal_source,target,normal_target,maxdist):
 if len(source)<20 or len(target)<20:return np.array([],int),np.array([],int),np.array([])
 tree=cKDTree(target);dist,idx=tree.query(source,workers=-1)
 _,reverse=cKDTree(source).query(target,workers=-1)
 valid=(dist<maxdist)&(np.sum(normal_source*normal_target[idx],axis=1)>.80)&(reverse[idx]==np.arange(len(source)))
 si=np.flatnonzero(valid);return si,idx[si],dist[si]


def refine_overlap(front,back,lf,lb,part,R,t,scale,settings):
 """Small rigid corrections from signed-normal, mutual, frozen same-side overlap."""
 fp=normal_proxy(front,lf,part,settings['voxel']);bp=normal_proxy(back,lb,part,settings['voxel']/scale)
 report={'method':'landmarks only','accepted':False,'reason':'insufficient same-surface overlap','initialPairs':0}
 if fp is None or bp is None:return R,t,report
 f,fn=fp;b,bn=bp;b=apply_transform(b,R,t,scale);bn=rotate(bn,R)
 # Require lateral side normals; anterior and posterior shells must never be paired.
 _,_,_,depthaxis,_=part_frame(lf,part)
 fs=abs(dot(fn,depthaxis))<settings['maxAnteriorNormal'];bs=abs(dot(bn,depthaxis))<settings['maxAnteriorNormal']
 f,fn=f[fs],fn[fs];b,bn=b[bs],bn[bs]
 si,ti,dist=match_overlap(b,bn,f,fn,settings['maxDistance'])
 report['initialPairs']=len(si)
 if len(si)<settings['minPairs']:return R,t,report
 # Freeze eligible overlap patches from the initial placement, expand only one voxel.
 smask=cKDTree(b[si]).query(b)[0]<settings['voxel']*1.5
 tmask=cKDTree(f[ti]).query(f)[0]<settings['voxel']*1.5
 b,bn=b[smask],bn[smask];f,fn=f[tmask],fn[tmask]
 si,ti,dist=match_overlap(b,bn,f,fn,settings['maxDistance'])
 before=float(np.median(dist));pivot=(np.array(lf[part[2]])+lf[part[3]])/2
 dR=np.eye(3);dt=np.zeros(3);length=np.linalg.norm(np.array(lf[part[3]])-lf[part[2]])
 lever=max(.07,length/2);spectrum=None
 for iteration in range(settings['iterations']):
  moved=apply_transform(b-pivot,dR,dt)+pivot;mn=rotate(bn,dR)
  si,ti,dist=match_overlap(moved,mn,f,fn,settings['maxDistance'])
  if len(si)<settings['minPairs']:break
  trim=dist<=np.quantile(dist,.8);si=si[trim];ti=ti[trim]
  p=moved[si];n=fn[ti];residual=np.sum((p-f[ti])*n,axis=1)
  robust=1/np.sqrt(1+(residual/.003)**2)
  J=np.c_[np.cross(p-pivot,n)/lever,n]
  A=J*robust[:,None];rhs=-residual*robust
  spectrum=np.linalg.svd(A,compute_uv=False)
  # Priors stabilize sliding on broad planar contact strips and unobservable twist.
  regularization=np.sqrt(len(si))*.2
  prior=np.r_[Rotation.from_matrix(dR).as_rotvec()*lever,dt]
  step=np.linalg.lstsq(np.vstack([A,np.eye(6)*regularization]),np.r_[rhs,-prior*regularization],rcond=None)[0]
  omega=step[:3]/lever;shift=step[3:]
  total_r=Rotation.from_rotvec(omega).as_matrix()@dR;total_t=Rotation.from_rotvec(omega).apply(dt)+shift
  if np.linalg.norm(Rotation.from_matrix(total_r).as_rotvec())>np.deg2rad(settings['maxRotationDegrees']) or np.linalg.norm(total_t)>settings['maxTranslation']:
   break
  dR,dt=total_r,total_t
  if np.linalg.norm(step)<1e-6:break
 moved=apply_transform(b-pivot,dR,dt)+pivot;mn=rotate(bn,dR)
 si,ti,dist=match_overlap(moved,mn,f,fn,settings['maxDistance'])
 after=float(np.median(dist)) if len(dist) else float('inf')
 acceptable=len(dist)>=settings['minPairs'] and after<before*.995
 report.update({'method':'bounded overlap point-to-plane ICP + landmark prior','accepted':bool(acceptable),
  'initialMedianDistance':before,'finalMedianDistance':after if np.isfinite(after) else None,
  'finalP90Distance':float(np.quantile(dist,.9)) if len(dist) else None,'finalPairs':len(si),
  'rotationCorrectionDegrees':float(np.rad2deg(np.linalg.norm(Rotation.from_matrix(dR).as_rotvec()))),
  'translationCorrection':float(np.linalg.norm(dt)),
  'normalizedSingularValues':(spectrum/spectrum[0]).tolist() if spectrum is not None else None,
  'reason':'overlap improved within bounded prior' if acceptable else 'refinement did not reliably improve overlap; retained landmark alignment'})
 if not acceptable:return R,t,report
 return dR@R,dR@(t-pivot)+dt+pivot,report


def transform_gaussians(data,R,t,scale=1):
 result=data.copy();result[:,:3]=apply_transform(data[:,:3],R,t,scale)
 q=data[:,3:7].astype(float);norm=np.linalg.norm(q,axis=1);bad=norm<1e-12;q[bad]=[1,0,0,0];norm[bad]=1;q/=norm[:,None]
 qr=Rotation.from_matrix(R).as_quat();rw=qr[3];rv=qr[:3]
 out=np.empty_like(q);out[:,0]=rw*q[:,0]-dot(q[:,1:],rv)
 out[:,1:]=rw*q[:,1:]+q[:,0,None]*rv+np.cross(rv,q[:,1:])
 result[:,3:7]=out;result[:,7:10]+=np.log(scale)
 return result


def coverage_weights(front,back,landmarks,part,settings):
 """Common anatomical cut, optical-density feather, preserve unsupported unique coverage."""
 settings={**settings,**settings.get('partOverrides',{}).get(part[0],{})}
 a,b,x,y,z=part_frame(landmarks,part)
 bias=settings.get('depthBias',0.)
 fdepth=dot(front[:,:3]-a,y)-bias;bdepth=dot(back[:,:3]-a,y)-bias
 w=settings['feather'];fw=1/(1+np.exp(np.clip(fdepth/w,-40,40)));bw=1/(1+np.exp(np.clip(-bdepth/w,-40,40)))
 # Donor coverage uses cross-section columns, not 3D NN (opposite shells must stay apart).
 fcolumns=np.c_[dot(front[:,:3]-a,x),dot(front[:,:3]-a,z)];bcolumns=np.c_[dot(back[:,:3]-a,x),dot(back[:,:3]-a,z)]
 if len(front) and len(back):
  donor_b=bcolumns[bw>.6];donor_f=fcolumns[fw>.6]
  if len(donor_b) and settings.get('preserveUniqueFront',True):
   dist=cKDTree(donor_b).query(fcolumns,workers=-1)[0];fw[dist>settings['donorColumnRadius']]=1
  elif not len(donor_b) and settings.get('preserveUniqueFront',True):fw[:]=1
  if len(donor_f) and settings.get('preserveUniqueBack',True):
   dist=cKDTree(donor_f).query(bcolumns,workers=-1)[0];bw[dist>settings['donorColumnRadius']]=1
  elif not len(donor_f) and settings.get('preserveUniqueBack',True):bw[:]=1
 else:
  if settings.get('preserveUniqueFront',True):fw[:]=1
  if settings.get('preserveUniqueBack',True):bw[:]=1
 return fw,bw


def attenuate(data,weight,minweight):
 keep=weight>=minweight;out=data[keep].copy();a=1/(1+np.exp(-np.clip(out[:,10].astype(float),-40,40)))
 a=-np.expm1(weight[keep]*np.log1p(-np.minimum(a,1-1e-9)))
 out[:,10]=np.log(np.clip(a,1e-8,1-1e-8)/(1-np.clip(a,1e-8,1-1e-8)))
 return out,keep


def write_ply(path,data):
 path.parent.mkdir(parents=True,exist_ok=True)
 header='ply\nformat binary_little_endian 1.0\ncomment local partwise Gaussian fusion\nelement vertex '+str(len(data))+'\n'+''.join('property float '+field+'\n' for field in FIELDS)+'end_header\n'
 temporary=path.with_suffix('.ply.tmp')
 with temporary.open('wb') as f:f.write(header.encode());np.asarray(data,dtype='<f4').tofile(f)
 temporary.replace(path)


def initialize_transforms(lf,lb,parts,settings):
 pairs=[('left_shoulder','right_shoulder'),('neck','waist'),('left_elbow','left_wrist'),('right_elbow','right_wrist'),('left_knee','left_ankle'),('right_knee','right_ankle')]
 ratios=[np.linalg.norm(np.array(lf[a])-lf[b])/np.linalg.norm(np.array(lb[a])-lb[b]) for a,b in pairs]
 scale=settings.get('scale') or float(np.median(ratios));Rbase=np.diag([-1.,-1.,1.]);transforms=[]
 for part in parts:
  name,_,a,b,_,_=part
  fa,fb=np.array(lf[a]),np.array(lf[b]);ba,bb=np.array(lb[a]),np.array(lb[b])
  if name in ('torso','pelvis'):
   keys=['neck','waist','left_shoulder','right_shoulder'] if name=='torso' else ['waist','pelvis','left_hip','right_hip']
   weights=[.4,1,1,1] if name=='torso' else [1,.4,.6,.6]
   R,t=fit_rigid(np.array([lb[k] for k in keys])*scale,np.array([lf[k] for k in keys]),weights)
  else:
   R=align_vectors(Rbase@(bb-ba),fb-fa)@Rbase
   t=(fa+fb)/2-scale*R@((ba+bb)/2)
  transforms.append((R,t))
 return scale,transforms,{'scale':scale,'boneScaleRatios':dict(zip([a+' to '+b for a,b in pairs],map(float,ratios))),'scalePolicy':'one global isotropic scale; all part refinements rigid'}


def axial_child_transform(parent_R,parent_t,constraint,extra_twist=0.):
 """One rotation about an inspected collar; no independent bend or translation."""
 axis=np.asarray(constraint['axisFrontRaw'],float);pivot=np.asarray(constraint['pivotFrontRaw'],float)
 angle=float(constraint['twistDegrees'])+float(extra_twist)
 if axis.shape!=(3,) or pivot.shape!=(3,) or not np.isfinite(axis).all() or not np.isfinite(pivot).all() or not np.isfinite(angle) or abs(np.linalg.norm(axis)-1)>1e-6:
  raise ValueError('Axial collar needs a finite pivot and unit axis')
 Q=Rotation.from_rotvec(axis*np.deg2rad(angle)).as_matrix()
 return Q@parent_R,Q@(parent_t-pivot)+pivot


def validate_rigid_parts(config,parts):
 """A rigid child has no independent pose controls or local deformation."""
 rigid=config.get('rigidParts',{})
 if not isinstance(rigid,dict):raise ValueError('rigidParts must map child part names to parent part names')
 known={part[0] for part in parts}
 for child,parent in rigid.items():
  if child not in known or not isinstance(parent,str) or parent not in known or child==parent:
   raise ValueError('Unknown or self-referencing rigid part: '+str(child))
  manual=config.get('adjustments',{}).get(child,{})
  for key,default,shape in [('rotationDegrees',[0,0,0],(3,)),('translation',[0,0,0],(3,)),('twistDegrees',0,())]:
   value=np.asarray(manual.get(key,default),float)
   if value.shape!=shape or not np.isfinite(value).all() or np.any(value!=0):
    raise ValueError('Rigid part '+child+' disallows independent adjustments.'+child+'.'+key+'; adjust '+parent)
  if child=='right_hand' and config.get('rightWristRegistration'):
   raise ValueError('Rigid right_hand cannot use rightWristRegistration')
  if child=='left_hand' and config.get('leftHandDigitAlignment',{}).get('amount',0):
   raise ValueError('Rigid left_hand cannot use leftHandDigitAlignment')
  if child=='head' and config.get('hairFullness',{}).get('amount',0):
   raise ValueError('Rigid head cannot use hairFullness deformation')
 # Reject cycles before reading large scans or starting registration.
 for child in rigid:
  visited=set();current=child
  while current in rigid:
   if current in visited:raise ValueError('Cyclic rigidParts constraint: '+child)
   visited.add(current);current=rigid[current]
 return dict(rigid)


def alignment_order(parts,rigid_parts,feature_fits,skip_icp=False):
 """Prepare effective parents first while retaining each part's original label."""
 indices={part[0]:i for i,part in enumerate(parts)};ordered=[];visiting=set();done=set()
 def visit(name):
  if name in visiting:raise ValueError('Cyclic part transform dependencies: '+name)
  if name in done:return
  visiting.add(name);parent=rigid_parts.get(name)
  if parent is None and not skip_icp:
   entry=feature_fits.get(name,(None,None,{}))[2]
   parent=entry.get('mechanicalConstraint',{}).get('parentPart') or entry.get('carriedByPart')
  if parent is not None:
   if parent not in indices:raise ValueError('Unknown transform parent: '+str(parent))
   visit(parent)
  visiting.remove(name);done.add(name);ordered.append(indices[name])
 for part in parts:visit(part[0])
 return ordered


def validate_feature_transforms(document,source_hashes,parts,scale):
 """Reject stale or non-rigid feature fits rather than silently corrupting splats."""
 if document.get('version')!=1:raise ValueError('Feature transforms require version 1')
 if document.get('sourceHashes')!=source_hashes:raise ValueError('Feature transforms were fitted to different source scans')
 known={part[0] for part in parts};result={}
 for name,entry in document.get('parts',{}).items():
  if name not in known:raise ValueError('Unknown feature-fit part: '+name)
  transform=entry['sourceToFrontRaw'];R=np.asarray(transform['rotation'],float);t=np.asarray(transform['translation'],float)
  if R.shape!=(3,3) or t.shape!=(3,) or not np.isfinite(R).all() or not np.isfinite(t).all():raise ValueError('Invalid feature transform: '+name)
  if not np.allclose(R.T@R,np.eye(3),atol=1e-6) or abs(np.linalg.det(R)-1)>1e-6:raise ValueError('Feature fit must be a proper rigid rotation: '+name)
  if not np.isclose(transform['scale'],scale,rtol=1e-7):raise ValueError('Feature fit must use the shared scan scale: '+name)
  result[name]=(R,t,entry)
 for name,(R,t,entry) in result.items():
  if entry.get('mechanicalConstraint'):
   constraint=entry['mechanicalConstraint'];parent=constraint['parentPart']
   if parent not in result or parent==name:raise ValueError('Unknown or self-referencing axial parent: '+name)
   expected_R,expected_t=axial_child_transform(result[parent][0],result[parent][1],constraint)
   if not np.allclose(R,expected_R,atol=1e-6,rtol=0) or not np.allclose(t,expected_t,atol=1e-6,rtol=0):
    raise ValueError('Feature fit violates axial collar constraint: '+name)
  if entry.get('carriedByPart') and (entry['carriedByPart'] not in result or entry['carriedByPart']==name):
   raise ValueError('Unknown or self-referencing carried parent: '+name)
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,default=ROOT/'tools/fusion/fusion-config.json');parser.add_argument('--skip-icp',action='store_true');args=parser.parse_args()
 config=json.loads(args.config.read_text());lf=json.loads((ROOT/config['frontLandmarks']).read_text())['landmarks'];lb=json.loads((ROOT/config['backLandmarks']).read_text())['landmarks']
 parts=[tuple(list(p[:4])+config.get('partRadii',{}).get(p[0],list(p[4:]))) for p in BODY_PARTS]
 rigid_parts=validate_rigid_parts(config,parts)
 print('Reading and filtering source scans',flush=True)
 front,meta_f=read_scan(ROOT/config['front']);back,meta_b=read_scan(ROOT/config['back'])
 wrist_config=config.get('rightWristRegistration')
 wrist_pad_rows=np.empty(0,np.uint32)
 if wrist_config:
  if wrist_config['sourceHashes']!={meta_f['file']:meta_f['sha256'],meta_b['file']:meta_b['sha256']}:
   raise ValueError('Wrist registration belongs to different source scans')
  wrist_masks,_=load_cleanup_masks([wrist_config['protectedPadSelection']],{'front':meta_f,'back':meta_b},ROOT)
  wrist_pad_rows=wrist_masks['back']
 cleanup_masks,cleanup_evidence=load_cleanup_masks(config.get('cleanupMasks',[]),{'front':meta_f,'back':meta_b},ROOT)
 color_overrides,color_evidence=load_color_overrides(config.get('colorOverrides',[]),{'front':meta_f,'back':meta_b},ROOT)
 restoration_rules=config.get('surfaceRestorations',[]);restoration_masks=[];restoration_evidence=[]
 for rule in restoration_rules:
  masks,evidence=load_cleanup_masks([rule['selection']],{'front':meta_f,'back':meta_b},ROOT)
  restoration_masks.append(masks);restoration_evidence.extend(evidence)
 assignment_rules=config.get('sourcePartAssignments',[]);assignment_masks=[];assignment_evidence=[]
 for rule in assignment_rules:
  masks,evidence=load_cleanup_masks([rule['selection']],{'front':meta_f,'back':meta_b},ROOT)
  assignment_masks.append(masks);assignment_evidence.extend(evidence)
 cervical={}
 if config.get('cervicalSegmentation'):
  document=json.loads((ROOT/config['cervicalSegmentation']).read_text())
  if document['sourceHashes']!={'front':meta_f['sha256'],'back':meta_b['sha256']}:raise ValueError('Cervical features were measured from different sources')
  cervical=document['cervicalSegmentation']
 front,fl,fs,fi=clean_scan(front,lf,config['filter'],parts,config.get('exclusions',{}).get('front',[]),config.get('segmentationOverrides',{}).get('front',[]),config['filter'].get('partOverrides',{}).get('front',{}),cervical.get('front'));back,bl,bs,bi=clean_scan(back,lb,config['filter'],parts,config.get('exclusions',{}).get('back',[]),config.get('segmentationOverrides',{}).get('back',[]),config['filter'].get('partOverrides',{}).get('back',{}),cervical.get('back'))
 fl,lf=apply_joint_boundaries(front[:,:3],fl,lf,config.get('jointBoundaries',{}).get('front',[]),parts)
 bl,lb=apply_joint_boundaries(back[:,:3],bl,lb,config.get('jointBoundaries',{}).get('back',[]),parts)
 fl=assign_source_parts(fl,fi,[m['front'] for m in assignment_masks],assignment_rules,parts)
 bl=assign_source_parts(bl,bi,[m['back'] for m in assignment_masks],assignment_rules,parts)
 validate_restorations(fl,fi,[m['front'] for m in restoration_masks],restoration_rules,parts,config['fusion']['minWeight'])
 validate_restorations(bl,bi,[m['back'] for m in restoration_masks],restoration_rules,parts,config['fusion']['minWeight'])
 print('Retained before fusion:',len(front),len(back),flush=True)
 scale,initial,scale_report=initialize_transforms(lf,lb,parts,config['alignment']);print('Global scale:',scale_report,flush=True)
 feature_fits={}
 if config.get('featureTransforms'):
  document=json.loads((ROOT/config['featureTransforms']).read_text())
  feature_fits=validate_feature_transforms(document,{'front':meta_f['sha256'],'back':meta_b['sha256']},parts,scale)
 fused_front=[];fused_back=[];front_labels=[];back_labels=[];front_indices=[];back_indices=[];raw_aligned=[];part_report=[None]*len(parts);prepared=np.diag([1.,-1.,-1.]);effective_transforms={}
 for i in alignment_order(parts,rigid_parts,feature_fits,args.skip_icp):
  part=parts[i]
  print('Aligning',part[0],flush=True);f=front[fl==i];b=back[bl==i];R,t=initial[i]
  rigid_parent=rigid_parts.get(part[0])
  if rigid_parent is not None:
   R,t=effective_transforms[rigid_parent]
   report={'method':'rigid shared parent transform','accepted':True,'reason':'No independent joint or pose for this part','rigidWithPart':rigid_parent}
  elif part[0] in feature_fits and not args.skip_icp:
   R,t,entry=feature_fits[part[0]]
   report={'method':entry.get('method','reviewed feature-guided rigid alignment'),'accepted':True,
    'reason':entry.get('reason','accepted after multi-angle Gaussian render review'),
    'featureEvidence':{key:value for key,value in entry.items() if key!='sourceToFrontRaw'}}
  elif not args.skip_icp:R,t,report=refine_overlap(f,b,lf,lb,part,R,t,scale,config['alignment'])
  else:report={'method':'landmarks only','accepted':False,'reason':'ICP disabled'}
  manual=config.get('adjustments',{}).get(part[0],{});pivot=(np.array(lf[part[2]])+lf[part[3]])/2
  active_entry=feature_fits.get(part[0],(None,None,{}))[2] if not args.skip_icp else {}
  if rigid_parent is not None:
   pass  # Exactly the effective parent pose; bypass historical child constraints.
  elif active_entry.get('mechanicalConstraint'):
   constraint=active_entry['mechanicalConstraint'];parent=constraint['parentPart']
   if parent not in effective_transforms:raise ValueError('Axial parent must be prepared before child: '+part[0])
   if np.any(np.asarray(manual.get('rotationDegrees',[0,0,0]))!=0) or np.any(np.asarray(manual.get('translation',[0,0,0]))!=0):
    raise ValueError('Axial collar disallows free bend/translation; tune adjustments.'+part[0]+'.twistDegrees')
   R,t=axial_child_transform(*effective_transforms[parent],constraint,manual.get('twistDegrees',0.))
   report['mechanicalConstraint']={**constraint,'twistDegrees':constraint['twistDegrees']+manual.get('twistDegrees',0.)}
  elif active_entry.get('carriedByPart'):
   parent=active_entry['carriedByPart']
   if parent not in effective_transforms:raise ValueError('Carried parent must be prepared before child: '+part[0])
   reference_R,reference_t,_=feature_fits[parent];parent_R,parent_t=effective_transforms[parent]
   delta_R=parent_R@reference_R.T;R,t=delta_R@R,delta_R@(t-reference_t)+parent_t
   report['carriedByPart']=parent
  if rigid_parent is None:
   correction=Rotation.from_euler('xyz',manual.get('rotationDegrees',[0,0,0]),degrees=True).as_matrix();shift=np.array(manual.get('translation',[0,0,0]));R,t=correction@R,correction@(t-pivot)+pivot+shift
  effective_transforms[part[0]]=(R,t)
  if part[0]=='right_hand' and wrist_config:
   parent_R,parent_t=effective_transforms['right_forearm']
   if not np.allclose(R,parent_R,atol=1e-8,rtol=0) or not np.allclose(t,parent_t,atol=1e-8,rtol=0):
    raise ValueError('Continuous right wrist pad requires the hand base to share the forearm transform')
   from right_wrist_transition import preserve_pad_align_fingers
   protected=np.isin(bi[bl==i],wrist_pad_rows)
   corrected,wrist_report=preserve_pad_align_fingers(b,wrist_config)
   if not np.array_equal(corrected[protected],b[protected]):
    raise ValueError('Finger transition moved the protected rigid wrist pad')
   wrist_report['protectedHandPadRows']=int(protected.sum())
   report['wristRegistration']=wrist_report
   b=corrected
  aligned=transform_gaussians(b,R,t,scale)
  if part[0]=='left_hand' and config.get('leftHandDigitAlignment',{}).get('amount',0):
   from hand_digits import align_left_digits
   aligned,digit_report=align_left_digits(aligned,config['leftHandDigitAlignment'])
   report['digitAlignment']=digit_report
  if part[0]=='head' and config.get('hairFullness',{}).get('amount',0):
   from hair_shape import deform_head_hair
   f,hair_front=deform_head_hair(f,config['hairFullness']);aligned,hair_back=deform_head_hair(aligned,config['hairFullness'])
   report['hairAppearanceCorrection']={'front':hair_front,'back':hair_back}
  fw,bw=coverage_weights(f,aligned,lf,part,config['fusion'])
  fw,frestored=restore_surface_weights(fw,fi[fl==i],[m['front'] for m in restoration_masks],restoration_rules)
  bw,brestored=restore_surface_weights(bw,bi[bl==i],[m['back'] for m in restoration_masks],restoration_rules)
  ff,fkeep=attenuate(f,fw,config['fusion']['minWeight']);bb,bkeep=attenuate(aligned,bw,config['fusion']['minWeight'])
  fids=fi[fl==i][fkeep];bids=bi[bl==i][bkeep]
  fclean=keep_source_rows(fids,cleanup_masks['front'])|frestored[fkeep];bclean=keep_source_rows(bids,cleanup_masks['back'])|brestored[bkeep]
  report['cleanupRemoved']={'front':int((~fclean).sum()),'back':int((~bclean).sum())}
  ff=ff[fclean];bb=bb[bclean];fids=fids[fclean];bids=bids[bclean]
  ff[:,11:14],nfcolor=apply_color_values(ff[:,11:14],fids,color_overrides['front'],config.get('colorRepairStrength',1.))
  bb[:,11:14],nbcolor=apply_color_values(bb[:,11:14],bids,color_overrides['back'],config.get('colorRepairStrength',1.))
  report['colorRepairAdjusted']={'front':nfcolor,'back':nbcolor}
  # Preserve audit samples before the complementary-source cut.
  raw_aligned.append((part[0],f[::max(1,len(f)//7000)],aligned[::max(1,len(aligned)//7000)]))
  ff=transform_gaussians(ff,prepared,np.zeros(3));bb=transform_gaussians(bb,prepared,np.zeros(3))
  fused_front.append(ff);fused_back.append(bb);front_labels.append(np.full(len(ff),i,np.uint8));back_labels.append(np.full(len(bb),i,np.uint8));front_indices.append(fids);back_indices.append(bids)
  sourceanchors=np.array([lb[part[2]],lb[part[3]]]);targetanchors=np.array([lf[part[2]],lf[part[3]]]);anchorerror=np.linalg.norm(apply_transform(sourceanchors,R,t,scale)-targetanchors,axis=1)
  report.update({'id':part[0],'label':part[1],'frontBefore':len(f),'backBefore':len(b),'frontRetained':len(ff),'backRetained':len(bb),'sourceToFrontRaw':{'scale':scale,'rotation':R.tolist(),'translation':t.tolist()},'endpointResiduals':anchorerror.tolist()})
  part_report[i]=report;print(' ',report['method'],report.get('initialPairs',0),'pairs; retained',len(ff),len(bb),flush=True)
 final_front=np.concatenate(fused_front);final_back=np.concatenate(fused_back);fl=np.concatenate(front_labels);bl=np.concatenate(back_labels)
 assignment_counts=[]
 for rule,masks in zip(assignment_rules,assignment_masks):
  for source,before,after in [('front',fi,np.concatenate(front_indices)),('back',bi,np.concatenate(back_indices))]:
   expected=np.sort(before[np.isin(before,masks[source])]);retained=np.sort(after[np.isin(after,masks[source])])
   if rule.get('requireRetained') and not np.array_equal(expected,retained):
    raise ValueError('Required crossing hardware disappeared during fusion: '+source+' '+rule['part'])
   assignment_counts.append({'selection':rule['selection'],'part':rule['part'],'source':source,'selectedAfterCleaning':len(expected),'exported':len(retained),'allSelectedRetained':bool(np.array_equal(expected,retained))})
 fused=np.concatenate([final_front,final_back]);out=ROOT/config['output'];out.mkdir(parents=True,exist_ok=True)
 write_ply(out/'mannequin_fused.ply',fused);np.save(out/'source-vertex-indices.npy',np.r_[np.concatenate(front_indices),np.concatenate(back_indices)]);np.save(out/'part-labels.npy',np.r_[fl,bl]);np.save(out/'capture-labels.npy',np.r_[np.zeros(len(fl),np.uint8),np.ones(len(bl),np.uint8)])
 report={'sources':[meta_f,meta_b],'filter':{'front':fs,'back':bs},'registration':scale_report,'parts':part_report,'fusedCount':len(fused),'frontCount':len(final_front),'backCount':len(final_back),'frame':'[x,-y,-z] of original front scan; no metric calibration','parameters':config,'cleanupEvidence':cleanup_evidence,'colorRepairEvidence':color_evidence}
 report['sourcePartAssignmentEvidence']=assignment_evidence
 report['sourcePartAssignmentCounts']=assignment_counts
 restoration_counts=[]
 for rule,masks in zip(restoration_rules,restoration_masks):
  for source,before,after in [('front',fi,np.concatenate(front_indices)),('back',bi,np.concatenate(back_indices))]:
   expected=np.sort(before[np.isin(before,masks[source])]);retained=np.sort(after[np.isin(after,masks[source])])
   if not np.array_equal(expected,retained):raise ValueError('Reviewed observed-surface rows disappeared: '+source+' '+rule['part'])
   restoration_counts.append({'selection':rule['selection'],'source':source,'part':rule['part'],'selectedAfterCleaning':len(expected),'exported':len(retained),'allSelectedRetained':True})
 report['surfaceRestorationEvidence']=restoration_evidence
 report['surfaceRestorationCounts']=restoration_counts
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
 review=ROOT/config['review'];review.mkdir(parents=True,exist_ok=True);captures=[]
 for name,data,labels in [('front',final_front,fl),('back',final_back,bl)]:
  budget=config['previewMaxPerSource'];idx=np.linspace(0,len(data)-1,min(budget,len(data)),dtype=int)
  data[idx].astype('<f4').tofile(review/(name+'.bin'));labels[idx].tofile(review/(name+'-labels.bin'));captures.append({'id':name,'label':name.title()+' contribution','url':name+'.bin','count':len(idx),'labelsUrl':name+'-labels.bin'})
 manifest={'version':1,'parts':[{'id':p[0],'label':p[1],'color':PALETTE[i]} for i,p in enumerate(parts)],'captures':captures,'bounds':{'min':fused[:,:3].min(axis=0).tolist(),'max':fused[:,:3].max(axis=0).tolist()},'report':report}
 (review/'fusion.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # Small source-correspondence set makes multi-angle residual QA reproducible.
 audit={}
 for name,f,b in raw_aligned:audit[name+'_front']=f;audit[name+'_back']=b
 np.savez_compressed(out/'alignment-audit.npz',**audit)
 print('Fused',len(fused),'Gaussians:',out/'mannequin_fused.ply',flush=True)
 return report

if __name__=='__main__':main()
