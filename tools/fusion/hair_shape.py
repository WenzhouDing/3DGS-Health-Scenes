#!/usr/bin/env python3
"""Editable local posterior-hair fullness field with covariance-correct splats.

Input and output use aligned front raw coordinates. Apply to the head part
only, equally to both captures, before source weighting. This is an aesthetic
correction to captured hair shape, not a rigid registration or recovered data.
No files are read or written by this module.
"""
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULT_FULLNESS={
 'amount':.014,'centerX':.006,'centerZ':-.225,
 'radiusX':.15,'radiusZ':.215,
 'depthFade':[-.012,.035],
 'napeFadeZ':[-.095,-.130],
 'crownLiftRatio':.5,'crownFadeSpan':.11,
 'jacobianStep':1e-5,
}

# Reviewed front-raw feature anchors. The default field is exactly zero here.
PROTECTED_FEATURES={
 'rightEarCanal':[-.08083317,-.0351,-.12765],
 'leftEarCanal':[.105,-.033,-.133],
 'nose':[0.,-.170,-.195],
 'chin':[0.,-.150,-.090],
 'anteriorNeck':[0.,-.140,-.010],
}

def _settings(settings):
 c={**DEFAULT_FULLNESS,**settings}
 for name in ['amount','centerX','centerZ','radiusX','radiusZ','crownLiftRatio','crownFadeSpan','jacobianStep']:
  c[name]=float(c[name])
  if not np.isfinite(c[name]):raise ValueError('Nonfinite hair parameter: '+name)
 if c['amount']<0 or c['amount']>.04:raise ValueError('Hair fullness amount must be between 0 and .04 scan units')
 if min(c['radiusX'],c['radiusZ'],c['crownFadeSpan'],c['jacobianStep'])<=0:raise ValueError('Hair radii and steps must be positive')
 for name in ['depthFade','napeFadeZ']:
  values=np.asarray(c[name],float)
  if values.shape!=(2,) or not np.isfinite(values).all() or values[0]==values[1]:raise ValueError('Invalid hair fade interval: '+name)
  c[name]=values.tolist()
 if c['depthFade'][1]<c['depthFade'][0] or c['napeFadeZ'][1]>c['napeFadeZ'][0]:raise ValueError('Hair fade intervals have reversed orientation')
 return c

def smoothstep(value):
 u=np.clip(value,0,1);return u*u*(3-2*u)

def hair_displacement(points,settings):
 """Smooth compact scalp-only field in FRONT RAW coordinates; no ear motion."""
 c=_settings(settings);p=np.asarray(points,float);amount=c['amount']
 x=(p[:,0]-c['centerX'])/c['radiusX'];z=(p[:,2]-c['centerZ'])/c['radiusZ']
 wx=np.maximum(0,1-x*x)**3;wz=np.maximum(0,1-z*z)**3
 yd=c['depthFade'];depth=smoothstep((p[:,1]-yd[0])/(yd[1]-yd[0]))
 zd=c['napeFadeZ'];nape=smoothstep((p[:,2]-zd[0])/(zd[1]-zd[0]))
 w=wx*wz*depth*nape;lift=smoothstep((c['centerZ']-p[:,2])/c['crownFadeSpan'])
 delta=np.zeros_like(p);delta[:,1]=amount*w;delta[:,2]=-amount*c['crownLiftRatio']*w*lift
 return delta

def hair_jacobian(points,settings):
 c=_settings(settings);p=np.asarray(points,float);eps=c['jacobianStep'];J=np.broadcast_to(np.eye(3),(len(p),3,3)).copy()
 for axis in range(3):
  step=np.zeros(3);step[axis]=eps
  J[:,:,axis]+=(hair_displacement(p+step,c)-hair_displacement(p-step,c))/(2*eps)
 return J

def deform_head_hair(data,settings):
 """Return (Gaussians, audit) after editable, non-rigid hair appearance correction.

 Apply to head part in aligned FRONT RAW frame, before source weighting, for
 both front and back sources. Rigid ear/skull alignment is kept; field support
 is only posterior hair above the nape. Covariance becomes J Sigma J^T.
 This is an explicit aesthetic correction, not a recovered scan observation.
 """
 c=_settings(settings);amount=c['amount']
 delta=hair_displacement(data[:,:3],c);changed=np.linalg.norm(delta,axis=1)>1e-12
 out=data.copy();d=data[changed]
 protected=float(np.linalg.norm(hair_displacement(np.asarray(list(PROTECTED_FEATURES.values())),c),axis=1).max(initial=0))
 report={'method':'smooth posterior hair dome; covariance J Sigma J^T','appearanceCorrection':True,'affected':int(changed.sum()),'amount':amount,'parameters':c,'maxMeanDisplacement':float(np.linalg.norm(delta,axis=1).max(initial=0)),'rigidFeaturesUnchanged':protected<1e-12,'protectedFeatureMaxDisplacement':protected}
 if len(d)==0:return out,report
 J=hair_jacobian(d[:,:3],c);q=d[:,3:7].astype(float);norm=np.linalg.norm(q,axis=1);bad=norm<1e-12;q[bad]=[1,0,0,0];norm[bad]=1;q/=norm[:,None]
 R=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix();axes=np.einsum('nij,njk->nik',J,R)*np.exp(d[:,None,7:10])
 # SVD of the covariance factor avoids squaring the extreme condition number
 # of thin trained surfels. U diag(s^2) U^T equals J Sigma J^T.
 vectors,sigma,_=np.linalg.svd(axes,full_matrices=False);floor=np.finfo(float).tiny;clipped=sigma<floor;sigma=np.maximum(sigma,floor);flip=np.linalg.det(vectors)<0;vectors[flip,:,0]*=-1
 rotation=Rotation.from_matrix(vectors).as_quat();out[changed,:3]=(d[:,:3]+delta[changed]);out[changed,3:7]=np.c_[rotation[:,3],rotation[:,:3]];out[changed,7:10]=np.log(sigma)
 singular=np.linalg.svd(J,compute_uv=False);report.update({'minJacobianDeterminant':float(np.linalg.det(J).min()),'maxJacobianStretch':float(singular.max()),'minJacobianStretch':float(singular.min()),'covarianceSingularValuesClipped':int(clipped.sum())})
 return out,report
