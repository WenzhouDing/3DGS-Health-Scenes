#!/usr/bin/env python3
"""Recover a fine matte laminate from observed iPhone grain, with local repair.

This deliberately repairs material appearance: removes coarse captured wall
layers, restores dense fine original phone grains, regularizes local mean and
contrast, confines grains to the fitted flat wall, and adds neutral sampled
material backing. It does not generate random texture. Capture fine-grain
positions and high-frequency brightness deviations supply the finish.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter,map_coordinates
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W

CF=np.array([-.08237276,.02981509,-.92014449])
UV=np.array([[-1.405,-.51],[-.345,.135]])
LABEL=np.array([[-1.50,-.52],[-1.315,.015]])
RECIPE={'uv_bounds':UV.tolist(),'plane':CF.tolist(),'uv_feather':.025,'label_protection_uv':LABEL.tolist(),'label_feather':.016,'clear_depth_bounds':[-.065,.065],'material_depth_bound':.012,'source_alpha_min':.06,'source_sigma_max':.0035,'source_sigma_mid_max':.0018,'local_material_radius':.018,'target_contrast':.22,'normal_sigma':.0001,'tangent_aspect_max':1.8,'tangent_max':.0016,'backing_spacing':.008,'backing_sigma':.007,'backing_plane_offset':-.0015,'backing_alpha':.97,'backing_boundary_sigma_cap':'3.2 sigma remains within wall and outside protected label box'}


def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)


def pos(v):return np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))
def alpha(v):return 1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
def logit(a):
 a=np.clip(a,1e-8,1-1e-8);return np.log(a/(1-a))
def uvweight(p):
 uv=p[:,:2];edge=np.minimum(uv-UV[0],UV[1]-uv).min(1);outside=np.maximum(np.maximum(LABEL[0]-uv,uv-LABEL[1]),0);labeldist=np.linalg.norm(outside,axis=1);return smooth(edge/RECIPE['uv_feather'])*smooth(labeldist/RECIPE['label_feather'])
def residual(p):return p[:,2]-np.einsum('ij,j->i',p[:,:2],CF[:2])-CF[2]
def set_columns(v,ids,names,data):
 for i,n in enumerate(names):v[n][ids]=data[:,i]


def local_field(p,lum,weight):
 step=.002;nx,ny=np.ceil((UV[1]-UV[0])/step).astype(int)+1;ij=np.clip(((p[:,:2]-UV[0])/step).astype(int),0,[nx-1,ny-1]);flat=ij[:,0]*ny+ij[:,1];shape=(nx,ny);s=RECIPE['local_material_radius']/step
 mass=np.bincount(flat,weights=weight,minlength=nx*ny).reshape(shape);mass=gaussian_filter(mass,s,mode='nearest')
 means=[]
 for values in [lum,lum*lum]:
  field=gaussian_filter(np.bincount(flat,weights=values*weight,minlength=nx*ny).reshape(shape),s,mode='nearest')/np.maximum(mass,1e-12);means.append(map_coordinates(field,((p[:,:2]-UV[0])/step).T,order=1,mode='nearest'))
 return means[0],np.sqrt(np.maximum(means[1]-means[0]**2,.07**2))


def grain_geometry(source,ids,p,w):
 rawsc=columns(source[ids],['scale_0','scale_1','scale_2']).astype(float);s=np.sort(np.exp(rawsc),axis=1);major=np.clip(np.sqrt(s[:,2]*s[:,1]*RECIPE['tangent_aspect_max']),.00048,RECIPE['tangent_max']);minor=np.clip(major/RECIPE['tangent_aspect_max'],.00032,None)
 normal=np.array([-CF[0],-CF[1],1.]);normal/=np.linalg.norm(normal);Q=Rotation.from_quat(columns(source[ids],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',W,Q);long=np.argmax(rawsc,axis=1);t=Q[np.arange(len(ids)),:,long];t-=np.einsum('ni,i->n',t,normal)[:,None]*normal;norm=np.linalg.norm(t,axis=1);t[norm<1e-5]=[1,0,CF[0]];t/=np.linalg.norm(t,axis=1)[:,None];minoraxis=np.cross(np.broadcast_to(normal,t.shape),t);worldq=np.stack([np.broadcast_to(normal,t.shape),t,minoraxis],axis=2);sigmas=np.c_[np.full(len(ids),RECIPE['normal_sigma']),major,minor]
 # Feather the covariance itself: rotating/scaling the entire Gaussian at
 # negligible selection weight would change visible grains outside the repair.
 oldB=Q*np.exp(rawsc)[:,None,:];oldC=np.einsum('nik,njk->nij',oldB,oldB);newB=worldq*sigmas[:,None,:];newC=np.einsum('nik,njk->nij',newB,newB);blend=(1-w[ids,None,None])*oldC+w[ids,None,None]*newC;vals,vecs=np.linalg.eigh(blend);vecs[np.linalg.det(vecs)<0,:,0]*=-1;rawq=np.einsum('ij,njk->nik',W.T,vecs);quat=Rotation.from_matrix(rawq).as_quat()[:,[3,0,1,2]];sigmas=np.sqrt(np.maximum(vals,1e-20))
 repaired=p[ids].copy();repaired[:,2]-=residual(repaired)-.0006
 # Match the original skin continuously at region/protected-label boundaries.
 repaired=p[ids]+w[ids,None]*(repaired-p[ids]);return np.einsum('ij,nj->ni',W.T,repaired),np.log(sigmas),quat


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
 rawpath=ROOT/'raw/ambulance_exp11_boot_sharp.ply';orig,_,_=read_ply(rawpath);base,_,_=read_ply(args.baseline/'iphone.ply');ref,_,_=read_ply(args.baseline/'reference-patches.ply');assert len(orig)==len(base)
 pp=pos(orig);rp=pos(ref);w=uvweight(pp);rw=uvweight(rp);pd=residual(pp);rd=residual(rp);ss=np.sort(np.exp(columns(orig,['scale_0','scale_1','scale_2']).astype(float)),axis=1);oa=alpha(orig);ba=alpha(base);ra=alpha(ref)
 # Coarse or displaced source material is explicitly removed within the wall
 # band; fine captured support is restored below. Labels/glass have weight0.
 clear=(abs(pd)<.065)&(w>0);rclr=(abs(rd)<.065)&(rw>0);grain=(w>0)&(abs(pd)<RECIPE['material_depth_bound'])&(oa>RECIPE['source_alpha_min'])&(ss[:,2]<RECIPE['source_sigma_max'])&(ss[:,1]<RECIPE['source_sigma_mid_max']);ids=np.flatnonzero(grain)
 rgb=.5+.28209479177387814*columns(orig[ids],['f_dc_0','f_dc_1','f_dc_2']).astype(float);lum=rgb.mean(1);areaweight=oa[ids]*ss[ids,1]*ss[ids,2];lm,ls=local_field(pp[ids],lum,areaweight)
 border=(pp[ids,1]>.093)&(pp[ids,0]>-1.30)&(pp[ids,0]<-.42)&(ss[ids,2]<.0023);target=np.average(rgb[border],axis=0,weights=areaweight[border]);target_luma=target.mean();chroma=target-target_luma
 # A gentle .02 lower-wall shading change preserves a light falloff while
 # removing localized broad dark/white/red clouds. Fine deviations survive.
 shade=.02*np.clip((.135-pp[ids,1])/.645,0,1);grain_luma=np.clip(target_luma-shade+(lum-lm)*np.clip(RECIPE['target_contrast']/ls,.55,1.8),.08,.985);colors=np.clip(grain_luma[:,None]+chroma[None,:],.03,.995)
 phone=base.copy();rr=ref.copy();newa=ba.copy();newa[clear]*=1-w[clear];newa[ids]+=oa[ids]*w[ids];phone['opacity'][clear]=logit(newa[clear]);rr['opacity'][rclr]=logit(ra[rclr]*(1-rw[rclr]));xyz,sc,quat=grain_geometry(orig,ids,pp,w);set_columns(phone,ids,['x','y','z'],xyz);set_columns(phone,ids,['scale_0','scale_1','scale_2'],sc);set_columns(phone,ids,['rot_0','rot_1','rot_2','rot_3'],quat)
 old=columns(orig[ids],['f_dc_0','f_dc_1','f_dc_2']).astype(float);dc=(colors-.5)/.28209479177387814;set_columns(phone,ids,['f_dc_0','f_dc_1','f_dc_2'],old+w[ids,None]*(dc-old))
 # Backing has no invented grain: it uses the mean of observed clean border
 # material. Captured phone fine texture remains in front of this thin skin.
 step=RECIPE['backing_spacing'];xx=np.arange(UV[0,0]+step/2,UV[1,0],step);yy=np.arange(UV[0,1]+step/2,UV[1,1],step);xy=np.stack(np.meshgrid(xx,yy),axis=-1).reshape(-1,2);bp=np.c_[xy,np.einsum('ij,j->i',xy,CF[:2])+CF[2]+RECIPE['backing_plane_offset']];bw=uvweight(bp);bp=bp[bw>0];bw=bw[bw>0];extra=np.zeros(len(bp),dtype=base.dtype);exyz=np.einsum('ij,nj->ni',W.T,bp);ei=np.arange(len(extra));set_columns(extra,ei,['x','y','z'],exyz);normal=np.array([-CF[0],-CF[1],1.]);normal/=np.linalg.norm(normal);t=np.array([1,0,CF[0]]);t/=np.linalg.norm(t);rmat=W.T@np.array([normal,t,np.cross(normal,t)]).T;q=Rotation.from_matrix(rmat).as_quat()[[3,0,1,2]]
 for n,x in zip(['rot_0','rot_1','rot_2','rot_3'],q):extra[n]=x
 edge=np.minimum(bp[:,:2]-UV[0],UV[1]-bp[:,:2]).min(1);labeldist=np.linalg.norm(np.maximum(np.maximum(LABEL[0]-bp[:,:2],bp[:,:2]-LABEL[1]),0),axis=1);bsigma=np.minimum(RECIPE['backing_sigma'],np.maximum(.0001,np.minimum(edge,labeldist)/3.2));extra['scale_0']=np.log(.00008);extra['scale_1']=np.log(bsigma);extra['scale_2']=np.log(bsigma)
 backing_colors=target[None,:]-.02*np.clip((.135-bp[:,1])/.645,0,1)[:,None];set_columns(extra,ei,['f_dc_0','f_dc_1','f_dc_2'],(backing_colors-.5)/.28209479177387814);extra['opacity']=logit(RECIPE['backing_alpha']*bw)
 changes={};summary={}
 for name,b,a in [('iphone',base,phone),('reference',ref,rr)]:
  union=np.zeros(len(b),bool);fields={}
  for n in b.dtype.names:
   ix=np.flatnonzero(b[n]!=a[n]);union[ix]=True
   if len(ix):changes[name+'_'+n+'_indices']=ix;changes[name+'_'+n+'_before']=b[n][ix];changes[name+'_'+n+'_after']=a[n][ix];fields[n]=len(ix)
  ui=np.flatnonzero(union);changes[name+'_indices']=ui;summary[name]={'changed_rows':len(ui),'changed_fields':fields,'world_bounds':[pos(b[ui]).min(0).tolist(),pos(b[ui]).max(0).tolist()],'unchanged_rows_exact':bool(np.array_equal(a[~union],b[~union])),'max_center_displacement':float(np.linalg.norm(pos(a[ui])-pos(b[ui]),axis=1).max())}
 # Report actual local contrast / parameter effects; no unqualified cleanup claim.
 np.savez_compressed(args.out/'changes.npz',**changes);write_ply(args.out/'iphone.ply',np.concatenate([phone,extra]));write_ply(args.out/'reference-patches.ply',rr);write_ply(args.out/'additions.ply',extra);shutil.copyfile(__file__,args.out/'generator.py')
 report={'status':'Unreviewed observed-grain material reconstruction','baseline':str(args.baseline),'baseline_hashes':{'iphone':sha256_file(args.baseline/'iphone.ply'),'reference':sha256_file(args.baseline/'reference-patches.ply')},'raw_phone':str(rawpath),'raw_phone_sha256':sha256_file(rawpath),'recipe':RECIPE,'source_grain_rows':len(ids),'source_grain_original_indices_file':'changes.npz','source_grain_indices_key':'grain_original_indices','clean_border_sample_rows':int(border.sum()),'sampled_target_rgb':target.tolist(),'local_mean_luma_quantiles':np.quantile(lm,[.01,.5,.99]).tolist(),'local_luma_std_quantiles':np.quantile(ls,[.01,.5,.99]).tolist(),'changes':summary,'added_rows':len(extra),'additions':{'file':'additions.ply','trailing_in':'iphone.ply','start_index':len(base),'purpose':'Thin continuous observed-mean laminate backing; source fine texture above it supplies grain.','geometry':'Fitted static white panel; no hardware/glass/labels.'},'reference_row_count':len(ref),'iphone_baseline_row_count':len(base),'checks':{'baseline_prefix_mapping_preserved':True,'protected_glass_labels_untouched':bool(np.array_equal(phone[w==0],base[w==0])),'reference_color_SH_geometry_unchanged':all(np.array_equal(ref[n],rr[n]) for n in ref.dtype.names if n!='opacity'),'all_values_finite':all(np.isfinite(np.concatenate([phone[n],extra[n]])).all() for n in phone.dtype.names) and all(np.isfinite(rr[n]).all() for n in rr.dtype.names)},'generator_sha256':sha256_file(args.out/'generator.py'),'limitations':['This intentionally changes wall positions,covariance,opacity,and DC within reported masks to meet requested uniform fine-grain appearance.','Matte backing carries no SH; donor SH support in this wall is suppressed, avoiding imported red view-dependent ghosts.','Borders, instruction labels, glass and nearby equipment require GPU controls.'],'remote_publish':False}
 changes['grain_original_indices']=ids;np.savez_compressed(args.out/'changes.npz',**changes);(args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
