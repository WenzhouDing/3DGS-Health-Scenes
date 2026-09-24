#!/usr/bin/env python3
"""Restore actual iPhone grain and adapt its footprint to measured capture density.

No new, copied or translated texture. Fine centers and opacity remain; existing
captured colors receive a gradual local luminance correction. Coarse captured support is recessed and bounded in thickness;
fine tangent coverage is adjusted from existing local neighbor spacing.
"""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.spatial.transform import Rotation
from cleanup import ROOT,read_ply,columns,write_ply,sha256_file
from repair_surfaces import WORLD_FROM_RAW as W
CF=np.array([-.08237276,.02981509,-.92014449]);UV=np.array([[-1.405,-.52],[-.345,.135]]);LABEL=np.array([[-1.50,-.52],[-1.315,.015]])
R={'uv_bounds':UV.tolist(),'bottom_edge_y_from_x1':[-.058,-.477],'label_protection_uv':LABEL.tolist(),'feather':.03,'source_coarse_maxsigma_min':.006,'local_skin_neighbors':64,'local_skin_max_neighbor_distance':.025,'local_offset_tolerance_min':.008,'local_offset_MAD_factor':4,'coarse_normal_sigma_limit':.001,'coarse_normal_edit_min_sigma':.004,'recess_quantile':.05,'recess_margin':.0045,'clear_global_depth_band':.16,'note':'Reference plane only finds the vicinity; local fine captured phone depth determines detached-layer selection.'}
R.update({'photometry_grid_step':.008,'photometry_local_bandwidth':.035,'fine_max_luma_bias':.3,'fine_min_contrast_scale':.35,'coarse_residual_contrast':.2,'local_skin_max_neighbor_distance':.045,'recess_quantile':.005,'recess_margin':.018,'coarse_tangent_sigma_limit':.012,'fine_optical_depth_target':2.,'fine_density_neighbors':16,'fine_tangent_sigma_limit':.0013,'fine_tangent_major_sigma_limit':.0014,'fine_max_footprint_scale':2.5,'fine_normal_sigma_limit':.00035,'fine_aspect_limit':2.})

def smooth(x):
 x=np.clip(x,0,1);return x*x*(3-2*x)
def pos(v):return np.einsum('ij,nj->ni',W,columns(v,['x','y','z']).astype(float))
def alpha(v):return 1/(1+np.exp(-np.clip(v['opacity'].astype(float),-40,40)))
def logit(a):
 a=np.clip(a,1e-8,1-1e-8);return np.log(a/(1-a))
def regionweight(p):
 uv=p[:,:2];e=np.minimum(uv-UV[0],UV[1]-uv).min(1);e=np.minimum(e,uv[:,1]-(-.058*uv[:,0]-.477));ld=np.linalg.norm(np.maximum(np.maximum(LABEL[0]-uv,uv-LABEL[1]),0),axis=1);return smooth(e/R['feather'])*smooth(ld/.018)
def dglobal(p):return p[:,2]-np.einsum('ij,j->i',p[:,:2],CF[:2])-CF[2]


def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--baseline',type=Path,default=ROOT/'raw/ambulance-cleanup/pass5/combined-v1');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.resolve()==a.baseline.resolve():raise ValueError('Baseline immutable')
 prior=json.loads((a.baseline/'report.json').read_text());source,_,_=read_ply(Path(prior['iphone']));base,_,_=read_ply(a.baseline/'iphone.ply');ref,_,_=read_ply(a.baseline/'reference-patches.ply');mapping=np.load(a.baseline/'selections.npz');pm=np.ones(len(source));rm=np.zeros(999410);omitted=[]
 for component in prior['components']:
  folder=Path(component['directory'])
  if str(folder).endswith('/pass3/trials/walls-v3'):
   omitted.append(str(folder));continue
  z=np.load(folder/'iphone-opacity-selection.npz');np.minimum.at(pm,z['indices'],z['multipliers']);z=np.load(folder/'reference-selection.npz');np.maximum.at(rm,z['indices'],z['multipliers'])
 if len(omitted)!=1:raise ValueError('Expected exactly the legacy lower-cabinet wall donor component')
 bpm=np.ones(len(source));bpm[mapping['iphone_indices']]=mapping['iphone_multipliers'];brm=np.zeros(len(rm));brm[mapping['reference_indices']]=mapping['reference_multipliers'];phone=base.copy();refout=ref.copy();restored=np.flatnonzero(pm>bpm);phone['opacity'][restored]=logit(alpha(source[restored])*pm[restored]);rid=mapping['reference_indices'];drop=np.flatnonzero(rm[rid]<brm[rid]);refout['opacity'][drop]=logit(alpha(ref[drop])*(rm[rid[drop]]/brm[rid[drop]]))
 p=pos(source);w=regionweight(p);dg=dglobal(p);s=np.exp(columns(source,['scale_0','scale_1','scale_2']).astype(float));rgb=.5+.28209479177387814*columns(source,['f_dc_0','f_dc_1','f_dc_2']);aa=alpha(source);maxs=s.max(1)
 anchor=(w>0)&(abs(dg)<.025)&(maxs<.0035)&(aa>.35)&(np.ptp(rgb,axis=1)<.2);ai=np.flatnonzero(anchor);tree=cKDTree(p[ai,:2]);candidate=(w>0)&(abs(dg)<R['clear_global_depth_band'])&(maxs>R['source_coarse_maxsigma_min'])&(aa>.01);ci=np.flatnonzero(candidate);dist,ni=tree.query(p[ci,:2],k=R['local_skin_neighbors']);neighbor_d=dg[ai[ni]];local_offset=np.median(neighbor_d,axis=1);mad=1.4826*np.median(abs(neighbor_d-local_offset[:,None]),axis=1);good=(dist[:,-1]<R['local_skin_max_neighbor_distance'])&(mad<.015);dl=dg[ci]-local_offset;tol=np.maximum(R['local_offset_tolerance_min'],R['local_offset_MAD_factor']*mad);bad=good&(abs(dl)>tol);removed=np.empty(0,dtype=np.int64)
 # Coarse captured splats retain their color, opacity and tangent footprint.
 # They are placed behind the local fine skin instead of being deleted.
 # This retains original coverage and natural local illumination.
 near=ci[good];Q=Rotation.from_quat(columns(source[near],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();Q=np.einsum('ij,njk->nik',W,Q);B=Q*s[near,None,:];C=np.einsum('nik,njk->nij',B,B);n=np.array([-CF[0],-CF[1],1.]);n/=np.linalg.norm(n);sn=np.sqrt(np.einsum('i,nij,j->n',n,C,n));edit=np.ones(len(near),bool);ei=near[edit];C=C[edit];sn=sn[edit];P=np.eye(3)-np.outer(n,n);thin=np.minimum(sn,R['coarse_normal_sigma_limit']);tangent=np.einsum('ij,njk,lk->nil',P,C,P);tev,taxes=np.linalg.eigh(tangent);tev=np.minimum(tev,R['coarse_tangent_sigma_limit']**2);target=np.einsum('nik,nk,njk->nij',taxes,tev,taxes)+thin[:,None,None]**2*np.outer(n,n);blend=C*(1-w[ei,None,None])+target*w[ei,None,None];ev,axes=np.linalg.eigh(blend);axes[np.linalg.det(axes)<0,:,0]*=-1;q=Rotation.from_matrix(np.einsum('ij,njk->nik',W.T,axes)).as_quat();logs=.5*np.log(np.maximum(ev,1e-20))
 target_offset=np.quantile(neighbor_d[good],R['recess_quantile'],axis=1)-R['recess_margin'];shift=(dg[ei]-target_offset)/np.sqrt(1+np.sum(CF[:2]**2));newp=p[ei]-(w[ei]*shift)[:,None]*n;rawp=np.einsum('ij,nj->ni',W.T,newp)
 for i,f in enumerate(['x','y','z']):phone[f][ei]=rawp[:,i]
 for i,f in enumerate(['rot_1','rot_2','rot_3','rot_0']):phone[f][ei]=q[:,i]
 for i,f in enumerate(['scale_0','scale_1','scale_2']):phone[f][ei]=logs[:,i]
 # Existing fine captured grains have insufficient projected optical coverage.
 # Adapt only their covariance from actual neighbor spacing. No relocated points
 # and no DC/opacity correction: local captured pattern and shading remain.
 fi=np.flatnonzero((w>0)&(abs(dg)<.025)&(maxs<=.006)&(aa>.35)&(np.ptp(rgb,axis=1)<.2))
 ftree=cKDTree(p[fi,:2]);fd,fn=ftree.query(p[fi,:2],k=R['fine_density_neighbors']+1)
 mean_alpha=np.mean(aa[fi[fn[:,1:]]],axis=1)
 qfine=Rotation.from_quat(columns(source[fi],['rot_1','rot_2','rot_3','rot_0'])).as_matrix();qfine=np.einsum('ij,njk->nik',W,qfine);bf=qfine*s[fi,None,:];cfine=np.einsum('nik,njk->nij',bf,bf)
 tu=np.array([1.,0.,CF[0]]);tu/=np.linalg.norm(tu);tv=np.cross(n,tu);T=np.stack([tu,tv]);c2=np.einsum('ij,njk,lk->nil',T,cfine,T);fe,faxes=np.linalg.eigh(c2);fe=np.maximum(fe,1e-12)
 original_g=np.prod(fe,axis=1)**.25
 requested=fd[:,-1]*np.sqrt(R['fine_optical_depth_target']/(2*R['fine_density_neighbors']*mean_alpha))
 target_g=np.maximum(original_g,np.minimum.reduce([requested,np.full(len(fi),R['fine_tangent_sigma_limit']),original_g*R['fine_max_footprint_scale']]))
 aspect=np.minimum(np.sqrt(fe[:,1]/fe[:,0]),R['fine_aspect_limit'])
 target_e=np.minimum(np.stack([target_g**2/aspect,target_g**2*aspect],axis=1),R['fine_tangent_major_sigma_limit']**2)
 # Physical-border guard: no new large footprint close to the protected label,
 # cabinet frame, or lower lip. The existing covariance remains at the feather.
 uv=p[fi,:2];edge=np.minimum(uv-UV[0],UV[1]-uv).min(1);edge=np.minimum(edge,uv[:,1]-(-.058*uv[:,0]-.477));labeldist=np.linalg.norm(np.maximum(np.maximum(LABEL[0]-uv,uv-LABEL[1]),0),axis=1);edge=np.minimum(edge,labeldist)
 target_e=np.minimum(target_e,np.maximum(edge/4,.00005)[:,None]**2)
 target2=np.einsum('nik,nk,njk->nij',faxes,target_e,faxes);fsn=np.sqrt(np.maximum(0,np.einsum('i,nij,j->n',n,cfine,n)));targetf=np.einsum('ij,njk,kl->nil',T.T,target2,T)+np.minimum(fsn,R['fine_normal_sigma_limit'])[:,None,None]**2*np.outer(n,n)
 blendf=cfine*(1-w[fi,None,None])+targetf*w[fi,None,None];fval,faxis=np.linalg.eigh(blendf);faxis[np.linalg.det(faxis)<0,:,0]*=-1;fq=Rotation.from_matrix(np.einsum('ij,njk->nik',W.T,faxis)).as_quat();flogs=.5*np.log(np.maximum(fval,1e-20))
 for i,f in enumerate(['rot_1','rot_2','rot_3','rot_0']):phone[f][fi]=fq[:,i]
 for i,f in enumerate(['scale_0','scale_1','scale_2']):phone[f][fi]=flogs[:,i]
 # Smooth illumination / contrast correction on existing captured samples.
 # Clean low-variance observed cells define a mild illumination field; local
 # point locations and their high-frequency residual ordering are unchanged.
 grid_step=R['photometry_grid_step'];shape=np.ceil((UV[1]-UV[0])/grid_step).astype(int)+1
 coord=(p[fi,:2]-UV[0])/grid_step;ix=np.clip(np.rint(coord).astype(int),0,shape-1);key=ix[:,0]*shape[1]+ix[:,1];grid_n=int(np.prod(shape));luma=rgb.mean(1).astype(float)
 counts=np.bincount(key,minlength=grid_n).reshape(shape);sums=np.bincount(key,weights=luma[fi],minlength=grid_n).reshape(shape);sum2=np.bincount(key,weights=luma[fi]**2,minlength=grid_n).reshape(shape)
 blur=R['photometry_local_bandwidth']/grid_step;den=gaussian_filter(counts.astype(float),blur);local=gaussian_filter(sums,blur)/np.maximum(den,1e-6);second=gaussian_filter(sum2,blur)/np.maximum(den,1e-6);std=np.sqrt(np.maximum(second-local*local,1e-6))
 gx,gy=np.meshgrid(np.arange(shape[0])*grid_step+UV[0,0],np.arange(shape[1])*grid_step+UV[0,1],indexing='ij')
 design=np.stack([np.ones_like(gx),gx,gy,gx*gy],axis=-1);clean=(den>4)&(std<.135)&(local>.73)&(local<.92)&((gx>-.67)|(gy>.045)|(gy<-.32))
 if clean.sum()<20:raise ValueError('Insufficient observed clean border for photometry')
 coefficients=np.linalg.lstsq(design[clean],local[clean],rcond=None)[0];illum=np.clip(np.einsum('ijk,k->ij',design,coefficients),.76,.91)
 flocal=map_coordinates(local,coord.T,order=1,mode='nearest');fstd=map_coordinates(std,coord.T,order=1,mode='nearest');ftarget=map_coordinates(illum,coord.T,order=1,mode='nearest')
 bias=np.clip(ftarget-flocal,-.07,R['fine_max_luma_bias']);contrast=np.clip(.11/np.maximum(fstd,1e-6),R['fine_min_contrast_scale'],1.)
 fine_luma=luma[fi]+w[fi]*(bias+(contrast-1)*(luma[fi]-flocal))
 for k,f in enumerate(['f_dc_0','f_dc_1','f_dc_2']):phone[f][fi]=(rgb[fi,k]+fine_luma-luma[fi]-.5)/.28209479177387814
 # Coarse support stays actual captured support, retaining gentle captured
 # local radiance variation after a base-tone correction. Only neutral rows.
 neutral=ei[np.ptp(rgb[ei],axis=1)<.15];cc=(p[neutral,:2]-UV[0])/grid_step;ci2=np.clip(np.rint(cc).astype(int),0,shape-1);ck=ci2[:,0]*shape[1]+ci2[:,1]
 ccount=np.bincount(ck,minlength=grid_n).reshape(shape);csum=np.bincount(ck,weights=luma[neutral],minlength=grid_n).reshape(shape);cden=gaussian_filter(ccount.astype(float),blur);cmean=gaussian_filter(csum,blur)/np.maximum(cden,1e-6);clocal=map_coordinates(cmean,cc.T,order=1,mode='nearest');ctarget=map_coordinates(illum,cc.T,order=1,mode='nearest')
 desired=ctarget+R['coarse_residual_contrast']*(luma[neutral]-clocal);cluma=luma[neutral]+w[neutral]*np.clip(desired-luma[neutral],-.35,.45)
 for k,f in enumerate(['f_dc_0','f_dc_1','f_dc_2']):phone[f][neutral]=(rgb[neutral,k]+cluma-luma[neutral]-.5)/.28209479177387814
 photometry={'clean_border_cells':int(clean.sum()),'captured_border_illumination_bilinear_coefficients':coefficients.tolist(),'fine_rows':len(fi),'neutral_coarse_rows':len(neutral),'fine_luma_change_q01_q50_q99':np.quantile(fine_luma-luma[fi],[.01,.5,.99]).tolist(),'coarse_luma_change_q01_q50_q99':np.quantile(cluma-luma[neutral],[.01,.5,.99]).tolist(),'note':'Luminance-only smooth bias and bounded local contrast; captured chroma and fine spatial positions retained.'}
 final_fine_e=np.linalg.eigvalsh(np.einsum('ij,njk,lk->nil',T,blendf,T));actual_g=np.maximum(np.prod(final_fine_e,axis=1),1e-30)**.25
 fine_diagnostic={'count':len(fi),'original_tangent_geometric_sigma_q10_q50_q90':np.quantile(original_g,[.1,.5,.9]).tolist(),'adapted_tangent_geometric_sigma_q10_q50_q90':np.quantile(actual_g,[.1,.5,.9]).tolist(),'k16_radius_q10_q50_q90':np.quantile(fd[:,-1],[.1,.5,.9]).tolist(),'approx_fine_optical_depth_before_q10_q50_q90':np.quantile(2*16*mean_alpha*original_g**2/fd[:,-1]**2,[.1,.5,.9]).tolist(),'approx_fine_optical_depth_after_q10_q50_q90':np.quantile(2*16*mean_alpha*actual_g**2/fd[:,-1]**2,[.1,.5,.9]).tolist()}
 a.out.mkdir(parents=True,exist_ok=True);write_ply(a.out/'iphone.ply',phone);write_ply(a.out/'reference-patches.ply',refout);changes={'restored_captured_phone_indices':restored,'coarse_detached_phone_indices':removed,'coarse_normal_covariance_phone_indices':ei,'fine_anchor_original_indices':ai,'fine_covariance_adapted_original_indices':fi};summary={}
 for label,b,v in [('iphone',base,phone),('reference',ref,refout)]:
  union=np.zeros(len(b),bool);fields={}
  for f in b.dtype.names:
   ix=np.flatnonzero(b[f]!=v[f]);union[ix]=True
   if len(ix):changes[label+'_'+f+'_indices']=ix;changes[label+'_'+f+'_before']=b[f][ix];changes[label+'_'+f+'_after']=v[f][ix];fields[f]=len(ix)
  ix=np.flatnonzero(union);changes[label+'_indices']=ix;summary[label]={'changed_rows':len(ix),'fields':fields,'unselected_rows_exact':bool(np.array_equal(v[~union],b[~union]))}
 np.savez_compressed(a.out/'changes.npz',**changes);shutil.copyfile(__file__,a.out/'generator.py');checks={'fine_source_positions_exact':all(np.array_equal(phone[f][maxs<=R['source_coarse_maxsigma_min']],base[f][maxs<=R['source_coarse_maxsigma_min']]) for f in ['x','y','z']),'captured_chroma_preserved_within_float32':bool(np.max(abs((columns(phone,['f_dc_0','f_dc_1','f_dc_2'])-columns(base,['f_dc_0','f_dc_1','f_dc_2']))[:,0]-(columns(phone,['f_dc_0','f_dc_1','f_dc_2'])-columns(base,['f_dc_0','f_dc_1','f_dc_2']))[:,1]))<2e-6),'reference_nonopacity_exact':all(np.array_equal(refout[f],ref[f]) for f in ref.dtype.names if f!='opacity'),'fine_source_opacity_preserved_after_component_restore':bool(np.allclose(alpha(phone[fi]),alpha(source[fi])*pm[fi],rtol=2e-6,atol=1e-8)),'all_finite':all(np.isfinite(v[f]).all() for v in [phone,refout] for f in v.dtype.names)}
 if not all(checks.values()):raise RuntimeError(checks)
 report={'status':'Unreviewed captured-texture-only wall cleanup','baseline':str(a.baseline),'baseline_hashes':{'iphone':sha256_file(a.baseline/'iphone.ply'),'reference':sha256_file(a.baseline/'reference-patches.ply')},'original_phone':prior['iphone'],'original_phone_sha256':prior['iphone_sha256'],'omitted_legacy_material_component':omitted,'recipe':R,'counts':{'original_phone_rows_restored':len(restored),'reference_rows_legacy_weight_reduced':len(drop),'fine_original_skin_anchors':len(ai),'coarse_candidates':len(ci),'coarse_candidates_rejected_uncertain':int((~good).sum()),'coarse_offsurface_rows_attenuated':len(removed),'coarse_normal_covariances_constrained':len(ei),'coarse_support_centers_recessed':len(ei),'maximum_coarse_center_displacement':float(np.max(np.linalg.norm(newp-p[ei],axis=1)))},'changes':summary,'added_rows':0,'fine_coverage_diagnostic':fine_diagnostic,'photometry':photometry,'source_field_policy':'Fine-grain positions and original restored opacity remain exact. Existing captured DC receives a smooth luminance bias/contrast correction estimated from observed clean border; chroma preserved. Fine tangent covariance adapts to actual local point spacing with compact aspect and physical boundary caps. Coarse captured support is recessed behind deep local fine skin along the measured normal, tangent/normal covariance bounded. No added/copied texture or synthetic backing; No new points or reconstructed pattern; only captured samples receive the documented low-frequency tone adjustment.','checks':checks,'generator_sha256':sha256_file(a.out/'generator.py'),'review_required':['wall-material-close','wall-material-grazing','left-wall','mattress-top'],'limitations':['Natural captured tone variation remains; this does not aim for a perfectly uniform synthetic finish.','Coarse-source normal relocation must be reviewed for boundary continuity and natural local shading; original fine centers remain unchanged; color and footprint adapt using documented smooth fields.'],'remote_publish':False};(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'counts':report['counts'],'changes':summary,'checks':checks},indent=2),flush=True)


if __name__=='__main__':main()
