#!/usr/bin/env python3
"""Apply only reviewed final covariance and exact residual source corrections."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from cleanup import ROOT,read_ply,write_ply,sha256_file,columns
from repair_surfaces import WORLD_FROM_RAW as W

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phone-ids',type=Path,required=True);ap.add_argument('--native-covariance',type=Path);a=ap.parse_args()
 if a.out.exists():raise ValueError('Output must be new')
 prior=json.loads((a.base/'report.json').read_text());baseline=Path(prior['baseline']);base_phone,_,_=read_ply(baseline/'iphone.ply');base_ref,_,_=read_ply(baseline/'reference-patches.ply');phone,_,_=read_ply(a.base/'iphone.ply');reference,_,_=read_ply(a.base/'reference-patches.ply');phone=phone.copy();candidate_ref=reference;reference=reference.copy();z=np.load(a.base/'changes.npz');changes={k:z[k] for k in z.files};mapping=changes['reference_source_ids'];ids_data=json.loads(a.phone_ids.read_text());ids=np.unique(np.asarray(ids_data['indices'],dtype=np.int64));world=np.einsum('ij,nj->ni',W,columns(phone[ids],['x','y','z']).astype(float));cf=np.asarray(prior['recipe']['plane']);residual=world[:,2]-np.einsum('ij,j->i',world[:,:2],cf[:2])-cf[2]
 if np.any(residual>-.02):raise ValueError('Exact residual list includes a point not behind wall; requires separate review')
 phone['opacity'][ids]=np.log(1e-8/(1-1e-8));covariance_report=None
 if a.native_covariance:
  native,_,_=read_ply(a.native_covariance/'native.ply');native_report=json.loads((a.native_covariance/'report.json').read_text());nz=np.load(a.native_covariance/'changes.npz');keys=[k for k in nz.files if k in ['reference_original_indices','reference_indices','indices','selected_indices','selected_original_ids']]
  if len(keys)!=1:raise ValueError('Ambiguous native covariance source IDs: '+str(nz.files))
  selected=nz[keys[0]];lookup=np.full(len(native),-1,np.int64);lookup[mapping]=np.arange(len(mapping));present=selected[lookup[selected]>=0];dest=lookup[present];fields=['scale_0','scale_1','scale_2','rot_0','rot_1','rot_2','rot_3']
  for f in fields:reference[f][dest]=native[f][present]
  changes['reviewed_normal_covariance_reference_source_ids']=present;covariance_report={'directory':str(a.native_covariance.resolve()),'source_sha256':sha256_file(a.native_covariance/'native.ply'),'report_sha256':sha256_file(a.native_covariance/'report.json'),'selected_original_ids':len(selected),'present_in_hybrid':len(present),'baseline_reference_rows':dest.tolist(),'fields':fields}
 changes['exact_residual_phone_ids']=ids;summary={}
 # Recalculate complete baseline-relative diffs so this is one standalone wall
 # component, preserving the row mapping used by the final composer.
 for label,b,v in [('iphone',base_phone,phone),('reference',base_ref,reference[:len(base_ref)])]:
  for key in list(changes):
   if key.startswith(label+'_') and (key.endswith('_indices') or key.endswith('_before') or key.endswith('_after')):del changes[key]
  union=np.zeros(len(b),bool);fields={}
  for f in b.dtype.names:
   selected=np.flatnonzero(b[f]!=v[f]);union[selected]=True
   if len(selected):changes[label+'_'+f+'_indices']=selected;changes[label+'_'+f+'_before']=b[f][selected];changes[label+'_'+f+'_after']=v[f][selected];fields[f]=len(selected)
  changes[label+'_indices']=np.flatnonzero(union);summary[label]={'changed_rows':int(union.sum()),'fields':fields,'unselected_exact':bool(np.array_equal(b[~union],v[~union]))}
 a.out.mkdir(parents=True);write_ply(a.out/'iphone.ply',phone);write_ply(a.out/'reference-patches.ply',reference);np.savez_compressed(a.out/'changes.npz',**changes);shutil.copyfile(__file__,a.out/'generator.py');checks={'phone_nonopacity_exact':all(np.array_equal(phone[f],base_phone[f]) for f in phone.dtype.names if f!='opacity'),'no_phone_opacity_increases':bool(np.all(phone['opacity']<=base_phone['opacity']+2e-6)),'reference_positions_DC_SH_opacity_unchanged_from_candidate':all(np.array_equal(reference[f],candidate_ref[f]) for f in ['x','y','z','opacity']+ [f for f in reference.dtype.names if f.startswith('f_')]),'all_finite':all(np.isfinite(v[f]).all() for v in [phone,reference] for f in v.dtype.names),'source_mapping_unique':len(np.unique(mapping))==len(mapping)}
 if not all(checks.values()):raise RuntimeError(checks)
 recipe={**prior['recipe']};recipe['physical_boundary_evidence']=recipe.get('physical_boundary_evidence','').replace('inset4mm','inset .004 scene units')
 report={**prior,'recipe':recipe,'status':'Unreviewed native wall final residual correction','preceding_candidate':str(a.base.resolve()),'preceding_candidate_phone_sha256':sha256_file(a.base/'iphone.ply'),'preceding_candidate_reference_sha256':sha256_file(a.base/'reference-patches.ply'),'exact_residual_correction':{'source':str(a.phone_ids.resolve()),'source_sha256':sha256_file(a.phone_ids),'count':len(ids),'indices':ids.tolist(),'plane_residuals':residual.tolist(),'world_positions':world.tolist(),'evidence':ids_data.get('evidence',[])},'reviewed_lower_normal_covariance':covariance_report,'changes':summary,'checks':checks,'generator_sha256':sha256_file(a.out/'generator.py'),'changes_sha256':sha256_file(a.out/'changes.npz'),'input_native_material_fields':'All captured positions/DC/SH and opacity retained from V2. If enabled, only approved native covariance changes are added.'};(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'changes':summary,'checks':checks,'exact_ids':ids.tolist(),'covariance':covariance_report},indent=2))
if __name__=='__main__':main()
