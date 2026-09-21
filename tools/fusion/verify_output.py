#!/usr/bin/env python3
"""Verify provenance, Gaussian parameters and finite payload of the current fusion."""
import json
from pathlib import Path
import numpy as np
from pipeline import ROOT, read_ply, sha256_file, FIELDS, transform_gaussians, axial_child_transform
from cleanup_masks import load_cleanup_masks, validate_restorations
from pipeline import BODY_PARTS
from appearance_colors import load_color_overrides, apply_color_values


def verify_rigid_part_constraints(report):
 """Independently check declared rigid ownership, shared poses and absent warps."""
 parameters=report['parameters'];rigid=parameters.get('rigidParts',{})
 assert isinstance(rigid,dict),'rigidParts is not a child-to-parent mapping'
 parts={part['id']:part for part in report['parts']}
 for child,parent in rigid.items():
  assert child in parts and isinstance(parent,str) and parent in parts and child!=parent,'invalid rigid parent'
  seen=set();current=child
  while current in rigid:
   assert current not in seen,'cyclic rigid parent constraint'
   seen.add(current);current=rigid[current]
  part=parts[child];assert part.get('rigidWithPart')==parent,'rigid parent missing from part report'
  for field in ['rotation','translation','scale']:
   np.testing.assert_array_equal(part['sourceToFrontRaw'][field],parts[parent]['sourceToFrontRaw'][field],err_msg='rigid child differs from its effective parent')
  manual=parameters.get('adjustments',{}).get(child,{})
  for key,default,shape in [('rotationDegrees',[0,0,0],(3,)),('translation',[0,0,0],(3,)),('twistDegrees',0,())]:
   value=np.asarray(manual.get(key,default),float)
   assert value.shape==shape and np.isfinite(value).all() and not np.any(value),'independent rigid child adjustment'
  assert not any(part.get(key) for key in ['mechanicalConstraint','wristRegistration','digitAlignment','hairAppearanceCorrection']),'rigid child reports an independent joint or warp'
  assert child!='right_hand' or not parameters.get('rightWristRegistration'),'rigid right hand has a wrist warp'
  assert child!='left_hand' or not parameters.get('leftHandDigitAlignment',{}).get('amount',0),'rigid left hand has a digit warp'
  assert child!='head' or not parameters.get('hairFullness',{}).get('amount',0),'rigid head has a hair warp'
 for part in report['parts']:
  assert not part.get('rigidWithPart') or rigid.get(part['id'])==part['rigidWithPart'],'undeclared rigid parent in report'
 return dict(rigid)


def main():
 out=ROOT/'raw/mannequin-fused'
 report=json.loads((out/'report.json').read_text())
 vertices,count,_=read_ply(out/'mannequin_fused.ply')
 labels=np.load(out/'part-labels.npy',mmap_mode='r');captures=np.load(out/'capture-labels.npy',mmap_mode='r');indices=np.load(out/'source-vertex-indices.npy',mmap_mode='r')
 assert len(labels)==len(captures)==len(indices)==count==report['fusedCount']
 assert set(np.unique(captures))=={0,1};assert labels.max()<len(report['parts'])
 rigid_parts=verify_rigid_part_constraints(report)
 by_part={p['id']:p for p in report['parts']};axial_checked=[]
 for part in report['parts']:
  if part.get('mechanicalConstraint'):
   constraint=part['mechanicalConstraint'];parent=by_part[constraint['parentPart']]['sourceToFrontRaw']
   expected_R,expected_t=axial_child_transform(np.array(parent['rotation']),np.array(parent['translation']),constraint)
   np.testing.assert_allclose(part['sourceToFrontRaw']['rotation'],expected_R,atol=1e-10,rtol=0)
   np.testing.assert_allclose(part['sourceToFrontRaw']['translation'],expected_t,atol=1e-10,rtol=0)
   axial_checked.append(part['id'])
 masks,_=load_cleanup_masks(report['parameters'].get('cleanupMasks',[]),dict(zip(['front','back'],report['sources'])),ROOT)
 color_overrides,_=load_color_overrides(report['parameters'].get('colorOverrides',[]),dict(zip(['front','back'],report['sources'])),ROOT)
 restoration_rules=report['parameters'].get('surfaceRestorations',[]);restoration_masks=[];restoration_checked=[]
 for rule in restoration_rules:
  selections,_=load_cleanup_masks([rule['selection']],dict(zip(['front','back'],report['sources'])),ROOT)
  restoration_masks.append(selections)
  for capture,source in enumerate(['front','back']):
   selected=(captures==capture)&np.isin(indices,selections[source])
   record=next(c for c in report['surfaceRestorationCounts'] if c['selection']==rule['selection'] and c['source']==source)
   assert int(selected.sum())==record['selectedAfterCleaning']==record['exported'] and record['allSelectedRetained'],'observed-surface restoration disappeared'
   restoration_checked.append({'part':rule['part'],'source':source,'exportedCount':int(selected.sum())})
 assignment_checked=[]
 for rule in report['parameters'].get('sourcePartAssignments',[]):
  selections,_=load_cleanup_masks([rule['selection']],dict(zip(['front','back'],report['sources'])),ROOT)
  expected_part=next(i for i,p in enumerate(report['parts']) if p['id']==rule['part'])
  for capture,source in enumerate(['front','back']):
   selected=(captures==capture)&np.isin(indices,selections[source])
   assert np.all(labels[selected]==expected_part),'crossing hardware was split across body parts'
   if rule.get('requireRetained'):
    record=next(c for c in report['sourcePartAssignmentCounts'] if c['selection']==rule['selection'] and c['source']==source)
    assert int(selected.sum())==record['selectedAfterCleaning']==record['exported'] and record['allSelectedRetained'],'crossing hardware disappeared'
   assignment_checked.append({'part':rule['part'],'source':source,'exportedCount':int(selected.sum())})
 for capture,source in enumerate(['front','back']):
  selected_masks=[m[source] for m in restoration_masks]
  validate_restorations(labels[captures==capture],indices[captures==capture],selected_masks,restoration_rules,BODY_PARTS,report['parameters']['fusion']['minWeight'])
  restored=np.unique(np.concatenate(selected_masks)) if selected_masks else np.empty(0,np.uint32)
  assert not np.isin(indices[captures==capture],np.setdiff1d(masks[source],restored)).any(),'unreviewed cleaned source vertices reappeared'
 assert all(np.isfinite(vertices[field]).all() for field in FIELDS)
 norm=np.sqrt(sum(vertices['rot_'+str(i)]**2 for i in range(4)))
 assert np.max(abs(norm-1))<1e-5
 rng=np.random.default_rng(601);verified=0;max_error=0.;original_hashes={};rigid_gaussians_checked={child:0 for child in rigid_parts}
 for capture,meta in enumerate(report['sources']):
  source=ROOT/meta['file'];assert sha256_file(source)==meta['sha256'];original_hashes[meta['file']]=meta['sha256']
  raw,source_count,_=read_ply(source)
  capture_rows=np.flatnonzero(captures==capture)
  for rule,selections in zip(restoration_rules,restoration_masks):
   restored_rows=capture_rows[np.isin(indices[capture_rows],selections[['front','back'][capture]])]
   original_alpha=1/(1+np.exp(-np.clip(raw['opacity'][indices[restored_rows]].astype(float),-40,40)))
   minimum_alpha=np.clip(-np.expm1(rule['minimumWeight']*np.log1p(-np.minimum(original_alpha,1-1e-9))),1e-8,1-1e-8)
   actual_alpha=1/(1+np.exp(-np.clip(vertices['opacity'][restored_rows].astype(float),-40,40)))
   assert np.all(actual_alpha>=minimum_alpha-2e-7),'restored surface opacity is below its configured floor'
  original_colors=np.column_stack([raw[name][indices[capture_rows]] for name in FIELDS[11:14]])
  expected_colors,_=apply_color_values(original_colors,indices[capture_rows],color_overrides[['front','back'][capture]],report['parameters'].get('colorRepairStrength',1.))
  actual_colors=np.column_stack([vertices[name][capture_rows] for name in FIELDS[11:14]])
  np.testing.assert_array_equal(expected_colors,actual_colors)
  for part,label in enumerate(report['parts']):
   rows=np.flatnonzero((captures==capture)&(labels==part))
   # Every rigid child's Gaussian must follow its parent, including rows beyond
   # a random sample where an accidental local finger/wrist warp could hide.
   sample=rows if label['id'] in rigid_parts else rng.choice(rows,min(500,len(rows)),replace=False)
   source_rows=indices[sample];assert np.all(source_rows<source_count)
   before=np.column_stack([raw[name][source_rows] for name in FIELDS]);after=np.column_stack([vertices[name][sample] for name in FIELDS])
   expected_source=before
   if capture and label['id']=='right_hand' and report['parameters'].get('rightWristRegistration'):
    from right_wrist_transition import preserve_pad_align_fingers
    expected_source,_=preserve_pad_align_fingers(before,report['parameters']['rightWristRegistration'])
   if capture:
    transform=by_part[rigid_parts[label['id']]]['sourceToFrontRaw'] if label['id'] in rigid_parts else label['sourceToFrontRaw']
    expected=transform_gaussians(expected_source,np.array(transform['rotation']),np.array(transform['translation']),transform['scale'])
   else:expected=expected_source
   if capture and label['id']=='left_hand' and report['parameters'].get('leftHandDigitAlignment',{}).get('amount',0):
    from hand_digits import align_left_digits
    expected,_=align_left_digits(expected,report['parameters']['leftHandDigitAlignment'])
   if label['id']=='head' and report['parameters'].get('hairFullness',{}).get('amount',0):
    from hair_shape import deform_head_hair
    expected,_=deform_head_hair(expected,report['parameters']['hairFullness'])
   expected=transform_gaussians(expected,np.diag([1.,-1.,-1.]),np.zeros(3))
   expected[:,11:14],_=apply_color_values(expected[:,11:14],source_rows,color_overrides[['front','back'][capture]],report['parameters'].get('colorRepairStrength',1.))
   error=float(np.max(abs(expected[:,:10]-after[:,:10]))) if len(sample) else 0.
   max_error=max(error,max_error);assert error<3e-6,(label['id'],error)
   np.testing.assert_array_equal(expected[:,11:14],after[:,11:14])
   assert np.all(after[:,10]<=before[:,10]+2e-4),'opacity should not increase'
   verified+=len(sample)
   if label['id'] in rigid_parts:rigid_gaussians_checked[label['id']]+=len(sample)
 result={'passed':True,'gaussianCount':count,'finiteFields':'all 14 fields checked for every Gaussian','maximumQuaternionNormError':float(np.max(abs(norm-1))),'provenanceSamplesChecked':verified,'maximumTransformParameterError':max_error,'originalHashesVerified':original_hashes,'cleanupMasksChecked':True,'hairAppearanceCorrectionChecked':report['parameters'].get('hairFullness',{}).get('amount',0)>0,'allExportedColorsChecked':True}
 result['axialConstraintsChecked']=axial_checked
 result['rigidPartsChecked']=rigid_parts
 result['rigidChildGaussiansChecked']=rigid_gaussians_checked
 result['sourcePartAssignmentsChecked']=assignment_checked
 result['surfaceRestorationsChecked']=restoration_checked
 result['leftHandDigitAlignmentChecked']=report['parameters'].get('leftHandDigitAlignment',{}).get('amount',0)>0
 result['rightWristRegistrationChecked']=bool(report['parameters'].get('rightWristRegistration'))
 (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
