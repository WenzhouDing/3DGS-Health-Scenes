#!/usr/bin/env python3
"""Constrain two individually attributed residual floor sheets, preserving radiance."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file, write_ply
from repair_surfaces import WORLD_FROM_RAW as W
from repair_opposite_floor_reference import NORMAL, point_geometry, covariances


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists(): ap.error('Output must be new.')
    ids=np.array([5051843,1225143],np.int64)
    parent=json.loads((args.parent/'report.json').read_text())
    phone,_,_=read_ply(args.parent/'iphone.ply')
    original=phone[ids].copy()
    positions,height=point_geometry(original)
    cov,sn,_=covariances(original)
    projection=np.eye(3)-np.outer(NORMAL,NORMAL)
    target_positions=positions+(-.001-height)[:,None]*NORMAL
    target_cov=np.einsum('ij,njk,lk->nil',projection,cov,projection)+.00015**2*np.outer(NORMAL,NORMAL)
    values,axes=np.linalg.eigh(target_cov)
    axes[np.linalg.det(axes)<0,:,0]*=-1
    raw_positions=np.einsum('ij,nj->ni',W.T,target_positions)
    quat=Rotation.from_matrix(np.einsum('ij,njk->nik',W.T,axes)).as_quat()
    out=phone.copy()
    for i,f in enumerate(['x','y','z']): out[f][ids]=raw_positions[:,i]
    for i,f in enumerate(['scale_0','scale_1','scale_2']): out[f][ids]=.5*np.log(np.maximum(values[:,i],1e-12))
    for i,f in enumerate(['rot_1','rot_2','rot_3','rot_0']): out[f][ids]=quat[:,i]
    unchanged=np.ones(len(out),bool);unchanged[ids]=False
    actual_p,actual_height=point_geometry(out[ids])
    actual_cov,actual_sn,_=covariances(out[ids])
    tangent_error=np.max(np.abs(np.einsum('ij,njk,lk->nil',projection,actual_cov-cov,projection)))
    checks={'all_other_parent_records_byte_exact':out[unchanged].tobytes()==phone[unchanged].tobytes(),
            'attributed_color_and_opacity_byte_exact':all(np.array_equal(out[f][ids],phone[f][ids]) for f in ['opacity','f_dc_0','f_dc_1','f_dc_2']),
            'tangential_covariance_preserved':bool(tangent_error<1e-9),
            'tangential_positions_preserved':bool(np.allclose(np.einsum('ij,nj->ni',projection,actual_p-positions),0,atol=2e-7)),
            'normal_sigma_000015':bool(np.allclose(actual_sn,.00015,atol=1e-9)),
            'all_three_sigma_below_floor':bool(np.all(actual_height+3*actual_sn<0)),
            'protected_dark_mark_3506802_byte_exact':out[3506802].tobytes()==phone[3506802].tobytes()}
    assert all(checks.values()),checks
    args.out.mkdir(parents=True)
    write_ply(args.out/'iphone.ply',out)
    for filename in ['reference-patches.ply','floor-additions.ply','sampling.npz']:
        shutil.copy2(args.parent/filename,args.out/filename)
    manifest=dict(np.load(args.parent/'changes.npz'))
    manifest['phone_changed_indices']=np.union1d(manifest['phone_changed_indices'],ids)
    manifest['phone_covariance_confined_indices']=ids
    np.savez_compressed(args.out/'changes.npz',**manifest)
    shutil.copy2(__file__,args.out/'generator.py')
    report=dict(parent)
    report['status']='Unreviewed final two-sheet confinement atop opposite-floor V3'
    report['parent_patch']={'directory':str(args.parent.resolve()),'report_sha256':sha256_file(args.parent/'report.json'),
                            'iphone_sha256':sha256_file(args.parent/'iphone.ply')}
    report['residual_confinement']={'original_phone_ids':ids.tolist(),'normal_depth':-.001,'normal_sigma':.00015,
                                  'original_signed_heights':height.tolist(),'original_normal_sigmas':sn.tolist(),
                                  'maximum_tangential_covariance_error':float(tangent_error),
                                  'actual_upper_three_sigma_heights':(actual_height+3*actual_sn).tolist(),
                                  'evidence':'raw/ambulance-cleanup/pass6/floor-audit/opposite-v3-residual-attribution.json',
                                  'protected_original_dark_mark':3506802}
    report['counts']={**parent['counts'],'individual_original_sheets_confined':2}
    report['checks']={**checks,'all_parent_additions_unchanged':True,
                      'reference_byte_exact_to_parent':sha256_file(args.out/'reference-patches.ply')==sha256_file(args.parent/'reference-patches.ply')}
    report['generator_sha256']=sha256_file(args.out/'generator.py')
    report['changes_sha256']=sha256_file(args.out/'changes.npz')
    report['frozen_ply_hashes']={f:sha256_file(args.out/f) for f in ['iphone.ply','reference-patches.ply']}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['residual_confinement'],indent=2))


if __name__=='__main__':main()
