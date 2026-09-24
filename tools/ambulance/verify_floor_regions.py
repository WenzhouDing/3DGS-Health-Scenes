#!/usr/bin/env python3
"""Independently audit both compacted floor repairs and their capture provenance.

Reads a completed surface composition; never changes scene/component data.
Writes independent-floor-compaction-audit.json using the installer schema.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cleanup import columns, read_ply, sha256_file
from repair_surfaces import WORLD_FROM_RAW as W


STREAMS = {'iphone': 'iphone.ply', 'reference': 'reference-patches.ply'}


def alpha(rows):
    return 1/(1+np.exp(-np.clip(rows['opacity'].astype(float), -40, 40)))


def actual_geometry(rows, plane):
    positions = np.einsum('ij,nj->ni', W, columns(rows, ['x', 'y', 'z']).astype(float))
    quaternion = columns(rows, ['rot_1', 'rot_2', 'rot_3', 'rot_0']).astype(float)
    orientation = np.einsum('ij,njk->nik', W, Rotation.from_quat(quaternion).as_matrix())
    scales = np.exp(columns(rows, ['scale_0', 'scale_1', 'scale_2']).astype(float))
    axes = orientation*scales[:, None, :]
    covariance = np.einsum('nik,njk->nij', axes, axes)
    normal = np.array([-plane[0], 1., -plane[1]])
    norm = np.linalg.norm(normal)
    normal /= norm
    height = (positions[:, 1]-np.einsum('ij,j->i', positions[:, [0, 2]], plane[:2])-plane[2])/norm
    sigma = np.sqrt(np.einsum('i,nij,j->n', normal, covariance, normal))
    return positions, covariance, height, sigma, normal, quaternion


def geometry_report(rows, plane, boundaries, expected_depth):
    positions, covariance, height, sigma, normal, quaternion = actual_geometry(rows, plane)
    boundary_stats = []
    for name, gradient, constant in boundaries:
        norm = np.linalg.norm(gradient)
        direction = gradient/norm
        distance = (np.einsum('ij,j->i', positions, gradient)+constant)/norm
        spread = np.sqrt(np.einsum('i,nij,j->n', direction, covariance, direction))
        margin = distance-3*spread
        boundary_stats.append({'boundary': name, 'minimum_three_sigma_clearance': float(margin.min()),
                               'three_sigma_outside_rows': int(np.sum(margin < 0))})
    checks = {
        'all_values_finite': all(bool(np.isfinite(rows[f]).all()) for f in rows.dtype.names),
        'all_quaternions_unit': bool(np.allclose(np.linalg.norm(quaternion, axis=1), 1, atol=1e-6)),
        'normal_sigma_000015': bool(np.allclose(sigma, .00015, atol=1e-9)),
        'signed_depth_matches': bool(np.allclose(height, expected_depth, atol=2e-7)),
        'all_three_sigma_support_below_fitted_floor': bool(np.all(height+3*sigma < 0)),
        'all_three_sigma_footprints_inside_semantic_boundaries': all(r['three_sigma_outside_rows'] == 0 for r in boundary_stats),
    }
    return {'count': len(rows), 'checks': checks,
            'signed_center_height_range': [float(height.min()), float(height.max())],
            'normal_sigma_range': [float(sigma.min()), float(sigma.max())],
            'upper_three_sigma_height_range': [float((height+3*sigma).min()), float((height+3*sigma).max())],
            'boundaries': boundary_stats}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--composition', type=Path, required=True)
    parser.add_argument('--main-floor', type=Path, default=Path('raw/ambulance-cleanup/pass6/floor-v7'))
    parser.add_argument('--opposite-floor', type=Path, default=Path('raw/ambulance-cleanup/pass6/opposite-floor-v4'))
    parser.add_argument('--original-phone', type=Path, default=Path('raw/ambulance_exp11_boot_sharp.ply'))
    args = parser.parse_args()
    folder = args.composition
    report_bytes = (folder/'report.json').read_bytes()
    composition_hash = hashlib.sha256(report_bytes).hexdigest()
    composition = json.loads(report_bytes)
    provenance = np.load(folder/'row-provenance.npz')
    components = {}
    checks = {}
    for label, path in [('main', args.main_floor), ('opposite', args.opposite_floor)]:
        index = next(i for i, item in enumerate(composition['patches']) if Path(item['directory']).resolve() == path.resolve())
        metadata = json.loads((path/'report.json').read_text())
        components[label] = {'directory': path, 'owner': index, 'metadata': metadata,
                             'manifest': np.load(path/'changes.npz'), 'stream_reports': {}, 'additions': {}}
        checks[f'{label}.component_report_frozen'] = sha256_file(path/'report.json') == composition['patches'][index]['report_sha256']
        checks[f'{label}.generator_snapshot_matches'] = sha256_file(path/'generator.py') == metadata['generator_sha256']
        checks[f'{label}.change_manifest_matches'] = sha256_file(path/'changes.npz') == metadata['changes_sha256']
        checks[f'{label}.same_baseline_as_composition'] = Path(metadata['baseline']).resolve() == Path(composition['baseline']).resolve()
    records, inverse_maps, streams = {}, {}, {}
    for stream, filename in STREAMS.items():
        current, _, _ = read_ply(folder/filename)
        mapping = provenance[f'{stream}_baseline_indices']
        owner = provenance[f'{stream}_addition_component']
        removed = provenance[f'{stream}_removed_baseline_indices']
        base_count = composition['streams'][stream]['baseline_count']
        kept = mapping >= 0
        inverse = np.full(base_count, -1, np.int64)
        inverse[mapping[kept]] = np.flatnonzero(kept)
        kept_ids = np.unique(mapping[kept])
        stream_checks = {
            'mapping_lengths_correct': len(mapping) == len(owner) == len(current),
            'baseline_mapping_unique': len(kept_ids) == int(kept.sum()),
            'kept_removed_partition_baseline': np.array_equal(np.union1d(kept_ids, removed), np.arange(base_count)) and len(np.intersect1d(kept_ids, removed)) == 0,
            'output_sha_matches_composition': sha256_file(folder/filename) == composition['streams'][stream]['output_sha256'],
            'no_effectively_zero_opacity_rows_remain': bool(np.all(alpha(current) > composition['remove_alpha_at_most'])),
        }
        deletion_sets, added_count = [], 0
        for label, component in components.items():
            candidate, _, _ = read_ply(component['directory']/filename)
            manifest = component['manifest']
            prefix = 'phone' if stream == 'iphone' else 'reference'
            deleted = manifest[f'{prefix}_delete_indices']
            changed = manifest[f'{prefix}_changed_indices']
            survivors = changed[inverse[changed] >= 0]
            expected = candidate[base_count:]
            expected = expected[alpha(expected) > composition['remove_alpha_at_most']]
            added = current[(mapping == -1) & (owner == component['owner'])]
            component_checks = {
                'all_delete_ids_physically_absent': bool(np.all(inverse[deleted] == -1)),
                'all_delete_ids_recorded_removed': bool(np.isin(deleted, removed).all()),
                'retained_changed_records_byte_exact': current[inverse[survivors]].tobytes() == candidate[survivors].tobytes(),
                'all_floor_additions_byte_exact': added.tobytes() == expected.tobytes(),
            }
            checks.update({f'{label}.{stream}.{k}': v for k, v in component_checks.items()})
            component['stream_reports'][stream] = {'checks': component_checks,
                'deletion_ids_physically_absent': len(deleted), 'retained_changed_count': len(survivors),
                'added_count': len(added)}
            component['additions'][stream] = added
            deletion_sets.append(deleted)
            added_count += len(added)
        checks.update({f'{stream}.{k}': v for k, v in stream_checks.items()})
        streams[stream] = {'output_count': len(current),
                           'floor_delete_ids_physically_absent': len(np.unique(np.concatenate(deletion_sets))),
                           'floor_additions_count': added_count, 'checks': stream_checks}
        records[stream], inverse_maps[stream] = current, inverse
    checks['main_reference_deletion_count_exactly_12859'] = len(components['main']['manifest']['reference_delete_indices']) == 12859
    checks['opposite_phone_deletion_count_exactly_470'] = len(components['opposite']['manifest']['phone_delete_indices']) == 470
    checks['row_provenance_hash_matches'] = sha256_file(folder/'row-provenance.npz') == composition['row_provenance_sha256']

    original_phone, _, _ = read_ply(args.original_phone)
    checks['original_phone_capture_hash_matches_main_report'] = sha256_file(args.original_phone) == components['main']['metadata']['source_phone_sha256']
    geometry, capture_provenance = {}, {}
    for label, component in components.items():
        meta, path = component['metadata'], component['directory']
        if label == 'main':
            plane = np.asarray(meta['floor_fit']['plane_y_from_xz1'])
            lo, hi = np.asarray(meta['region']['xz_bounds'])
            bench = np.asarray(meta['region']['bench_face_z_from_xy1'])
            extra_boundary = ('bench_face', np.array([bench[0], bench[1], -1.]), bench[2]-meta['region']['bench_face_minimum_front_clearance'])
            base_count, tread_count = 29918, 147294
            sample_path = path/'tread-sampling.npz'
            expected_sample_hash = meta['tread_reconstruction']['sampling_sha256']
            parent_dir = Path(meta['parent_patch']['directory'])
            gain = np.ones(3)
        else:
            plane = np.asarray(meta['floor_y_from_xz1'])
            lo = np.array([meta['region']['x'][0], meta['region']['z'][0]])
            hi = np.array([meta['region']['x'][1], meta['region']['z'][1]])
            cabinet = np.asarray(meta['region']['cabinet_z_from_xy1'])
            extra_boundary = ('cabinet_front', np.array([-cabinet[0], -cabinet[1], 1.]), -cabinet[2]-meta['region']['cabinet_clearance'])
            base_count, tread_count = 13437, 65406
            sample_path = path/'sampling.npz'
            expected_sample_hash = meta['sampling_sha256']
            parent_dir = Path(meta['template_source'])
            gain = np.asarray(meta['shading']['tread_rgb_gain'])
        boundaries = [('x_min', np.array([1.,0,0]), -lo[0]), ('x_max', np.array([-1.,0,0]), hi[0]),
                      ('z_min', np.array([0.,0,1]), -lo[1]), ('z_max', np.array([0.,0,-1]), hi[1]), extra_boundary]
        extra = component['additions']['iphone']
        base, tread = extra[:base_count], extra[base_count:]
        geometry[label] = {'base': geometry_report(base, plane, boundaries, -.009),
                           'tread': geometry_report(tread, plane, boundaries, -.002)}
        checks[f'{label}.base_count_exact'] = len(base) == base_count
        checks[f'{label}.tread_count_exact'] = len(tread) == tread_count
        for kind, entry in geometry[label].items():
            checks.update({f'{label}.{kind}.{k}': v for k, v in entry['checks'].items()})
        sampling = np.load(sample_path)
        ids = sampling['added_original_phone_ids']
        parent, _, _ = read_ply(parent_dir/'iphone.ply')
        source = parent[ids]
        p, _, _, _, _, _ = actual_geometry(tread, plane)
        source_p = np.einsum('ij,nj->ni', W, columns(source, ['x','y','z']).astype(float))
        dc_errors = []
        for i, field in enumerate(['f_dc_0','f_dc_1','f_dc_2']):
            expected = original_phone[field][ids].astype(float)
            if label == 'opposite':
                expected = ((.5+.28209479177387814*expected)*gain[i]-.5)/.28209479177387814
            dc_errors.append(float(np.max(np.abs(tread[field].astype(float)-expected))))
        provenance_checks = {
            'sampling_file_hash_matches_frozen_report': sha256_file(sample_path) == expected_sample_hash,
            'sampling_count_matches_tread': len(ids) == len(tread),
            'all_source_ids_valid_original_capture_rows': bool(np.all((ids >= 0) & (ids < len(original_phone)))),
            'captured_colors_match_declared_gain': max(dc_errors) < 2e-7,
            'source_quaternions_byte_exact': all(np.array_equal(tread[f], source[f]) for f in ['rot_0','rot_1','rot_2','rot_3']),
            'world_translations_match_sampling': bool(np.allclose(p, source_p+sampling['added_world_translations'], atol=2e-7, rtol=0)),
            'tangent_scales_not_increased': all(bool(np.all(tread[f] <= source[f]+1e-7)) for f in ['scale_1','scale_2']),
        }
        checks.update({f'{label}.provenance.{k}': v for k, v in provenance_checks.items()})
        capture_provenance[label] = {'checks': provenance_checks, 'unique_template_rows': len(np.unique(ids)),
            'maximum_dc_error': max(dc_errors),
            'maximum_translation_error': float(np.max(np.abs(p-source_p-sampling['added_world_translations'])))}
        if label == 'opposite':
            confined = component['manifest']['phone_covariance_confined_indices']
            checks['opposite.exact_confinement_ids'] = np.array_equal(confined, [5051843,1225143])
            indices = inverse_maps['iphone'][confined]
            checks['opposite.original_confined_records_survive'] = bool(np.all(indices >= 0))
            if np.any(indices < 0): raise ValueError('Individually confined floor sheets are absent.')
            final = records['iphone'][indices]
            geometry[label]['original_confined_sheets'] = geometry_report(final, plane, boundaries, -.001)
            checks.update({f'opposite.original_sheets.{k}': v for k, v in geometry[label]['original_confined_sheets']['checks'].items()})
            baseline, _, _ = read_ply(Path(composition['baseline'])/'iphone.ply')
            checks['opposite.confined_original_radiance_exact'] = all(np.array_equal(final[f], baseline[f][confined]) for f in ['opacity','f_dc_0','f_dc_1','f_dc_2'])
            fp, fc, _, _, normal, _ = actual_geometry(final, plane)
            bp, bc, _, _, _, _ = actual_geometry(baseline[confined], plane)
            projection = np.eye(3)-np.outer(normal, normal)
            checks['opposite.confined_tangent_covariance_preserved'] = bool(np.max(np.abs(np.einsum('ij,njk,lk->nil', projection, fc-bc, projection))) < 1e-9)
            checks['opposite.confined_tangent_positions_preserved'] = bool(np.max(np.abs(np.einsum('ij,nj->ni', projection, fp-bp))) < 2e-7)
            dark_index = inverse_maps['iphone'][3506802]
            checks['opposite.protected_dark_mark_survives_exact'] = bool(dark_index >= 0 and records['iphone'][dark_index].tobytes() == baseline[3506802].tobytes())

    checks['composition_report_unchanged_during_audit'] = sha256_file(folder/'report.json') == composition_hash
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL',
              'scope': 'Both physically compacted floor regions, captured-grain provenance, original sheet corrections and measured support bounds.',
              'checked_composition_report_sha256': composition_hash,
              'checked_floor_component_reports': {label: sha256_file(c['directory']/'report.json') for label, c in components.items()},
              'checks': checks, 'streams': streams,
              'components': {label: c['stream_reports'] for label, c in components.items()},
              'geometry': geometry, 'capture_provenance': capture_provenance,
              'limitations': ['All lengths are uncalibrated scene units.',
                             'Support bounds refer to actual PLY records before lossy SOG encoding.',
                             'Captured tread repetition and shadow correction are explicit material reconstruction.',
                             'Gaussian 3-sigma support is a convention, not a watertight collision mesh.',
                             'Whole-scene collision geometry and the separate legacy collider are not certified.']}
    (folder/'independent-floor-compaction-audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'failed_checks': [k for k,v in checks.items() if not v],
                      'streams': streams, 'audit': str((folder/'independent-floor-compaction-audit.json').resolve())}, indent=2))
    if result['status'] != 'PASS': raise SystemExit(1)


if __name__ == '__main__': main()
