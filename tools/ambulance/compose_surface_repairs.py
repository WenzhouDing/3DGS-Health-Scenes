#!/usr/bin/env python3
"""Merge baseline-indexed surface edits and physically omit suppressed splats.

Unlike the earlier opacity-only capture transfer, these explicitly reviewed
repairs may change geometry, covariance or material. All field changes are
recorded; independent patches that disagree on a field are rejected. Baseline
and source captures are never overwritten. Visual review is still required.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from cleanup import read_ply, sha256_file, write_ply

STREAMS = {'iphone': 'iphone.ply', 'reference': 'reference-patches.ply'}


def alpha(values):
    return 1 / (1 + np.exp(-np.clip(values.astype(np.float64), -40, 40)))


def finite(records):
    return all(bool(np.isfinite(records[name]).all()) for name in records.dtype.names)


def merge_stream(baseline, patches, name, threshold):
    filename = STREAMS[name]
    base, _, _ = read_ply(baseline / filename)
    if not finite(base):
        raise ValueError(f'Nonfinite baseline: {filename}')
    out = base.copy()
    changed = np.zeros(len(base), bool)
    additions, addition_owners, components = [], [], []
    for owner, directory in enumerate(patches):
        path = directory / filename
        candidate, _, _ = read_ply(path)
        if candidate.dtype != base.dtype or len(candidate) < len(base):
            raise ValueError(f'Patch must retain the baseline prefix and dtype: {path}')
        if not finite(candidate):
            raise ValueError(f'Nonfinite patch: {path}')
        prefix = candidate[:len(base)]
        affected = np.zeros(len(base), bool)
        counts = {}
        for field in base.dtype.names:
            selected = prefix[field] != base[field]
            prior = out[field] != base[field]
            conflict = selected & prior & (out[field] != prefix[field])
            if conflict.any():
                ids = np.flatnonzero(conflict)[:10].tolist()
                raise ValueError(f'Conflicting {name}.{field} edits in {directory}: {ids}')
            out[field][selected] = prefix[field][selected]
            counts[field] = int(selected.sum())
            affected |= selected
        changed |= affected
        extra = candidate[len(base):].copy()
        if len(extra):
            additions.append(extra)
            addition_owners.append(np.full(len(extra), owner, np.int16))
        components.append({'directory': str(directory.resolve()),
            'file_sha256': sha256_file(path), 'changed_baseline_rows': int(affected.sum()),
            'changed_fields': {k: v for k, v in counts.items() if v}, 'added_rows': len(extra)})
    unchanged_exact = out[~changed].tobytes() == base[~changed].tobytes()
    assert unchanged_exact
    all_rows = np.concatenate([out, *additions]) if additions else out
    baseline_ids = np.r_[np.arange(len(base), dtype=np.int64),
                         np.full(len(all_rows)-len(base), -1, dtype=np.int64)]
    owners = np.concatenate([np.full(len(base), -1, np.int16), *addition_owners])
    keep = alpha(all_rows['opacity']) > threshold
    output = all_rows[keep]
    assert finite(output)
    baseline_suppressed = alpha(base['opacity']) <= threshold
    result = {'baseline': str((baseline / filename).resolve()),
        'baseline_sha256': sha256_file(baseline / filename), 'baseline_count': len(base),
        'changed_baseline_rows': int(changed.sum()), 'unmodified_baseline_rows': int((~changed).sum()),
        'unmodified_baseline_records_byte_exact_before_compaction': unchanged_exact,
        'already_suppressed_baseline_rows_removed': int((baseline_suppressed & ~keep[:len(base)]).sum()),
        'newly_suppressed_baseline_rows_removed': int((~baseline_suppressed & ~keep[:len(base)]).sum()),
        'added_rows_before_compaction': len(all_rows)-len(base),
        'total_removed': int((~keep).sum()), 'output_count': len(output), 'components': components}
    mapping = {'baseline_indices': baseline_ids[keep], 'addition_component': owners[keep],
               'changed_baseline_indices': np.flatnonzero(changed),
               'removed_baseline_indices': np.flatnonzero(~keep[:len(base)])}
    return output, mapping, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--patch', type=Path, action='append', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--remove-alpha-at-most', type=float, default=1.01e-8,
                    help='Default removes only the earlier cleanup opacity floor, not faint visible detail')
    args = ap.parse_args()
    if not 0 <= args.remove_alpha_at_most <= 1.01e-8:
        ap.error('Only effectively zero opacity may be globally compacted')
    protected = {args.baseline.resolve(), *(p.resolve() for p in args.patch)}
    if args.out.resolve() in protected:
        ap.error('Output must be distinct from every input')
    if any((args.out / name).exists() for name in [*STREAMS.values(), 'report.json']):
        ap.error('Use a fresh output directory; existing candidate outputs are immutable')
    baseline_hashes = {name: sha256_file(args.baseline / filename) for name, filename in STREAMS.items()}
    for patch in args.patch:
        metadata = json.loads((patch / 'report.json').read_text())
        if Path(metadata['baseline']).resolve() != args.baseline.resolve():
            raise ValueError(f'Patch identifies a different baseline: {patch}')
        for stream, field in [('iphone', 'baseline_phone_sha256'), ('reference', 'baseline_reference_sha256')]:
            recorded = metadata.get(field, metadata.get('sources', {}).get(stream, {}).get('input_sha256'))
            if recorded is None:
                recorded = metadata.get('baseline_hashes', {}).get(stream)
            if recorded != baseline_hashes[stream]:
                raise ValueError(f'Patch baseline hash differs: {patch}/{stream}')
        if metadata.get('generator_sha256') and sha256_file(patch / 'generator.py') != metadata['generator_sha256']:
            raise ValueError(f'Patch generator snapshot changed: {patch}')
        if metadata.get('changes_sha256') and sha256_file(patch / 'changes.npz') != metadata['changes_sha256']:
            raise ValueError(f'Patch change manifest changed: {patch}')
    args.out.mkdir(parents=True, exist_ok=True)
    report = {'status': 'Numerically verified; visual review required',
        'method': 'Explicit baseline-relative geometry/material edits plus physical removal of suppressed rows',
        'baseline': str(args.baseline.resolve()),
        'baseline_report_sha256': sha256_file(args.baseline / 'report.json'),
        'remove_alpha_at_most': args.remove_alpha_at_most,
        'patches': [{'directory': str(p.resolve()), 'report_sha256': sha256_file(p / 'report.json')}
                    for p in args.patch], 'streams': {}, 'remote_publish': False}
    mapping = {}
    for stream, filename in STREAMS.items():
        records, indices, result = merge_stream(args.baseline, args.patch, stream, args.remove_alpha_at_most)
        write_ply(args.out / filename, records)
        # Re-read serialized data: serialization must preserve all chosen fields.
        restored, _, _ = read_ply(args.out / filename)
        assert restored.dtype == records.dtype and restored.tobytes() == records.tobytes()
        result['serialized_records_exact'] = True
        result['output_sha256'] = sha256_file(args.out / filename)
        report['streams'][stream] = result
        mapping.update({f'{stream}_{k}': v for k, v in indices.items()})
        del records, restored
    np.savez_compressed(args.out / 'row-provenance.npz', **mapping)
    report['row_provenance_sha256'] = sha256_file(args.out / 'row-provenance.npz')
    report['combined_count'] = sum(s['output_count'] for s in report['streams'].values())
    report['checks'] = {'all_records_finite': True, 'patch_baseline_hashes_verified': True,
        'conflicting_field_edits_rejected': True,
        'unmodified_baseline_records_exact': all(s['unmodified_baseline_records_byte_exact_before_compaction']
                                               for s in report['streams'].values()),
        'serialized_output_exact': all(s['serialized_records_exact'] for s in report['streams'].values())}
    report['limitations'] = ['Surface edits are manual reconstruction corrections, not recovered ground truth.',
        'Removing floor haze does not certify a complete scene collision model; the previous voxel collider is independent.']
    (args.out / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'combined_count': report['combined_count'], 'checks': report['checks'],
                      'output': str(args.out.resolve())}, indent=2))


if __name__ == '__main__':
    main()
