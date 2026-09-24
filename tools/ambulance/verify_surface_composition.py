#!/usr/bin/env python3
"""Audit serialized geometry/material edits using their frozen row provenance."""
import argparse
import json
from pathlib import Path

import numpy as np

from cleanup import read_ply, sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    args = ap.parse_args()
    folder = args.directory
    report = json.loads((folder / 'report.json').read_text())
    provenance = np.load(folder / 'row-provenance.npz')
    baseline = Path(report['baseline'])
    threshold = report['remove_alpha_at_most']
    checks = {'baseline_report_frozen': sha256_file(baseline / 'report.json') == report['baseline_report_sha256'],
        'provenance_frozen': sha256_file(folder / 'row-provenance.npz') == report['row_provenance_sha256']}
    counts = {}
    for kind, filename in [('iphone', 'iphone.ply'), ('reference', 'reference-patches.ply')]:
        base, _, _ = read_ply(baseline / filename)
        output, _, _ = read_ply(folder / filename)
        indices = provenance[kind + '_baseline_indices']
        owner = provenance[kind + '_addition_component']
        changed = provenance[kind + '_changed_baseline_indices']
        removed = provenance[kind + '_removed_baseline_indices']
        stream = report['streams'][kind]
        checks[kind + '_baseline_frozen'] = sha256_file(baseline / filename) == stream['baseline_sha256']
        checks[kind + '_output_frozen'] = sha256_file(folder / filename) == stream['output_sha256']
        checks[kind + '_mapping_count'] = len(indices) == len(output) == len(owner)
        retained = indices >= 0
        source_ids = indices[retained]
        checks[kind + '_baseline_partition'] = (np.array_equal(np.sort(np.r_[source_ids, removed]), np.arange(len(base)))
                                              and len(np.unique(source_ids)) == len(source_ids))
        untouched = retained & ~np.isin(indices, changed)
        checks[kind + '_unchanged_rows_byte_exact'] = output[untouched].tobytes() == base[indices[untouched]].tobytes()
        expected = base[source_ids].copy()
        expected_opacity = base['opacity'].copy()
        changes_seen = np.zeros(len(base), bool)
        for component_index, component in enumerate(stream['components']):
            directory = Path(component['directory'])
            patch, _, _ = read_ply(directory / filename)
            checks[f'{kind}_component_{component_index}_frozen'] = sha256_file(directory / filename) == component['file_sha256']
            for field in base.dtype.names:
                different = patch[field][:len(base)] != base[field]
                changes_seen |= different
                use = different[source_ids]
                expected[field][use] = patch[field][source_ids[use]]
                if field == 'opacity':
                    expected_opacity[different] = patch[field][:len(base)][different]
            added = patch[len(base):]
            keep_added = 1/(1+np.exp(-np.clip(added['opacity'].astype(float), -40, 40))) > threshold
            checks[f'{kind}_component_{component_index}_additions_exact'] = (
                output[owner == component_index].tobytes() == added[keep_added].tobytes())
        checks[kind + '_changed_set_exact'] = np.array_equal(np.flatnonzero(changes_seen), changed)
        checks[kind + '_edited_kept_rows_exact'] = expected.tobytes() == output[retained].tobytes()
        expected_removed = np.flatnonzero(1/(1+np.exp(-np.clip(expected_opacity.astype(float), -40, 40))) <= threshold)
        checks[kind + '_only_suppressed_baseline_records_removed'] = np.array_equal(expected_removed, removed)
        checks[kind + '_addition_owner_mapping_valid'] = bool(np.all(owner[retained] == -1)
            and np.all((owner[~retained] >= 0) & (owner[~retained] < len(stream['components']))))
        a = 1/(1+np.exp(-np.clip(output['opacity'].astype(float), -40, 40)))
        checks[kind + '_no_suppressed_records_in_output'] = bool(np.all(a > threshold))
        checks[kind + '_all_fields_finite'] = all(bool(np.isfinite(output[f]).all()) for f in output.dtype.names)
        counts[kind] = {'output': len(output), 'baseline_removed': len(removed),
                       'unchanged_exact': int(untouched.sum()), 'added': int((~retained).sum())}
    for i, patch in enumerate(report['patches']):
        checks[f'patch_{i}_report_frozen'] = sha256_file(Path(patch['directory']) / 'report.json') == patch['report_sha256']
    result = {'status': 'PASS' if all(checks.values()) else 'FAIL',
              'scope': 'Independent serialized-record/provenance audit; visual quality and physical surface bounds are reviewed separately',
              'checks': checks, 'counts': counts}
    (folder / 'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
