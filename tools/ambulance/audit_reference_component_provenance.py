#!/usr/bin/env python3
"""Audit component capture identities and recorded generator snapshots.

Complements the all-record verifier without treating unavailable historical
snapshots as verified. Does not modify any component or source asset.
"""
import argparse
import json
from pathlib import Path
from cleanup import sha256_file
from verify_reference_composition import EXPECTED_PHONE, EXPECTED_REFERENCE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    args = ap.parse_args()
    combined = json.loads((args.directory / 'report.json').read_text())
    sources = {
        'iphone': {'recorded': combined['iphone_sha256'],
                   'actual': sha256_file(Path(combined['iphone'])),
                   'expected': EXPECTED_PHONE},
        'reference': {'recorded': combined['reference_sha256'],
                      'actual': sha256_file(Path(combined['reference'])),
                      'expected': EXPECTED_REFERENCE},
    }
    rows = []
    for entry in combined['components']:
        directory = Path(entry['directory'])
        report = json.loads((directory / 'report.json').read_text())
        source_checks = {name: report.get(name + '_sha256') == data['expected']
                         for name, data in sources.items()}
        recorded = report.get('generator_sha256')
        snapshot = directory / 'generator.py'
        if recorded is None:
            generator = {'status': 'unavailable: no recorded generator hash',
                         'snapshot_present': snapshot.is_file(), 'verified': None}
        elif not snapshot.is_file():
            generator = {'status': 'FAIL: recorded generator snapshot is missing',
                         'recorded_sha256': recorded, 'verified': False}
        else:
            actual = sha256_file(snapshot)
            generator = {'status': 'PASS' if actual == recorded else 'FAIL: hash mismatch',
                         'snapshot': str(snapshot), 'recorded_sha256': recorded,
                         'actual_sha256': actual, 'verified': actual == recorded}
        rows.append({'directory': str(directory), 'status_recorded': report.get('status'),
                     'source_identity_checks': source_checks, 'generator': generator})
    all_sources = all(d['recorded'] == d['actual'] == d['expected'] for d in sources.values())
    all_component_sources = all(all(r['source_identity_checks'].values()) for r in rows)
    all_recorded_generators = all(r['generator']['verified'] is not False for r in rows)
    missing = [r['directory'] for r in rows if r['generator']['verified'] is None]
    result = {
        'status': 'PASS' if all_sources and all_component_sources and all_recorded_generators else 'FAIL',
        'scope': 'Capture identity plus recorded component-generator snapshots; complements verification.json.',
        'sources': sources,
        'checks': {'source_files_match_expected_captures': all_sources,
                   'all_component_source_hashes_match_expected_captures': all_component_sources,
                   'all_recorded_generator_snapshots_match': all_recorded_generators},
        'counts': {'components': len(rows), 'verified_generator_snapshots': sum(r['generator']['verified'] is True for r in rows),
                   'historical_generator_hash_unavailable': len(missing)},
        'generator_unavailable_components': missing,
        'components': rows,
    }
    (args.directory / 'component-provenance-verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ['status', 'checks', 'counts', 'generator_unavailable_components']}, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
