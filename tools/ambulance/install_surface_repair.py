#!/usr/bin/env python3
"""Install an exact locally reviewed surface repair; no remote operations."""
import argparse
import json
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from cleanup import ROOT, sha256_file


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--directory', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--source-sha256', required=True)
    ap.add_argument('--camera-file', type=Path, required=True)
    args = ap.parse_args()
    folder, review = args.directory.resolve(), args.review.resolve()
    source = folder / 'hybrid.sog'
    installed_report = json.loads((folder / 'report.json').read_text())
    reviewed_report = folder / 'reviewed-composition-report.json'
    report_path = folder / 'report.json'
    if installed_report.get('accepted_from_report_sha256'):
        assert sha256_file(reviewed_report) == installed_report['accepted_from_report_sha256']
        report_path = reviewed_report
    report = json.loads(report_path.read_text())
    baseline = Path(report['baseline'])
    previous = sha256_file(baseline / 'hybrid.sog')
    canonical = ROOT / 'viewers/ambulance/index.sog'
    assert sha256_file(source) == args.source_sha256
    assert sha256_file(canonical) in {previous, args.source_sha256}
    assert sha256_file(baseline / 'report.json') == report['baseline_report_sha256']
    assert all(report['checks'].values())
    independent = json.loads((folder / 'verification.json').read_text())
    assert independent['status'] == 'PASS' and all(independent['checks'].values())
    floor_audit_path = folder / 'independent-floor-compaction-audit.json'
    floor_audit = json.loads(floor_audit_path.read_text())
    assert floor_audit['status'] == 'PASS' and all(floor_audit['checks'].values())
    assert floor_audit['checked_composition_report_sha256'] == sha256_file(report_path)
    for name, filename in [('iphone', 'iphone.ply'), ('reference', 'reference-patches.ply')]:
        stream = report['streams'][name]
        assert sha256_file(baseline / filename) == stream['baseline_sha256']
        assert sha256_file(folder / filename) == stream['output_sha256']
        for component in stream['components']:
            assert sha256_file(Path(component['directory']) / filename) == component['file_sha256']
    for patch in report['patches']:
        directory = Path(patch['directory'])
        assert sha256_file(directory / 'report.json') == patch['report_sha256']
        component = json.loads((directory / 'report.json').read_text())
        if component.get('generator_sha256'):
            assert sha256_file(directory / 'generator.py') == component['generator_sha256']
        if component.get('changes_sha256'):
            assert sha256_file(directory / 'changes.npz') == component['changes_sha256']
    assert sha256_file(folder / 'row-provenance.npz') == report['row_provenance_sha256']
    with zipfile.ZipFile(source) as archive:
        assert archive.testzip() is None
        meta = json.loads(archive.read('meta.json'))
        assert meta['count'] == report['combined_count'] and meta['shN']['bands'] == 3
    comparison = json.loads((review / 'final-comparison-manifest.json').read_text())
    qa = json.loads((review / 'final-qa.json').read_text())
    assert comparison['candidate_sha256'] == args.source_sha256
    assert comparison['baseline_sha256'] == previous
    candidate = comparison['models'][comparison['candidate']]
    camera_definitions = {c['id']: c for c in comparison['cameras']}
    expected_definitions = {c['id']: c for c in json.loads(args.camera_file.read_text())}
    cameras = set(camera_definitions)
    assert len(cameras) == len(comparison['cameras']) == len(expected_definitions) == 24
    assert camera_definitions == expected_definitions and set(candidate) == cameras
    for camera_id, image in candidate.items():
        assert image['camera'] == expected_definitions[camera_id]
        assert image['near'] == .02 and image['fov_axis'] == 'vertical'
        assert image['resolution'] == '1000x750'
        assert image['input_sha256'] == args.source_sha256
        assert sha256_file(review / image['image']) == image['sha256']
    assert qa['source_sha256'] == args.source_sha256 and qa['baseline_sha256'] == previous
    assert qa['view_count'] == len(cameras) and qa['visible_regressions_found'] == []
    assert qa['batch_verified'] and qa['image_hashes_verified'] and qa['matched_camera_and_clipping_verified']
    assert qa['verdict'].startswith('Accepted')
    # All gates above are read-only. Commit local assets only after they pass.
    if report_path != reviewed_report:
        shutil.copy2(report_path, reviewed_report)
    report['accepted_from_report_sha256'] = sha256_file(reviewed_report)
    report['status'] = f'Accepted after targeted geometry/material and final {len(cameras)}-view exported-model review; local only'
    report['limitations'] = qa['limitations']
    report['browser_export'] = {'file': 'viewers/ambulance/index.sog', 'bytes': source.stat().st_size,
        'sha256': args.source_sha256, 'gaussians': meta['count'], 'SH_bands': 3,
        'converter': '@playcanvas/splat-transform 3.4.2', 'near_clip_minimum': .08}
    report['visual_review'] = {'camera_count': len(cameras), 'candidate': comparison['candidate'], 'near_plane': .02,
        'camera_file_sha256': sha256_file(args.camera_file),
        'comparison': str((review / 'index.html').relative_to(ROOT)),
        'comparison_manifest_sha256': sha256_file(review / 'final-comparison-manifest.json'),
        'final_qa_sha256': sha256_file(review / 'final-qa.json'),
        'previous_accepted_model_sha256': previous}
    report['collision_status'] = {'scope': 'Targeted floor geometry cleanup; not whole-scene collision certification',
        'suppressed_rows_physically_omitted': sum(s['total_removed'] for s in report['streams'].values()),
        'floor_compaction_audit_sha256': sha256_file(floor_audit_path),
        'floor_reference_haze_rows_physically_omitted': floor_audit['streams']['reference']['floor_delete_ids_physically_absent'],
        'legacy_voxel_file': 'viewers/ambulance/index.voxel.json',
        'legacy_voxel_updated': False,
        'note': 'The earlier separate voxel collider does not incorporate this surface repair.'}
    report['remote_publish'] = False
    (folder / 'report.json').write_text(json.dumps(report, indent=2)+'\n')
    temporary = canonical.with_suffix('.sog.tmp')
    shutil.copy2(source, temporary)
    assert sha256_file(temporary) == args.source_sha256
    temporary.replace(canonical)
    shutil.copy2(folder / 'report.json', ROOT / 'viewers/ambulance/cleanup-report.json')
    with Image.open(review / candidate['default']['image']) as image:
        image.convert('RGB').save(ROOT / 'assets/ambulance.jpg', quality=95)
    landing = ROOT / 'index.html'
    html, count = re.subn(r'(<h3>Ambulance &mdash; iPhone</h3><span class="size">).*?(</span>)',
        lambda m: m[1]+f"{meta['count']/1e6:.1f}M splats · {source.stat().st_size/1048576:.0f} MB"+m[2],
        landing.read_text(), count=1)
    assert count == 1
    landing.write_text(html)
    readme = ROOT / 'README.md'
    text = readme.read_text().replace('/pass5/review/', '/pass6/review/')
    text = text.replace('current corner/wall repair', 'current floor/wall surface repair')
    readme.write_text(text)
    assert sha256_file(canonical) == args.source_sha256
    print(json.dumps({'installed': str(canonical), 'sha256': args.source_sha256,
        'gaussians': meta['count'], 'bytes': source.stat().st_size,
        'reviewed_views': len(cameras), 'remote_publish': False}, indent=2))


if __name__ == '__main__':
    main()
