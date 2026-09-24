#!/usr/bin/env python3
"""Build a verified, matched pass5/pass7 comparison without rendering anything."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'raw/ambulance-cleanup/pass7/review'
BASELINE_MANIFEST = ROOT / 'raw/ambulance-cleanup/pass6/review/final-comparison-manifest.json'
BASELINE_SOURCE = ROOT / 'raw/ambulance-cleanup/pass5/combined-v1/hybrid.sog'
BASELINE_SHA = '6e7aa0f7463a3e297531394aaba7054f0b7c7554099c18b168e0dcce5695da47'
SETTINGS = {'near': .02, 'resolution': '1000x750', 'fov_axis': 'vertical'}


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def local_path(value, origin):
    """Resolve recorded relative paths and old absolute repository prefixes."""
    path = Path(value)
    candidates = [path] if path.is_absolute() else [origin / path, ROOT / path]
    marker = 'raw/ambulance-cleanup/'
    if marker in str(value):
        candidates.append(ROOT / (marker + str(value).split(marker, 1)[1]))
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate.is_relative_to(ROOT) and candidate.is_file():
            return candidate
    raise ValueError(f'Missing repository artifact: {value}')


def relative_url(path):
    return quote(os.path.relpath(path, REVIEW).replace(os.sep, '/'), safe='/')


def check_camera(actual, expected, context):
    # Permit an omitted default up vector, but no changed camera definition.
    a, b = dict(actual), dict(expected)
    a.setdefault('up', [0, 1, 0])
    b.setdefault('up', [0, 1, 0])
    if a != b:
        raise ValueError(f'Camera definition mismatch: {context}')


def check_image(path, expected_hash):
    if not re.fullmatch(r'[0-9a-f]{64}', str(expected_hash)):
        raise ValueError(f'Invalid recorded image SHA256: {path}')
    if sha256(path) != expected_hash:
        raise ValueError(f'Image SHA256 mismatch: {path}')
    with Image.open(path) as image:
        if image.size != (1000, 750):
            raise ValueError(f'Image dimensions differ from 1000x750: {path}')
        image.verify()


def build_data(label, source, camera_file):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', label) or label == 'baseline':
        raise ValueError('Use a simple candidate label distinct from baseline')
    source = source.resolve(strict=True)
    camera_file = camera_file.resolve(strict=True)
    if source.suffix.lower() != '.sog':
        raise ValueError('--source must be the exact exported SOG')
    cameras = json.loads(camera_file.read_text())
    if not isinstance(cameras, list) or len(cameras) != 24:
        raise ValueError('Exactly 24 fixed camera definitions are required')
    by_id = {camera['id']: camera for camera in cameras}
    if len(by_id) != 24 or 'front-wall' not in by_id:
        raise ValueError('Camera IDs must be unique and include front-wall')
    baseline_doc = json.loads(BASELINE_MANIFEST.read_text())
    baseline = baseline_doc['models']['baseline']
    if sha256(BASELINE_SOURCE) != BASELINE_SHA:
        raise ValueError('Preserved baseline SOG differs from restored pass5')
    if set(baseline) != set(by_id):
        raise ValueError('Baseline camera set differs from the 24 requested views')
    candidate_path = REVIEW / f'{label}-manifest.json'
    candidate = json.loads(candidate_path.read_text())
    if not candidate.get('batch_verified') or candidate.get('invalid_reason'):
        raise ValueError('Candidate render batch is incomplete or invalid')
    source_hash = sha256(source)
    if candidate.get('input_sha256') != source_hash:
        raise ValueError('Candidate source SHA256 differs from --source')
    if candidate.get('input_frame') != 'sog' or candidate.get('rotation_degrees') is not None:
        raise ValueError('Candidate must render the exported SOG directly')
    if candidate.get('camera_file_sha256') != sha256(camera_file):
        raise ValueError('Candidate camera-file SHA256 mismatch')
    for field, value in SETTINGS.items():
        if candidate.get(field) != value:
            raise ValueError(f'Candidate rendering setting mismatch: {field}')
    renders = candidate['renders']
    after_by_id = {record['camera']['id']: record for record in renders}
    if len(renders) != 24 or set(after_by_id) != set(by_id):
        raise ValueError('Candidate must contain exactly one render for every camera')
    models = {'baseline': {}, label: {}}
    for camera in cameras:
        key = camera['id']
        before, after = baseline[key], after_by_id[key]
        check_camera(before['camera'], camera, f'baseline/{key}')
        check_camera(after['camera'], camera, f'{label}/{key}')
        if before.get('input_sha256') != BASELINE_SHA:
            raise ValueError(f'Baseline source differs from restored pass5: {key}')
        for field, value in SETTINGS.items():
            if before.get(field) != value:
                raise ValueError(f'Baseline rendering setting mismatch: {key}/{field}')
        before_path = local_path(before['image'], BASELINE_MANIFEST.parent)
        after_path = local_path(after['output'], candidate_path.parent)
        check_image(before_path, before['sha256'])
        check_image(after_path, after['sha256'])
        models['baseline'][key] = {**before, 'image': relative_url(before_path)}
        models[label][key] = {'camera': camera, 'image': relative_url(after_path),
                             'sha256': after['sha256'], 'input_sha256': source_hash,
                             **SETTINGS}
    # Catch a source/render manifest changing while its images were checked.
    candidate_manifest_hash = sha256(candidate_path)
    if json.loads(candidate_path.read_text()) != candidate or sha256(source) != source_hash:
        raise ValueError('Candidate changed during gallery verification')
    return {'candidate': label, 'status': 'Matched comparison; visual approval is separate',
            'default_camera': 'front-wall', 'cameras': cameras, 'models': models,
            'names': {'baseline': 'Before · restored version', label: 'After · ceiling cleanup'},
            'baseline_sha256': BASELINE_SHA, 'candidate_sha256': source_hash,
            'source': str(source), 'settings': SETTINGS,
            'notes': ['The floor and lower cabinet wall are restored to pass5.',
                      'This candidate removes 12 distant source splats for a modest ceiling haze cleanup.',
                      'No new materials, copied texture, or surface fills are introduced.',
                      'Residual corner and ceiling haze remains; these matched images do not imply visual approval.'],
            'provenance': {'baseline_manifest': str(BASELINE_MANIFEST),
                           'baseline_manifest_sha256': sha256(BASELINE_MANIFEST),
                           'baseline_source': str(BASELINE_SOURCE),
                           'candidate_manifest': str(candidate_path),
                           'candidate_manifest_sha256': candidate_manifest_hash,
                           'camera_file': str(camera_file),
                           'camera_file_sha256': sha256(camera_file),
                           'generator_sha256': sha256(Path(__file__))},
            'checks': {'all_24_camera_definitions_match': True,
                       'all_48_image_hashes_and_dimensions_match': True,
                       'candidate_source_hash_matches': True,
                       'baseline_source_is_pass5': True,
                       'near_002_vertical_fov_1000x750': True},
            'viewer_url': relative_url(ROOT / 'viewers/ambulance/index.html')}


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ambulance · ceiling cleanup comparison</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#10151b;color:#e9eef3;font:16px/1.5 system-ui,sans-serif}
main{max-width:1140px;margin:auto;padding:28px 20px 50px}h1{font-size:clamp(23px,4vw,34px);line-height:1.2;margin:0 0 14px}p{color:#bdc8d2;max-width:950px}a{color:#99d7ff}label{display:block;font-weight:600;margin:18px 0 7px}select{width:100%;max-width:700px;padding:10px;background:#1d2833;color:inherit;border:1px solid #617080;border-radius:5px;font:inherit}
.stage{position:relative;aspect-ratio:4/3;background:#000;margin-top:20px;overflow:hidden;border:1px solid #44525f;--split:50%}.stage img{display:block;position:absolute;width:100%;height:100%;object-fit:contain}#after{clip-path:inset(0 0 0 var(--split))}.divider{position:absolute;left:var(--split);top:0;bottom:0;border-left:2px solid white;pointer-events:none}.badge{position:absolute;top:10px;background:#10151bdb;padding:4px 9px;font-size:13px;border-radius:3px}.before-badge{left:10px}.after-badge{right:10px}input[type=range]{width:100%;margin:7px 0 0;accent-color:#94d7ff}.links{display:flex;flex-wrap:wrap;gap:20px;margin-top:15px}.settings{font-size:13px;color:#a7b6c3}.hash{font:12px/1.6 ui-monospace,monospace;overflow-wrap:anywhere}details{margin-top:24px;border-top:1px solid #344250;padding-top:14px}summary{cursor:pointer}#loading{min-height:24px;font-size:14px}noscript{display:block;padding:20px;background:#33291c}
</style><main>
<h1>Ceiling haze comparison</h1>
<p>The floor and lower cabinet wall have been restored. This small ceiling cleanup removes distant stray splats while preserving the captured textures. Some corner and ceiling haze remains.</p>
<p>Choose a viewpoint and move the divider to inspect the two renders.</p>
<label for="camera">Viewpoint</label><select id="camera"></select>
<div class="stage" id="stage"><img id="before" alt="Restored pass5 before view"><img id="after" alt="Ceiling cleanup candidate after view"><span class="divider" aria-hidden="true"></span><span class="badge before-badge">Before · restored version</span><span class="badge after-badge">After · ceiling cleanup</span></div>
<label for="split">Before / after divider</label><input id="split" type="range" min="0" max="100" value="50" aria-label="Divider: left shows before, right shows after">
<div id="loading" role="status" aria-live="polite"></div><p id="settings" class="settings"></p>
<div class="links"><a id="before-link" target="_blank" rel="noopener">Open before image</a><a id="after-link" target="_blank" rel="noopener">Open after image</a><a id="viewer">Open local viewer</a><a href="final-comparison-manifest.json">Verification manifest</a></div>
<details><summary>Comparison provenance</summary><p>All 24 views use identical camera definitions, a 0.02 near plane, vertical field of view, and 1000 × 750 images. Image hashes and the candidate exported SOG hash are verified when this page is built.</p><p class="hash" id="hashes"></p></details>
<noscript>JavaScript is needed for the comparison slider. The verification manifest links every before and after image.</noscript>
</main><script id="comparison-data" type="application/json">__DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('comparison-data').textContent);
const select=document.getElementById('camera'),stage=document.getElementById('stage');
const before=document.getElementById('before'),after=document.getElementById('after'),status=document.getElementById('loading');
for(const camera of D.cameras){select.add(new Option(camera.label+' · '+camera.id,camera.id));}
let revision=0;
async function update(){
 const token=++revision,key=select.value,camera=D.cameras.find(c=>c.id===key);
 const a=D.models.baseline[key],b=D.models[D.candidate][key];
 status.textContent='Loading matched views…';
 before.src=a.image;after.src=b.image;
 before.alt='Before: '+camera.label;after.alt='After candidate: '+camera.label;
 document.getElementById('before-link').href=a.image;document.getElementById('after-link').href=b.image;
 document.getElementById('settings').textContent=key+' · vertical FOV '+camera.fov.toFixed(2)+'° · near 0.02 · 1000 × 750';
 try{await Promise.all([before.decode(),after.decode()]);if(token===revision)status.textContent='Matched views loaded.';}
 catch(error){if(token===revision)status.textContent='An image could not be loaded. Use the image links or serve the repository locally.';}
}
select.value=D.default_camera;select.addEventListener('change',update);
document.getElementById('split').addEventListener('input',event=>stage.style.setProperty('--split',event.target.value+'%'));
document.getElementById('viewer').href=D.viewer_url;
document.getElementById('hashes').textContent='Before SOG SHA256: '+D.baseline_sha256+'\nAfter SOG SHA256: '+D.candidate_sha256;
update();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--label', required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--camera-file', type=Path, required=True)
    args = parser.parse_args()
    data = build_data(args.label, args.source, args.camera_file)
    # Escaping '<' prevents a camera label/path from closing the inert JSON script.
    embedded = json.dumps(data, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    outputs = {'index.html': HTML.replace('__DATA__', embedded),
               'final-comparison-manifest.json': json.dumps(data, indent=2) + '\n'}
    REVIEW.mkdir(parents=True, exist_ok=True)
    for filename, content in outputs.items():
        temporary = REVIEW / (filename + '.tmp')
        temporary.write_text(content)
        temporary.replace(REVIEW / filename)
    print(REVIEW / 'index.html')


if __name__ == '__main__':
    main()
