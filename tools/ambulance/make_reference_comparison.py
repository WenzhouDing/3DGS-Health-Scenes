#!/usr/bin/env python3
"""Build the final local review only after all matched candidate renders exist."""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REVIEW = ROOT / 'raw/ambulance-cleanup/pass3/review'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect(label, review, strict=False, source_cache=None):
    records = {}
    manifests = []
    for directory in [review, review.parent / 'default-review']:
        path = directory / f'{label}-manifest.json'
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        if manifest.get('invalid_reason'):
            raise ValueError(f'Invalid render batch: {path}')
        if strict:
            if not manifest.get('batch_verified'):
                raise ValueError(f'Candidate batch was not verified: {path}')
            source = Path(manifest['input'])
            if source_cache is None:
                source_cache = {}
            if str(source) not in source_cache:
                source_cache[str(source)] = sha(source)
            if source_cache[str(source)] != manifest['input_sha256']:
                raise ValueError(f'Candidate source differs from rendered source: {source}')
        manifests.append(manifest)
        for record in manifest['renders']:
            image = Path(record['output'])
            if sha(image) != record['sha256']:
                raise ValueError(f'Image differs from render manifest: {image}')
            records[record['camera']['id']] = {
                'camera': record['camera'],
                'image': os.path.relpath(image, review),
                'sha256': record['sha256'],
                'resolution': manifest['resolution'],
                'near': manifest['near'],
                'fov_axis': manifest.get('fov_axis'),
                'input_sha256': manifest['input_sha256'],
            }
    return records, manifests


def make_roof_comparison(review, models, candidate):
    from PIL import Image, ImageDraw, ImageFont
    canvas = Image.new('RGB', (1616, 696), '#10151b')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 27)
        small = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 18)
    except OSError:
        font = small = ImageFont.load_default()
    for i, (label, caption) in enumerate([('original', 'Original iPhone'),
                                         (candidate, 'After: observed roof material')]):
        draw.text((i * 816 + 16, 14), caption, font=font, fill='#e9eef3')
        path = review / models[label]['ceiling-front']['image']
        with Image.open(path) as image:
            canvas.paste(image.resize((800, 600), Image.Resampling.LANCZOS), (i * 816, 60))
    draw.text((16, 671), 'Same camera and clipping · full Gaussian renders · resized for comparison only',
              font=small, fill='#bdc9d4')
    canvas.save(review / 'comparison-roof.jpg', quality=96)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', default='reference-repair')
    parser.add_argument('--review', type=Path, default=REVIEW)
    parser.add_argument('--notes', type=Path)
    args = parser.parse_args()
    cameras = json.loads((ROOT / 'tools/ambulance/pass2-cameras.json').read_text())
    default = json.loads((args.review.parent / 'diagnostic-cameras.json').read_text())
    cameras = default + cameras
    models = {}
    source_cache = {}
    for label in ['original', 'baseline', 'refined', 'insta-aligned', args.candidate]:
        models[label], _ = collect(label, args.review, label == args.candidate, source_cache)
    for camera in cameras:
        key = camera['id']
        for label in ['original', args.candidate, 'insta-aligned']:
            if key not in models[label]:
                raise ValueError(f'Missing required final render: {label}/{key}')
        a, b = models['original'][key], models[args.candidate][key]
        for field in ['position', 'target', 'fov']:
            if a['camera'][field] != b['camera'][field]:
                raise ValueError(f'Candidate camera mismatch: {key}/{field}')
        if a['camera'].get('up', [0, 1, 0]) != b['camera'].get('up', [0, 1, 0]):
            raise ValueError(f'Candidate up direction mismatch: {key}')
        if a['fov_axis'] != b['fov_axis']:
            raise ValueError(f'Candidate field-of-view axis mismatch: {key}')
        if a['near'] != b['near'] or a['resolution'] != b['resolution']:
            raise ValueError(f'Candidate render settings mismatch: {key}')
    candidate_hashes = {r['input_sha256'] for r in models[args.candidate].values()}
    if len(candidate_hashes) != 1:
        raise ValueError('Candidate views do not use the same frozen source.')
    notes = json.loads(args.notes.read_text()) if args.notes else {}
    names = {'original': 'Original iPhone', 'baseline': 'First cleanup (historical)',
             'refined': 'Rejected gray-roof refinement', 'insta-aligned': 'Aligned Insta360 reference',
             args.candidate: 'Reference-guided repair'}
    data = {'candidate': args.candidate, 'names': names, 'cameras': cameras,
            'models': models, 'notes': notes,
            'candidate_sha256': next(iter(candidate_hashes))}
    (args.review / 'final-comparison-manifest.json').write_text(json.dumps(data, indent=2) + '\n')
    page = HTML.replace('__DATA__', json.dumps(data).replace('<', '\\u003c'))
    existing = args.review / 'index.html'
    if existing.exists() and not (args.review / 'reassessment.html').exists():
        (args.review / 'reassessment.html').write_text(existing.read_text())
    existing.write_text(page)
    make_roof_comparison(args.review, models, args.candidate)
    print(existing)


HTML = '''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ambulance · reference-guided repair</title>
<style>
:root{color-scheme:dark;font:15px/1.5 system-ui;background:#10151b;color:#e9eef3}*{box-sizing:border-box}body{margin:0}main{max-width:1450px;margin:auto;padding:28px}h1{font-size:30px;letter-spacing:-.02em;margin:0 0 8px}p{max-width:980px;color:#bdc9d4;margin:8px 0 18px}a{color:#8fcef6}button,select{font:inherit;background:#1a2632;border:1px solid #475868;border-radius:7px;color:#fff;padding:9px 12px}button{cursor:pointer}label{display:grid;gap:5px;color:#b9c6d1;font-size:13px}.controls{display:flex;gap:16px;align-items:end;flex-wrap:wrap;margin:23px 0 18px}.controls label:first-child{flex:1;min-width:260px}#stage{position:relative;aspect-ratio:4/3;width:100%;background:#000;overflow:hidden;border:1px solid #415265;border-radius:10px}#stage img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}#after{clip-path:inset(0 50% 0 0)}#divider{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid #fff;pointer-events:none}.tag{position:absolute;top:12px;background:#10151bd9;border:1px solid #ffffff40;border-radius:6px;padding:5px 9px;font-size:13px;pointer-events:none}#afterTag{left:12px}#beforeTag{right:12px}#range{width:100%;margin:16px 0 3px;accent-color:#8fcef6}#status{min-height:28px;color:#cfdbe5;margin-top:12px}.hint{font-size:13px;color:#9baebc}details{border-top:1px solid #36424e;margin-top:22px;padding-top:15px}summary{cursor:pointer;color:#cbd9e5}#reference{width:100%;display:block;margin-top:14px;border-radius:8px}#referencePanel{max-width:1100px}.links{display:flex;gap:18px;flex-wrap:wrap;font-size:14px}code{overflow-wrap:anywhere;font-size:12px}#unavailable{position:absolute;inset:0;display:none;align-items:center;justify-content:center;background:#10151bed}footer{color:#97aab9;font-size:12px;margin-top:30px}@media(max-width:700px){main{padding:16px}h1{font-size:25px}.controls{gap:10px}.controls label{width:100%}.tag{font-size:11px;max-width:44%}}
</style>
<main>
<h1>Ambulance: reference-guided repair</h1>
<p>Compare the original iPhone capture with the revised result from the same viewpoint. The Insta360 capture provides observed surface material and geometry; it is also available below as a reference.</p>
<div class="links"><a href="/viewers/ambulance/">Open local 3D viewer</a><a href="findings.md">Review findings and remaining artifacts</a><a href="reassessment.html">Earlier comparison and rejected refinement</a></div>
<div class="controls">
<label>View<select id="camera"></select></label>
<label>Before<select id="beforeSelect"></select></label>
<label>After<select id="afterSelect"></select></label>
<button id="previous" aria-label="Previous view">←</button><button id="next" aria-label="Next view">→</button>
</div>
<div id="stage"><img id="before" alt="Original comparison view"><img id="after" alt="Repaired comparison view"><div id="divider"></div><span id="afterTag" class="tag"></span><span id="beforeTag" class="tag"></span><div id="unavailable">This historical version has no render for this view.</div></div>
<input id="range" type="range" min="0" max="100" value="50" aria-label="Reveal amount of the repaired view">
<div class="hint">Drag the divider or use the slider. Left: after · right: before.</div>
<div id="status" role="status"></div>
<details id="referencePanel" open><summary>Aligned Insta360 reference</summary><p class="hint">Same physical viewpoint, preserving the reference capture's original appearance. Movable equipment and bedding may differ between captures.</p><img id="reference" alt="Aligned Insta360 reference"><a id="referenceLink" href="">Open reference at full resolution</a></details>
<details><summary>Comparison provenance</summary><p>All before/after images use full Gaussian rendering at 1000 × 750 with the same camera, field of view and near plane of 0.02. The native Insta360 reference uses the measured alignment, transformed camera up direction and a scale-adjusted near distance. No image denoising, recoloring or photographic cleanup is applied to these review images.</p><p>Final model SHA256: <code id="hash"></code></p><a href="final-comparison-manifest.json">Camera settings, source and image hashes</a></details>
<footer>Local review artifact. Historical failed attempts remain available so improvements can be judged against the original, not only against previous experiments.</footer>
</main>
<script>
const D=__DATA__;const $=id=>document.getElementById(id);const camera=$('camera'),beforeSelect=$('beforeSelect'),afterSelect=$('afterSelect');
for(const c of D.cameras){const o=new Option(c.label||c.id,c.id);camera.add(o)}
for(const key of ['original','baseline','refined','insta-aligned'])beforeSelect.add(new Option(D.names[key],key));
for(const key of [D.candidate,'original','baseline','refined','insta-aligned'])afterSelect.add(new Option(D.names[key],key));
$('hash').textContent=D.candidate_sha256;
function update(){const c=camera.value,b=D.models[beforeSelect.value]?.[c],a=D.models[afterSelect.value]?.[c],r=D.models['insta-aligned'][c];$('unavailable').style.display=b&&a?'none':'flex';if(b)$('before').src=b.image;if(a)$('after').src=a.image;$('beforeTag').textContent=D.names[beforeSelect.value];$('afterTag').textContent=D.names[afterSelect.value];$('reference').src=r.image;$('referenceLink').href=r.image;$('status').textContent=D.notes[c]||'Inspect surface texture, edges and nearby equipment in both views; consult the findings for remaining limitations.';history.replaceState(null,'','#'+encodeURIComponent(c))}
function reveal(v){v=Math.min(100,Math.max(0,v));$('range').value=v;$('after').style.clipPath=`inset(0 ${100-v}% 0 0)`;$('divider').style.left=v+'%'}
$('range').oninput=e=>reveal(+e.target.value);for(const el of [camera,beforeSelect,afterSelect])el.onchange=update;
$('previous').onclick=()=>{camera.selectedIndex=(camera.selectedIndex-1+D.cameras.length)%D.cameras.length;update()};$('next').onclick=()=>{camera.selectedIndex=(camera.selectedIndex+1)%D.cameras.length;update()};
let dragging=false;const stage=$('stage');stage.onpointerdown=e=>{dragging=true;stage.setPointerCapture(e.pointerId);move(e)};stage.onpointermove=e=>{if(dragging)move(e)};stage.onpointerup=()=>dragging=false;function move(e){const r=stage.getBoundingClientRect();reveal((e.clientX-r.left)/r.width*100)}
const initial=decodeURIComponent(location.hash.slice(1));if(D.cameras.some(c=>c.id===initial))camera.value=initial;update();
</script></html>'''


if __name__ == '__main__':
    main()
