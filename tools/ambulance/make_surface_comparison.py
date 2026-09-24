#!/usr/bin/env python3
"""Build a verified, fixed-camera comparison for all 16 surface inspections."""
import argparse
import hashlib
import html
import json
from pathlib import Path

from pass2_review import CAMERAS, OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=OUT)
    parser.add_argument('--before', default='baseline')
    parser.add_argument('--after', default='refined')
    args = parser.parse_args()
    cameras = json.loads(CAMERAS.read_text())
    manifests = [json.loads((args.directory / f'{label}-manifest.json').read_text())
                 for label in [args.before, args.after]]
    for field in ['camera_file_sha256', 'near', 'resolution', 'fov_axis', 'renderer']:
        if manifests[0][field] != manifests[1][field]:
            raise ValueError(f'Comparison settings differ: {field}')
    if manifests[0]['camera_file_sha256'] != hashlib.sha256(CAMERAS.read_bytes()).hexdigest():
        raise ValueError('Camera recipe differs from rendered manifests')
    sections = []
    for camera in cameras:
        files = []
        for label, manifest in zip([args.before, args.after], manifests):
            records = [r for r in manifest['renders'] if r['camera']['id'] == camera['id']]
            if len(records) != 1 or records[0]['camera'] != camera:
                raise ValueError(f'{label}: missing or mismatched camera {camera["id"]}')
            name = f'{label}-{camera["id"]}.webp'
            path = args.directory / name
            if records[0]['sha256'] != hashlib.sha256(path.read_bytes()).hexdigest():
                raise ValueError(f'Render hash mismatch: {name}')
            files.append(html.escape(name))
        title = html.escape(camera['label'])
        region = html.escape(camera['region'])
        sections.append(f'''<section data-region="{region}" id="{camera['id']}">
<h2>{title}</h2><div class="pair">
<figure class="before"><figcaption>Previous cleanup</figcaption><a href="{files[0]}"><img src="{files[0]}" loading="lazy" alt="Previous cleanup: {title}" width="1000" height="750"></a></figure>
<figure class="after"><figcaption>Surface repairs</figcaption><a href="{files[1]}"><img src="{files[1]}" loading="lazy" alt="Surface repairs: {title}" width="1000" height="750"></a></figure>
</div></section>''')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ambulance · surface repair comparison</title><style>
*{box-sizing:border-box}body{margin:0;padding:28px;background:#10171e;color:#e6edf3;font:16px/1.5 system-ui,sans-serif}header{max-width:1000px}h1{font-size:30px;font-weight:550}h2{font-size:20px;font-weight:500;margin:36px 0 12px}p,figcaption{color:#b7c6d2}a{color:#afd9ff}.controls{position:sticky;top:0;z-index:1;display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:12px 0;background:#10171ef5}button,select{padding:9px 12px;background:#253746;border:1px solid #526879;color:#f2f6f9;border-radius:6px;font:inherit}button{cursor:pointer}button:focus-visible,select:focus-visible,a:focus-visible{outline:2px solid #85cbff;outline-offset:3px}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}figcaption{padding-bottom:8px}img{display:block;width:100%;height:auto;border:1px solid #34444f;border-radius:4px}body.toggle .pair{display:block;max-width:1200px}body.toggle .before{display:none}body.toggle.show-before .before{display:block}body.toggle.show-before .after{display:none}[hidden]{display:none!important}footer{margin-top:48px;color:#b7c6d2;font-size:14px}@media(max-width:780px){body{padding:16px}h1{font-size:26px}.pair{grid-template-columns:1fr}}
</style><header><a href="../../../../viewers/ambulance/?noanim">Open repaired 3D scene</a>
<h1>Ambulance surface repairs</h1>
<p>Sixteen close and grazing views compare the previous cleanup with the new surface repairs. All images render the full Gaussian scene with identical cameras, resolution, field of view, background and clipping. Click any image to inspect it at full resolution.</p>
<p>Reconstructed surface support fills missing coverage using the scan’s observed shapes and materials. It is manual appearance repair; it does not restore missing photographic detail. Fine edge artifacts remain around some straps, light rims and tubing.</p></header>
<div class="controls"><label for="region">Inspect </label><select id="region"><option value="all">All 16 views</option value="architecture">Walls and ceiling</option><option value="upholstery">Seats, pads and bench</option><option value="objects">Stretcher and equipment</option></select><button id="mode" aria-pressed="false">Compare by switching</button><button id="flip" hidden>Show previous cleanup</button></div>
'''
    footer = f'''<footer>Fixed near plane: {manifests[0]['near']} scene units · Resolution: {html.escape(manifests[0]['resolution'])} · Full anisotropic GPU renderer.<br>
<a href="{html.escape(args.before)}-manifest.json">Before render provenance</a> · <a href="{html.escape(args.after)}-manifest.json">After render provenance</a> · <a href="findings.md">Inspection notes</a></footer>'''
    script = '''<script>
const mode=document.querySelector('#mode'),flip=document.querySelector('#flip'),region=document.querySelector('#region');
mode.onclick=()=>{const active=document.body.classList.toggle('toggle');document.body.classList.remove('show-before');mode.textContent=active?'Compare side by side':'Compare by switching';mode.setAttribute('aria-pressed',String(active));flip.hidden=!active;flip.textContent='Show previous cleanup'};
flip.onclick=()=>{const previous=document.body.classList.toggle('show-before');flip.textContent=previous?'Show surface repairs':'Show previous cleanup'};
region.onchange=()=>document.querySelectorAll('section').forEach(section=>section.hidden=region.value!=='all'&&section.dataset.region!==region.value);
</script></html>'''
    (args.directory / 'index.html').write_text(document + '\n'.join(sections) + footer + script)
    (args.directory / 'comparison.json').write_text(json.dumps({
        'before': args.before, 'after': args.after, 'cameras': cameras,
        'sameCameraSettingsVerified': True, 'renderHashesVerified': True,
        'beforeInputSha256': manifests[0]['input_sha256'],
        'afterInputSha256': manifests[1]['input_sha256'],
        'near': manifests[0]['near'], 'resolution': manifests[0]['resolution'],
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
