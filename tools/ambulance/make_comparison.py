#!/usr/bin/env python3
"""Create a static, same-camera original/cleaned Gaussian comparison page."""
import argparse
import html
import json
from pathlib import Path

from render_review import CAMERAS, DEFAULT_OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--before', default='original')
    parser.add_argument('--after', default='cleaned')
    args = parser.parse_args()
    sections = []
    for camera in CAMERAS:
        a = f'{args.before}-{camera["id"]}.webp'
        b = f'{args.after}-{camera["id"]}.webp'
        for name in (a, b):
            if not (args.directory / name).is_file():
                raise FileNotFoundError(args.directory / name)
        sections.append(f'''<section><h2>{html.escape(camera['label'])}</h2>
<div class="pair"><figure><figcaption>Original</figcaption><a href="{html.escape(a)}"><img src="{html.escape(a)}" alt="Original {html.escape(camera['label'])}"></a></figure>
<figure><figcaption>Cleaned</figcaption><a href="{html.escape(b)}"><img src="{html.escape(b)}" alt="Cleaned {html.escape(camera['label'])}"></a></figure></div></section>''')
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>iPhone ambulance — cleanup comparison</title><style>
*{box-sizing:border-box}body{margin:0;padding:32px;background:#11161c;color:#e8ecf0;font:16px/1.5 system-ui,sans-serif}header{max-width:880px}h1{font-weight:550;font-size:30px}h2{font-weight:500;font-size:19px;margin-top:36px}p{color:#aebcc9}a{color:#a9d2ff}.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}figure{margin:0}figcaption{margin-bottom:8px;color:#aebcc9}img{display:block;width:100%;height:auto;border:1px solid #35414e;border-radius:4px}@media(max-width:800px){body{padding:20px}.pair{grid-template-columns:1fr}}button{padding:9px 14px;background:#273749;color:#e8ecf0;border:1px solid #4b6278;border-radius:6px;cursor:pointer}body.toggle .pair{display:block}body.toggle figure:first-child{display:none}body.toggle.before figure:first-child{display:block}body.toggle.before figure:last-child{display:none}
</style><header><a href="../../../viewers/ambulance/">Open cleaned 3D scene</a><h1>iPhone ambulance cleanup</h1>
<p>Matched camera positions and original lighting with full Gaussian rendering. The cleanup reduces detached surface haze and adds locally fitted support to sparse dark upholstery. Added support is an appearance repair, not recovered photographic detail. Click an image for its full resolution.</p>
<p>The cleaned views also include the viewer's near-plane fix (0.08 scene units, versus 0.02 in the original renders), which removes the near-eye white veil. Files named cleaned-model use the original near plane to isolate geometry changes.</p>
<p><button id="mode">Compare by switching</button> <button id="flip" hidden>Show original</button></p></header>'''
    script = '''<script>const mode=document.querySelector('#mode'),flip=document.querySelector('#flip');mode.onclick=()=>{document.body.classList.toggle('toggle');document.body.classList.remove('before');const active=document.body.classList.contains('toggle');mode.textContent=active?'Compare side by side':'Compare by switching';flip.hidden=!active;flip.textContent='Show original'};flip.onclick=()=>{const before=document.body.classList.toggle('before');flip.textContent=before?'Show cleaned':'Show original'};</script></html>'''
    (args.directory/'index.html').write_text(document+'\n'.join(sections)+script)
    (args.directory/'comparison.json').write_text(json.dumps({'before':args.before,'after':args.after,'cameras':CAMERAS},indent=2)+'\n')


if __name__ == '__main__':
    main()
