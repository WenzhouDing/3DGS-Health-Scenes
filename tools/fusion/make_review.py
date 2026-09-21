#!/usr/bin/env python3
"""Render matched before/after cameras for every revised rigid body region."""
import argparse
from pathlib import Path
import json
import html
import numpy as np
from render_gaussians import ROOT,load_fused,atlas

REGIONS=[
 ('head-neck','Head, hair and neck',['head','neck','torso'],[0,-.035,-.12],.56),
 ('left-shoulder','Left shoulder pad and elbow',['torso','left_upper_arm','left_forearm'],[.27,-.005,.19],.48),
 ('right-shoulder','Right shoulder pad and elbow',['torso','right_upper_arm','right_forearm'],[-.27,-.035,.20],.48),
 ('left-leg','Left knee, lower leg and ankle',['left_thigh','left_shin','left_foot'],[.395,-.02,1.34],.86),
 ('right-leg','Right knee, lower leg and ankle',['right_thigh','right_shin','right_foot'],[-.39,-.02,1.34],.86),
 ('left-foot','Left ankle and foot',['left_shin','left_foot'],[.49,-.04,1.59],.37),
 ('right-foot','Right ankle and foot',['right_shin','right_foot'],[-.49,-.04,1.59],.37),
 ('shorts','Shorts and lower hems',['pelvis','left_thigh','right_thigh','left_shin','right_shin'],[0,-.035,.89],.88),
 ('left-hand','Left palm and back of hand',['left_forearm','left_hand'],[.76,.09,.64],.30),
 ('head-profile','Hair profile and crown',['head','neck'],[0,.015,-.17],.40),
 ('left-sole','Left sole and heel',['left_shin','left_foot'],[.49,-.04,1.59],.37),
 ('right-sole','Right sole and heel',['right_shin','right_foot'],[-.49,-.04,1.59],.37),
 ('right-arm','Right shoulder, arm swivel and hand',['torso','right_upper_arm','right_forearm','right_hand'],[-.47,-.025,.39],.90),
 ('right-collar','Right arm axial collar',['right_upper_arm','right_forearm'],[-.355,-.04,.255],.34),
 ('grey-cable-arm','Grey-cable arm, shoulder and hand',['torso','left_upper_arm','left_forearm','left_hand'],[.48,.035,.37],.86),
 ('grey-cable-collar','Grey-cable arm collar',['left_upper_arm','left_forearm'],[.38,.025,.27],.34),
 ('grey-cable-detail','Grey cable continuity at the collar',['left_upper_arm','left_forearm'],[.43,.04,.29],.32),
 ('left-hand-under','Left hand and wrist from underneath',['left_forearm','left_hand'],[.764,.088,.643],.30),
 ('right-hand-under','Right hand and wrist from underneath',['right_forearm','right_hand'],[-.764,.050,.665],.31),
 ('right-wrist-pad','Right wrist pad, hand and forearm',['right_forearm','right_hand'],[-.68,.055,.57],.36),
 ('right-finger-sides','Right finger side seams',['right_forearm','right_hand'],[-.785,.055,.705],.23),
]
DETAIL_VIEWS={
 'right-finger-sides':[('Palm outer grazing',[-1,-.12,.15]),('Palm inner grazing',[1,-.12,.15]),('Dorsal outer grazing',[-1,.12,.15]),('Dorsal inner grazing',[1,.12,.15]),('Outer side',[-1,0,0]),('Inner side',[1,0,0]),('Below',[0,0,1]),('Distal',[-.626,0,.78])],
 'right-wrist-pad':[('Dorsal pad',[0,1,0]),('Dorsal outer',[-.6,1,.15]),('Dorsal inner',[.6,1,.15]),('Outer profile',[-1,.2,0]),('Inner profile',[1,.2,0]),('Palm',[0,-1,0]),('Along wrist',[-.7,.2,.7]),('Low dorsal',[-.4,.6,.8])],
 'left-hand-under':[('Distal bottom-up',[.7,0,.7]),('Front underside',[.45,-.85,.65]),('Back underside',[.45,.85,.65]),('Below',[0,0,1]),('Front',[0,-1,0]),('Back',[0,1,0]),('Outer side',[1,0,0]),('Inner side',[-1,0,0])],
 'right-hand-under':[('Distal bottom-up',[-.7,0,.7]),('Front underside',[-.45,-.85,.65]),('Back underside',[-.45,.85,.65]),('Below',[0,0,1]),('Front',[0,-1,0]),('Back',[0,1,0]),('Outer side',[-1,0,0]),('Inner side',[1,0,0])],
 'grey-cable-detail':[('Back',[0,1,0]),('Back outer',[.7,1,0]),('Back inner',[-.7,1,0]),('Down cable',[.4,1,.6]),('Upper oblique',[1,1,-.3]),('Front',[0,-1,0])],
 'head-profile':[('Front',[0,-1,0]),('Back',[0,1,0]),('Right',[-1,0,0]),('Left',[1,0,0]),('High rear',[-.6,1,-.65]),('Crown',[.3,.25,-1])],
 'left-sole':[('Sole',[0,.4,1]),('Heel oblique',[.7,1,.5]),('Outer sole',[-1,.4,.7])],
 'right-sole':[('Sole',[0,.4,1]),('Heel oblique',[-.7,1,.5]),('Outer sole',[1,.4,.7])],
}

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--before',type=Path,default=ROOT/'raw/fusion-work/iteration-v2/baseline')
 p.add_argument('--after',type=Path,default=ROOT/'raw/mannequin-fused')
 p.add_argument('--out',type=Path,default=ROOT/'raw/mannequin-fused/render-review')
 p.add_argument('--regions',default='all');p.add_argument('--size',type=int,default=500)
 p.add_argument('--title',default='Alignment review')
 a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 sources={name:load_fused(path) for name,path in [('Before',a.before),('After',a.after)]}
 sections=[]
 for slug,title,parts,center,span in REGIONS:
  if a.regions!='all' and slug not in a.regions.split(','):continue
  captures={}
  for name,(data,labels,_,report) in sources.items():
   chosen=[i for i,part in enumerate(report['parts']) if part['id'] in parts]
   captures[name]=data[np.isin(labels,chosen)]
  cameras={'views':DETAIL_VIEWS[slug]} if slug in DETAIL_VIEWS else {}
  atlas(captures,a.out/(slug+'.png'),center,span,a.size,title=title+' / previous version above, revised below / identical cameras',**cameras)
  sections.append(f'<section id="{slug}"><h2>{title}</h2><a href="{slug}.png" target="_blank"><img loading="lazy" src="{slug}.png" alt="{title}: matching before and after Gaussian renders"></a></section>')
 doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Manikin alignment review</title><style>
body{margin:0;background:#10171c;color:#e4ece9;font:16px/1.5 system-ui,sans-serif}header,section{padding:24px 4vw}header{border-bottom:1px solid #394942;max-width:1100px}h1{font-weight:500}h2{font-size:20px;font-weight:500}a{color:#c3eba2}img{display:block;width:100%;height:auto;border:1px solid #394942}p{color:#b2c4bc;max-width:850px}section{border-bottom:1px solid #394942}</style>
<header><a href="../../../viewers/mannequin-fusion/">Open interactive fusion viewer</a><h1>REVIEW_TITLE</h1><p>Each comparison uses identical cameras and scale. The previous export is the upper row; the revised export is the lower row. These images render the full retained Gaussian data, including its anisotropic shape, color and opacity. Click any image for its full resolution.</p><p>Standard rows show front, back, right side, left side, front oblique and back oblique. Detail rows include crown and sole cameras as labeled. Missing capture coverage and differences in the wig or clothing remain visible.</p><p><a href="../render-review/">Earlier alignment comparison</a></p></header>'''
 doc=doc.replace('REVIEW_TITLE',html.escape(a.title)).replace('<title>Manikin alignment review</title>','<title>'+html.escape(a.title)+'</title>')
 (a.out/'index.html').write_text(doc+'\n'.join(sections)+'</html>')
 (a.out/'cameras.json').write_text(json.dumps({'before':str(a.before),'after':str(a.after),'regions':REGIONS,'detailViews':DETAIL_VIEWS,'size':a.size},indent=2)+'\n')

if __name__=='__main__':main()
