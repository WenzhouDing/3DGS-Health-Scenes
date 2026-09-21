#!/usr/bin/env python3
"""Select inspected lower-shorts artifacts without changing scans or fused output.

Outputs a sorted original-source index mask plus a human-readable recipe. Apply
that mask only AFTER fusion confidence/coverage weighting; earlier deletion can
activate donor fallback and resurrect low-quality contact-side fragments.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from pipeline import ROOT,sha256_file,read_ply
from render_gaussians import load_fused,atlas

PARTS=('pelvis','left_thigh','right_thigh')

def select(data,labels,sources,indices,parts,native_back_y):
    """Inputs are fused Gaussians in FRONT RAW XYZ, with original vertex IDs."""
    body_indices=[i for i,p in enumerate(parts) if p['id'] in PARTS]
    eligible=(sources==1)&np.isin(labels,body_indices)
    native_y=np.zeros(len(data),np.float32)
    native_y[sources==1]=native_back_y[indices[sources==1]]
    rgb=np.clip(.5+.28209479177387814*data[:,11:14],0,1)
    luminance=np.einsum('ni,i->n',rgb,[.2126,.7152,.0722])
    chroma=rgb.max(axis=1)-rgb.min(axis=1)
    contact=eligible&(data[:,2]>.88)&(data[:,2]<1.13)&(native_y>.10)
    crotch=eligible&(abs(data[:,0])<.225)&(data[:,2]>1.045)&(data[:,2]<1.215)&(luminance>.22)&(chroma<.35)
    return contact|crotch,{'backLowerShortContact':int(contact.sum()),'neutralBrightCrotch':int(crotch.sum()),'bothRules':int((contact&crotch).sum())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,default=ROOT/'raw/fusion-work/cleanup-v3/baseline')
    parser.add_argument('--out',type=Path,default=ROOT/'raw/fusion-work/cleanup-v3/pants')
    parser.add_argument('--mask',type=Path,default=ROOT/'tools/fusion/cleanup-masks/pants-back.npy')
    parser.add_argument('--skip-render',action='store_true')
    args=parser.parse_args()
    data,labels,sources,report=load_fused(args.baseline)
    indices=np.load(args.baseline/'source-vertex-indices.npy')
    source=next(x for x in report['sources'] if x['file'].endswith('mannequin_back_119999.ply'))
    source_path=ROOT/source['file'];digest=sha256_file(source_path)
    if digest!=source['sha256']:raise ValueError('Original back PLY differs from reviewed source')
    columns,_,_=read_ply(source_path)
    rejected,counts=select(data,labels,sources,indices,report['parts'],columns['y'])
    original_indices=np.unique(indices[rejected]).astype(np.uint32)
    if np.any(sources[rejected]!=1):raise AssertionError('Cleanup must be back-only')
    args.mask.parent.mkdir(parents=True,exist_ok=True);np.save(args.mask,original_indices)
    args.out.mkdir(parents=True,exist_ok=True);np.save(args.out/'accepted-reject.npy',rejected)
    evidence={'version':1,'purpose':'remove scanned bed-contact remnants and floating white fragments around lower shorts',
      'sourceCapture':'back','sourceFile':source['file'],'sourceSha256':digest,
      'sourceHashes':{s['file']:s['sha256'] for s in report['sources']},
      'baseline':str(args.baseline.relative_to(ROOT)),'maskFile':str(args.mask.relative_to(ROOT)),
      'maskDtype':'uint32','maskMeaning':'sorted unique ORIGINAL back PLY vertex indices; hash guarded',
      'applyStage':'after coverage confidence weighting and opacity attenuation, before final export; do not trigger donor rescue again',
      'rejectedFusedCount':int(rejected.sum()),'uniqueSourceIndices':len(original_indices),'ruleCounts':counts,
      'rejectedByPart':{p['id']:int(np.sum(rejected&(labels==i))) for i,p in enumerate(report['parts']) if np.any(rejected&(labels==i))},
      'rules':[{'id':'back-lower-short-contact','source':'back','parts':list(PARTS),
                'frontRawZ':{'gt':.88,'lt':1.13},'nativeBackY':{'gt':.10}},
               {'id':'neutral-bright-crotch-remnants','source':'back','parts':list(PARTS),
                'frontRawAbsX':{'lt':.225},'frontRawZ':{'gt':1.045,'lt':1.215},
                'dcLuminance':{'gt':.22},'dcMaxMinusMin':{'lt':.35}}],
      'colorFormula':'RGB=clip(0.5+0.28209479177387814*f_dc,0,1); luminance=0.2126R+0.7152G+0.0722B',
      'review':'Six actual anisotropic Gaussian views: front, back, both sides, front/rear oblique; plus source-isolated crotch closeups. Preserves front fabric/logo, rear cloth folds, and knee screw heads outside the central X bound.',
      'limitations':'Different shorts folds remain because captures have different cloth shapes. This removes artifact contributions without inventing or warping cloth.',
      'proof':str((args.out/'candidate-c.png').relative_to(ROOT)),
      'reproduce':'.venv-fusion/bin/python -B tools/fusion/cleanup_pants.py'}
    (args.mask.parent/'pants-back.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (args.out/'cleanup-evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not args.skip_render:
        crop=(data[:,2]>.56)&(data[:,2]<1.26)&(abs(data[:,0])<.52)
        atlas({'Baseline':data[crop],'Cleaned':data[crop&~rejected]},args.out/'candidate-c.png',
              [0,-.045,.90],.96,540,title='Localized lower-short contact/bright remnant cleanup')
    print(json.dumps({'rejected':int(rejected.sum()),'indices':len(original_indices),'mask':str(args.mask),'counts':counts},indent=2),flush=True)

if __name__=='__main__':main()
