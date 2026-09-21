"""Sparse, reversible Gaussian coverage repair in the original source frame.

Adds a recorded tangential covariance component to existing splats. This is
reconstructed surface coverage, not a new observation or a registration warp.
Centers, opacity, color and source vertex identity remain unchanged.
"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def load_coverage_repairs(paths, sources, root):
    hashes={meta['file']:meta['sha256'] for meta in sources.values()}
    chunks={source:[] for source in sources};evidence=[]
    for path in paths:
        doc=json.loads((Path(root)/path).read_text());source=doc.get('sourceCapture')
        if doc.get('version')!=1 or source not in sources or doc.get('part')!='right_hand':
            raise ValueError('Invalid finger surface coverage selection: '+str(path))
        meta=sources[source]
        if (doc.get('sourceFile')!=meta['file'] or doc.get('sourceSha256')!=meta['sha256']
                or doc.get('sourceHashes')!=hashes):
            raise ValueError('Coverage repair belongs to different source scans: '+str(path))
        with np.load(Path(root)/doc['repairFile'],allow_pickle=False) as pack:
            item={key:pack[key].copy() for key in ['indices','factors','original_parameters']}
        ids=item['indices'];n=len(ids)
        if (ids.ndim!=1 or ids.dtype!=np.dtype('uint32') or n!=doc['uniqueSourceIndices']
                or (n and (ids[-1]>=meta['count'] or np.any(ids[1:]<=ids[:-1])))):
            raise ValueError('Coverage repair requires sorted unique source IDs')
        if (item['factors'].shape!=(n,3) or item['original_parameters'].shape!=(n,7)
                or not all(np.isfinite(item[key]).all() for key in ['factors','original_parameters'])):
            raise ValueError('Invalid coverage repair factors or original parameters')
        if np.any(np.linalg.norm(item['factors'],axis=1)>.006):
            raise ValueError('Coverage repair exceeds the bounded native footprint extension')
        chunks[source].append(item);evidence.append({'document':str(path),**doc})
    merged={}
    for source,items in chunks.items():
        if not items:
            merged[source]={'indices':np.empty(0,np.uint32),'factors':np.empty((0,3)),
                            'original_parameters':np.empty((0,7),np.float32)}
            continue
        ids=np.concatenate([item['indices'] for item in items]);order=np.argsort(ids)
        if np.any(ids[order][1:]==ids[order][:-1]):
            raise ValueError('Coverage repair selections overlap: '+source)
        merged[source]={key:np.concatenate([item[key] for item in items])[order]
                        for key in ['indices','factors','original_parameters']}
    return merged,evidence


def apply_coverage_repair(data, source_indices, repair, strength=1.):
    """Add strength * factor * factor.T to selected native covariances."""
    if not np.isfinite(strength) or not 0<=strength<=1:
        raise ValueError('Coverage strength must be between zero and one')
    out=data.copy();ids=repair['indices']
    if not len(ids) or strength==0:return out,0
    position=np.searchsorted(ids,source_indices);safe=np.minimum(position,len(ids)-1)
    rows=np.flatnonzero((position<len(ids))&(ids[safe]==source_indices));chosen=position[rows]
    if not np.array_equal(data[rows,3:10],repair['original_parameters'][chosen]):
        raise ValueError('Coverage repair original parameters do not match the source')
    if not len(rows):return out,0
    q=data[rows,3:7].astype(float);norm=np.linalg.norm(q,axis=1)
    q[norm<1e-12]=[1,0,0,0];norm[norm<1e-12]=1;q/=norm[:,None]
    rotation=Rotation.from_quat(np.c_[q[:,1:],q[:,0]]).as_matrix()
    axes=rotation*np.exp(data[rows,None,7:10].astype(float))
    factor=repair['factors'][chosen]*np.sqrt(strength)
    vectors,sigma,_=np.linalg.svd(np.concatenate([axes,factor[:,:,None]],axis=2),full_matrices=False)
    vectors[np.linalg.det(vectors)<0,:,0]*=-1
    quat=Rotation.from_matrix(vectors).as_quat()
    out[rows,3:7]=np.c_[quat[:,3],quat[:,:3]]
    out[rows,7:10]=np.log(np.maximum(sigma,np.finfo(float).tiny))
    return out,len(rows)
