"""Sparse, reversible DC-color repairs keyed by original source vertex rows."""
import json
from pathlib import Path
import numpy as np


def load_color_overrides(paths, sources, root):
    hashes = {meta['file']: meta['sha256'] for meta in sources.values()}
    chunks = {source: [] for source in sources}
    evidence = []
    for path in paths:
        document = json.loads((Path(root) / path).read_text())
        source = document.get('sourceCapture')
        if document.get('version') != 1 or source not in sources:
            raise ValueError('Invalid color repair: ' + str(path))
        meta = sources[source]
        if (document.get('sourceFile') != meta['file'] or document.get('sourceSha256') != meta['sha256'] or
                document.get('sourceHashes') != hashes):
            raise ValueError('Color repair belongs to different source scans: ' + str(path))
        with np.load(Path(root) / document['overrideFile'], allow_pickle=False) as packed:
            item = {name: packed[name].copy() for name in ['indices', 'f_dc', 'original_f_dc']}
        indices = item['indices']
        if (indices.ndim != 1 or indices.dtype != np.dtype('uint32') or len(indices) != document['uniqueSourceIndices'] or
                (len(indices) and (indices[-1] >= meta['count'] or np.any(indices[1:] <= indices[:-1])))):
            raise ValueError('Color repair requires sorted unique uint32 source rows: ' + str(path))
        for name in ['f_dc', 'original_f_dc']:
            if item[name].shape != (len(indices), 3) or not np.isfinite(item[name]).all():
                raise ValueError('Invalid DC values in color repair: ' + str(path))
        rgb = .5 + .28209479177387814 * item['f_dc']
        if np.any(rgb < -1e-6) or np.any(rgb > 1 + 1e-6):
            raise ValueError('Repaired DC colors must lie in the visible RGB range: ' + str(path))
        chunks[source].append(item)
        evidence.append({'document': str(path), **document})
    merged = {}
    for source, items in chunks.items():
        if not items:
            merged[source] = {'indices': np.empty(0, np.uint32), 'f_dc': np.empty((0, 3), np.float32),
                              'original_f_dc': np.empty((0, 3), np.float32)}
            continue
        indices = np.concatenate([item['indices'] for item in items])
        order = np.argsort(indices)
        if np.any(indices[order][1:] == indices[order][:-1]):
            raise ValueError('Color repairs overlap; resolve their source rows explicitly: ' + source)
        merged[source] = {name: np.concatenate([item[name] for item in items])[order]
                          for name in ['indices', 'f_dc', 'original_f_dc']}
    return merged, evidence


def apply_color_values(colors, source_indices, override, strength=1.):
    """Change only selected colors; verify the expected original values first."""
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Color repair strength must be between zero and one')
    result = colors.copy()
    if not len(override['indices']) or strength == 0:
        return result, 0
    position = np.searchsorted(override['indices'], source_indices)
    safe = np.minimum(position, len(override['indices']) - 1)
    rows = np.flatnonzero((position < len(override['indices'])) & (override['indices'][safe] == source_indices))
    chosen = position[rows]
    if not np.array_equal(result[rows], override['original_f_dc'][chosen]):
        raise ValueError('Color repair original values do not match the source')
    result[rows] = override['f_dc'][chosen] if strength == 1 else result[rows] + strength * (override['f_dc'][chosen] - result[rows])
    return result, len(rows)
