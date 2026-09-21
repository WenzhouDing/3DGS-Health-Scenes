"""Load reviewed cleanup selections addressed by immutable source vertex index."""
import json
from pathlib import Path
import numpy as np


def load_cleanup_masks(paths, sources, root):
    """Validate both source identities before using a frozen local selection."""
    hashes = {meta['file']: meta['sha256'] for meta in sources.values()}
    selections = {source: [] for source in sources}
    evidence = []
    for path in paths:
        document = json.loads((Path(root) / path).read_text())
        source = document.get('sourceCapture')
        if document.get('version') != 1 or source not in sources:
            raise ValueError('Invalid cleanup selection: ' + str(path))
        meta = sources[source]
        if (document.get('sourceFile') != meta['file'] or
                document.get('sourceSha256') != meta['sha256'] or
                document.get('sourceHashes') != hashes):
            raise ValueError('Cleanup selection belongs to different source scans: ' + str(path))
        indices = np.load(Path(root) / document['maskFile'], allow_pickle=False)
        if (indices.ndim != 1 or indices.dtype != np.dtype('uint32') or
                (len(indices) and (indices[-1] >= meta['count'] or np.any(indices[1:] <= indices[:-1])))):
            raise ValueError('Cleanup indices must be sorted unique uint32 source rows: ' + str(path))
        if len(indices) != document['uniqueSourceIndices']:
            raise ValueError('Cleanup selection count does not match its evidence: ' + str(path))
        selections[source].append(indices)
        evidence.append({'document': str(path), **document})
    return {source: np.unique(np.concatenate(arrays)) if arrays else np.empty(0, dtype=np.uint32)
            for source, arrays in selections.items()}, evidence


def keep_source_rows(indices, removed):
    """Apply after coverage so removing a ghost never rescues its opposite ghost."""
    return ~np.isin(indices, removed)


def validate_restorations(labels, indices, selections, rules, parts, minimum_weight):
    """Restored observed surfaces must stay within their reviewed body part."""
    known = {part[0]: i for i, part in enumerate(parts)}
    claimed = np.zeros(len(indices), dtype=bool)
    for rows, rule in zip(selections, rules):
        weight = rule['minimumWeight']
        if (rule['part'] not in known or not np.isfinite(weight)
                or not minimum_weight <= weight <= 1):
            raise ValueError('Invalid observed-surface restoration rule')
        selected = np.isin(indices, rows)
        if np.any(claimed & selected):
            raise ValueError('Observed-surface restoration selections overlap')
        if np.any(labels[selected] != known[rule['part']]):
            raise ValueError('Observed-surface restoration crosses a body-part boundary')
        claimed |= selected


def restore_surface_weights(weights, indices, selections, rules):
    """Restore only reviewed source rows; preserve other fusion weights exactly."""
    result = weights.copy()
    restored = np.zeros(len(indices), dtype=bool)
    for rows, rule in zip(selections, rules):
        selected = np.isin(indices, rows)
        result[selected] = np.maximum(result[selected], rule['minimumWeight'])
        restored |= selected
    return result, restored
