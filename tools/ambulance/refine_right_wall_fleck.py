#!/usr/bin/env python3
"""Extend the right-paint trial only across the attributed unpadded wall stain.

Original rows 802001/2272553 form its dark mark; 1449382/4233869
form the broad pale layer. All lie in the bare gap between fixed pads.
The actual reference capture supplies material, without changing its geometry,
colors or SH. Existing pad, strap and oxygen-label exclusion boxes still apply.
"""
import copy
import json
import shutil
import sys
from pathlib import Path
import transfer_reference_white_walls as walls
from cleanup import sha256_file


if __name__ == '__main__':
    group = copy.deepcopy(walls.GROUPS['right-paint'])
    group['regions'].append(walls.region(
        'attributed-bare-gap-stain', 2, [0, 1],
        [[.512, .294], [.633, .390]],
        [-.08850324, .01756119, .86515643], [-.045, .12],
        [13, 11], [.70, 5.5], 1, .012))
    group['description'] += (
        ' A narrow bare-wall gap extension covers the existing gray/pale iPhone '
        'blotch identified by exact visible contributors. Lower/upper pad and '
        'restraint exclusions remain enforced.')
    walls.GROUPS['right-paint'] = group
    walls.main()
    out = Path(sys.argv[sys.argv.index('--out') + 1])
    shutil.copyfile(__file__, out / 'generator.py')
    shutil.copyfile(walls.__file__, out / 'white-wall-builder.py')
    report = json.loads((out / 'report.json').read_text())
    report['generator_sha256'] = sha256_file(out / 'generator.py')
    report['generator_dependencies'] = {
        'transfer_reference_white_walls.py': sha256_file(out / 'white-wall-builder.py')}
    report['attributed_gap_stain'] = {
        'diagnostic': 'pass5/walls-diagnostics/right-fleck-current-pixel-contributors.json',
        'gray_original_ids': [802001, 2272553],
        'pale_original_ids': [1449382, 4233869],
        'source_only_trace_evidence': 'Visible original layers in the bare gap; no pad or harness component.',
        'preserved_pad_boundary_y': [.29, .415]}
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
