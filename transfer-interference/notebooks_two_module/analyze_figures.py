#!/usr/bin/env python3
import json
import re
import sys

notebooks = [
    'figure2_transfer_interference_comparison.ipynb',
    'figure3_anns_comparison.ipynb',
    'figure4_individual_differences_comparison.ipynb'
]

for notebook_name in notebooks:
    print(f"\n{'='*60}")
    print(f"{notebook_name}")
    print('='*60)
    
    with open(notebook_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    savefig_cells = []
    for i, cell in enumerate(data['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            if 'savefig' in source.lower():
                has_show = bool(re.search(r'plt\.show\(\)|display\(|fig\.show\(\)', source, re.IGNORECASE))
                savefig_cells.append((i, source, has_show))
    
    print(f"Total cells with savefig: {len(savefig_cells)}")
    
    cells_needing_fix = []
    for cell_idx, source, has_show in savefig_cells:
        if not has_show:
            cells_needing_fix.append(cell_idx)
            # Find the savefig line
            lines = source.split('\n')
            for j, line in enumerate(lines):
                if 'savefig' in line.lower():
                    print(f"\nCell {cell_idx} - Missing display:")
                    # Show context
                    start = max(0, j - 3)
                    end = min(len(lines), j + 4)
                    for k in range(start, end):
                        marker = ">>> " if k == j else "    "
                        print(f"{marker}{k}: {lines[k]}")
                    break
    
    print(f"\nCells needing fix: {len(cells_needing_fix)}")
    if cells_needing_fix:
        print(f"Cell indices: {cells_needing_fix}")
