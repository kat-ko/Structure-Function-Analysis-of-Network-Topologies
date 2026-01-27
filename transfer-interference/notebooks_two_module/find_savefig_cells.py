#!/usr/bin/env python3
"""Find all cells that save figures but don't display them."""
import json
import re

notebooks = [
    'figure2_transfer_interference_comparison.ipynb',
    'figure3_anns_comparison.ipynb',
    'figure4_individual_differences_comparison.ipynb'
]

results = {}

for nb_name in notebooks:
    print(f"\n{'='*70}")
    print(f"Analyzing: {nb_name}")
    print('='*70)
    
    with open(nb_name, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    cells_to_fix = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell.get('source', []))
            
            # Check if this cell has savefig
            if '.savefig' in source or 'savefig(' in source:
                # Check if it has plt.show() or display()
                has_show = bool(re.search(r'plt\.show\(\)|display\(|fig\.show\(\)', source, re.IGNORECASE))
                
                if not has_show:
                    cells_to_fix.append(i)
                    # Show context
                    lines = source.split('\n')
                    savefig_idx = next((j for j, line in enumerate(lines) if '.savefig' in line or 'savefig(' in line), None)
                    if savefig_idx is not None:
                        print(f"\nCell {i}: MISSING display")
                        start = max(0, savefig_idx - 2)
                        end = min(len(lines), savefig_idx + 5)
                        for j in range(start, end):
                            marker = ">>> " if j == savefig_idx else "    "
                            print(f"{marker}{lines[j]}")
    
    results[nb_name] = cells_to_fix
    print(f"\nSummary: {len(cells_to_fix)} cells need fixing")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
for nb_name, cells in results.items():
    print(f"{nb_name}: {len(cells)} cells need fixing")
    if cells:
        print(f"  Cell indices: {cells}")
