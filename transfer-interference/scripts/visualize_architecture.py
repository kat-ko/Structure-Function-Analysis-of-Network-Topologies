#!/usr/bin/env python3
"""
Script to visualize network architectures, particularly two-module MLP architectures.

Usage:
    python scripts/visualize_architecture.py two_module_rich_50_none
    python scripts/visualize_architecture.py two_module_rich_50_high
    python scripts/visualize_architecture.py rich_50  # Single-module (will show simple architecture)
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt

# Get the script's directory and the project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)  # transfer-interference directory

# Add project root to Python path
sys.path.insert(0, PROJECT_ROOT)

from src.models.neural_network import simpleLinearNet
from src.models.two_module_mlp import TwoModuleMLP
from src.utils.visualize_architecture import (
    visualize_two_module_mlp, 
    visualize_two_module_mlp_nodes,
    visualize_single_module_nodes
)
from src.analysis import ann


def visualize_network_architecture(condition_name, base_folder='./', save_path=None, show=True, nodes=False):
    """
    Visualize a network architecture based on condition configuration.
    
    Args:
        condition_name: Name of the condition from ann_experiments.json
        base_folder: Base project folder path
        save_path: Optional path to save the figure (if None, saves to figures/)
        show: Whether to display the figure
        nodes: If True, show individual nodes; if False, show high-level modules
    """
    # Setup paths
    data_folder = os.path.join(base_folder, 'data')
    config_path = os.path.join(base_folder, 'src', 'models', 'ann_experiments.json')
    figures_folder = os.path.join(base_folder, 'figures')
    
    # Load settings and find specified condition
    with open(config_path, 'r') as f:
        settings = json.load(f)
    
    condition = next((c for c in settings['conditions'] if c['name'] == condition_name), None)
    if not condition:
        raise ValueError(f"Condition '{condition_name}' not found in settings")
    
    # Setup parameters
    task_parameters = ann.setup_task_parameters()
    dim_input = task_parameters['nStim_perTask'] * 2
    dim_hidden = condition['dim_hidden']
    dim_output = 4
    
    # Check if this is a two-module architecture
    architecture_config = condition.get('architecture', None)
    is_two_module = (architecture_config is not None and 
                     architecture_config.get('type') == 'two_module')
    
    if is_two_module:
        # Create two-module network
        comm_bandwidth = architecture_config.get('comm_bandwidth', 'none')
        comm_scale = architecture_config.get('comm_scale', None)
        network = TwoModuleMLP(dim_input, dim_hidden, dim_output, 
                              comm_bandwidth=comm_bandwidth, comm_scale=comm_scale)
        
        # Visualize two-module architecture
        if save_path is None:
            os.makedirs(figures_folder, exist_ok=True)
            suffix = '_nodes' if nodes else ''
            save_path = os.path.join(figures_folder, f'architecture_{condition_name}{suffix}.png')
        
        if nodes:
            fig = visualize_two_module_mlp_nodes(network, save_path=save_path)
            print(f"✓ Visualized two-module architecture (node-level): {condition_name}")
        else:
            fig = visualize_two_module_mlp(network, save_path=save_path)
            print(f"✓ Visualized two-module architecture: {condition_name}")
        
        print(f"  Communication: {comm_bandwidth}")
        print(f"  Dimensions: input={dim_input}, hidden={dim_hidden} (split: {dim_hidden//2} per module), output={dim_output}")
        
    else:
        # Create single-module network for reference
        network = simpleLinearNet(dim_input, dim_hidden, dim_output)
        
        # Create a simple visualization for single-module
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Simple diagram
        ax.text(0.1, 0.5, 'Input\n(dim_input)', ha='center', va='center', 
                fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', edgecolor='black'))
        ax.text(0.5, 0.5, 'Hidden\n(dim_hidden)', ha='center', va='center',
                fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', edgecolor='black'))
        ax.text(0.9, 0.5, 'Output\n(dim_output)', ha='center', va='center',
                fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='black'))
        
        # Arrows
        ax.annotate('', xy=(0.5, 0.5), xytext=(0.1, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        ax.annotate('', xy=(0.9, 0.5), xytext=(0.5, 0.5),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Single-Module MLP Architecture\n{condition_name}', 
                    fontsize=14, weight='bold')
        
        if save_path is None:
            os.makedirs(figures_folder, exist_ok=True)
            save_path = os.path.join(figures_folder, f'architecture_{condition_name}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualized single-module architecture: {condition_name}")
        print(f"  Dimensions: input={dim_input}, hidden={dim_hidden}, output={dim_output}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    print(f"  Saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize network architectures')
    parser.add_argument('condition', type=str, help='Condition name from ann_experiments.json')
    parser.add_argument('--base-folder', type=str, default=None,
                       help='Base project folder path (default: script parent directory)')
    parser.add_argument('--save-path', type=str, default=None,
                       help='Path to save the figure (default: figures/architecture_{condition}.png)')
    parser.add_argument('--no-show', action='store_true',
                       help='Do not display the figure (only save)')
    parser.add_argument('--nodes', action='store_true',
                       help='Show individual nodes (node-level visualization)')
    
    args = parser.parse_args()
    
    # Use PROJECT_ROOT if base_folder not specified
    base_folder = args.base_folder if args.base_folder else PROJECT_ROOT
    
    visualize_network_architecture(
        args.condition,
        base_folder=base_folder,
        save_path=args.save_path,
        show=not args.no_show,
        nodes=args.nodes
    )


if __name__ == "__main__":
    main()
