"""
Architecture visualization utilities for neural networks.

This module provides functions to visualize network architectures, particularly
for two-module MLP architectures with inter-module communication.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def visualize_two_module_mlp(network, save_path=None, figsize=(12, 8)):
    """
    Visualize a two-module MLP architecture.
    
    Args:
        network: TwoModuleMLP instance
        save_path: Optional path to save the figure
        figsize: Figure size tuple
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Define positions
    input_pos = (1, 5)
    mod_a_pos = (4, 7)
    mod_b_pos = (4, 3)
    comm_pos = (6, 5)
    readout_pos = (8, 5)
    output_pos = (9.5, 5)
    
    # Draw input layer
    input_box = FancyBboxPatch(
        (input_pos[0] - 0.3, input_pos[1] - 0.4),
        0.6, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='lightblue',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(input_pos[0], input_pos[1], 'Input\n(dim_input)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw Module A
    mod_a_box = FancyBboxPatch(
        (mod_a_pos[0] - 0.5, mod_a_pos[1] - 0.5),
        1.0, 1.0,
        boxstyle="round,pad=0.1",
        facecolor='lightgreen',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(mod_a_box)
    ax.text(mod_a_pos[0], mod_a_pos[1], 'Module A\n(dim_hidden/2)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw Module B
    mod_b_box = FancyBboxPatch(
        (mod_b_pos[0] - 0.5, mod_b_pos[1] - 0.5),
        1.0, 1.0,
        boxstyle="round,pad=0.1",
        facecolor='lightcoral',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(mod_b_box)
    ax.text(mod_b_pos[0], mod_b_pos[1], 'Module B\n(dim_hidden/2)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw communication layer (if enabled)
    if network.comm_bandwidth != "none":
        comm_box = FancyBboxPatch(
            (comm_pos[0] - 0.4, comm_pos[1] - 0.6),
            0.8, 1.2,
            boxstyle="round,pad=0.1",
            facecolor='yellow',
            edgecolor='black',
            linewidth=2,
            alpha=0.7
        )
        ax.add_patch(comm_box)
        ax.text(comm_pos[0], comm_pos[1], f'Comm\n({network.comm_bandwidth})', 
                ha='center', va='center', fontsize=9, weight='bold')
    
    # Draw readout layer
    readout_box = FancyBboxPatch(
        (readout_pos[0] - 0.4, readout_pos[1] - 0.4),
        0.8, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='lightyellow',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(readout_box)
    ax.text(readout_pos[0], readout_pos[1], 'Readout\n(shared)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw output layer
    output_box = FancyBboxPatch(
        (output_pos[0] - 0.3, output_pos[1] - 0.4),
        0.6, 0.8,
        boxstyle="round,pad=0.1",
        facecolor='lightpink',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(output_box)
    ax.text(output_pos[0], output_pos[1], 'Output\n(dim_output)', 
            ha='center', va='center', fontsize=10, weight='bold')
    
    # Draw connections: Input -> Module A (Task A routing)
    arrow1 = FancyArrowPatch(
        (input_pos[0] + 0.3, input_pos[1] + 0.2),
        (mod_a_pos[0] - 0.5, mod_a_pos[1] - 0.2),
        arrowstyle='->', lw=2, color='green',
        connectionstyle="arc3,rad=0.2"
    )
    ax.add_patch(arrow1)
    ax.text((input_pos[0] + mod_a_pos[0]) / 2, input_pos[1] + 1.5, 'Task A', 
            ha='center', fontsize=8, color='green', weight='bold')
    
    # Draw connections: Input -> Module B (Task B routing)
    arrow2 = FancyArrowPatch(
        (input_pos[0] + 0.3, input_pos[1] - 0.2),
        (mod_b_pos[0] - 0.5, mod_b_pos[1] + 0.2),
        arrowstyle='->', lw=2, color='red',
        connectionstyle="arc3,rad=-0.2"
    )
    ax.add_patch(arrow2)
    ax.text((input_pos[0] + mod_b_pos[0]) / 2, input_pos[1] - 1.5, 'Task B', 
            ha='center', fontsize=8, color='red', weight='bold')
    
    # Draw communication connections (if enabled)
    if network.comm_bandwidth != "none":
        # A -> B communication
        arrow3 = FancyArrowPatch(
            (mod_a_pos[0] + 0.5, mod_a_pos[1] - 0.3),
            (comm_pos[0] - 0.4, comm_pos[1] - 0.3),
            arrowstyle='->', lw=1.5, color='orange',
            connectionstyle="arc3,rad=0.1"
        )
        ax.add_patch(arrow3)
        
        arrow4 = FancyArrowPatch(
            (comm_pos[0] + 0.4, comm_pos[1] - 0.3),
            (mod_b_pos[0] + 0.5, mod_b_pos[1] + 0.3),
            arrowstyle='->', lw=1.5, color='orange',
            connectionstyle="arc3,rad=0.1"
        )
        ax.add_patch(arrow4)
        
        # B -> A communication
        arrow5 = FancyArrowPatch(
            (mod_b_pos[0] + 0.5, mod_b_pos[1] + 0.3),
            (comm_pos[0] - 0.4, comm_pos[1] + 0.3),
            arrowstyle='->', lw=1.5, color='orange',
            connectionstyle="arc3,rad=-0.1"
        )
        ax.add_patch(arrow5)
        
        arrow6 = FancyArrowPatch(
            (comm_pos[0] + 0.4, comm_pos[1] + 0.3),
            (mod_a_pos[0] + 0.5, mod_a_pos[1] - 0.3),
            arrowstyle='->', lw=1.5, color='orange',
            connectionstyle="arc3,rad=-0.1"
        )
        ax.add_patch(arrow6)
        
        # Update module positions for readout connections
        mod_a_readout_start = (mod_a_pos[0] + 0.5, mod_a_pos[1])
        mod_b_readout_start = (mod_b_pos[0] + 0.5, mod_b_pos[1])
        comm_readout_end = (comm_pos[0] + 0.4, comm_pos[1])
        
        # Module A -> Communication -> Readout
        arrow7 = FancyArrowPatch(
            mod_a_readout_start,
            comm_readout_end,
            arrowstyle='->', lw=2, color='blue',
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow7)
        
        # Module B -> Communication -> Readout
        arrow8 = FancyArrowPatch(
            mod_b_readout_start,
            comm_readout_end,
            arrowstyle='->', lw=2, color='blue',
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow8)
        
        # Communication -> Readout
        arrow9 = FancyArrowPatch(
            (comm_pos[0] + 0.4, comm_pos[1]),
            (readout_pos[0] - 0.4, readout_pos[1]),
            arrowstyle='->', lw=2, color='blue'
        )
        ax.add_patch(arrow9)
    else:
        # Direct connections from modules to readout (no communication)
        arrow7 = FancyArrowPatch(
            (mod_a_pos[0] + 0.5, mod_a_pos[1]),
            (readout_pos[0] - 0.4, readout_pos[1] + 0.2),
            arrowstyle='->', lw=2, color='blue',
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow7)
        
        arrow8 = FancyArrowPatch(
            (mod_b_pos[0] + 0.5, mod_b_pos[1]),
            (readout_pos[0] - 0.4, readout_pos[1] - 0.2),
            arrowstyle='->', lw=2, color='blue',
            connectionstyle="arc3,rad=0"
        )
        ax.add_patch(arrow8)
    
    # Readout -> Output
    arrow10 = FancyArrowPatch(
        (readout_pos[0] + 0.4, readout_pos[1]),
        (output_pos[0] - 0.3, output_pos[1]),
        arrowstyle='->', lw=2, color='purple'
    )
    ax.add_patch(arrow10)
    
    # Add title
    title = f"Two-Module MLP Architecture\n"
    title += f"Communication: {network.comm_bandwidth}"
    if network.comm_bandwidth != "none":
        title += f" (scale={network.comm_scale:.3f})"
    ax.text(5, 9.5, title, ha='center', va='top', fontsize=12, weight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightgreen', label='Module A (Task A)'),
        mpatches.Patch(color='lightcoral', label='Module B (Task B)'),
        mpatches.Patch(color='yellow', label='Communication (if enabled)'),
        mpatches.Patch(color='lightyellow', label='Shared Readout'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Architecture visualization saved to {save_path}")
    
    return fig
