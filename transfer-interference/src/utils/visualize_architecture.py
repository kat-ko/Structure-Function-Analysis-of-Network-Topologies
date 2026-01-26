"""
Architecture visualization utilities for neural networks.

This module provides functions to visualize network architectures, particularly
for two-module MLP architectures with inter-module communication.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, FancyArrow
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


def visualize_two_module_mlp_nodes(network, save_path=None, figsize=(20, 12), max_nodes_per_layer=200):
    """
    Visualize a two-module MLP architecture with individual nodes shown.
    
    Args:
        network: TwoModuleMLP instance
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        max_nodes_per_layer: Maximum nodes to show per layer (for readability)
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Get dimensions
    dim_input = network.input_dim
    dim_hidden_per_module = network.dim_hidden_per_module
    dim_output = network.output_dim
    
    # Limit nodes for visualization if too many
    n_input_show = min(dim_input, max_nodes_per_layer)
    n_hidden_show = min(dim_hidden_per_module, max_nodes_per_layer)
    n_output_show = min(dim_output, max_nodes_per_layer)
    
    # Define x positions for layers
    x_input = 1
    x_mod_a = 4
    x_mod_b = 4
    x_comm = 7
    x_readout = 10
    x_output = 13
    
    # Node spacing
    node_radius = 0.15
    spacing = 0.4
    
    # Draw input nodes
    input_y_start = 6 - (n_input_show - 1) * spacing / 2
    input_nodes = []
    for i in range(n_input_show):
        y = input_y_start + i * spacing
        circle = Circle((x_input, y), node_radius, facecolor='lightblue', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        input_nodes.append((x_input, y))
        if i < 3 or i >= n_input_show - 3:
            ax.text(x_input, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_input > max_nodes_per_layer:
        ax.text(x_input, input_y_start - 0.5, f'... ({dim_input} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw Module A nodes
    mod_a_y_start = 9 - (n_hidden_show - 1) * spacing / 2
    mod_a_nodes = []
    for i in range(n_hidden_show):
        y = mod_a_y_start + i * spacing
        circle = Circle((x_mod_a, y), node_radius, facecolor='lightgreen', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        mod_a_nodes.append((x_mod_a, y))
        if i < 3 or i >= n_hidden_show - 3:
            ax.text(x_mod_a, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_hidden_per_module > max_nodes_per_layer:
        ax.text(x_mod_a, mod_a_y_start - 0.5, f'... ({dim_hidden_per_module} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw Module B nodes
    mod_b_y_start = 3 - (n_hidden_show - 1) * spacing / 2
    mod_b_nodes = []
    for i in range(n_hidden_show):
        y = mod_b_y_start + i * spacing
        circle = Circle((x_mod_b, y), node_radius, facecolor='lightcoral', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        mod_b_nodes.append((x_mod_b, y))
        if i < 3 or i >= n_hidden_show - 3:
            ax.text(x_mod_b, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_hidden_per_module > max_nodes_per_layer:
        ax.text(x_mod_b, mod_b_y_start - 0.5, f'... ({dim_hidden_per_module} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw communication nodes (if enabled)
    comm_nodes_a = []
    comm_nodes_b = []
    if network.comm_bandwidth != "none":
        comm_y_center = 6
        comm_spacing = 0.3
        n_comm_show = min(n_hidden_show, max_nodes_per_layer)
        comm_y_start = comm_y_center - (n_comm_show - 1) * comm_spacing / 2
        
        for i in range(n_comm_show):
            y = comm_y_start + i * comm_spacing
            # Communication A->B nodes
            circle1 = Circle((x_comm - 0.2, y), node_radius * 0.8, 
                           facecolor='yellow', edgecolor='orange', linewidth=1)
            ax.add_patch(circle1)
            comm_nodes_a.append((x_comm - 0.2, y))
            
            # Communication B->A nodes
            circle2 = Circle((x_comm + 0.2, y), node_radius * 0.8, 
                           facecolor='yellow', edgecolor='orange', linewidth=1)
            ax.add_patch(circle2)
            comm_nodes_b.append((x_comm + 0.2, y))
    
    # Draw readout nodes
    readout_y_start = 6 - (n_output_show - 1) * spacing / 2
    readout_nodes = []
    for i in range(n_output_show):
        y = readout_y_start + i * spacing
        circle = Circle((x_readout, y), node_radius, facecolor='lightyellow', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        readout_nodes.append((x_readout, y))
        if i < 3 or i >= n_output_show - 3:
            ax.text(x_readout, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_output > max_nodes_per_layer:
        ax.text(x_readout, readout_y_start - 0.5, f'... ({dim_output} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw output nodes
    output_y_start = 6 - (n_output_show - 1) * spacing / 2
    output_nodes = []
    for i in range(n_output_show):
        y = output_y_start + i * spacing
        circle = Circle((x_output, y), node_radius, facecolor='lightpink', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        output_nodes.append((x_output, y))
        if i < 3 or i >= n_output_show - 3:
            ax.text(x_output, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_output > max_nodes_per_layer:
        ax.text(x_output, output_y_start - 0.5, f'... ({dim_output} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw connections: Input -> Module A (Task A routing) - ALL connections
    for input_node in input_nodes:
        for mod_a_node in mod_a_nodes:
            ax.plot([input_node[0] + node_radius, mod_a_node[0] - node_radius],
                   [input_node[1], mod_a_node[1]], 
                   'g-', linewidth=0.2, alpha=0.15)
    ax.text((x_input + x_mod_a) / 2, 10, 'Task A\n(all-to-all)', 
           ha='center', fontsize=8, color='green', weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw connections: Input -> Module B (Task B routing) - ALL connections
    for input_node in input_nodes:
        for mod_b_node in mod_b_nodes:
            ax.plot([input_node[0] + node_radius, mod_b_node[0] - node_radius],
                   [input_node[1], mod_b_node[1]], 
                   'r-', linewidth=0.2, alpha=0.15)
    ax.text((x_input + x_mod_b) / 2, 1, 'Task B\n(all-to-all)', 
           ha='center', fontsize=8, color='red', weight='bold',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    # Draw communication connections (if enabled) - ALL connections
    if network.comm_bandwidth != "none" and comm_nodes_a:
        # A -> B communication - ALL connections
        for mod_a_node in mod_a_nodes:
            for comm_node in comm_nodes_a:
                ax.plot([mod_a_node[0] + node_radius, comm_node[0] - node_radius * 0.8],
                       [mod_a_node[1], comm_node[1]], 
                       'orange', linewidth=0.15, alpha=0.1)
        for comm_node in comm_nodes_a:
            for mod_b_node in mod_b_nodes:
                ax.plot([comm_node[0] + node_radius * 0.8, mod_b_node[0] - node_radius],
                       [comm_node[1], mod_b_node[1]], 
                       'orange', linewidth=0.15, alpha=0.1)
        
        # B -> A communication - ALL connections
        for mod_b_node in mod_b_nodes:
            for comm_node in comm_nodes_b:
                ax.plot([mod_b_node[0] + node_radius, comm_node[0] - node_radius * 0.8],
                       [mod_b_node[1], comm_node[1]], 
                       'orange', linewidth=0.15, alpha=0.1)
        for comm_node in comm_nodes_b:
            for mod_a_node in mod_a_nodes:
                ax.plot([comm_node[0] + node_radius * 0.8, mod_a_node[0] - node_radius],
                       [comm_node[1], mod_a_node[1]], 
                       'orange', linewidth=0.15, alpha=0.1)
        
        ax.text(x_comm, 11, f'Comm\n({network.comm_bandwidth})', 
               ha='center', fontsize=9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # Draw connections: Modules -> Readout - ALL connections
    if network.comm_bandwidth == "none":
        # Direct connections from modules to readout
        for mod_a_node in mod_a_nodes:
            for readout_node in readout_nodes:
                ax.plot([mod_a_node[0] + node_radius, readout_node[0] - node_radius],
                       [mod_a_node[1], readout_node[1]], 
                       'b-', linewidth=0.2, alpha=0.15)
        for mod_b_node in mod_b_nodes:
            for readout_node in readout_nodes:
                ax.plot([mod_b_node[0] + node_radius, readout_node[0] - node_radius],
                       [mod_b_node[1], readout_node[1]], 
                       'b-', linewidth=0.2, alpha=0.15)
    else:
        # Through communication layer
        for mod_a_node in mod_a_nodes:
            for comm_node in comm_nodes_a:
                ax.plot([mod_a_node[0] + node_radius, comm_node[0] - node_radius * 0.8],
                       [mod_a_node[1], comm_node[1]], 
                       'b-', linewidth=0.15, alpha=0.1)
        for comm_node in comm_nodes_a:
            for readout_node in readout_nodes:
                ax.plot([comm_node[0] + node_radius * 0.8, readout_node[0] - node_radius],
                       [comm_node[1], readout_node[1]], 
                       'b-', linewidth=0.15, alpha=0.1)
        for mod_b_node in mod_b_nodes:
            for comm_node in comm_nodes_b:
                ax.plot([mod_b_node[0] + node_radius, comm_node[0] - node_radius * 0.8],
                       [mod_b_node[1], comm_node[1]], 
                       'b-', linewidth=0.15, alpha=0.1)
        for comm_node in comm_nodes_b:
            for readout_node in readout_nodes:
                ax.plot([comm_node[0] + node_radius * 0.8, readout_node[0] - node_radius],
                       [comm_node[1], readout_node[1]], 
                       'b-', linewidth=0.15, alpha=0.1)
    
    # Draw connections: Readout -> Output - ALL connections
    for readout_node in readout_nodes:
        for output_node in output_nodes:
            ax.plot([readout_node[0] + node_radius, output_node[0] - node_radius],
                   [readout_node[1], output_node[1]], 
                   'purple', linewidth=0.2, alpha=0.15)
    
    # Add layer labels
    ax.text(x_input, 11.5, f'Input\n({dim_input})', ha='center', fontsize=10, weight='bold')
    ax.text(x_mod_a, 11.5, f'Module A\n({dim_hidden_per_module})', ha='center', fontsize=10, weight='bold')
    ax.text(x_mod_b, 0.5, f'Module B\n({dim_hidden_per_module})', ha='center', fontsize=10, weight='bold')
    if network.comm_bandwidth != "none":
        ax.text(x_comm, 0.5, 'Comm', ha='center', fontsize=9, weight='bold')
    ax.text(x_readout, 11.5, f'Readout\n({dim_output})', ha='center', fontsize=10, weight='bold')
    ax.text(x_output, 11.5, f'Output\n({dim_output})', ha='center', fontsize=10, weight='bold')
    
    # Add title
    title = f"Two-Module MLP Architecture (Node-Level View)\n"
    title += f"Communication: {network.comm_bandwidth}"
    if network.comm_bandwidth != "none":
        title += f" (scale={network.comm_scale:.3f})"
    ax.text(8, 11.8, title, ha='center', va='top', fontsize=12, weight='bold')
    
    # Add note about connectivity
    note = "All connections shown (all-to-all connectivity).\n"
    note += f"Input→Modules: {dim_input}×{dim_hidden_per_module} each, "
    if network.comm_bandwidth != "none":
        note += f"Modules↔Comm: {dim_hidden_per_module}×{dim_hidden_per_module}, "
    note += f"Modules→Readout: {dim_hidden_per_module}×{dim_output} each, "
    note += f"Readout→Output: {dim_output}×{dim_output}"
    ax.text(8, 0.2, note, ha='center', fontsize=7, style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Input nodes'),
        mpatches.Patch(color='lightgreen', label='Module A nodes'),
        mpatches.Patch(color='lightcoral', label='Module B nodes'),
        mpatches.Patch(color='yellow', label='Communication nodes'),
        mpatches.Patch(color='lightyellow', label='Readout nodes'),
        mpatches.Patch(color='lightpink', label='Output nodes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Node-level architecture visualization saved to {save_path}")
    
    return fig


def visualize_single_module_nodes(network, save_path=None, figsize=(16, 10), max_nodes_per_layer=200):
    """
    Visualize a single-module MLP architecture with individual nodes shown.
    
    Args:
        network: simpleLinearNet instance
        save_path: Optional path to save the figure
        figsize: Figure size tuple
        max_nodes_per_layer: Maximum nodes to show per layer (for readability)
    
    Returns:
        matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Get dimensions from network weights
    dim_input = network.in_hid.weight.shape[1]
    dim_hidden = network.in_hid.weight.shape[0]
    dim_output = network.hid_out.weight.shape[0]
    
    # Limit nodes for visualization if too many
    n_input_show = min(dim_input, max_nodes_per_layer)
    n_hidden_show = min(dim_hidden, max_nodes_per_layer)
    n_output_show = min(dim_output, max_nodes_per_layer)
    
    # Define x positions for layers
    x_input = 1
    x_hidden = 5
    x_output = 9
    
    # Node spacing
    node_radius = 0.15
    spacing = 0.4
    
    # Draw input nodes
    input_y_start = 5 - (n_input_show - 1) * spacing / 2
    input_nodes = []
    for i in range(n_input_show):
        y = input_y_start + i * spacing
        circle = Circle((x_input, y), node_radius, facecolor='lightblue', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        input_nodes.append((x_input, y))
        if i < 3 or i >= n_input_show - 3:
            ax.text(x_input, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_input > max_nodes_per_layer:
        ax.text(x_input, input_y_start - 0.5, f'... ({dim_input} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw hidden nodes
    hidden_y_start = 5 - (n_hidden_show - 1) * spacing / 2
    hidden_nodes = []
    for i in range(n_hidden_show):
        y = hidden_y_start + i * spacing
        circle = Circle((x_hidden, y), node_radius, facecolor='lightgreen', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        hidden_nodes.append((x_hidden, y))
        if i < 3 or i >= n_hidden_show - 3:
            ax.text(x_hidden, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_hidden > max_nodes_per_layer:
        ax.text(x_hidden, hidden_y_start - 0.5, f'... ({dim_hidden} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw output nodes
    output_y_start = 5 - (n_output_show - 1) * spacing / 2
    output_nodes = []
    for i in range(n_output_show):
        y = output_y_start + i * spacing
        circle = Circle((x_output, y), node_radius, facecolor='lightyellow', 
                       edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        output_nodes.append((x_output, y))
        if i < 3 or i >= n_output_show - 3:
            ax.text(x_output, y, str(i), ha='center', va='center', fontsize=6)
    
    if dim_output > max_nodes_per_layer:
        ax.text(x_output, output_y_start - 0.5, f'... ({dim_output} total)', 
               ha='center', fontsize=7, style='italic')
    
    # Draw connections: Input -> Hidden - ALL connections
    for input_node in input_nodes:
        for hidden_node in hidden_nodes:
            ax.plot([input_node[0] + node_radius, hidden_node[0] - node_radius],
                   [input_node[1], hidden_node[1]], 
                   'g-', linewidth=0.2, alpha=0.15)
    
    # Draw connections: Hidden -> Output - ALL connections
    for hidden_node in hidden_nodes:
        for output_node in output_nodes:
            ax.plot([hidden_node[0] + node_radius, output_node[0] - node_radius],
                   [hidden_node[1], output_node[1]], 
                   'b-', linewidth=0.2, alpha=0.15)
    
    # Add layer labels
    ax.text(x_input, 9, f'Input\n({dim_input})', ha='center', fontsize=10, weight='bold')
    ax.text(x_hidden, 9, f'Hidden\n({dim_hidden})', ha='center', fontsize=10, weight='bold')
    ax.text(x_output, 9, f'Output\n({dim_output})', ha='center', fontsize=10, weight='bold')
    
    # Add title
    title = "Single-Module MLP Architecture (Node-Level View)"
    ax.text(5, 9.5, title, ha='center', va='top', fontsize=12, weight='bold')
    
    # Add note about connectivity
    note = "All connections shown (all-to-all connectivity).\n"
    note += f"Input→Hidden: {dim_input}×{dim_hidden}, Hidden→Output: {dim_hidden}×{dim_output}"
    ax.text(5, 0.2, note, ha='center', fontsize=7, style='italic',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightblue', label='Input nodes'),
        mpatches.Patch(color='lightgreen', label='Hidden nodes'),
        mpatches.Patch(color='lightyellow', label='Output nodes'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Node-level architecture visualization saved to {save_path}")
    
    return fig
