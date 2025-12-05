"""
Visualize Small-World Network Topology

This script demonstrates the structure of a small-world network topology
by creating and visualizing it using the row-based layout.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List
from topology_playground import create_small_world_topology


def visualize_topology_structure(topology: nx.DiGraph, input_nodes: List[int], 
                                 output_nodes: List[int], 
                                 figsize: Tuple[int, int] = (16, 12),
                                 save_path: str = None) -> plt.Figure:
    """
    Visualize network topology structure with row-based layout.
    
    Args:
        topology: NetworkX DiGraph representing the topology
        input_nodes: List of input node indices
        output_nodes: List of output node indices
        figsize: Figure size tuple
        save_path: Optional path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Separate nodes into rows based on connectivity
    hidden_nodes = [n for n in topology.nodes() 
                    if n not in input_nodes and n not in output_nodes]
    
    # Categorize hidden nodes
    hidden_connected_to_inputs = [n for n in hidden_nodes 
                                    if any(pred in input_nodes for pred in topology.predecessors(n))]
    hidden_connected_to_outputs = [n for n in hidden_nodes 
                                     if any(succ in output_nodes for succ in topology.successors(n))]
    # Middle hidden nodes (not directly connected to inputs or outputs)
    middle_hidden_nodes = [n for n in hidden_nodes 
                           if n not in hidden_connected_to_inputs and n not in hidden_connected_to_outputs]
    
    # Calculate node positions in rows
    pos = {}
    row_spacing = 4.0  # Vertical spacing between rows
    node_x_spacing = 1.2  # Horizontal spacing between nodes in same row
    
    # Row 1: Input nodes
    row1_y = 4 * row_spacing
    for i, node in enumerate(sorted(input_nodes)):
        x_offset = (i - (len(input_nodes) - 1) / 2) * node_x_spacing
        pos[node] = (x_offset, row1_y)
    
    # Row 2: Hidden nodes connected to inputs
    row2_y = 3 * row_spacing
    for i, node in enumerate(sorted(hidden_connected_to_inputs)):
        x_offset = (i - (len(hidden_connected_to_inputs) - 1) / 2) * node_x_spacing
        pos[node] = (x_offset, row2_y)
    
    # Row 3: Middle hidden nodes
    row3_y = 2 * row_spacing
    for i, node in enumerate(sorted(middle_hidden_nodes)):
        x_offset = (i - (len(middle_hidden_nodes) - 1) / 2) * node_x_spacing
        pos[node] = (x_offset, row3_y)
    
    # Row 4: Hidden nodes connected to outputs
    row4_y = 1 * row_spacing
    for i, node in enumerate(sorted(hidden_connected_to_outputs)):
        x_offset = (i - (len(hidden_connected_to_outputs) - 1) / 2) * node_x_spacing
        pos[node] = (x_offset, row4_y)
    
    # Row 5: Output nodes
    row5_y = 0
    for i, node in enumerate(sorted(output_nodes)):
        x_offset = (i - (len(output_nodes) - 1) / 2) * node_x_spacing
        pos[node] = (x_offset, row5_y)
    
    # Draw edges
    for edge in topology.edges():
        source, target = edge
        x_coords = [pos[source][0], pos[target][0]]
        y_coords = [pos[source][1], pos[target][1]]
        ax.plot(x_coords, y_coords, color='gray', alpha=0.4, linewidth=1.0, zorder=1)
        
        # Draw arrowhead
        dx = x_coords[1] - x_coords[0]
        dy = y_coords[1] - y_coords[0]
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            arrow_length = 0.2
            arrow_dx = dx / length * arrow_length
            arrow_dy = dy / length * arrow_length
            ax.arrow(x_coords[1] - arrow_dx, y_coords[1] - arrow_dy,
                    arrow_dx, arrow_dy, head_width=0.15, head_length=0.1,
                    fc='gray', ec='gray', alpha=0.4, zorder=2)
    
    # Draw nodes with different colors for each row type
    # Input nodes - green
    for node in input_nodes:
        ax.scatter(pos[node][0], pos[node][1], s=400, c='green', 
                  edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center',
               fontsize=10, fontweight='bold', zorder=4, color='white')
    
    # Hidden nodes connected to inputs - light blue
    for node in hidden_connected_to_inputs:
        ax.scatter(pos[node][0], pos[node][1], s=300, c='lightblue', 
                  edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=4)
    
    # Middle hidden nodes - blue
    for node in middle_hidden_nodes:
        ax.scatter(pos[node][0], pos[node][1], s=300, c='blue', 
                  edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=4, color='white')
    
    # Hidden nodes connected to outputs - light blue
    for node in hidden_connected_to_outputs:
        ax.scatter(pos[node][0], pos[node][1], s=300, c='lightblue', 
                  edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center',
               fontsize=9, fontweight='bold', zorder=4)
    
    # Output nodes - red
    for node in output_nodes:
        ax.scatter(pos[node][0], pos[node][1], s=400, c='red', 
                  edgecolors='black', linewidths=2, zorder=3, alpha=0.8)
        ax.text(pos[node][0], pos[node][1], str(node), ha='center', va='center',
               fontsize=10, fontweight='bold', zorder=4, color='white')
    
    # Set axis limits based on node positions
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    x_margin = 2.0
    y_margin = 1.0
    ax.set_xlim(min(all_x) - x_margin, max(all_x) + x_margin)
    ax.set_ylim(min(all_y) - y_margin, max(all_y) + y_margin)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.8, label='Input Nodes'),
        Patch(facecolor='lightblue', alpha=0.8, label='Hidden (Input-connected)'),
        Patch(facecolor='blue', alpha=0.8, label='Hidden (Middle)'),
        Patch(facecolor='lightblue', alpha=0.8, label='Hidden (Output-connected)'),
        Patch(facecolor='red', alpha=0.8, label='Output Nodes')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    ax.set_title('Small-World Network Topology Structure', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Topology visualization saved to: {save_path}")
    
    return fig


if __name__ == "__main__":
    # Topology parameters
    input_dim = 4  # CartPole observation space
    hidden_size = 32
    output_dim = 2  # CartPole action space
    k = 9
    p = 0.2
    seed = 47
    
    print("Creating small-world topology...")
    print(f"Parameters: input_dim={input_dim}, hidden_size={hidden_size}, output_dim={output_dim}")
    print(f"k={k}, p={p}, seed={seed}")
    
    # Create topology
    topology, input_nodes, output_nodes = create_small_world_topology(
        input_dim, hidden_size, output_dim, k, p, seed
    )
    
    print(f"\nTopology created:")
    print(f"  Total nodes: {len(topology.nodes())}")
    print(f"  Total edges: {len(topology.edges())}")
    print(f"  Input nodes: {input_nodes}")
    print(f"  Output nodes: {output_nodes}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(topology)}")
    
    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_topology_structure(topology, input_nodes, output_nodes, 
                                       figsize=(18, 14),
                                       save_path="topology_structure.png")
    
    print("\nVisualization complete!")
    print("Figure saved to: topology_structure.png")
    
    # Optionally display (if running interactively)
    # plt.show()

