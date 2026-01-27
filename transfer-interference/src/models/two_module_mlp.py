"""
Two-module MLP architecture for interference task.

This module implements a two-module MLP with task-based routing, optional
inter-module communication, and shared readout for studying structural biases
under continual learning.
"""
import torch
from torch import nn
from typing import Tuple, Optional
from src.models.network_interface import InterferenceTaskNetwork


class TwoModuleMLP(InterferenceTaskNetwork):
    """Two-module MLP with task-based routing and optional inter-module communication.
    
    Architecture:
    - Module A: Processes Task A inputs (A1, A2 phases)
    - Module B: Processes Task B inputs (B phase)
    - Inter-module communication: Optional bidirectional connections
    - Shared readout: Both modules contribute to final output
    
    Args:
        dim_input: Input dimension
        dim_hidden: Total hidden dimension (will be split between modules)
        dim_output: Output dimension
        comm_bandwidth: Communication bandwidth preset ("none", "low", "high")
        comm_scale: Explicit communication scale (overrides bandwidth preset if provided)
    """
    
    def __init__(self, dim_input, dim_hidden, dim_output, comm_bandwidth="none", comm_scale=None):
        super(TwoModuleMLP, self).__init__()
        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._dim_output = dim_output
        self._comm_bandwidth = comm_bandwidth
        self._dim_hidden_per_module = dim_hidden // 2  # Split hidden units between modules
        
        # Set communication scale based on bandwidth preset
        if comm_scale is not None:
            self._comm_scale = comm_scale
        elif comm_bandwidth == "none":
            self._comm_scale = 0.0
        elif comm_bandwidth == "low":
            self._comm_scale = 0.05
        elif comm_bandwidth == "high":
            self._comm_scale = 0.1
        else:
            raise ValueError(f"Unknown comm_bandwidth: {comm_bandwidth}. Must be 'none', 'low', or 'high'")
        
        # Module A: processes Task A inputs
        self.mod_A = nn.Linear(dim_input, self._dim_hidden_per_module, bias=False)
        
        # Module B: processes Task B inputs
        self.mod_B = nn.Linear(dim_input, self._dim_hidden_per_module, bias=False)
        
        # Inter-module communication (only if communication is enabled)
        if self._comm_bandwidth != "none":
            self.comm_AB = nn.Linear(self._dim_hidden_per_module, self._dim_hidden_per_module, bias=False)
            self.comm_BA = nn.Linear(self._dim_hidden_per_module, self._dim_hidden_per_module, bias=False)
        else:
            self.comm_AB = None
            self.comm_BA = None
        
        # Shared readout layers
        self.readout_A = nn.Linear(self._dim_hidden_per_module, dim_output, bias=False)
        self.readout_B = nn.Linear(self._dim_hidden_per_module, dim_output, bias=False)
    
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None, 
                task_id: Optional[str] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the two-module network.
        
        Args:
            x: Input tensor (batch_size, input_dim)
            hidden: Ignored for FFN (kept for interface compatibility)
            task_id: Task identifier ("A" or "B") to route input to appropriate module
        
        Returns:
            output: Network output (batch_size, output_dim)
            hidden_combined: Concatenated hidden states [h_A, h_B] (batch_size, dim_hidden)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Route input to appropriate module based on task_id
        if task_id == "A":
            h_A = self.mod_A(x)
            h_B = torch.zeros(batch_size, self._dim_hidden_per_module, device=device)
        elif task_id == "B":
            h_B = self.mod_B(x)
            h_A = torch.zeros(batch_size, self._dim_hidden_per_module, device=device)
        else:
            # If no task_id provided, default behavior: route to both modules
            # This maintains backward compatibility but may not be desired behavior
            h_A = self.mod_A(x)
            h_B = self.mod_B(x)
        
        # Apply inter-module communication if enabled
        if self._comm_bandwidth != "none" and self.comm_AB is not None and self.comm_BA is not None:
            # Bidirectional communication
            h_A = h_A + self._comm_scale * self.comm_BA(h_B)
            h_B = h_B + self._comm_scale * self.comm_AB(h_A)
        
        # Shared readout: combine outputs from both modules
        output_A = self.readout_A(h_A)
        output_B = self.readout_B(h_B)
        output = output_A + output_B
        
        # Concatenate hidden states for compatibility with existing analysis code
        hidden_combined = torch.cat([h_A, h_B], dim=1)
        
        # Store per-module hidden states for analysis
        self._last_h_A = h_A
        self._last_h_B = h_B
        
        return output, hidden_combined
    
    def get_module_hidden_states(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the last computed per-module hidden states.
        
        Returns:
            h_A: Module A hidden states from last forward pass
            h_B: Module B hidden states from last forward pass
        """
        if not hasattr(self, '_last_h_A') or not hasattr(self, '_last_h_B'):
            raise RuntimeError("Must call forward() before getting module hidden states")
        return self._last_h_A, self._last_h_B
    
    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """Get current hidden state. Returns None for FFNs."""
        return None
    
    def reset_hidden_state(self, batch_size: int = 1):
        """Reset hidden state. No-op for FFNs."""
        pass
    
    def get_embeddings(self) -> Optional[torch.Tensor]:
        """
        Get input embedding weights for both modules.
        
        Returns:
            Concatenated embedding weights [mod_A.weight, mod_B.weight]
            Shape: (dim_hidden, input_dim)
        """
        return torch.cat([self.mod_A.weight, self.mod_B.weight], dim=0)
    
    def get_readouts(self) -> Optional[torch.Tensor]:
        """
        Get output readout weights for both modules.
        
        Returns:
            Concatenated readout weights [readout_A.weight, readout_B.weight]
            Shape: (output_dim, dim_hidden)
        """
        return torch.cat([self.readout_A.weight, self.readout_B.weight], dim=1)
    
    @property
    def supports_sequences(self) -> bool:
        """FFN does not support sequences."""
        return False
    
    @property
    def input_dim(self) -> int:
        """Input dimension."""
        return self._dim_input
    
    @property
    def output_dim(self) -> int:
        """Output dimension."""
        return self._dim_output
    
    @property
    def dim_hidden_per_module(self) -> int:
        """Hidden dimension per module."""
        return self._dim_hidden_per_module
    
    @property
    def comm_bandwidth(self) -> str:
        """Communication bandwidth setting."""
        return self._comm_bandwidth
    
    @property
    def comm_scale(self) -> float:
        """Communication scale factor."""
        return self._comm_scale
