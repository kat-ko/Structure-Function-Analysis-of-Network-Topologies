"""
RNN network implementations for interference task.

This module provides RNN-based networks that implement the InterferenceTaskNetwork
interface, enabling sequence processing capabilities in the interference task.
"""
import torch
from torch import nn
from typing import Tuple, Optional
from src.models.network_interface import InterferenceTaskNetwork


class SimpleRNN(InterferenceTaskNetwork):
    """Simple RNN network for interference task.
    
    Architecture:
    input -> RNN layer(s) -> output layer
    
    Supports both single timestep and sequence processing modes.
    """
    
    def __init__(self, dim_input, dim_hidden, dim_output, num_layers=1, 
                 cell_type='RNN', dropout=0.0, sequence_mode='single'):
        """
        Initialize SimpleRNN.
        
        Args:
            dim_input: Input dimension
            dim_hidden: Hidden state dimension
            dim_output: Output dimension
            num_layers: Number of RNN layers
            cell_type: Type of RNN cell ('RNN', 'GRU', 'LSTM')
            dropout: Dropout probability (0.0 = no dropout)
            sequence_mode: 'single' (each trial is seq_len=1) or 'sequence' (process multiple trials)
        """
        super(SimpleRNN, self).__init__()
        self._dim_input = dim_input
        self._dim_hidden = dim_hidden
        self._dim_output = dim_output
        self._num_layers = num_layers
        self._cell_type = cell_type.upper()
        self._sequence_mode = sequence_mode
        
        # Create RNN cell
        if self._cell_type == 'RNN':
            self.rnn = nn.RNN(dim_input, dim_hidden, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self._cell_type == 'GRU':
            self.rnn = nn.GRU(dim_input, dim_hidden, num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        elif self._cell_type == 'LSTM':
            self.rnn = nn.LSTM(dim_input, dim_hidden, num_layers,
                              batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        else:
            raise ValueError(f"Unsupported cell_type: {cell_type}. Must be 'RNN', 'GRU', or 'LSTM'")
        
        # Output layer
        self.output_layer = nn.Linear(dim_hidden, dim_output, bias=False)
        
        # Hidden state storage
        self._hidden_state = None
        
    def forward(self, x, hidden=None):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
                - Single timestep: (batch_size, input_dim)
                - Sequence: (batch_size, seq_len, input_dim)
            hidden: Optional hidden state (if None, uses internal state or initializes)
        
        Returns:
            output: Network output
                - Single timestep: (batch_size, output_dim)
                - Sequence: (batch_size, seq_len, output_dim)
            hidden: Hidden state (for compatibility with interface)
                - Single timestep: (batch_size, hidden_dim)
                - Sequence: (batch_size, seq_len, hidden_dim)
        """
        # Handle input shape
        if x.dim() == 2:
            # Single timestep: (batch_size, input_dim) -> (batch_size, 1, input_dim)
            x = x.unsqueeze(1)
            single_timestep = True
        else:
            # Sequence: (batch_size, seq_len, input_dim)
            single_timestep = False
        
        # Use provided hidden or internal state
        if hidden is None:
            hidden = self._hidden_state
        
        # RNN forward pass
        rnn_out, new_hidden = self.rnn(x, hidden)
        
        # Store new hidden state
        self._hidden_state = new_hidden
        
        # Get last timestep for output (or all timesteps if sequence mode)
        if single_timestep:
            # Take last (and only) timestep
            rnn_out = rnn_out[:, -1, :]  # (batch_size, hidden_dim)
            hidden_for_return = rnn_out  # Use RNN output as hidden representation
        else:
            # For sequences, use all timesteps
            hidden_for_return = rnn_out  # (batch_size, seq_len, hidden_dim)
        
        # Output layer
        out = self.output_layer(rnn_out)
        
        return out, hidden_for_return
    
    def get_hidden_state(self):
        """Get current hidden state."""
        if self._hidden_state is None:
            return None
        
        # For LSTM, return only hidden (not cell state)
        if self._cell_type == 'LSTM':
            return self._hidden_state[0]  # (num_layers, batch_size, hidden_dim)
        else:
            return self._hidden_state  # (num_layers, batch_size, hidden_dim)
    
    def reset_hidden_state(self, batch_size=1):
        """Reset hidden state to zeros."""
        device = next(self.parameters()).device
        if self._cell_type == 'LSTM':
            h_0 = torch.zeros(self._num_layers, batch_size, self._dim_hidden, device=device)
            c_0 = torch.zeros(self._num_layers, batch_size, self._dim_hidden, device=device)
            self._hidden_state = (h_0, c_0)
        else:
            self._hidden_state = torch.zeros(self._num_layers, batch_size, self._dim_hidden, device=device)
    
    def get_embeddings(self):
        """Get input-to-hidden weights (first layer only)."""
        # RNN doesn't have a clear "embedding" layer, but we can return input weights
        if hasattr(self.rnn, 'weight_ih_l0'):
            return self.rnn.weight_ih_l0  # (hidden_dim, input_dim)
        return None
    
    def get_readouts(self):
        """Get hidden-to-output weights."""
        return self.output_layer.weight  # (output_dim, hidden_dim)
    
    @property
    def supports_sequences(self):
        """RNN supports sequence processing."""
        return True
    
    @property
    def input_dim(self):
        """Input dimension."""
        return self._dim_input
    
    @property
    def output_dim(self):
        """Output dimension."""
        return self._dim_output

