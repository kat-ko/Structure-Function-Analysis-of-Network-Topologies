"""
Network interface for interference task.

This module defines the base interface that all networks used in the interference
task must implement. This allows the training pipeline to work with arbitrary
network architectures (FFN, RNN, etc.).
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import torch


class InterferenceTaskNetwork(ABC, torch.nn.Module):
    """Base interface for networks used in interference task.
    
    All networks used in the interference task must inherit from this class
    and implement the required abstract methods. This ensures compatibility
    with the training and analysis pipeline.
    """
    
    @abstractmethod
    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor 
                - For FFN: (batch_size, input_dim)
                - For RNN: (batch_size, input_dim) or (batch_size, seq_len, input_dim)
            hidden: Optional hidden state for RNNs
                - For FFN: Ignored (should be None)
                - For RNN: (batch_size, hidden_dim) or (num_layers, batch_size, hidden_dim)
        
        Returns:
            output: Network output
                - For FFN: (batch_size, output_dim)
                - For RNN: (batch_size, output_dim) or (batch_size, seq_len, output_dim)
            hidden: Hidden state
                - For FFN: (batch_size, hidden_dim) - intermediate representation
                - For RNN: (batch_size, hidden_dim) or (num_layers, batch_size, hidden_dim)
        """
        pass
    
    @abstractmethod
    def get_hidden_state(self) -> Optional[torch.Tensor]:
        """
        Get current hidden state (for RNNs).
        
        Returns:
            Hidden state tensor for RNNs, None for FFNs.
        """
        pass
    
    @abstractmethod
    def reset_hidden_state(self, batch_size: int = 1):
        """
        Reset hidden state (for RNNs).
        
        Args:
            batch_size: Batch size for the hidden state initialization.
            
        Note:
            This is a no-op for FFNs.
        """
        pass
    
    def get_embeddings(self) -> Optional[torch.Tensor]:
        """
        Optional: Get input embedding weights for analysis.
        
        Returns:
            Embedding weight matrix if applicable, None otherwise.
            Shape: (hidden_dim, input_dim) or similar.
        """
        return None
    
    def get_readouts(self) -> Optional[torch.Tensor]:
        """
        Optional: Get output readout weights for analysis.
        
        Returns:
            Readout weight matrix if applicable, None otherwise.
            Shape: (output_dim, hidden_dim) or similar.
        """
        return None
    
    @property
    @abstractmethod
    def supports_sequences(self) -> bool:
        """
        Whether network supports sequence processing.
        
        Returns:
            True if network can process sequences (RNN), False otherwise (FFN).
        """
        pass
    
    @property
    @abstractmethod
    def input_dim(self) -> int:
        """
        Input dimension.
        
        Returns:
            Size of input dimension.
        """
        pass
    
    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        Output dimension.
        
        Returns:
            Size of output dimension.
        """
        pass

