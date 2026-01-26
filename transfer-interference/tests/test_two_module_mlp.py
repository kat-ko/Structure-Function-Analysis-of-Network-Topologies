"""
Unit tests for TwoModuleMLP architecture.
"""
import torch
import numpy as np
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.two_module_mlp import TwoModuleMLP


class TestTwoModuleMLP:
    """Test suite for TwoModuleMLP."""
    
    def test_initialization(self):
        """Test network initialization."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="none"
        )
        
        assert network.input_dim == 12
        assert network.output_dim == 4
        assert network.dim_hidden_per_module == 25
        assert network.comm_bandwidth == "none"
        assert network.comm_scale == 0.0
        assert network.comm_AB is None
        assert network.comm_BA is None
    
    def test_initialization_with_communication(self):
        """Test network initialization with communication."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="high"
        )
        
        assert network.comm_bandwidth == "high"
        assert network.comm_scale == 0.1
        assert network.comm_AB is not None
        assert network.comm_BA is not None
    
    def test_forward_task_a(self):
        """Test forward pass with Task A routing."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="none"
        )
        
        x = torch.randn(2, 12)  # batch_size=2
        output, hidden = network(x, task_id="A")
        
        assert output.shape == (2, 4)
        assert hidden.shape == (2, 50)  # Concatenated h_A and h_B
        
        # Check that Module B is zero (no input routed to it)
        h_A, h_B = network.get_module_hidden_states()
        assert h_A.shape == (2, 25)
        assert h_B.shape == (2, 25)
        assert torch.allclose(h_B, torch.zeros_like(h_B), atol=1e-6)
    
    def test_forward_task_b(self):
        """Test forward pass with Task B routing."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="none"
        )
        
        x = torch.randn(2, 12)
        output, hidden = network(x, task_id="B")
        
        assert output.shape == (2, 4)
        assert hidden.shape == (2, 50)
        
        # Check that Module A is zero
        h_A, h_B = network.get_module_hidden_states()
        assert torch.allclose(h_A, torch.zeros_like(h_A), atol=1e-6)
        assert h_B.shape == (2, 25)
    
    def test_communication_effect(self):
        """Test that communication affects outputs."""
        network_none = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="none"
        )
        
        network_high = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="high"
        )
        
        # Initialize with same weights for fair comparison
        with torch.no_grad():
            for (name_none, param_none), (name_high, param_high) in zip(
                network_none.named_parameters(),
                network_high.named_parameters()
            ):
                if "comm" not in name_high:
                    param_high.data.copy_(param_none.data)
        
        x = torch.randn(1, 12)
        output_none, _ = network_none(x, task_id="A")
        output_high, _ = network_high(x, task_id="A")
        
        # Outputs should be different when communication is enabled
        # (unless communication weights happen to be exactly zero)
        assert not torch.allclose(output_none, output_high, atol=1e-5)
    
    def test_shared_readout(self):
        """Test that shared readout combines both modules."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="none"
        )
        
        # Test with Task A: only Module A should contribute
        x_a = torch.randn(1, 12)
        output_a, _ = network(x_a, task_id="A")
        
        # Test with Task B: only Module B should contribute
        x_b = torch.randn(1, 12)
        output_b, _ = network(x_b, task_id="B")
        
        # Outputs should be different (different inputs, different modules)
        assert not torch.allclose(output_a, output_b, atol=1e-5)
    
    def test_interface_compliance(self):
        """Test that TwoModuleMLP implements InterferenceTaskNetwork interface."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4
        )
        
        # Check required properties
        assert hasattr(network, 'input_dim')
        assert hasattr(network, 'output_dim')
        assert hasattr(network, 'supports_sequences')
        assert network.supports_sequences == False
        
        # Check required methods
        assert hasattr(network, 'forward')
        assert hasattr(network, 'get_hidden_state')
        assert hasattr(network, 'reset_hidden_state')
        assert hasattr(network, 'get_embeddings')
        assert hasattr(network, 'get_readouts')
        
        # Test method calls
        x = torch.randn(1, 12)
        output, hidden = network(x, task_id="A")
        assert output is not None
        assert hidden is not None
        
        assert network.get_hidden_state() is None  # FFN doesn't have hidden state
        network.reset_hidden_state()  # Should not raise error
        
        embeddings = network.get_embeddings()
        assert embeddings is not None
        assert embeddings.shape[0] == 50  # Total hidden dim
        
        readouts = network.get_readouts()
        assert readouts is not None
        assert readouts.shape[1] == 50  # Total hidden dim
    
    def test_comm_bandwidth_presets(self):
        """Test communication bandwidth presets."""
        network_low = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="low"
        )
        assert network_low.comm_scale == 0.05
        
        network_high = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="high"
        )
        assert network_high.comm_scale == 0.1
        
        network_custom = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="high",
            comm_scale=0.2
        )
        assert network_custom.comm_scale == 0.2  # Custom scale overrides preset
    
    def test_invalid_comm_bandwidth(self):
        """Test that invalid communication bandwidth raises error."""
        with pytest.raises(ValueError):
            TwoModuleMLP(
                dim_input=12,
                dim_hidden=50,
                dim_output=4,
                comm_bandwidth="invalid"
            )
    
    def test_backward_compatibility_no_task_id(self):
        """Test that forward works without task_id (backward compatibility)."""
        network = TwoModuleMLP(
            dim_input=12,
            dim_hidden=50,
            dim_output=4,
            comm_bandwidth="none"
        )
        
        x = torch.randn(1, 12)
        # Should work without task_id (routes to both modules)
        output, hidden = network(x)
        
        assert output.shape == (1, 4)
        assert hidden.shape == (1, 50)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
