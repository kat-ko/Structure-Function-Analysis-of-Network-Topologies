#!/usr/bin/env python3
"""
Test script to validate the Quantization implementation.

This script tests the Quantization class to ensure it correctly quantizes
input tensors to the nearest values in quant_values.
"""

import numpy as np
import torch
import sys
from dynspec.models import Quantization


def test_basic_quantization():
    """Test 1: Basic quantization to nearest values"""
    print("Test 1: Basic quantization...")
    
    # Simple quantization levels
    quant_values = np.array([-1.0, 0.0, 1.0, 2.0])
    quant = Quantization(quant_values)
    
    # Test values that should map to nearest quant levels
    x = torch.tensor([-0.8, -0.3, 0.3, 0.8, 1.5, 1.9])
    result = quant(x)
    
    expected = torch.tensor([-1.0, 0.0, 0.0, 1.0, 1.0, 2.0])
    assert torch.allclose(result, expected), \
        f"Failed: {result} != {expected}"
    print(f"  ✓ Basic quantization works correctly")
    
    # Verify all output values are in quant_values
    result_np = result.numpy()
    for val in result_np.flatten():
        assert val in quant_values, \
            f"Failed: Output value {val} not in quant_values {quant_values}"
    print(f"  ✓ All output values are in quant_values")
    
    print("  ✓ Test 1 passed\n")


def test_exact_quantization_levels():
    """Test 2: Values exactly on quantization levels"""
    print("Test 2: Exact quantization levels...")
    
    quant_values = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    quant = Quantization(quant_values)
    
    # Test with values exactly on quant levels
    x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
    result = quant(x)
    
    assert torch.allclose(result, x), \
        f"Failed: Values on quant levels should remain unchanged, got {result}"
    print(f"  ✓ Values exactly on quant levels remain unchanged")
    
    print("  ✓ Test 2 passed\n")


def test_boundary_values():
    """Test 3: Boundary values (midpoints between quant levels)"""
    print("Test 3: Boundary values...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    # Test midpoints - when equidistant, argmin picks first (lower index)
    x = torch.tensor([-0.5, 0.5])  # Midpoints between -1,0 and 0,1
    result = quant(x)
    
    # -0.5 is equidistant from -1.0 and 0.0, so picks -1.0 (first match)
    # 0.5 is equidistant from 0.0 and 1.0, so picks 0.0 (first match)
    # This is correct behavior for argmin with ties
    assert all(val in quant_values for val in result.numpy()), \
        f"Failed: All values should be in quant_values, got {result}"
    assert torch.allclose(result, torch.tensor([-1.0, 0.0])), \
        f"Failed: Expected [-1.0, 0.0] for ties, got {result}"
    print(f"  ✓ Boundary values quantized correctly (ties handled by argmin)")
    
    # Test values closer to one quant level
    x2 = torch.tensor([-0.6, 0.4])  # Closer to -1.0 and 0.0 respectively
    result2 = quant(x2)
    assert torch.allclose(result2, torch.tensor([-1.0, 0.0])), \
        f"Failed: Should map to nearest, got {result2}"
    print(f"  ✓ Values closer to quant levels map correctly")
    
    print("  ✓ Test 3 passed\n")


def test_shape_preservation():
    """Test 4: Shape preservation"""
    print("Test 4: Shape preservation...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    # Test various shapes
    shapes = [(10,), (5, 3), (2, 3, 4), (1, 1, 1, 1)]
    
    for shape in shapes:
        x = torch.randn(*shape)
        result = quant(x)
        assert result.shape == x.shape, \
            f"Failed: Shape {result.shape} != {x.shape}"
        print(f"  ✓ Shape {shape} preserved")
    
    print("  ✓ Test 4 passed\n")


def test_numpy_array_input():
    """Test 5: Numpy array input"""
    print("Test 5: Numpy array input...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    # Test with numpy array
    x_np = np.array([-0.8, 0.3, 0.9])
    result = quant(x_np)
    
    assert isinstance(result, torch.Tensor), \
        f"Failed: Result should be torch.Tensor, got {type(result)}"
    assert result.shape == x_np.shape, \
        f"Failed: Shape mismatch"
    
    # Verify quantization worked
    result_np = result.numpy()
    for val in result_np:
        assert val in quant_values, \
            f"Failed: Output value {val} not in quant_values"
    
    print(f"  ✓ Numpy array input converted and quantized correctly")
    print("  ✓ Test 5 passed\n")


def test_different_bit_levels():
    """Test 6: Different bit quantization levels (as used in Quantized_weight)"""
    print("Test 6: Different bit levels...")
    
    for n_bits in [2, 3, 4, 8]:
        quant_values = np.linspace(
            -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1, 2**n_bits
        )
        quant = Quantization(quant_values)
        
        # Test with random values
        x = torch.randn(10) * 10  # Values in range roughly [-10, 10]
        result = quant(x)
        
        # Verify all outputs are in quant_values
        result_np = result.numpy()
        for val in result_np:
            assert val in quant_values, \
                f"Failed: Output {val} not in quant_values for {n_bits}-bit quantization"
        
        # Verify min/max are within quant_values range
        assert result.min().item() >= quant_values.min(), \
            f"Failed: Min value {result.min()} < {quant_values.min()}"
        assert result.max().item() <= quant_values.max(), \
            f"Failed: Max value {result.max()} > {quant_values.max()}"
        
        print(f"  ✓ {n_bits}-bit quantization works correctly ({2**n_bits} levels)")
    
    print("  ✓ Test 6 passed\n")


def test_device_consistency():
    """Test 7: Device consistency (CPU/GPU)"""
    print("Test 7: Device consistency...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    # Test on CPU
    x_cpu = torch.tensor([-0.5, 0.0, 0.5])
    result_cpu = quant(x_cpu)
    assert result_cpu.device.type == 'cpu', \
        f"Failed: Result should be on CPU, got {result_cpu.device}"
    print(f"  ✓ CPU quantization works")
    
    # Test on GPU if available
    if torch.cuda.is_available():
        x_gpu = torch.tensor([-0.5, 0.0, 0.5]).cuda()
        result_gpu = quant(x_gpu)
        assert result_gpu.device.type == 'cuda', \
            f"Failed: Result should be on CUDA, got {result_gpu.device}"
        
        # Verify results are the same (moved to CPU for comparison)
        assert torch.allclose(result_cpu, result_gpu.cpu()), \
            f"Failed: CPU and GPU results differ"
        print(f"  ✓ GPU quantization works and matches CPU")
    else:
        print(f"  ⚠ GPU not available, skipping GPU test")
    
    print("  ✓ Test 7 passed\n")


def test_extreme_values():
    """Test 8: Extreme values (outside quant range)"""
    print("Test 8: Extreme values...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    # Test values far outside range
    x = torch.tensor([-100.0, 100.0, -1000.0, 1000.0])
    result = quant(x)
    
    # Should map to nearest quant values (-1.0 or 1.0)
    expected = torch.tensor([-1.0, 1.0, -1.0, 1.0])
    assert torch.allclose(result, expected), \
        f"Failed: Extreme values should map to boundaries, got {result}"
    print(f"  ✓ Extreme values quantized to boundaries correctly")
    
    print("  ✓ Test 8 passed\n")


def test_empty_tensor():
    """Test 9: Empty tensor"""
    print("Test 9: Empty tensor...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    # Test empty tensor
    x = torch.tensor([])
    result = quant(x)
    
    assert result.shape == x.shape, \
        f"Failed: Empty tensor shape {result.shape} != {x.shape}"
    assert len(result) == 0, \
        f"Failed: Empty tensor should remain empty"
    print(f"  ✓ Empty tensor handled correctly")
    
    print("  ✓ Test 9 passed\n")


def test_quantized_weight_usage():
    """Test 10: Usage pattern as in Quantized_weight"""
    print("Test 10: Quantized_weight usage pattern...")
    
    # Simulate how Quantized_weight uses Quantization
    n_bits = 4
    quant_values = np.linspace(
        -(2 ** (n_bits - 1)), 2 ** (n_bits - 1) - 1, 2**n_bits
    )
    quant = Quantization(quant_values=quant_values)
    
    # Simulate weight matrix
    W = torch.randn(10, 20) * 5  # Random weights
    quantized_W = quant(W)
    
    # Verify all values are in quant_values
    quantized_W_np = quantized_W.numpy()
    unique_values = np.unique(quantized_W_np)
    for val in unique_values:
        assert val in quant_values, \
            f"Failed: Quantized value {val} not in quant_values"
    
    # Verify shape preserved
    assert quantized_W.shape == W.shape, \
        f"Failed: Shape mismatch {quantized_W.shape} != {W.shape}"
    
    print(f"  ✓ Quantized_weight usage pattern works correctly")
    print(f"  ✓ Output has {len(unique_values)} unique quantized values")
    print("  ✓ Test 10 passed\n")


def test_consistency_across_calls():
    """Test 11: Consistency across multiple calls"""
    print("Test 11: Consistency across calls...")
    
    quant_values = np.array([-1.0, 0.0, 1.0])
    quant = Quantization(quant_values)
    
    x = torch.tensor([-0.5, 0.0, 0.5])
    
    # Call multiple times
    result1 = quant(x)
    result2 = quant(x)
    result3 = quant(x)
    
    # All should be identical
    assert torch.allclose(result1, result2), \
        f"Failed: Results differ between calls"
    assert torch.allclose(result2, result3), \
        f"Failed: Results differ between calls"
    
    print(f"  ✓ Consistent results across multiple calls")
    print("  ✓ Test 11 passed\n")


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("Quantization Implementation Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_quantization,
        test_exact_quantization_levels,
        test_boundary_values,
        test_shape_preservation,
        test_numpy_array_input,
        test_different_bit_levels,
        test_device_consistency,
        test_extreme_values,
        test_empty_tensor,
        test_quantized_weight_usage,
        test_consistency_across_calls,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  ✗ FAILED: {e}\n")
            failed += 1
        except Exception as e:
            print(f"  ✗ ERROR: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

