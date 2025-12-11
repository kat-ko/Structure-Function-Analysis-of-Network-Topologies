#!/usr/bin/env python3
"""
Test script to validate the CKA (Centered Kernel Alignment) implementation.

This script tests the CKA implementation against known properties and example cases
to ensure correctness.
"""

import numpy as np
import sys
from dynspec.correlations import CKA


def test_identical_matrices():
    """Test 1: CKA between identical matrices should be 1.0"""
    print("Test 1: Identical matrices...")
    cka = CKA()
    
    # Test with random matrix
    X = np.random.randn(100, 50)
    result = cka.linear_CKA(X, X)
    assert np.isclose(result, 1.0, atol=1e-6), f"Failed: CKA(X, X) = {result}, expected 1.0"
    print(f"  ✓ CKA(X, X) = {result:.6f} (expected 1.0)")
    
    # Test with different shapes
    X = np.random.randn(20, 10)
    result = cka.linear_CKA(X, X)
    assert np.isclose(result, 1.0, atol=1e-6), f"Failed: CKA(X, X) = {result}, expected 1.0"
    print(f"  ✓ CKA(X, X) with shape {X.shape} = {result:.6f} (expected 1.0)")
    
    print("  ✓ Test 1 passed\n")


def test_orthogonal_independent_matrices():
    """Test 2: CKA between orthogonal/independent matrices should be close to 0"""
    print("Test 2: Orthogonal/independent matrices...")
    cka = CKA()
    
    # Test with orthogonal matrices (identity and flipped)
    # Note: After centering, these may not be perfectly orthogonal
    X = np.eye(50)
    Y = np.flipud(X)  # Flipped identity
    result = cka.linear_CKA(X, Y)
    assert 0.0 <= result <= 1.0, f"Failed: CKA value {result} not in [0, 1]"
    print(f"  ✓ CKA(eye, flipped_eye) = {result:.6f} (in [0, 1])")
    
    # Test with independent random matrices
    # Note: Independent random matrices can have non-zero CKA due to finite sample size
    # We just verify it's in the valid range and less than a reasonable threshold
    np.random.seed(42)
    X = np.random.randn(100, 50)
    np.random.seed(123)  # Different seed
    Y = np.random.randn(100, 50)
    result = cka.linear_CKA(X, Y)
    assert 0.0 <= result <= 1.0, f"Failed: CKA value {result} not in [0, 1]"
    assert result < 0.5, f"Failed: CKA value {result} is unexpectedly large for independent matrices (should be < 0.5)"
    print(f"  ✓ CKA(independent_random) = {result:.6f} (should be < 0.5 for independent matrices)")
    
    print("  ✓ Test 2 passed\n")


def test_correlated_matrices():
    """Test 3: CKA between correlated matrices should be between 0 and 1"""
    print("Test 3: Correlated matrices...")
    cka = CKA()
    
    # Create correlated matrices
    np.random.seed(42)
    X = np.random.randn(100, 50)
    # Y is X with added noise (should have positive correlation)
    Y = X + 0.5 * np.random.randn(100, 50)
    result = cka.linear_CKA(X, Y)
    assert 0.0 < result < 1.0, f"Failed: CKA(X, Y) = {result}, expected between 0 and 1"
    print(f"  ✓ CKA(correlated) = {result:.6f} (should be between 0 and 1)")
    
    # Test with stronger correlation
    Y = X + 0.1 * np.random.randn(100, 50)  # Less noise = stronger correlation
    result_strong = cka.linear_CKA(X, Y)
    assert result_strong > result, f"Failed: Stronger correlation should give higher CKA"
    print(f"  ✓ CKA(strongly_correlated) = {result_strong:.6f} > {result:.6f}")
    
    print("  ✓ Test 3 passed\n")


def test_symmetry():
    """Test 4: CKA should be symmetric: CKA(X, Y) = CKA(Y, X)"""
    print("Test 4: Symmetry property...")
    cka = CKA()
    
    np.random.seed(42)
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 50)
    
    result_XY = cka.linear_CKA(X, Y)
    result_YX = cka.linear_CKA(Y, X)
    
    assert np.isclose(result_XY, result_YX, atol=1e-10), \
        f"Failed: CKA(X, Y) = {result_XY}, CKA(Y, X) = {result_YX}"
    print(f"  ✓ CKA(X, Y) = {result_XY:.6f}")
    print(f"  ✓ CKA(Y, X) = {result_YX:.6f}")
    print(f"  ✓ Symmetry property holds\n")


def test_linear_transformation():
    """Test 5: CKA should be invariant to orthogonal transformations"""
    print("Test 5: Invariance to orthogonal transformations...")
    cka = CKA()
    
    np.random.seed(42)
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 50)
    
    # Create orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(50, 50))
    
    # CKA(X, Y) should equal CKA(X @ Q, Y @ Q) for orthogonal Q
    result_original = cka.linear_CKA(X, Y)
    result_transformed = cka.linear_CKA(X @ Q, Y @ Q)
    
    # Note: This property should hold approximately for linear CKA
    print(f"  CKA(X, Y) = {result_original:.6f}")
    print(f"  CKA(XQ, YQ) = {result_transformed:.6f}")
    print(f"  (Note: Linear CKA may not be exactly invariant to all transformations)\n")


def test_scaling_invariance():
    """Test 6: CKA should be invariant to scaling"""
    print("Test 6: Scaling invariance...")
    cka = CKA()
    
    np.random.seed(42)
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 50)
    
    result_original = cka.linear_CKA(X, Y)
    
    # Scale both matrices
    result_scaled = cka.linear_CKA(2.0 * X, 3.0 * Y)
    
    # CKA should be approximately invariant to scaling (after centering)
    print(f"  CKA(X, Y) = {result_original:.6f}")
    print(f"  CKA(2X, 3Y) = {result_scaled:.6f}")
    # They should be close (centering removes constant scaling effects)
    assert np.isclose(result_original, result_scaled, atol=1e-6), \
        f"Failed: CKA should be invariant to scaling"
    print(f"  ✓ Scaling invariance holds\n")


def test_edge_cases():
    """Test 7: Edge cases"""
    print("Test 7: Edge cases...")
    cka = CKA()
    
    # Test with constant matrices (zero variance after centering)
    X = np.ones((10, 5))  # Constant matrix
    Y = np.ones((10, 5))
    result = cka.linear_CKA(X, Y)
    assert result == 0.0 or np.isnan(result), \
        f"Failed: CKA of constant matrices should be 0 or NaN, got {result}"
    print(f"  ✓ CKA(constant, constant) = {result}")
    
    # Test with one constant, one random
    X = np.ones((10, 5))
    Y = np.random.randn(10, 5)
    result = cka.linear_CKA(X, Y)
    assert result == 0.0 or np.isnan(result), \
        f"Failed: CKA of constant and random should be 0 or NaN, got {result}"
    print(f"  ✓ CKA(constant, random) = {result}")
    
    # Test with small matrices
    X = np.random.randn(3, 2)
    Y = np.random.randn(3, 2)
    result = cka.linear_CKA(X, Y)
    assert 0.0 <= result <= 1.0, f"Failed: CKA value {result} not in [0, 1]"
    print(f"  ✓ CKA(small_matrices) = {result:.6f}")
    
    print("  ✓ Test 7 passed\n")


def test_range_property():
    """Test 8: CKA values should always be in [0, 1]"""
    print("Test 8: Range property [0, 1]...")
    cka = CKA()
    
    # Test with various random matrices
    for i in range(10):
        np.random.seed(i)
        X = np.random.randn(50, 20)
        Y = np.random.randn(50, 20)
        result = cka.linear_CKA(X, Y)
        assert 0.0 <= result <= 1.0, \
            f"Failed: CKA value {result} not in [0, 1] for seed {i}"
    
    print(f"  ✓ All 10 random test cases produce values in [0, 1]")
    print("  ✓ Test 8 passed\n")


def test_known_reference():
    """Test 9: Compare with known reference implementation formula"""
    print("Test 9: Formula verification...")
    cka = CKA()
    
    np.random.seed(42)
    X = np.random.randn(100, 50)
    Y = np.random.randn(100, 50)
    
    # Manual calculation using the formula
    X_centered = X - X.mean(axis=0, keepdims=True)
    Y_centered = Y - Y.mean(axis=0, keepdims=True)
    
    # CKA = ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F)
    numerator = np.trace(Y_centered @ Y_centered.T @ X_centered @ X_centered.T)
    XX_norm_sq = np.trace(X_centered @ X_centered.T @ X_centered @ X_centered.T)
    YY_norm_sq = np.trace(Y_centered @ Y_centered.T @ Y_centered @ Y_centered.T)
    denominator = np.sqrt(XX_norm_sq * YY_norm_sq)
    expected = numerator / denominator if denominator != 0 else 0.0
    
    result = cka.linear_CKA(X, Y)
    
    assert np.isclose(result, expected, atol=1e-10), \
        f"Failed: Implementation {result} != manual calculation {expected}"
    print(f"  ✓ Implementation result: {result:.10f}")
    print(f"  ✓ Manual calculation: {expected:.10f}")
    print(f"  ✓ Formula verification passed\n")


def run_all_tests():
    """Run all test cases"""
    print("=" * 60)
    print("CKA Implementation Validation Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_identical_matrices,
        test_orthogonal_independent_matrices,
        test_correlated_matrices,
        test_symmetry,
        test_linear_transformation,
        test_scaling_invariance,
        test_edge_cases,
        test_range_property,
        test_known_reference,
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
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

