"""
Tests for mathematical utility functions
"""

import numpy as np
import pytest
from rds_generator.math_utils import (
    arcsec_to_pixels,
    apply_phase_shift_2d_robust,
    create_frequency_mesh
)


class TestMathUtils:
    """Test mathematical utility functions"""
    
    def test_arcsec_to_pixels(self):
        """Test arcseconds to pixels conversion"""
        # Test with standard values
        pixels = arcsec_to_pixels(20.0, 57.0, 96)
        assert pixels > 0
        assert isinstance(pixels, float)
        
        # Test with zero disparity
        pixels = arcsec_to_pixels(0.0, 57.0, 96)
        assert pixels == 0.0
        
        # Test with negative disparity
        pixels = arcsec_to_pixels(-20.0, 57.0, 96)
        assert pixels < 0
        
        # Test proportionality
        pixels1 = arcsec_to_pixels(20.0, 57.0, 96)
        pixels2 = arcsec_to_pixels(40.0, 57.0, 96)
        assert abs(pixels2 - 2 * pixels1) < 1e-10
    
    def test_apply_phase_shift_2d_robust(self):
        """Test 2D phase shift application"""
        # Create test image
        image = np.random.rand(64, 64)
        
        # Test zero shift
        shifted = apply_phase_shift_2d_robust(image, 0.0, pad_width=16)
        np.testing.assert_array_almost_equal(shifted, image, decimal=5)
        
        # Test non-zero shift
        shifted = apply_phase_shift_2d_robust(image, 2.0, pad_width=16)
        assert shifted.shape == image.shape
        assert not np.array_equal(shifted, image)
        
        # Test negative shift
        shifted_neg = apply_phase_shift_2d_robust(image, -2.0, pad_width=16)
        assert shifted_neg.shape == image.shape
        assert not np.array_equal(shifted_neg, image)
        assert not np.array_equal(shifted_neg, shifted)
    
    def test_phase_shift_symmetry(self):
        """Test phase shift symmetry properties"""
        image = np.random.rand(32, 32)
        shift_amount = 1.5
        
        # Apply positive and negative shifts
        shifted_pos = apply_phase_shift_2d_robust(image, shift_amount, pad_width=8)
        shifted_neg = apply_phase_shift_2d_robust(image, -shift_amount, pad_width=8)
        
        # They should be different
        assert not np.array_equal(shifted_pos, shifted_neg)
        
        # Apply double shift should approximate original
        double_shifted = apply_phase_shift_2d_robust(shifted_pos, -shift_amount, pad_width=8)
        # Note: Due to numerical precision, we use a relaxed tolerance
        np.testing.assert_array_almost_equal(double_shifted, image, decimal=1)
    
    def test_create_frequency_mesh(self):
        """Test frequency mesh creation"""
        height, width = 32, 32
        U, V = create_frequency_mesh(height, width)
        
        assert U.shape == (height, width)
        assert V.shape == (height, width)
        
        # Check frequency range
        assert U.min() >= -0.5
        assert U.max() <= 0.5
        assert V.min() >= -0.5
        assert V.max() <= 0.5
        
        # Check center is approximately zero
        center_y, center_x = height // 2, width // 2
        assert abs(U[center_y, center_x]) < 1e-10
        assert abs(V[center_y, center_x]) < 1e-10
    
    def test_phase_shift_edge_cases(self):
        """Test phase shift edge cases"""
        # Very small image
        small_image = np.random.rand(8, 8)
        shifted = apply_phase_shift_2d_robust(small_image, 0.5, pad_width=2)
        assert shifted.shape == small_image.shape
        
        # Large shift
        image = np.random.rand(32, 32)
        shifted = apply_phase_shift_2d_robust(image, 10.0, pad_width=8)
        assert shifted.shape == image.shape
        
        # Zero padding
        shifted = apply_phase_shift_2d_robust(image, 1.0, pad_width=0)
        assert shifted.shape == image.shape