"""
Tests for image utility functions
"""

import numpy as np
import pytest
from rds_generator.image_utils import (
    generate_random_dots,
    create_shape_mask,
    add_minimal_background_noise,
    combine_stereo_images
)


class TestImageUtils:
    """Test image utility functions"""
    
    def test_generate_random_dots(self):
        """Test random dot generation"""
        width, height = 64, 64
        density = 25.0
        dot_size = 2
        
        image = generate_random_dots(
            width, height, density, dot_size, 
            "四角", "#FFFFFF", "#000000"
        )
        
        assert image.shape == (height, width)
        assert image.dtype == np.float64
        assert image.min() >= 0
        assert image.max() <= 255
        
        # Test with circular dots
        image_circle = generate_random_dots(
            width, height, density, dot_size, 
            "円", "#FFFFFF", "#000000"
        )
        
        assert image_circle.shape == (height, width)
        assert image_circle.dtype == np.float64
    
    def test_create_shape_mask_rectangle(self):
        """Test rectangular shape mask creation"""
        width, height = 64, 64
        
        # Test filled rectangle
        mask = create_shape_mask(
            width, height, "四角形", "面", 2, 
            32, 32, 32, 32
        )
        
        assert mask.shape == (height, width)
        assert mask.dtype == bool
        assert mask.sum() > 0  # Should have some True values
        
        # Test rectangle outline
        mask_outline = create_shape_mask(
            width, height, "四角形", "枠線", 2, 
            32, 32, 32, 32
        )
        
        assert mask_outline.shape == (height, width)
        assert mask_outline.dtype == bool
        assert mask_outline.sum() > 0
        assert mask_outline.sum() < mask.sum()  # Outline should be smaller
    
    def test_create_shape_mask_circle(self):
        """Test circular shape mask creation"""
        width, height = 64, 64
        
        # Test filled circle
        mask = create_shape_mask(
            width, height, "円", "面", 2, 
            32, 32, 32, 32
        )
        
        assert mask.shape == (height, width)
        assert mask.dtype == bool
        assert mask.sum() > 0
        
        # Test circle outline
        mask_outline = create_shape_mask(
            width, height, "円", "枠線", 2, 
            32, 32, 32, 32
        )
        
        assert mask_outline.shape == (height, width)
        assert mask_outline.dtype == bool
        assert mask_outline.sum() > 0
        assert mask_outline.sum() < mask.sum()
    
    def test_add_minimal_background_noise(self):
        """Test background noise addition"""
        width, height = 32, 32
        image = np.ones((height, width)) * 128  # Gray image
        mask = np.zeros((height, width), dtype=bool)
        mask[10:20, 10:20] = True  # Small shape region
        
        noisy_image = add_minimal_background_noise(image, mask, noise_std=1.0)
        
        assert noisy_image.shape == image.shape
        assert noisy_image.dtype == image.dtype
        
        # Shape region should be unchanged
        np.testing.assert_array_equal(noisy_image[mask], image[mask])
        
        # Background region should be different
        background_mask = ~mask
        assert not np.array_equal(noisy_image[background_mask], image[background_mask])
    
    def test_combine_stereo_images(self):
        """Test stereo image combination"""
        left_image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        right_image = np.random.randint(0, 255, (32, 32), dtype=np.uint8)
        
        combined = combine_stereo_images(left_image, right_image, separator_width=5)
        
        expected_width = left_image.shape[1] + right_image.shape[1] + 5
        assert combined.shape == (32, expected_width)
        assert combined.dtype == np.uint8
        
        # Check that left image is at the beginning
        np.testing.assert_array_equal(combined[:, :32], left_image)
        
        # Check that right image is at the end
        np.testing.assert_array_equal(combined[:, -32:], right_image)
        
        # Check separator is zeros
        separator_region = combined[:, 32:37]
        assert np.all(separator_region == 0)
    
    def test_shape_mask_edge_cases(self):
        """Test shape mask edge cases"""
        width, height = 32, 32
        
        # Very small shape
        mask = create_shape_mask(
            width, height, "四角形", "面", 1, 
            2, 2, 16, 16
        )
        assert mask.shape == (height, width)
        assert mask.sum() > 0
        
        # Shape at edge
        mask_edge = create_shape_mask(
            width, height, "四角形", "面", 1, 
            20, 20, 2, 2
        )
        assert mask_edge.shape == (height, width)
        assert mask_edge.sum() > 0
        
        # Shape larger than image
        mask_large = create_shape_mask(
            width, height, "四角形", "面", 1, 
            100, 100, 16, 16
        )
        assert mask_large.shape == (height, width)
        assert mask_large.sum() > 0