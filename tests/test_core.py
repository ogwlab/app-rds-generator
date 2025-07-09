"""
Tests for core RDS generation functionality
"""

import numpy as np
import pytest
from rds_generator.core import RDSGenerator
from rds_generator.config import RDSConfig


class TestRDSGenerator:
    """Test RDS generator core functionality"""
    
    @pytest.fixture
    def generator(self):
        """Test fixture for RDS generator"""
        return RDSGenerator()
    
    @pytest.fixture
    def config(self):
        """Test fixture for RDS configuration"""
        return RDSConfig(
            width=128, height=128, density=25.0,
            disparity_arcsec=20.0, distance_cm=57.0, ppi=96
        )
    
    def test_generator_initialization(self):
        """Test RDS generator initialization"""
        generator = RDSGenerator()
        assert generator._cached_base_image is None
        assert generator._cached_params is None
    
    def test_generate_stereo_pair(self):
        """Test stereo pair generation"""
        generator = RDSGenerator()
        
        # Create test base image
        base_image = np.random.rand(32, 32) * 255
        disparity_pixels = 2.0
        
        # Generate without mask
        left, right = generator.generate_stereo_pair(base_image, disparity_pixels)
        
        assert left.shape == base_image.shape
        assert right.shape == base_image.shape
        assert left.dtype == np.uint8
        assert right.dtype == np.uint8
        assert not np.array_equal(left, right)
        
        # Generate with mask
        mask = np.zeros((32, 32), dtype=bool)
        mask[10:20, 10:20] = True
        
        left_masked, right_masked = generator.generate_stereo_pair(
            base_image, disparity_pixels, mask
        )
        
        assert left_masked.shape == base_image.shape
        assert right_masked.shape == base_image.shape
        assert left_masked.dtype == np.uint8
        assert right_masked.dtype == np.uint8
    
    def test_generate_rds_full(self):
        """Test full RDS generation"""
        generator = RDSGenerator()
        config = RDSConfig(
            width=64, height=64, density=25.0,
            disparity_arcsec=20.0, distance_cm=57.0, ppi=96,
            shape_width=32, shape_height=32,
            random_seed=42, left_noise_seed=100, right_noise_seed=200
        )
        
        result = generator.generate_rds(config)
        
        assert 'left_image' in result
        assert 'right_image' in result
        assert 'disparity_pixels' in result
        assert 'base_image' in result
        assert 'shape_mask' in result
        
        assert result['left_image'].shape == (64, 64)
        assert result['right_image'].shape == (64, 64)
        assert result['left_image'].dtype == np.uint8
        assert result['right_image'].dtype == np.uint8
        assert isinstance(result['disparity_pixels'], float)
        assert result['shape_mask'].dtype == bool
    
    def test_generate_rds_from_dict(self):
        """Test RDS generation from dictionary (backward compatibility)"""
        generator = RDSGenerator()
        params = {
            'width': 64, 'height': 64, 'density': 25.0,
            'dot_size': 2, 'dot_shape': '四角',
            'bg_color': '#FFFFFF', 'dot_color': '#000000',
            'disparity_arcsec': 20.0, 'distance_cm': 57.0, 'ppi': 96,
            'shape_type': '四角形', 'shape_mode': '面',
            'border_width': 2, 'shape_width': 32, 'shape_height': 32,
            'center_x': 32, 'center_y': 32
        }
        
        result = generator.generate_rds_from_dict(params)
        
        assert 'left_image' in result
        assert 'right_image' in result
        assert 'disparity_pixels' in result
        assert result['left_image'].shape == (64, 64)
        assert result['right_image'].shape == (64, 64)
    
    def test_zero_disparity(self):
        """Test RDS generation with zero disparity"""
        generator = RDSGenerator()
        config = RDSConfig(
            width=32, height=32, density=25.0,
            disparity_arcsec=0.0
        )
        
        result = generator.generate_rds(config)
        
        assert result['disparity_pixels'] == 0.0
        assert result['left_image'].shape == (32, 32)
        assert result['right_image'].shape == (32, 32)
    
    def test_negative_disparity(self):
        """Test RDS generation with negative disparity"""
        generator = RDSGenerator()
        config = RDSConfig(
            width=32, height=32, density=25.0,
            disparity_arcsec=-20.0
        )
        
        result = generator.generate_rds(config)
        
        assert result['disparity_pixels'] < 0.0
        assert result['left_image'].shape == (32, 32)
        assert result['right_image'].shape == (32, 32)
        assert not np.array_equal(result['left_image'], result['right_image'])
    
    def test_cache_reset(self):
        """Test cache reset functionality"""
        generator = RDSGenerator()
        
        # Set some cache values
        generator._cached_base_image = np.ones((10, 10))
        generator._cached_params = {'test': 'value'}
        
        # Reset cache
        generator.reset_cache()
        
        assert generator._cached_base_image is None
        assert generator._cached_params is None
    
    def test_different_shapes(self):
        """Test RDS generation with different shapes"""
        generator = RDSGenerator()
        
        # Test rectangle
        config_rect = RDSConfig(
            width=32, height=32, density=25.0,
            shape_type='四角形', shape_mode='面',
            shape_width=16, shape_height=16,
            random_seed=42
        )
        result_rect = generator.generate_rds(config_rect)
        
        # Test circle
        config_circle = RDSConfig(
            width=32, height=32, density=25.0,
            shape_type='円', shape_mode='面',
            shape_width=16, shape_height=16,
            random_seed=42
        )
        result_circle = generator.generate_rds(config_circle)
        
        # Both should generate valid results
        assert result_rect['left_image'].shape == (32, 32)
        assert result_circle['left_image'].shape == (32, 32)
        assert not np.array_equal(result_rect['shape_mask'], result_circle['shape_mask'])
    
    def test_reproducibility_with_seeds(self):
        """Test reproducibility with fixed seeds"""
        generator = RDSGenerator()
        config = RDSConfig(
            width=32, height=32, density=25.0,
            disparity_arcsec=20.0,
            random_seed=42, left_noise_seed=100, right_noise_seed=200
        )
        
        # Generate twice with same seeds
        result1 = generator.generate_rds(config)
        result2 = generator.generate_rds(config)
        
        # Results should be identical
        np.testing.assert_array_equal(result1['left_image'], result2['left_image'])
        np.testing.assert_array_equal(result1['right_image'], result2['right_image'])
        np.testing.assert_array_equal(result1['base_image'], result2['base_image'])
    
    def test_different_seeds_produce_different_results(self):
        """Test that different seeds produce different results"""
        generator = RDSGenerator()
        config1 = RDSConfig(width=32, height=32, density=25.0, random_seed=42)
        config2 = RDSConfig(width=32, height=32, density=25.0, random_seed=123)
        
        result1 = generator.generate_rds(config1)
        result2 = generator.generate_rds(config2)
        
        # Base images should be different
        assert not np.array_equal(result1['base_image'], result2['base_image'])