"""
Tests for RDS configuration
"""

import pytest
from rds_generator.config import RDSConfig


class TestRDSConfig:
    """Test RDS configuration class"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = RDSConfig()
        
        assert config.width == 512
        assert config.height == 512
        assert config.density == 50.0
        assert config.dot_size == 2
        assert config.disparity_arcsec == 20.0
        assert config.distance_cm == 57.0
        assert config.ppi == 96
    
    def test_custom_config(self):
        """Test custom configuration values"""
        config = RDSConfig(
            width=256,
            height=256,
            density=30.0,
            disparity_arcsec=40.0,
            random_seed=100,
            left_noise_seed=200,
            right_noise_seed=300
        )
        
        assert config.width == 256
        assert config.height == 256
        assert config.density == 30.0
        assert config.disparity_arcsec == 40.0
        assert config.random_seed == 100
        assert config.left_noise_seed == 200
        assert config.right_noise_seed == 300
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid width
        with pytest.raises(ValueError, match="Width must be between 128 and 1024"):
            RDSConfig(width=50)
        
        # Test invalid height
        with pytest.raises(ValueError, match="Height must be between 128 and 1024"):
            RDSConfig(height=2000)
        
        # Test invalid density
        with pytest.raises(ValueError, match="Density must be between 1 and 100"):
            RDSConfig(density=150)
        
        # Test invalid dot size
        with pytest.raises(ValueError, match="Dot size must be between 1 and 10"):
            RDSConfig(dot_size=0)
        
        with pytest.raises(ValueError, match="Dot size must be between 1 and 10"):
            RDSConfig(dot_size=11)
        
        # Test invalid disparity
        with pytest.raises(ValueError, match="Disparity must be between -600 and 600"):
            RDSConfig(disparity_arcsec=1000)
        
        # Test invalid distance
        with pytest.raises(ValueError, match="Distance must be between 30 and 200"):
            RDSConfig(distance_cm=300)
        
        # Test invalid PPI
        with pytest.raises(ValueError, match="PPI must be between 72 and 400"):
            RDSConfig(ppi=500)
        
        # Test invalid shape parameters
        with pytest.raises(ValueError, match="Shape width must be positive"):
            RDSConfig(shape_width=0)
        
        with pytest.raises(ValueError, match="Shape height must be positive"):
            RDSConfig(shape_height=0)
        
        with pytest.raises(ValueError, match="Shape width cannot exceed image width"):
            RDSConfig(width=256, shape_width=300)
        
        with pytest.raises(ValueError, match="Shape height cannot exceed image height"):
            RDSConfig(height=256, shape_height=300)
        
        with pytest.raises(ValueError, match="Shape center X must be within image boundaries"):
            RDSConfig(width=256, center_x=300)
        
        with pytest.raises(ValueError, match="Shape center Y must be within image boundaries"):
            RDSConfig(height=256, center_y=300)
        
        with pytest.raises(ValueError, match="Border width must be positive"):
            RDSConfig(border_width=0)
    
    def test_to_dict(self):
        """Test conversion to dictionary"""
        config = RDSConfig(width=256, height=256, density=30.0, random_seed=123)
        config_dict = config.to_dict()
        
        assert config_dict['width'] == 256
        assert config_dict['height'] == 256
        assert config_dict['density'] == 30.0
        assert config_dict['random_seed'] == 123
        assert 'disparity_arcsec' in config_dict
        assert 'distance_cm' in config_dict
        assert 'left_noise_seed' in config_dict
        assert 'right_noise_seed' in config_dict
    
    def test_from_dict(self):
        """Test creation from dictionary"""
        config_dict = {
            'width': 256,
            'height': 256,
            'density': 30.0,
            'disparity_arcsec': 40.0,
            'distance_cm': 50.0,
            'ppi': 100,
            'random_seed': 999,
            'left_noise_seed': 888,
            'right_noise_seed': 777
        }
        
        config = RDSConfig.from_dict(config_dict)
        
        assert config.width == 256
        assert config.height == 256
        assert config.density == 30.0
        assert config.disparity_arcsec == 40.0
        assert config.distance_cm == 50.0
        assert config.ppi == 100
        assert config.random_seed == 999
        assert config.left_noise_seed == 888
        assert config.right_noise_seed == 777
    
    def test_random_seed_none(self):
        """Test configuration with None random seeds"""
        config = RDSConfig(random_seed=None, left_noise_seed=None, right_noise_seed=None)
        
        assert config.random_seed is None
        assert config.left_noise_seed is None
        assert config.right_noise_seed is None