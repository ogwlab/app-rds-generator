"""
Configuration module for RDS Generator
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class RDSConfig:
    """Configuration class for RDS generation parameters"""
    
    # Image parameters
    width: int = 512
    height: int = 512
    density: float = 50.0  # Dot density percentage
    dot_size: int = 2
    dot_shape: str = "四角"  # "四角" or "円"
    bg_color: str = "#FFFFFF"
    dot_color: str = "#000000"
    
    # Stereo parameters
    disparity_arcsec: float = 20.0  # Disparity in arcseconds
    distance_cm: float = 57.0  # Viewing distance in cm
    ppi: int = 96  # Pixels per inch
    
    # Shape parameters
    shape_type: str = "四角形"  # "四角形" or "円"
    shape_mode: str = "面"  # "面" or "枠線"
    border_width: int = 2
    shape_width: int = 256
    shape_height: int = 256
    center_x: int = 256
    center_y: int = 256
    
    # Noise parameters
    noise_std: float = 0.5
    
    # FFT parameters
    pad_width: int = 32
    
    # Random seed parameters
    random_seed: Optional[int] = 42  # Main random seed for dot generation (None for random)
    left_noise_seed: Optional[int] = 42  # Left eye noise seed
    right_noise_seed: Optional[int] = 43  # Right eye noise seed
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if not (128 <= self.width <= 1024):
            raise ValueError("Width must be between 128 and 1024")
        if not (128 <= self.height <= 1024):
            raise ValueError("Height must be between 128 and 1024")
        if not (1 <= self.density <= 100):
            raise ValueError("Density must be between 1 and 100")
        if not (1 <= self.dot_size <= 10):
            raise ValueError("Dot size must be between 1 and 10")
        if not (-600 <= self.disparity_arcsec <= 600):
            raise ValueError("Disparity must be between -600 and 600 arcsec")
        if not (30 <= self.distance_cm <= 200):
            raise ValueError("Distance must be between 30 and 200 cm")
        if not (72 <= self.ppi <= 400):
            raise ValueError("PPI must be between 72 and 400")
        
        # Validate shape parameters
        if self.shape_width <= 0:
            raise ValueError("Shape width must be positive")
        if self.shape_height <= 0:
            raise ValueError("Shape height must be positive")
        if self.shape_width > self.width:
            raise ValueError("Shape width cannot exceed image width")
        if self.shape_height > self.height:
            raise ValueError("Shape height cannot exceed image height")
        if not (0 <= self.center_x <= self.width):
            raise ValueError("Shape center X must be within image boundaries")
        if not (0 <= self.center_y <= self.height):
            raise ValueError("Shape center Y must be within image boundaries")
        if self.border_width <= 0:
            raise ValueError("Border width must be positive")
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary"""
        return {
            'width': self.width,
            'height': self.height,
            'density': self.density,
            'dot_size': self.dot_size,
            'dot_shape': self.dot_shape,
            'bg_color': self.bg_color,
            'dot_color': self.dot_color,
            'disparity_arcsec': self.disparity_arcsec,
            'distance_cm': self.distance_cm,
            'ppi': self.ppi,
            'shape_type': self.shape_type,
            'shape_mode': self.shape_mode,
            'border_width': self.border_width,
            'shape_width': self.shape_width,
            'shape_height': self.shape_height,
            'center_x': self.center_x,
            'center_y': self.center_y,
            'random_seed': self.random_seed,
            'left_noise_seed': self.left_noise_seed,
            'right_noise_seed': self.right_noise_seed,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'RDSConfig':
        """Create configuration from dictionary"""
        return cls(**config_dict)