"""
Core RDS generation functionality
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .config import RDSConfig
from .math_utils import arcsec_to_pixels, apply_phase_shift_2d_robust
from .image_utils import (
    generate_random_dots, 
    create_shape_mask, 
    add_minimal_background_noise
)


class RDSGenerator:
    """Random Dot Stereogram Generator"""
    
    def __init__(self):
        self.reset_cache()
    
    def reset_cache(self):
        """Reset internal cache"""
        self._cached_base_image = None
        self._cached_params = None
    
    def generate_stereo_pair(self, base_image: np.ndarray, disparity_pixels: float,
                           shape_mask: Optional[np.ndarray] = None,
                           left_noise_seed: Optional[int] = None,
                           right_noise_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stereo pair from base image
        
        Args:
            base_image: Base random dot image
            disparity_pixels: Disparity in pixels
            shape_mask: Optional shape mask
            left_noise_seed: Random seed for left eye noise (None for random)
            right_noise_seed: Random seed for right eye noise (None for random)
            
        Returns:
            Tuple of (left_image, right_image)
        """
        # Add minimal background noise (different for left and right)
        left_base = add_minimal_background_noise(base_image, shape_mask, 
                                               noise_std=0.5, random_seed=left_noise_seed)
        
        right_base = add_minimal_background_noise(base_image, shape_mask, 
                                                noise_std=0.5, random_seed=right_noise_seed)
        
        # Apply FFT phase shift to entire image
        left_shifted = apply_phase_shift_2d_robust(left_base, -disparity_pixels / 2)
        right_shifted = apply_phase_shift_2d_robust(right_base, disparity_pixels / 2)

        # If mask is provided, apply shift only to shape region
        if shape_mask is not None:
            left_image = np.where(shape_mask, left_shifted, left_base)
            right_image = np.where(shape_mask, right_shifted, right_base)
        else:
            # No mask - entire image is shifted
            left_image = left_shifted
            right_image = right_shifted

        # Clip values to 0-255 range
        left_image = np.clip(left_image, 0, 255)
        right_image = np.clip(right_image, 0, 255)
        
        return left_image.astype(np.uint8), right_image.astype(np.uint8)
    
    def generate_rds(self, config: RDSConfig) -> Dict[str, np.ndarray]:
        """
        Generate RDS from configuration
        
        Args:
            config: RDS configuration
            
        Returns:
            Dictionary containing generated images and metadata
        """
        # Generate base random dot image
        base_image = generate_random_dots(
            config.width, config.height, config.density,
            config.dot_size, config.dot_shape, 
            config.bg_color, config.dot_color,
            config.random_seed
        )
        
        # Create shape mask
        shape_mask = create_shape_mask(
            config.width, config.height, config.shape_type,
            config.shape_mode, config.border_width, 
            config.shape_width, config.shape_height,
            config.center_x, config.center_y
        )
        
        # Convert disparity to pixels
        disparity_pixels = arcsec_to_pixels(
            config.disparity_arcsec, config.distance_cm, config.ppi
        )
        
        # Generate stereo pair
        left_image, right_image = self.generate_stereo_pair(
            base_image, disparity_pixels, shape_mask,
            config.left_noise_seed, config.right_noise_seed
        )
        
        return {
            'left_image': left_image,
            'right_image': right_image,
            'disparity_pixels': disparity_pixels,
            'base_image': base_image.astype(np.uint8),
            'shape_mask': shape_mask,
        }
    
    def generate_rds_from_dict(self, params: Dict) -> Dict[str, np.ndarray]:
        """
        Generate RDS from parameter dictionary (for backward compatibility)
        
        Args:
            params: Parameter dictionary
            
        Returns:
            Dictionary containing generated images and metadata
        """
        config = RDSConfig.from_dict(params)
        return self.generate_rds(config)