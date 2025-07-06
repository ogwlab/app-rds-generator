"""
Random Dot Stereogram (RDS) Generator

Main class for generating RDS images with FFT phase shift method.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from .utils.math_utils import arcsec_to_pixels, clip_to_uint8
from .utils.image_utils import (
    generate_random_dots, create_shape_mask, add_background_noise,
    apply_fft_phase_shift
)
from .config.settings import (
    DEFAULT_PAD_WIDTH, DEFAULT_NOISE_STD, RANDOM_SEED_LEFT, 
    RANDOM_SEED_RIGHT, RANDOM_SEED_NOISE
)


class RDSGenerator:
    """Random Dot Stereogram Generator Class"""
    
    def __init__(self):
        """Initialize RDS Generator"""
        self.reset_cache()
    
    def reset_cache(self):
        """Reset cache"""
        self._cached_base_image = None
        self._cached_params = None
    
    def generate_stereo_pair(self, base_image: np.ndarray, disparity_pixels: float,
                           shape_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate stereo pair from base image.
        
        Args:
            base_image: Base random dot image
            disparity_pixels: Disparity in pixels
            shape_mask: Optional shape mask for hidden figure
            
        Returns:
            Tuple of (left_image, right_image) as uint8 arrays
        """
        # Add different noise to left and right images
        left_base = add_background_noise(
            base_image, shape_mask, 
            noise_std=DEFAULT_NOISE_STD, seed=RANDOM_SEED_LEFT
        )
        right_base = add_background_noise(
            base_image, shape_mask,
            noise_std=DEFAULT_NOISE_STD, seed=RANDOM_SEED_RIGHT
        )
        
        # Apply FFT phase shift to entire images
        left_shifted = apply_fft_phase_shift(left_base, -disparity_pixels / 2)
        right_shifted = apply_fft_phase_shift(right_base, disparity_pixels / 2)
        
        # If shape mask is provided, apply shifts only to shape regions
        if shape_mask is not None:
            left_image = np.where(shape_mask, left_shifted, left_base)
            right_image = np.where(shape_mask, right_shifted, right_base)
        else:
            # No mask - entire image is shifted
            left_image = left_shifted
            right_image = right_shifted
        
        # Convert to uint8
        left_image = clip_to_uint8(left_image)
        right_image = clip_to_uint8(right_image)
        
        return left_image, right_image
    
    def generate_rds(self, params: Dict) -> Dict:
        """
        Generate complete RDS from parameters.
        
        Args:
            params: Dictionary containing all generation parameters
            
        Returns:
            Dictionary containing generated images and metadata
        """
        # Generate base random dot image
        base_image = generate_random_dots(
            params['width'], params['height'], params['density'],
            params['dot_size'], params['dot_shape'], 
            params['bg_color'], params['dot_color']
        )
        
        # Create shape mask
        shape_mask = create_shape_mask(
            params['width'], params['height'], params['shape_type'],
            params['shape_mode'], params['border_width'], 
            params['shape_width'], params['shape_height'],
            params['center_x'], params['center_y']
        )
        
        # Convert disparity to pixels
        disparity_pixels = arcsec_to_pixels(
            params['disparity_arcsec'], params['distance_cm'], params['ppi']
        )
        
        # Generate stereo pair
        left_image, right_image = self.generate_stereo_pair(
            base_image, disparity_pixels, shape_mask
        )
        
        return {
            'left_image': left_image,
            'right_image': right_image,
            'disparity_pixels': disparity_pixels,
            'base_image': base_image,
            'shape_mask': shape_mask
        }