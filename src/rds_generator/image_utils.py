"""
Image processing utilities for RDS generation
"""

import numpy as np
from PIL import Image, ImageDraw
from typing import Tuple, Optional


def generate_random_dots(width: int, height: int, density: float, 
                        dot_size: int, dot_shape: str, bg_color: str, 
                        dot_color: str, random_seed: Optional[int] = None) -> np.ndarray:
    """
    Generate random dot image
    
    Args:
        width: Image width
        height: Image height
        density: Dot density percentage
        dot_size: Size of each dot
        dot_shape: Shape of dots ("四角" or "円")
        bg_color: Background color
        dot_color: Dot color
        random_seed: Random seed for reproducibility (None for random)
        
    Returns:
        Generated random dot image as numpy array
    """
    # Create PIL image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate number of dots
    total_pixels = width * height
    num_dots = int(total_pixels * density / 100.0 / (dot_size * dot_size))
    
    # Set seed for reproducibility (if specified)
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Place dots at random positions (fixed boundary issue)
    for _ in range(num_dots):
        x = np.random.randint(0, width - dot_size + 1)
        y = np.random.randint(0, height - dot_size + 1)
        
        if dot_shape == '四角':
            draw.rectangle([x, y, x + dot_size, y + dot_size], fill=dot_color)
        else:  # 円
            draw.ellipse([x, y, x + dot_size, y + dot_size], fill=dot_color)
    
    # Convert to numpy array (grayscale)
    img_gray = img.convert('L')
    return np.array(img_gray, dtype=np.float64)


def create_shape_mask(width: int, height: int, shape_type: str,
                     shape_mode: str, border_width: int, shape_width: int,
                     shape_height: int, center_x: int, center_y: int) -> np.ndarray:
    """
    Create shape mask for RDS generation
    
    Args:
        width: Image width
        height: Image height
        shape_type: Type of shape ("四角形" or "円")
        shape_mode: Mode of shape ("面" or "枠線")
        border_width: Border width for outline mode
        shape_width: Width of shape
        shape_height: Height of shape
        center_x: Center X coordinate
        center_y: Center Y coordinate
        
    Returns:
        Boolean mask array
    """
    mask = np.zeros((height, width), dtype=bool)
    
    if shape_type == '四角形':
        # Rectangle bounds
        left = max(0, center_x - shape_width // 2)
        right = min(width, center_x + shape_width // 2)
        top = max(0, center_y - shape_height // 2)
        bottom = min(height, center_y + shape_height // 2)
        
        if shape_mode == '面':
            mask[top:bottom, left:right] = True
        else:  # 枠線
            # Outer frame
            mask[top:bottom, left:right] = True
            # Remove inner area
            inner_left = left + border_width
            inner_right = right - border_width
            inner_top = top + border_width
            inner_bottom = bottom - border_width
            if inner_left < inner_right and inner_top < inner_bottom:
                mask[inner_top:inner_bottom, inner_left:inner_right] = False
    
    else:  # 円
        y_coords, x_coords = np.ogrid[:height, :width]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        radius = min(shape_width, shape_height) // 2
        
        if shape_mode == '面':
            mask = distances <= radius
        else:  # 枠線
            mask = (distances <= radius) & (distances >= radius - border_width)
    
    return mask


def add_minimal_background_noise(image: np.ndarray, mask: np.ndarray, 
                               noise_std: float = 0.5, 
                               random_seed: Optional[int] = None) -> np.ndarray:
    """
    Add minimal Gaussian noise to background region
    
    Args:
        image: Input image
        mask: Shape mask (True for shape region)
        noise_std: Standard deviation of noise
        random_seed: Random seed for noise generation (None for random)
        
    Returns:
        Image with added noise
    """
    noisy_image = image.copy()
    
    # Background region (inverse of mask)
    background_mask = ~mask
    
    # Generate noise with optional seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    noise = np.random.normal(0, noise_std, image.shape)
    
    # Apply noise only to background region
    noisy_image[background_mask] += noise[background_mask]
    
    return noisy_image


def combine_stereo_images(left_image: np.ndarray, right_image: np.ndarray,
                         separator_width: int = 5) -> np.ndarray:
    """
    Combine left and right images into stereo pair
    
    Args:
        left_image: Left eye image
        right_image: Right eye image
        separator_width: Width of separator between images
        
    Returns:
        Combined stereo pair image
    """
    separator = np.zeros((left_image.shape[0], separator_width), dtype=np.uint8)
    stereo_pair = np.hstack([left_image, separator, right_image])
    return stereo_pair