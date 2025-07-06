"""
Image utility functions for RDS generation
"""

import numpy as np
import scipy.fft as fft
from PIL import Image, ImageDraw
from typing import Optional
from .math_utils import create_2d_frequency_mesh, calculate_phase_shift


def generate_random_dots(width: int, height: int, density: float, 
                        dot_size: int, dot_shape: str, bg_color: str, 
                        dot_color: str) -> np.ndarray:
    """
    Generate random dot image.
    
    Args:
        width: Image width in pixels
        height: Image height in pixels
        density: Dot density percentage
        dot_size: Size of each dot in pixels
        dot_shape: Shape of dots ('四角' or '円')
        bg_color: Background color
        dot_color: Dot color
        
    Returns:
        Random dot image as numpy array
    """
    # Create PIL image
    img = Image.new('RGB', (width, height), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Calculate number of dots
    total_pixels = width * height
    num_dots = int(total_pixels * density / 100.0 / (dot_size * dot_size))
    
    # Place dots at random positions
    np.random.seed(42)  # For reproducibility
    for _ in range(num_dots):
        x = np.random.randint(0, width - dot_size)
        y = np.random.randint(0, height - dot_size)
        
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
    Create shape mask for hidden figure.
    
    Args:
        width: Image width
        height: Image height
        shape_type: Type of shape ('四角形' or '円')
        shape_mode: Mode ('面' or '枠線')
        border_width: Border width for outline mode
        shape_width: Width/diameter of shape
        shape_height: Height of shape
        center_x: X coordinate of center
        center_y: Y coordinate of center
        
    Returns:
        Boolean mask array
    """
    mask = np.zeros((height, width), dtype=bool)
    
    if shape_type == '四角形':
        # Calculate rectangle bounds
        left = max(0, center_x - shape_width // 2)
        right = min(width, center_x + shape_width // 2)
        top = max(0, center_y - shape_height // 2)
        bottom = min(height, center_y + shape_height // 2)
        
        if shape_mode == '面':
            mask[top:bottom, left:right] = True
        else:  # 枠線
            # Outer rectangle
            mask[top:bottom, left:right] = True
            # Cut out inner rectangle
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


def add_background_noise(image: np.ndarray, mask: Optional[np.ndarray] = None, 
                        noise_std: float = 0.5, seed: int = 123) -> np.ndarray:
    """
    Add Gaussian noise to background region.
    
    Args:
        image: Input image
        mask: Shape mask (noise applied to background if provided)
        noise_std: Standard deviation of noise
        seed: Random seed for reproducibility
        
    Returns:
        Noisy image
    """
    noisy_image = image.copy()
    
    if mask is not None:
        # Apply noise only to background region
        background_mask = ~mask
        
        # Set seed for reproducibility
        np.random.seed(seed)
        noise = np.random.normal(0, noise_std, image.shape)
        
        # Apply noise to background only
        noisy_image[background_mask] += noise[background_mask]
    else:
        # Apply noise to entire image
        np.random.seed(seed)
        noise = np.random.normal(0, noise_std, image.shape)
        noisy_image += noise
    
    return noisy_image


def apply_fft_phase_shift(image: np.ndarray, shift_x: float, 
                         pad_width: int = 32) -> np.ndarray:
    """
    Apply FFT-based phase shift to image.
    
    Args:
        image: Input image
        shift_x: X-direction shift in pixels
        pad_width: Padding width to prevent boundary artifacts
        
    Returns:
        Shifted image
    """
    # Add padding to prevent boundary artifacts
    padded_image = np.pad(image, pad_width, mode='reflect')
    
    # 2D FFT
    fft_result = fft.fft2(padded_image)
    
    # Shift spectrum to center
    fft_shifted = fft.fftshift(fft_result)
    
    # Create frequency mesh
    height, width = fft_shifted.shape
    U, V = create_2d_frequency_mesh(height, width)
    
    # Apply phase shift
    phase_shift = calculate_phase_shift(U, shift_x)
    shifted_fft = fft_shifted * phase_shift
    
    # Shift back and inverse FFT
    shifted_fft_uncentered = fft.ifftshift(shifted_fft)
    shifted_padded = fft.ifft2(shifted_fft_uncentered)
    
    # Take real part
    shifted_padded_real = np.real(shifted_padded)
    
    # Remove padding
    if pad_width > 0:
        shifted_image = shifted_padded_real[pad_width:-pad_width, 
                                          pad_width:-pad_width]
    else:
        shifted_image = shifted_padded_real
        
    return shifted_image