"""
Mathematical utility functions for RDS generation
"""

import math
import numpy as np


def arcsec_to_pixels(arcsec: float, distance_cm: float, ppi: int) -> float:
    """
    Convert arc seconds to pixels.
    
    Args:
        arcsec: Arc seconds to convert
        distance_cm: Viewing distance in centimeters
        ppi: Pixels per inch of the display
        
    Returns:
        Equivalent distance in pixels
    """
    # arcsec → radians → cm → pixels
    radians = arcsec / 3600.0 * (math.pi / 180.0)
    displacement_cm = radians * distance_cm
    ppcm = ppi / 2.54
    return displacement_cm * ppcm


def create_2d_frequency_mesh(height: int, width: int) -> tuple:
    """
    Create 2D frequency mesh for FFT operations.
    
    Args:
        height: Height of the image
        width: Width of the image
        
    Returns:
        Tuple of (U, V) frequency meshes
    """
    # Generate frequency coordinates
    freq_x = np.fft.fftfreq(width, d=1.0)
    freq_y = np.fft.fftfreq(height, d=1.0)
    
    # Adjust for fftshift
    freq_x = np.fft.fftshift(freq_x)
    freq_y = np.fft.fftshift(freq_y)
    
    # Create 2D mesh grid
    U, V = np.meshgrid(freq_x, freq_y)
    
    return U, V


def calculate_phase_shift(U: np.ndarray, shift_x: float) -> np.ndarray:
    """
    Calculate phase shift for FFT-based image translation.
    
    Args:
        U: X frequency mesh
        shift_x: Shift amount in X direction
        
    Returns:
        Phase shift array
    """
    return np.exp(-2j * np.pi * U * shift_x)


def clip_to_uint8(array: np.ndarray) -> np.ndarray:
    """
    Clip array values to 0-255 range and convert to uint8.
    
    Args:
        array: Input array
        
    Returns:
        Clipped uint8 array
    """
    return np.clip(array, 0, 255).astype(np.uint8)