"""
Mathematical utility functions for RDS generation
"""

import math
import numpy as np
import scipy.fft as fft
from typing import Tuple


def arcsec_to_pixels(arcsec: float, distance_cm: float, ppi: int) -> float:
    """
    Convert arcseconds to pixels
    
    視差の単位変換計算：
    1. 秒角をラジアンに変換: rad = arcsec × (π/180) / 3600
    2. 視差距離を計算: d_cm = tan(rad) × distance_cm ≈ rad × distance_cm (小角近似)
    3. ピクセルに変換: d_px = d_cm × (ppi / 2.54)
    
    Args:
        arcsec: 視差（秒角）
        distance_cm: 観察距離（cm）
        ppi: ディスプレイのピクセル密度（pixels per inch）
        
    Returns:
        視差（ピクセル）
    """
    # Step 1: 秒角をラジアンに変換
    radians = arcsec / 3600.0 * (math.pi / 180.0)
    
    # Step 2: 小角近似による視差距離の計算
    # tan(θ) ≈ θ (ラジアン) for small angles
    displacement_cm = radians * distance_cm
    
    # Step 3: cmをピクセルに変換
    ppcm = ppi / 2.54  # pixels per cm
    return displacement_cm * ppcm


def apply_phase_shift_2d_robust(image: np.ndarray, shift_x: float, 
                              pad_width: int = 32) -> np.ndarray:
    """
    Apply robust 2D phase shift using FFT
    
    フーリエ変換のシフト定理を利用した堅牢な2次元位相シフト実装：
    f(x - dx) ↔ F(u) × exp(-i2πu×dx)
    
    境界アーチファクト対策としてパディング処理を含む。
    
    Args:
        image: 入力画像 (2D ndarray)
        shift_x: X方向のシフト量（ピクセル単位）
        pad_width: 境界アーチファクト防止用のパディング幅
        
    Returns:
        シフトされた画像 (2D ndarray)
    """
    # Step 1: Add padding to prevent boundary artifacts
    padded_image = np.pad(image, pad_width, mode='reflect')
    
    # Step 2: Apply 2D FFT
    fft_result = fft.fft2(padded_image)
    
    # Step 3: Center the frequency spectrum
    fft_shifted = fft.fftshift(fft_result)
    
    # Step 4: Generate 2D frequency coordinates
    height, width = fft_shifted.shape
    
    # Generate frequency coordinates (adjusted for fftshift)
    freq_x = fft.fftfreq(width, d=1.0)
    freq_y = fft.fftfreq(height, d=1.0)
    
    # Adjust frequency layout after fftshift
    freq_x = fft.fftshift(freq_x)
    freq_y = fft.fftshift(freq_y)
    
    # Create 2D mesh grid
    U, V = np.meshgrid(freq_x, freq_y)
    
    # Step 5: Apply phase shift
    phase_shift_2d = np.exp(-2j * np.pi * U * shift_x)
    
    # Apply phase shift to frequency spectrum
    shifted_fft = fft_shifted * phase_shift_2d
    
    # Step 6: Restore frequency spectrum layout
    shifted_fft_uncentered = fft.ifftshift(shifted_fft)
    
    # Step 7: Apply inverse 2D FFT
    shifted_padded = fft.ifft2(shifted_fft_uncentered)
    
    # Take only real part (imaginary part is numerical error)
    shifted_padded_real = np.real(shifted_padded)
    
    # Step 8: Crop padding to restore original size
    if pad_width > 0:
        shifted_image = shifted_padded_real[pad_width:-pad_width, 
                                         pad_width:-pad_width]
    else:
        shifted_image = shifted_padded_real
        
    return shifted_image


def create_frequency_mesh(height: int, width: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create 2D frequency mesh for FFT operations
    
    Args:
        height: Image height
        width: Image width
        
    Returns:
        Tuple of (U, V) frequency meshes
    """
    freq_x = fft.fftfreq(width, d=1.0)
    freq_y = fft.fftfreq(height, d=1.0)
    
    freq_x = fft.fftshift(freq_x)
    freq_y = fft.fftshift(freq_y)
    
    U, V = np.meshgrid(freq_x, freq_y)
    return U, V