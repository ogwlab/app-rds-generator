"""
Main area UI components for the RDS Generator
"""

import streamlit as st
import numpy as np
from typing import Dict, Optional


def render_main_area(rds_generator, params: Dict) -> Optional[Dict]:
    """
    Render the main area with preview and generation controls.
    
    Args:
        rds_generator: RDS generator instance
        params: Current parameter values
        
    Returns:
        Generated RDS result dictionary if generation occurred, None otherwise
    """
    st.subheader("🖼️ プレビュー")
    
    # Display existing result if available
    if 'latest_result' in st.session_state:
        _display_existing_result()
    else:
        # Initial message
        st.info("パラメータを設定して「設定を反映」ボタンを押してください")
    
    # Generate button
    generate_preview = st.button("🔄 設定を反映", type="primary", use_container_width=True)
    
    # Handle generation
    if generate_preview:
        return _generate_rds(rds_generator, params)
    
    return None


def _display_existing_result():
    """Display the existing RDS result from session state"""
    result = st.session_state.latest_result
    saved_params = st.session_state.latest_params
    
    # Display stereo pair
    if saved_params['display_mode'] == 'ステレオペア (交差法)':
        left_img, right_img = result['right_image'], result['left_image']
        caption = "ステレオペア (交差法)"
    else:
        left_img, right_img = result['left_image'], result['right_image']
        caption = "ステレオペア (平行法)"
    
    # Create stereo pair image with separator
    separator_width = 5
    separator = np.zeros((left_img.shape[0], separator_width), dtype=np.uint8)
    stereo_pair = np.hstack([left_img, separator, right_img])
    
    st.image(stereo_pair, caption=caption, use_container_width=True)
    
    # Display info
    st.info(f"視差: {saved_params['disparity_arcsec']} arcsec = {result['disparity_pixels']:.2f} pixels")


def _generate_rds(rds_generator, params: Dict) -> Optional[Dict]:
    """
    Generate RDS with the given parameters.
    
    Args:
        rds_generator: RDS generator instance
        params: Parameter dictionary
        
    Returns:
        Generated result dictionary or None if error occurred
    """
    try:
        with st.spinner('RDS生成中...'):
            result = rds_generator.generate_rds(params)
        
        # Save result to session state
        st.session_state.latest_result = result
        st.session_state.latest_params = params
        
        # Reload to display the result
        st.rerun()
        
    except Exception as e:
        st.error(f"エラーが発生しました: {str(e)}")
        return None