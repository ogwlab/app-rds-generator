"""
Download UI components for the RDS Generator
"""

import streamlit as st
import numpy as np
import io
import zipfile
from PIL import Image
from typing import Dict


def render_download_area(rds_generator, params: Dict):
    """
    Render the download area with single and batch download options.
    
    Args:
        rds_generator: RDS generator instance
        params: Current parameter values
    """
    st.subheader("💾 ダウンロード")
    
    # Single image download
    _render_single_download()
    
    st.markdown("---")
    
    # Batch download
    _render_batch_download(rds_generator, params)


def _render_single_download():
    """Render single image download section"""
    if 'latest_result' in st.session_state and 'latest_params' in st.session_state:
        try:
            result = st.session_state.latest_result
            params = st.session_state.latest_params
            
            # Create ZIP file
            zip_buffer = _create_single_zip(result, params)
            
            # Generate filename
            sign = "pos" if params['disparity_arcsec'] >= 0 else "neg"
            base_name = f"rds_{sign}{abs(params['disparity_arcsec']):03d}"
            
            # Download button
            st.download_button(
                label="📦 画像をダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"{base_name}.zip",
                mime="application/zip"
            )
                
        except Exception as e:
            st.error(f"ダウンロードエラー: {str(e)}")
    else:
        st.info("⬅️ まず画像を生成してください")


def _render_batch_download(rds_generator, params: Dict):
    """Render batch download section"""
    if st.button("📦 一括生成 (ZIP)"):
        _handle_batch_generation(rds_generator, params)


def _create_single_zip(result: Dict, params: Dict) -> io.BytesIO:
    """
    Create ZIP file for single image download.
    
    Args:
        result: RDS generation result
        params: Generation parameters
        
    Returns:
        ZIP file buffer
    """
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Generate base filename
        sign = "pos" if params['disparity_arcsec'] >= 0 else "neg"
        base_name = f"rds_{sign}{abs(params['disparity_arcsec']):03d}"
        
        # Save stereo pair
        left_img = Image.fromarray(result['left_image'])
        right_img = Image.fromarray(result['right_image'])
        
        left_buffer = io.BytesIO()
        right_buffer = io.BytesIO()
        
        left_img.save(left_buffer, format='PNG')
        right_img.save(right_buffer, format='PNG')
        
        zip_file.writestr(f"{base_name}_L.png", left_buffer.getvalue())
        zip_file.writestr(f"{base_name}_R.png", right_buffer.getvalue())
    
    zip_buffer.seek(0)
    return zip_buffer


def _handle_batch_generation(rds_generator, params: Dict):
    """
    Handle batch generation and download.
    
    Args:
        rds_generator: RDS generator instance
        params: Current parameter values
    """
    try:
        # Calculate disparity range
        disparities = np.arange(
            params['batch_start'], 
            params['batch_end'] + params['batch_step'], 
            params['batch_step']
        )
        total_images = len(disparities)
        
        if total_images == 0:
            st.error("視差の範囲が無効です")
            return
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Create batch ZIP
        zip_buffer = _create_batch_zip(
            rds_generator, params, disparities, 
            progress_bar, status_text
        )
        
        status_text.text("✅ 生成完了！")
        
        # Download button
        st.download_button(
            label="📦 ZIPファイルをダウンロード",
            data=zip_buffer.getvalue(),
            file_name=f"rds_batch_{params['batch_start']}_{params['batch_end']}_{params['batch_step']}.zip",
            mime="application/zip"
        )
        
    except Exception as e:
        st.error(f"バッチ生成エラー: {str(e)}")


def _create_batch_zip(rds_generator, params: Dict, disparities: np.ndarray,
                     progress_bar, status_text) -> io.BytesIO:
    """
    Create ZIP file for batch download.
    
    Args:
        rds_generator: RDS generator instance
        params: Base parameters
        disparities: Array of disparity values
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text element
        
    Returns:
        ZIP file buffer
    """
    zip_buffer = io.BytesIO()
    total_images = len(disparities)
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, disp in enumerate(disparities):
            status_text.text(f"処理中: {i+1}/{total_images} (視差: {disp} arcsec)")
            
            # Update parameters
            batch_params = params.copy()
            batch_params['disparity_arcsec'] = disp
            
            # Generate RDS
            batch_result = rds_generator.generate_rds(batch_params)
            
            # Generate filename
            sign = "pos" if disp >= 0 else "neg"
            base_name = f"rds_{sign}{abs(disp):03d}"
            
            # Save stereo pair
            left_img = Image.fromarray(batch_result['left_image'])
            right_img = Image.fromarray(batch_result['right_image'])
            
            left_buffer = io.BytesIO()
            right_buffer = io.BytesIO()
            
            left_img.save(left_buffer, format='PNG')
            right_img.save(right_buffer, format='PNG')
            
            zip_file.writestr(f"{base_name}_L.png", left_buffer.getvalue())
            zip_file.writestr(f"{base_name}_R.png", right_buffer.getvalue())
            
            # Update progress
            progress_bar.progress((i + 1) / total_images)
    
    zip_buffer.seek(0)
    return zip_buffer