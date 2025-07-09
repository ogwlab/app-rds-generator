"""
Streamlit application for RDS Generator
"""

import streamlit as st
import numpy as np
from PIL import Image
import io
import zipfile
from typing import Dict

from .core import RDSGenerator
from .config import RDSConfig
from .image_utils import combine_stereo_images


def main():
    """Main Streamlit application"""
    
    # Page configuration
    st.set_page_config(
        page_title="RDS生成ツール",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Application title
    st.title("🔬 ランダムドットステレオグラム (RDS) 生成ツール")
    st.markdown("**知覚・認知心理学研究用 - FFT位相シフト法対応**")
    
    # Initialize RDS generator
    @st.cache_resource
    def get_rds_generator():
        return RDSGenerator()
    
    rds_gen = get_rds_generator()
    
    # Sidebar parameters
    st.sidebar.header("📋 パラメータ設定")
    
    # Overall settings
    st.sidebar.subheader("🎯 全体の設定")
    width = st.sidebar.number_input("画像サイズ (幅) px", min_value=128, max_value=1024, value=512, step=128)
    height = st.sidebar.number_input("画像サイズ (高さ) px", min_value=128, max_value=1024, value=512, step=128)
    density = st.sidebar.slider("ドット密度 (%)", min_value=1, max_value=100, value=50)
    dot_size = st.sidebar.slider("ドットサイズ (px)", min_value=1, max_value=10, value=2)
    dot_shape = st.sidebar.selectbox("ドット形状", ['四角', '円'])
    bg_color = st.sidebar.color_picker("背景色", "#FFFFFF")
    dot_color = st.sidebar.color_picker("ドット色", "#000000")
    
    # Stereo settings
    st.sidebar.subheader("👁️ 立体視の設定")
    disparity_arcsec = st.sidebar.slider("視差 (arcsec)", min_value=-600, max_value=600, value=20)
    distance_cm = st.sidebar.number_input("観察距離 (cm)", min_value=30.0, max_value=200.0, value=57.0, step=1.0)
    ppi = st.sidebar.number_input("モニターPPI", min_value=72, max_value=400, value=96)
    display_mode = st.sidebar.radio("表示形式", ['ステレオペア (平行法)', 'ステレオペア (交差法)'])
    
    # Hidden shape settings
    st.sidebar.subheader("🔷 隠された図形")
    shape_type = st.sidebar.selectbox("図形の形状", ['四角形', '円'])
    shape_mode = st.sidebar.selectbox("図形の表示モード", ['面', '枠線'])
    border_width = st.sidebar.slider("枠線の太さ (px)", min_value=1, max_value=20, value=2)
    shape_width = st.sidebar.slider("サイズ (幅/直径) px", min_value=10, max_value=width, value=min(256, width))
    shape_height = st.sidebar.slider("サイズ (高さ) px", min_value=10, max_value=height, value=min(256, height))
    center_x = st.sidebar.slider("中心位置 (X座標) px", min_value=0, max_value=width, value=width//2)
    center_y = st.sidebar.slider("中心位置 (Y座標) px", min_value=0, max_value=height, value=height//2)
    
    # Random seed settings
    st.sidebar.subheader("🎲 乱数設定")
    use_random_seed = st.sidebar.checkbox("乱数シードを指定", value=True, help="再現性を保つために乱数シードを指定します")
    if use_random_seed:
        random_seed = st.sidebar.number_input("ドット配置シード", min_value=0, max_value=2**31-1, value=42, help="ランダムドット配置用のシード値")
        left_noise_seed = st.sidebar.number_input("左眼ノイズシード", min_value=0, max_value=2**31-1, value=42, help="左眼用背景ノイズのシード値")
        right_noise_seed = st.sidebar.number_input("右眼ノイズシード", min_value=0, max_value=2**31-1, value=43, help="右眼用背景ノイズのシード値")
    else:
        random_seed = None
        left_noise_seed = None
        right_noise_seed = None
    
    # Batch generation settings
    st.sidebar.subheader("📦 バッチ生成")
    batch_start = st.sidebar.number_input("開始視差 (arcsec)", min_value=-1000, max_value=1000, value=-100, step=10)
    batch_end = st.sidebar.number_input("終了視差 (arcsec)", min_value=-1000, max_value=1000, value=100, step=10)
    batch_step = st.sidebar.number_input("ステップ (arcsec)", min_value=1, max_value=200, value=20)
    
    # Validate batch parameters
    if batch_start > batch_end:
        st.sidebar.error("開始視差は終了視差以下である必要があります")
    if batch_step <= 0:
        st.sidebar.error("ステップは正の値である必要があります")
    
    # Main area
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🖼️ プレビュー")
        
        # Create configuration
        config = RDSConfig(
            width=width, height=height, density=density,
            dot_size=dot_size, dot_shape=dot_shape,
            bg_color=bg_color, dot_color=dot_color,
            disparity_arcsec=disparity_arcsec, distance_cm=distance_cm, ppi=ppi,
            shape_type=shape_type, shape_mode=shape_mode,
            border_width=border_width, shape_width=shape_width, shape_height=shape_height,
            center_x=center_x, center_y=center_y,
            random_seed=random_seed, left_noise_seed=left_noise_seed, right_noise_seed=right_noise_seed
        )
        
        # Display existing result if available
        if 'latest_result' in st.session_state:
            result = st.session_state.latest_result
            saved_config = st.session_state.latest_config
            
            # Display stereo pair
            if display_mode == 'ステレオペア (交差法)':
                left_img, right_img = result['right_image'], result['left_image']
                caption = "ステレオペア (交差法)"
            else:
                left_img, right_img = result['left_image'], result['right_image']
                caption = "ステレオペア (平行法)"
            
            stereo_pair = combine_stereo_images(left_img, right_img)
            st.image(stereo_pair, caption=caption, use_container_width=True)
            
            st.info(f"視差: {saved_config.disparity_arcsec} arcsec = {result['disparity_pixels']:.2f} pixels")
        else:
            # Initial display message
            st.info("パラメータを設定して「設定を反映」ボタンを押してください")
        
        # Apply settings button
        generate_preview = st.button("🔄 設定を反映", type="primary", use_container_width=True)
        
        # Button click handler
        if generate_preview:
            try:
                with st.spinner('RDS生成中...'):
                    result = rds_gen.generate_rds(config)
                
                # Save result to session state
                st.session_state.latest_result = result
                st.session_state.latest_config = config
                
                # Reload page to show latest result
                st.rerun()
                
            except Exception as e:
                st.error(f"エラーが発生しました: {str(e)}")
    
    with col2:
        st.subheader("💾 ダウンロード")
        
        # Single image download
        if 'latest_result' in st.session_state and 'latest_config' in st.session_state:
            result = st.session_state.latest_result
            config = st.session_state.latest_config
            
            # Create ZIP file
            zip_buffer = create_zip_file(result, config)
            
            # Download button
            st.download_button(
                label="📦 画像をダウンロード (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=create_filename(config),
                mime="application/zip"
            )
        else:
            st.info("⬅️ まず画像を生成してください")
        
        st.markdown("---")
        
        # Batch generation
        if st.button("📦 一括生成 (ZIP)"):
            try:
                batch_generate(rds_gen, config, batch_start, batch_end, batch_step)
            except Exception as e:
                st.error(f"バッチ生成エラー: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("**Powered by Ogwlab**")


def create_filename(config: RDSConfig) -> str:
    """Create filename for download"""
    sign = "pos" if config.disparity_arcsec >= 0 else "neg"
    return f"rds_{sign}{abs(config.disparity_arcsec):03d}.zip"


def create_zip_file(result: Dict, config: RDSConfig) -> io.BytesIO:
    """Create ZIP file with stereo pair"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        sign = "pos" if config.disparity_arcsec >= 0 else "neg"
        base_name = f"rds_{sign}{abs(config.disparity_arcsec):03d}"
        
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


def batch_generate(rds_gen: RDSGenerator, base_config: RDSConfig, 
                  start: int, end: int, step: int):
    """Generate batch of RDS images"""
    # Validate parameters
    if step == 0:
        st.error("ステップは0にできません")
        return
    
    if step > 0 and start > end:
        st.error("ステップが正の場合、開始視差は終了視差以下である必要があります")
        return
    
    if step < 0 and start < end:
        st.error("ステップが負の場合、開始視差は終了視差以上である必要があります")
        return
    
    disparities = np.arange(start, end + step, step)
    total_images = len(disparities)
    
    if total_images == 0:
        st.error("視差の範囲が無効です")
        return
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Create ZIP file
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        
        for i, disp in enumerate(disparities):
            status_text.text(f"処理中: {i+1}/{total_images} (視差: {disp} arcsec)")
            
            # Update configuration
            batch_config = RDSConfig(
                width=base_config.width, height=base_config.height,
                density=base_config.density, dot_size=base_config.dot_size,
                dot_shape=base_config.dot_shape, bg_color=base_config.bg_color,
                dot_color=base_config.dot_color, disparity_arcsec=disp,
                distance_cm=base_config.distance_cm, ppi=base_config.ppi,
                shape_type=base_config.shape_type, shape_mode=base_config.shape_mode,
                border_width=base_config.border_width, shape_width=base_config.shape_width,
                shape_height=base_config.shape_height, center_x=base_config.center_x,
                center_y=base_config.center_y,
                random_seed=base_config.random_seed, left_noise_seed=base_config.left_noise_seed,
                right_noise_seed=base_config.right_noise_seed
            )
            
            # Generate RDS
            batch_result = rds_gen.generate_rds(batch_config)
            
            # Create filename
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
    status_text.text("✅ 生成完了！")
    
    # Download button
    st.download_button(
        label="📦 ZIPファイルをダウンロード",
        data=zip_buffer.getvalue(),
        file_name=f"rds_batch_{start}_{end}_{step}.zip",
        mime="application/zip"
    )


if __name__ == "__main__":
    main()