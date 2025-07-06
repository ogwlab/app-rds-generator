"""
Sidebar UI components for the RDS Generator
"""

import streamlit as st
from ..config.settings import *


def render_sidebar():
    """
    Render the sidebar with all parameter controls.
    
    Returns:
        Dictionary containing all parameter values
    """
    st.sidebar.header("📋 パラメータ設定")
    
    # Overall settings
    st.sidebar.subheader("🎯 全体の設定")
    width = st.sidebar.number_input(
        "画像サイズ (幅) px", 
        min_value=MIN_IMAGE_SIZE, 
        max_value=MAX_IMAGE_SIZE, 
        value=DEFAULT_IMAGE_WIDTH, 
        step=IMAGE_SIZE_STEP
    )
    height = st.sidebar.number_input(
        "画像サイズ (高さ) px", 
        min_value=MIN_IMAGE_SIZE, 
        max_value=MAX_IMAGE_SIZE, 
        value=DEFAULT_IMAGE_HEIGHT, 
        step=IMAGE_SIZE_STEP
    )
    density = st.sidebar.slider(
        "ドット密度 (%)", 
        min_value=MIN_DENSITY, 
        max_value=MAX_DENSITY, 
        value=DEFAULT_DENSITY
    )
    dot_size = st.sidebar.slider(
        "ドットサイズ (px)", 
        min_value=MIN_DOT_SIZE, 
        max_value=MAX_DOT_SIZE, 
        value=DEFAULT_DOT_SIZE
    )
    dot_shape = st.sidebar.selectbox("ドット形状", ['四角', '円'])
    bg_color = st.sidebar.color_picker("背景色", DEFAULT_BG_COLOR)
    dot_color = st.sidebar.color_picker("ドット色", DEFAULT_DOT_COLOR)
    
    # Stereo vision settings
    st.sidebar.subheader("👁️ 立体視の設定")
    disparity_arcsec = st.sidebar.slider(
        "視差 (arcsec)", 
        min_value=MIN_DISPARITY, 
        max_value=MAX_DISPARITY, 
        value=DEFAULT_DISPARITY_ARCSEC
    )
    distance_cm = st.sidebar.number_input(
        "観察距離 (cm)", 
        min_value=MIN_DISTANCE, 
        max_value=MAX_DISTANCE, 
        value=DEFAULT_DISTANCE_CM, 
        step=1.0
    )
    ppi = st.sidebar.number_input(
        "モニターPPI", 
        min_value=MIN_PPI, 
        max_value=MAX_PPI, 
        value=DEFAULT_PPI
    )
    display_mode = st.sidebar.radio(
        "表示形式", 
        ['ステレオペア (平行法)', 'ステレオペア (交差法)']
    )
    
    # Hidden figure settings
    st.sidebar.subheader("🔷 隠された図形")
    shape_type = st.sidebar.selectbox("図形の形状", ['四角形', '円'])
    shape_mode = st.sidebar.selectbox("図形の表示モード", ['面', '枠線'])
    border_width = st.sidebar.slider(
        "枠線の太さ (px)", 
        min_value=MIN_BORDER_WIDTH, 
        max_value=MAX_BORDER_WIDTH, 
        value=DEFAULT_BORDER_WIDTH
    )
    shape_width = st.sidebar.slider(
        "サイズ (幅/直径) px", 
        min_value=MIN_SHAPE_SIZE, 
        max_value=width, 
        value=min(DEFAULT_SHAPE_WIDTH, width)
    )
    shape_height = st.sidebar.slider(
        "サイズ (高さ) px", 
        min_value=MIN_SHAPE_SIZE, 
        max_value=height, 
        value=min(DEFAULT_SHAPE_HEIGHT, height)
    )
    center_x = st.sidebar.slider(
        "中心位置 (X座標) px", 
        min_value=0, 
        max_value=width, 
        value=width//2
    )
    center_y = st.sidebar.slider(
        "中心位置 (Y座標) px", 
        min_value=0, 
        max_value=height, 
        value=height//2
    )
    
    # Batch generation settings
    st.sidebar.subheader("📦 バッチ生成")
    batch_start = st.sidebar.number_input(
        "開始視差 (arcsec)", 
        min_value=MIN_BATCH_DISPARITY, 
        max_value=MAX_BATCH_DISPARITY, 
        value=DEFAULT_BATCH_START, 
        step=10
    )
    batch_end = st.sidebar.number_input(
        "終了視差 (arcsec)", 
        min_value=MIN_BATCH_DISPARITY, 
        max_value=MAX_BATCH_DISPARITY, 
        value=DEFAULT_BATCH_END, 
        step=10
    )
    batch_step = st.sidebar.number_input(
        "ステップ (arcsec)", 
        min_value=MIN_BATCH_STEP, 
        max_value=MAX_BATCH_STEP, 
        value=DEFAULT_BATCH_STEP
    )
    
    # Return all parameters as dictionary
    return {
        'width': width,
        'height': height,
        'density': density,
        'dot_size': dot_size,
        'dot_shape': dot_shape,
        'bg_color': bg_color,
        'dot_color': dot_color,
        'disparity_arcsec': disparity_arcsec,
        'distance_cm': distance_cm,
        'ppi': ppi,
        'display_mode': display_mode,
        'shape_type': shape_type,
        'shape_mode': shape_mode,
        'border_width': border_width,
        'shape_width': shape_width,
        'shape_height': shape_height,
        'center_x': center_x,
        'center_y': center_y,
        'batch_start': batch_start,
        'batch_end': batch_end,
        'batch_step': batch_step
    }