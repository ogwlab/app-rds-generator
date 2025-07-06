"""
RDS Generator - Main Streamlit Application

A modular Random Dot Stereogram generator for perception and cognitive psychology research.
"""

import streamlit as st
from rds_generator import RDSGenerator
from ui.sidebar import render_sidebar
from ui.main_area import render_main_area
from ui.download import render_download_area


def setup_page():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title="RDS生成ツール",
        page_icon="👁️",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def render_header():
    """Render application header"""
    st.title("🔬 ランダムドットステレオグラム (RDS) 生成ツール")
    st.markdown("**知覚・認知心理学研究用 - FFT位相シフト法対応**")


def render_footer():
    """Render application footer"""
    st.markdown("---")
    st.markdown("**Powered by Ogwlab**")


@st.cache_resource
def get_rds_generator():
    """Get cached RDS generator instance"""
    return RDSGenerator()


def main():
    """Main application function"""
    # Setup page
    setup_page()
    
    # Render header
    render_header()
    
    # Get RDS generator
    rds_generator = get_rds_generator()
    
    # Render sidebar and get parameters
    params = render_sidebar()
    
    # Create main layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Render main area with preview
        render_main_area(rds_generator, params)
    
    with col2:
        # Render download area
        render_download_area(rds_generator, params)
    
    # Render footer
    render_footer()


if __name__ == "__main__":
    main()