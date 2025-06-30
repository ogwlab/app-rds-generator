import streamlit as st
import numpy as np
import scipy.fft as fft
from PIL import Image, ImageDraw
import io
import zipfile
import json
from typing import Tuple, Optional
import math

# ページ設定
st.set_page_config(
    page_title="RDS生成ツール",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# アプリケーションタイトル
st.title("🔬 ランダムドットステレオグラム (RDS) 生成ツール")
st.markdown("**知覚・認知心理学研究用 - FFT位相シフト法対応**")

class RDSGenerator:
    """ランダムドットステレオグラム生成クラス"""
    
    def __init__(self):
        self.reset_cache()
    
    def reset_cache(self):
        """キャッシュをリセット"""
        self._cached_base_image = None
        self._cached_params = None
    
    def arcsec_to_pixels(self, arcsec: float, distance_cm: float, ppi: int) -> float:
        """秒角をピクセルに変換"""
        # arcsec → radians → cm → pixels
        radians = arcsec / 3600.0 * (math.pi / 180.0)
        displacement_cm = radians * distance_cm
        ppcm = ppi / 2.54
        return displacement_cm * ppcm
    
    def generate_random_dots(self, width: int, height: int, density: float, 
                           dot_size: int, dot_shape: str, bg_color: str, 
                           dot_color: str) -> np.ndarray:
        """ランダムドット画像を生成"""
        # PIL画像を作成
        img = Image.new('RGB', (width, height), bg_color)
        draw = ImageDraw.Draw(img)
        
        # ドット数を計算
        total_pixels = width * height
        num_dots = int(total_pixels * density / 100.0 / (dot_size * dot_size))
        
        # ランダムな位置にドットを配置
        np.random.seed(42)  # 再現性のため
        for _ in range(num_dots):
            x = np.random.randint(0, width - dot_size)
            y = np.random.randint(0, height - dot_size)
            
            if dot_shape == '四角':
                draw.rectangle([x, y, x + dot_size, y + dot_size], fill=dot_color)
            else:  # 円
                draw.ellipse([x, y, x + dot_size, y + dot_size], fill=dot_color)
        
        # NumPy配列に変換（グレースケール）
        img_gray = img.convert('L')
        return np.array(img_gray, dtype=np.float64)
    
    def create_shape_mask(self, width: int, height: int, shape_type: str,
                         shape_mode: str, border_width: int, shape_width: int,
                         shape_height: int, center_x: int, center_y: int) -> np.ndarray:
        """図形マスクを作成"""
        mask = np.zeros((height, width), dtype=bool)
        
        if shape_type == '四角形':
            # 四角形の範囲を計算
            left = max(0, center_x - shape_width // 2)
            right = min(width, center_x + shape_width // 2)
            top = max(0, center_y - shape_height // 2)
            bottom = min(height, center_y + shape_height // 2)
            
            if shape_mode == '面':
                mask[top:bottom, left:right] = True
            else:  # 枠線
                # 外枠
                mask[top:bottom, left:right] = True
                # 内側をくり抜く
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
    
    def add_minimal_background_noise(self, image: np.ndarray, mask: np.ndarray, 
                                   noise_std: float = 0.5) -> np.ndarray:
        """背景領域に軽微なガウシアンノイズを追加（現在の標準手法）"""
        noisy_image = image.copy()
        
        # 背景領域（マスクの逆）に軽微なノイズ追加
        background_mask = ~mask
        
        # シード固定で再現性を保持
        np.random.seed(123)
        noise = np.random.normal(0, noise_std, image.shape)
        
        # 背景領域のみにノイズ適用
        noisy_image[background_mask] += noise[background_mask]
        
        return noisy_image
    
    def apply_phase_shift_2d_robust(self, image: np.ndarray, shift_x: float, 
                                  pad_width: int = 32) -> np.ndarray:
        """
        2次元FFTによる堅牢な位相シフト実装
        
        Args:
            image: 入力画像 (2D ndarray)
            shift_x: X方向のシフト量（ピクセル単位）
            pad_width: 境界アーチファクト防止用のパディング幅
            
        Returns:
            シフトされた画像 (2D ndarray)
        """
        # Step 1: パディングを追加（境界アーチファクト対策）
        # reflect モードで画像境界を反射させてパディング
        padded_image = np.pad(image, pad_width, mode='reflect')
        
        # Step 2: 2次元FFTを実行
        # 画像全体を周波数領域に変換
        fft_result = fft.fft2(padded_image)
        
        # Step 3: 周波数スペクトルを中心化
        # DC成分（ゼロ周波数）を中央に移動
        fft_shifted = fft.fftshift(fft_result)
        
        # Step 4: 2次元周波数座標を生成
        height, width = fft_shifted.shape
        
        # 周波数座標を生成（fftshift後の配置に対応）
        freq_x = fft.fftfreq(width, d=1.0)
        freq_y = fft.fftfreq(height, d=1.0)
        
        # fftshift後の周波数配置に調整
        freq_x = fft.fftshift(freq_x)
        freq_y = fft.fftshift(freq_y)
        
        # 2次元メッシュグリッドを作成
        U, V = np.meshgrid(freq_x, freq_y)
        
        # Step 5: 位相シフトを適用
        # X方向のシフトに対応する位相項を計算
        phase_shift_2d = np.exp(-2j * np.pi * U * shift_x)
        
        # 周波数スペクトルに位相シフトを適用
        shifted_fft = fft_shifted * phase_shift_2d
        
        # Step 6: 周波数スペクトルの配置を元に戻す
        # 中心化を解除してFFTの標準配置に戻す
        shifted_fft_uncentered = fft.ifftshift(shifted_fft)
        
        # Step 7: 逆2次元FFTで空間領域に戻す
        # 周波数領域から画像領域に変換
        shifted_padded = fft.ifft2(shifted_fft_uncentered)
        
        # 実数成分のみを取得（虚数成分は数値誤差）
        shifted_padded_real = np.real(shifted_padded)
        
        # Step 8: パディングをクロッピングして元のサイズに戻す
        # パディングした領域を除去
        if pad_width > 0:
            shifted_image = shifted_padded_real[pad_width:-pad_width, 
                                             pad_width:-pad_width]
        else:
            shifted_image = shifted_padded_real
            
        return shifted_image
    
    def apply_fft_phase_shift(self, image: np.ndarray, shift_pixels: float) -> np.ndarray:
        """FFT位相シフト法で画像をシフト（2次元FFT版・全体処理）"""
        # マスクの有無にかかわらず、画像全体をシフトする
        return self.apply_phase_shift_2d_robust(image, shift_pixels)
    
    def generate_stereo_pair(self, base_image: np.ndarray, disparity_pixels: float,
                           shape_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """ステレオペアを生成（現在の標準手法）"""
        
        # 背景に軽微なノイズを追加（左右で異なるノイズ）
        np.random.seed(42)  # 左眼用のシード
        left_base = self.add_minimal_background_noise(base_image, shape_mask, noise_std=0.5)
        
        np.random.seed(43)  # 右眼用のシード（異なるノイズパターン）
        right_base = self.add_minimal_background_noise(base_image, shape_mask, noise_std=0.5)
        
        # 画像全体をFFT位相シフト
        left_shifted = self.apply_fft_phase_shift(left_base, -disparity_pixels / 2)
        right_shifted = self.apply_fft_phase_shift(right_base, disparity_pixels / 2)

        # マスクが指定されている場合、図形領域のみシフト後の画像で置き換える
        if shape_mask is not None:
            left_image = np.where(shape_mask, left_shifted, left_base)
            right_image = np.where(shape_mask, right_shifted, right_base)
        else:
            # マスクがない場合は全体がシフトされる
            left_image = left_shifted
            right_image = right_shifted

        # 値を0-255の範囲にクリップ
        left_image = np.clip(left_image, 0, 255)
        right_image = np.clip(right_image, 0, 255)
        
        return left_image.astype(np.uint8), right_image.astype(np.uint8)
    
    def generate_rds(self, params: dict) -> dict:
        """RDSを生成"""
        # ベース画像を生成
        base_image = self.generate_random_dots(
            params['width'], params['height'], params['density'],
            params['dot_size'], params['dot_shape'], 
            params['bg_color'], params['dot_color']
        )
        
        # 図形マスクを作成
        shape_mask = self.create_shape_mask(
            params['width'], params['height'], params['shape_type'],
            params['shape_mode'], params['border_width'], 
            params['shape_width'], params['shape_height'],
            params['center_x'], params['center_y']
        )
        
        # 視差をピクセルに変換
        disparity_pixels = self.arcsec_to_pixels(
            params['disparity_arcsec'], params['distance_cm'], params['ppi']
        )
        
        # ステレオペアを生成（現在の標準手法）
        left_image, right_image = self.generate_stereo_pair(
            base_image, disparity_pixels, shape_mask
        )
        
        result = {
            'left_image': left_image,
            'right_image': right_image,
            'disparity_pixels': disparity_pixels
        }
        
        return result

# RDSジェネレータのインスタンスを作成
@st.cache_resource
def get_rds_generator():
    return RDSGenerator()

rds_gen = get_rds_generator()

# サイドバーでパラメータを設定
st.sidebar.header("📋 パラメータ設定")

# 全体の設定
st.sidebar.subheader("🎯 全体の設定")
width = st.sidebar.number_input("画像サイズ (幅) px", min_value=128, max_value=1024, value=512, step=128)
height = st.sidebar.number_input("画像サイズ (高さ) px", min_value=128, max_value=1024, value=512, step=128)
density = st.sidebar.slider("ドット密度 (%)", min_value=1, max_value=100, value=50)
dot_size = st.sidebar.slider("ドットサイズ (px)", min_value=1, max_value=10, value=2)
dot_shape = st.sidebar.selectbox("ドット形状", ['四角', '円'])
bg_color = st.sidebar.color_picker("背景色", "#FFFFFF")
dot_color = st.sidebar.color_picker("ドット色", "#000000")

# 立体視の設定
st.sidebar.subheader("👁️ 立体視の設定")
disparity_arcsec = st.sidebar.slider("視差 (arcsec)", min_value=-600, max_value=600, value=20)
distance_cm = st.sidebar.number_input("観察距離 (cm)", min_value=30.0, max_value=200.0, value=57.0, step=1.0)
ppi = st.sidebar.number_input("モニターPPI", min_value=72, max_value=400, value=96)
display_mode = st.sidebar.radio("表示形式", ['ステレオペア (平行法)', 'ステレオペア (交差法)'])

# 隠された図形
st.sidebar.subheader("🔷 隠された図形")
shape_type = st.sidebar.selectbox("図形の形状", ['四角形', '円'])
shape_mode = st.sidebar.selectbox("図形の表示モード", ['面', '枠線'])
border_width = st.sidebar.slider("枠線の太さ (px)", min_value=1, max_value=20, value=2)
shape_width = st.sidebar.slider("サイズ (幅/直径) px", min_value=10, max_value=width, value=min(256, width))
shape_height = st.sidebar.slider("サイズ (高さ) px", min_value=10, max_value=height, value=min(256, height))
center_x = st.sidebar.slider("中心位置 (X座標) px", min_value=0, max_value=width, value=width//2)
center_y = st.sidebar.slider("中心位置 (Y座標) px", min_value=0, max_value=height, value=height//2)

# バッチ生成
st.sidebar.subheader("📦 バッチ生成")
batch_start = st.sidebar.number_input("開始視差 (arcsec)", min_value=-1000, max_value=1000, value=-100, step=10)
batch_end = st.sidebar.number_input("終了視差 (arcsec)", min_value=-1000, max_value=1000, value=100, step=10)
batch_step = st.sidebar.number_input("ステップ (arcsec)", min_value=1, max_value=200, value=20)

# メインエリア
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("🖼️ プレビュー")
    
    # パラメータをまとめる
    params = {
        'width': width, 'height': height, 'density': density,
        'dot_size': dot_size, 'dot_shape': dot_shape,
        'bg_color': bg_color, 'dot_color': dot_color,
        'disparity_arcsec': disparity_arcsec, 'distance_cm': distance_cm, 'ppi': ppi,
        'display_mode': display_mode, 'shape_type': shape_type, 'shape_mode': shape_mode,
        'border_width': border_width, 'shape_width': shape_width, 'shape_height': shape_height,
        'center_x': center_x, 'center_y': center_y
    }
    
    # 既存の結果がある場合は表示、ない場合は初回メッセージ
    if 'latest_result' in st.session_state:
        result = st.session_state.latest_result
        saved_params = st.session_state.latest_params
        
        # ステレオペア表示
        if saved_params['display_mode'] == 'ステレオペア (交差法)':
            left_img, right_img = result['right_image'], result['left_image']
            caption = "ステレオペア (交差法)"
        else:
            left_img, right_img = result['left_image'], result['right_image']
            caption = "ステレオペア (平行法)"
        
        separator_width = 5
        separator = np.zeros((left_img.shape[0], separator_width), dtype=np.uint8)
        stereo_pair = np.hstack([left_img, separator, right_img])
        st.image(stereo_pair, caption=caption, use_container_width=True)
        
        st.info(f"視差: {saved_params['disparity_arcsec']} arcsec = {result['disparity_pixels']:.2f} pixels")
    else:
        # 初回表示時のメッセージ
        st.info("パラメータを設定して「設定を反映」ボタンを押してください")
    
    # 設定を反映ボタン（画像の下に配置）
    generate_preview = st.button("🔄 設定を反映", type="primary", use_container_width=True)
    
    # ボタンが押された時の処理
    if generate_preview:
        try:
            with st.spinner('RDS生成中...'):
                result = rds_gen.generate_rds(params)
            
            # 生成結果をセッション状態に保存
            st.session_state.latest_result = result
            st.session_state.latest_params = params
            
            # ページをリロードして最新の結果を表示
            st.rerun()
            
        except Exception as e:
            st.error(f"エラーが発生しました: {str(e)}")

with col2:
    st.subheader("💾 ダウンロード")
    
    # 単一画像ダウンロード
    if 'latest_result' in st.session_state and 'latest_params' in st.session_state:
        try:
            result = st.session_state.latest_result
            params = st.session_state.latest_params
            
            # ZIPファイルを作成
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                
                # ファイル名の基本部分を生成
                sign = "pos" if params['disparity_arcsec'] >= 0 else "neg"
                base_name = f"rds_{sign}{abs(params['disparity_arcsec']):03d}"
                
                if params['display_mode'] == 'アナグリフ':
                    # アナグリフを保存
                    img = Image.fromarray(result['anaglyph'])
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG')
                    zip_file.writestr(f"{base_name}_anaglyph.png", img_buffer.getvalue())
                else:
                    # ステレオペアを保存
                    left_img = Image.fromarray(result['left_image'])
                    right_img = Image.fromarray(result['right_image'])
                    
                    left_buffer = io.BytesIO()
                    right_buffer = io.BytesIO()
                    
                    left_img.save(left_buffer, format='PNG')
                    right_img.save(right_buffer, format='PNG')
                    
                    zip_file.writestr(f"{base_name}_L.png", left_buffer.getvalue())
                    zip_file.writestr(f"{base_name}_R.png", right_buffer.getvalue())
            
            zip_buffer.seek(0)
            
            # ワンクリックZIPダウンロードボタン
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
    
    st.markdown("---")
    
    # バッチ生成
    if st.button("📦 一括生成 (ZIP)"):
        try:
            # 視差の範囲を計算
            disparities = np.arange(batch_start, batch_end + batch_step, batch_step)
            total_images = len(disparities)
            
            if total_images == 0:
                st.error("視差の範囲が無効です")
            else:
                # プログレスバー
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # ZIPファイルを作成
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    
                    for i, disp in enumerate(disparities):
                        status_text.text(f"処理中: {i+1}/{total_images} (視差: {disp} arcsec)")
                        
                        # パラメータを更新
                        batch_params = params.copy()
                        batch_params['disparity_arcsec'] = disp
                        
                        # RDS生成
                        batch_result = rds_gen.generate_rds(batch_params)
                        
                        # ファイル名生成
                        sign = "pos" if disp >= 0 else "neg"
                        base_name = f"rds_{sign}{abs(disp):03d}"
                        
                        # ステレオペアを保存
                        left_img = Image.fromarray(batch_result['left_image'])
                        right_img = Image.fromarray(batch_result['right_image'])
                        
                        left_buffer = io.BytesIO()
                        right_buffer = io.BytesIO()
                        
                        left_img.save(left_buffer, format='PNG')
                        right_img.save(right_buffer, format='PNG')
                        
                        zip_file.writestr(f"{base_name}_L.png", left_buffer.getvalue())
                        zip_file.writestr(f"{base_name}_R.png", right_buffer.getvalue())
                        
                        # プログレス更新
                        progress_bar.progress((i + 1) / total_images)
                
                zip_buffer.seek(0)
                status_text.text("✅ 生成完了！")
                
                # ZIPダウンロードボタン
                st.download_button(
                    label="📦 ZIPファイルをダウンロード",
                    data=zip_buffer.getvalue(),
                    file_name=f"rds_batch_{batch_start}_{batch_end}_{batch_step}.zip",
                    mime="application/zip"
                )
                
        except Exception as e:
            st.error(f"バッチ生成エラー: {str(e)}")

# フッター
st.markdown("---")
st.markdown("**Powered by Ogwlab**")