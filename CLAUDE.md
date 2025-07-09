# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## プロジェクト概要

このプロジェクトは、心理学・認知科学研究用のランダムドットステレオグラム（RDS）生成ツールです。2次元FFT（高速フーリエ変換）による位相シフト法を用いて、サブピクセル精度での視差制御を実現しています。

## 技術スタック

- **言語**: Python 3.8+
- **フレームワーク**: Streamlit（Webアプリケーション）
- **主要ライブラリ**: NumPy, SciPy, Pillow
- **依存関係**: requirements.txt で管理

## 環境セットアップ

```bash
# 仮想環境の作成・有効化
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# 依存関係のインストール
pip install -r requirements.txt

# 開発用依存関係のインストール（開発時）
pip install -r requirements-dev.txt

# アプリケーションの起動
streamlit run src/app.py

# テストの実行
pytest

# コードフォーマット
black src/ tests/
```

## アーキテクチャ

### プロジェクト構造

```
src/
├── rds_generator/
│   ├── __init__.py          # パッケージ初期化
│   ├── core.py              # RDSGenerator メインクラス
│   ├── config.py            # 設定管理（RDSConfig）
│   ├── math_utils.py        # 数学計算関数
│   ├── image_utils.py       # 画像処理関数
│   └── app.py               # Streamlit UI
├── app.py                   # アプリケーション起動
tests/                       # テストコード
examples/                    # 使用例
docs/                       # API仕様書
```

### メインクラス: `RDSGenerator`

- `src/rds_generator/core.py` - RDS生成のメインロジック
- **主要メソッド**:
  - `generate_rds(config)` - 設定からRDS生成
  - `generate_stereo_pair()` - ステレオペア生成
  - `generate_rds_from_dict()` - 辞書からRDS生成（後方互換性）

### 設定管理: `RDSConfig`

- `src/rds_generator/config.py` - 型安全な設定クラス
- パラメータの検証と型チェック
- 辞書との相互変換機能

### 技術的特徴

1. **2次元FFT位相シフト法**: サブピクセル精度での視差制御
2. **境界アーチファクト対策**: パディング処理による堅牢な実装
3. **視差の単位変換**: 秒角（arcsec）からピクセルへの正確な変換
4. **モジュール化**: 責務分離による保守性向上
5. **型安全性**: 型ヒントとデータクラスの活用

## 開発時の注意点

### コアロジック

- FFT処理は `math_utils.apply_phase_shift_2d_robust()` で実装
- 境界アーチファクト防止のため、必ずパディング処理を行う
- 視差計算では物理的単位（秒角）を正確にピクセルに変換

### パラメータ制約

- 最小視差: 約0.5ピクセル（モニター依存）
- 最大視差: 融像限界内（一般的に600 arcsec以下）
- 画像サイズ: FFT処理のため2の累乗サイズが効率的

### テスト

```bash
# 全テストの実行
pytest

# カバレッジ付きテスト
pytest --cov=src/rds_generator

# 特定のテストファイル
pytest tests/test_core.py
```

### 実行例

```bash
# 開発サーバーの起動
source venv/bin/activate
streamlit run src/app.py

# パッケージとしてのインストール
pip install -e .

# 本番デプロイはStreamlit Cloudで実行
# URL: https://ogwlab-app-rds-generator.streamlit.app/
```

## ファイル構成

```
/
├── src/
│   ├── rds_generator/   # メインパッケージ
│   └── app.py           # アプリケーション起動
├── tests/               # テストコード
├── examples/            # 使用例
├── docs/               # ドキュメント
├── requirements.txt     # Python依存関係
├── requirements-dev.txt # 開発用依存関係
├── pyproject.toml      # プロジェクト設定
├── LICENSE             # ライセンス
└── README.md           # プロジェクト詳細ドキュメント
```

## 研究用途での使用

このツールは関西学院大学 小川洋和研究室で開発された研究用ツールです。
- 視差検出閾値の測定
- 発達研究での応用
- 両眼視差による奥行き知覚の研究

## 引用

研究成果発表時の引用形式：
```
小川洋和研究室 (2025). RDS生成ツール (v1.0.0). 関西学院大学. 
Retrieved from https://github.com/hrkzogw/ogwlab
```