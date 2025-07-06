# フォルダ構造の整理について

## 整理前の構造
```
/workspace/
├── .git/
├── README.md
├── requirements.txt
└── src/
    ├── .gitkeep
    └── app.py (481行の巨大なファイル)
```

## 整理後の構造
```
/workspace/
├── .git/
├── README.md
├── requirements.txt
├── FOLDER_STRUCTURE.md (このファイル)
└── src/
    ├── __init__.py
    ├── app.py (メイン実行ファイル - 約70行)
    ├── rds_generator.py (RDS生成クラス)
    ├── config/
    │   ├── __init__.py
    │   └── settings.py (設定定数)
    ├── utils/
    │   ├── __init__.py
    │   ├── math_utils.py (数学関数)
    │   └── image_utils.py (画像処理関数)
    └── ui/
        ├── __init__.py
        ├── sidebar.py (サイドバーUI)
        ├── main_area.py (メイン表示エリア)
        └── download.py (ダウンロード機能)
```

## 改善点

### 1. モジュール化
- 単一の巨大なファイルを機能別に分割
- 各モジュールが単一の責任を持つ
- インポート関係が明確になった

### 2. 再利用性向上
- UI コンポーネントが独立
- 数学関数やimage処理関数の再利用が容易
- 設定値の一元管理

### 3. 保守性向上
- コードの可読性向上
- デバッグが容易
- テストの作成が簡単

### 4. 拡張性向上
- 新機能の追加が容易
- 既存機能の修正が他に影響しにくい
- 他のプロジェクトへの移植が容易

## ファイル説明

### メインファイル
- `src/app.py`: Streamlitアプリケーションのメイン実行ファイル

### 核となるロジック
- `src/rds_generator.py`: RDS生成のメインクラス

### 設定
- `src/config/settings.py`: デフォルト値と制約値の定義

### ユーティリティ
- `src/utils/math_utils.py`: 数学計算関数
- `src/utils/image_utils.py`: 画像処理関数

### UI コンポーネント
- `src/ui/sidebar.py`: サイドバーのパラメータ設定
- `src/ui/main_area.py`: メイン表示エリア
- `src/ui/download.py`: ダウンロード機能

## 実行方法

従来通り、以下のコマンドで実行できます：

```bash
streamlit run src/app.py
```

## 開発のメリット

1. **チーム開発**: 複数人での開発が容易
2. **単体テスト**: 各モジュールの個別テストが可能
3. **コードレビュー**: 変更箇所の特定が容易
4. **デバッグ**: 問題の特定と修正が迅速
5. **ドキュメント**: 各モジュールの役割が明確

このモジュール化により、コードの品質、保守性、拡張性が大幅に向上しました。