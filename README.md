# DECAES.jl Runner (Streamlit, DICOM/NIfTI/CSV, CSV[gzip], Robust)

## 概要
- ブラウザから **DICOM**（1枚以上）/ **NIfTI** / **CSV(TE[ms], Signal)** をアップロード  
- DICOMは内部で **dcm2niix** により NIfTI 変換  
- **DECAES.jl**（Julia, **MIT**）を CLI で実行  
- **.mat** 出力に加え、選択方式で **CSV** を **gzip** で自動生成（long / slice / summary、変数名フィルタあり）  
- 改善点: 一時ディレクトリ掃除、依存チェック、引数サニタイズ、シリーズ選択、TE根拠表示、CSV対応、プリコンパイル、可変タイムアウト、など

## デプロイ
1. 本リポジトリを GitHub にプッシュ  
2. Streamlit Community Cloud → **New app** → リポジトリ指定 → Deploy  
3. （任意）Secrets に `DECAES_SOURCE` を設定すると、フォーク/ブランチへ差し替え可能  
4. （任意）環境変数 `APP_SOURCE_URL` にこのリポURLを設定すると、アプリ内の告知欄に表示されます

## ライセンス / OSS 告知
- **DECAES.jl**: MIT  
  - ソース: https://github.com/jondeuce/DECAES.jl  
  - LICENSE: https://github.com/jondeuce/DECAES.jl/blob/master/LICENSE
- **dcm2niix**: BSD-2-Clause（大部分）※一部 Public Domain / MIT  
  - ソース: https://github.com/rordenlab/dcm2niix  
  - LICENSE: https://github.com/rordenlab/dcm2niix/blob/master/license.txt  
- 詳細は `NOTICE.md` と `THIRD_PARTY_NOTICES/` を参照してください。

> **配布形態**: 本アプリは dcm2niix を外部プログラムとして呼び出し、バイナリを本リポジトリで再配布しません。
