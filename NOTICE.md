# Open Source Notices

本アプリは以下のオープンソースを利用しています。それぞれ **元ライセンス**に従います。

## DECAES.jl
- ライセンス: **MIT**
- ソース: https://github.com/jondeuce/DECAES.jl
- LICENSE: https://github.com/jondeuce/DECAES.jl/blob/master/LICENSE  
- 備考: 本アプリは DECAES を外部呼び出し（`julia --project=@decaes`）で使用します。`DECAES_SOURCE`（環境変数 or `st.secrets`）でフォーク/ブランチ等へ差し替え可能。

## dcm2niix（DICOM→NIfTI 変換）
- ライセンス: **BSD-2-Clause**（大部分）  
  - 一部ファイルは **Public Domain**（`nifti*.*`, `miniz.c`）・**MIT**（`ujpeg.cpp`）
- ソース: https://github.com/rordenlab/dcm2niix
- LICENSE: https://github.com/rordenlab/dcm2niix/blob/master/license.txt
- 備考:
  - 本アプリは dcm2niix を **外部プログラム（`subprocess`）として実行**し、このリポジトリでバイナリ再配布は行いません。
  - 透明性のため、上流の BSD-2-Clause 文章を `THIRD_PARTY_NOTICES/dcm2niix-LICENSE.txt` に同梱しています（参考用）。
