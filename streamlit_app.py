# -*- coding: utf-8 -*-
"""
DECAES.jl Runner (Streamlit, DICOM/NIfTI/CSV→NIfTI, CSV[gzip]出力, 改善版)
- 入力:
  A) DICOM（1枚以上）→ dcm2niix で NIfTI へ変換
  B) 既存 NIfTI（.nii/.nii.gz）
  C) CSV（1列目=TE[ms], 2列目=Signal）→ 1x1x1xN の4D NIfTIに変換
- 改善点（抜粋）:
  * 一時ディレクトリの確実な掃除（TemporaryDirectory）
  * julia / dcm2niix の存在チェック
  * アップロードのサイズ検証
  * 追加引数の shlex サニタイズ
  * CSV gzip圧縮 + 変数名フィルタ（正規表現）
  * dcm2niix 複数シリーズの選択UI
  * TE推定の根拠（候補/差分中央値/CSV近似誤差）の可視化
  * Julia Pkg.precompile + 任意タイムアウト
  * ライセンス告知強化（DECAES=MIT, dcm2niix=BSD-2-Clause）
"""

import os, json, shlex, tempfile, shutil, zipfile, io, re, subprocess, math
from tempfile import TemporaryDirectory, mkstemp
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import streamlit as st
import numpy as np
import pandas as pd
import h5py
from scipy.io import loadmat as scipy_loadmat
import nibabel as nib  # ← 追加（CSV→NIfTI用）

# ====== 定数 / 設定 ======
JULIA = "julia"
DCM2NIIX = "dcm2niix"

APP_SOURCE_URL = os.environ.get("APP_SOURCE_URL", "")  # このアプリのソースURL（任意）
DECAES_REPO_URL = "https://github.com/jondeuce/DECAES.jl"
DECAES_LICENSE_URL = f"{DECAES_REPO_URL}/blob/master/LICENSE"  # MIT
DCM2NIIX_REPO_URL = "https://github.com/rordenlab/dcm2niix"
DCM2NIIX_LICENSE_URL = f"{DCM2NIIX_REPO_URL}/blob/master/license.txt"

TPN_DIR = Path("THIRD_PARTY_NOTICES")
TPN_DCM2NIIX = TPN_DIR / "dcm2niix-LICENSE.txt"

# アップロード制限（合計）
MAX_TOTAL_BYTES = 500 * 1024 * 1024  # 500MB 目安


# ====== 共通ユーティリティ ======
def _run(cmd: List[str], cwd: Optional[str] = None, env: Optional[dict] = None) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr

def assert_cli_available():
    """julia / dcm2niix が PATH にあるかを確認"""
    import shutil as _shutil
    missing = []
    for exe in (JULIA, DCM2NIIX):
        if _shutil.which(exe) is None:
            missing.append(exe)
    if missing:
        st.error(
            "必須コマンドが見つかりませんでした: " + ", ".join(missing) +
            "\nStreamlit Cloud のビルドログ（packages.txt: julia / dcm2niix）をご確認ください。"
        )
        st.stop()

def validate_uploads(files):
    """合計サイズの上限チェック（DICOM/NIfTI/CSVアップロード共通）"""
    total = 0
    for f in files or []:
        total += len(f.getbuffer())
    if total > MAX_TOTAL_BYTES:
        st.error(f"アップロード合計 {total/1e6:.1f} MB が上限（{MAX_TOTAL_BYTES/1e6:.0f} MB）を超過しました。")
        st.stop()

def _save_upload_to_tempfile(uploaded, suffix: str) -> Path:
    """StreamlitのUploadedFileをNamedTemporaryFileに保存してパス返却"""
    fd, fp = mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(uploaded.getbuffer())
    return Path(fp)

def parse_cli_extra(s: str) -> List[str]:
    """フリーフォーム引数を shlex で安全にトークン化"""
    s = s.strip()
    if not s:
        return []
    try:
        toks = [t for t in shlex.split(s) if t.strip()]
        return toks
    except ValueError:
        st.warning("追加引数の構文に問題があります（引用符の閉じ忘れ等）。該当部分を無視して続行します。")
        return []


# ====== Julia(DECAES) セットアップ ======
@st.cache_resource(show_spinner=False)
def ensure_decaes() -> None:
    """
    初回: named project(@decaes) に DECAES をインストール＆ビルド＆プリコンパイル。
    """
    st.info("初回セットアップ: Julia 用プロジェクト @decaes に DECAES をインストール＆プリコンパイル中…")
    decaes_source = ""
    try:
        decaes_source = st.secrets.get("DECAES_SOURCE", "")
    except Exception:
        pass
    if not decaes_source:
        decaes_source = os.environ.get("DECAES_SOURCE", "")

    jl = (
        'import Pkg; '
        'src = get(ENV, "DECAES_SOURCE", ""); '
        'try src = isempty(src) ? "" : src catch; end; '
        'if isempty(src); Pkg.add("DECAES"); '
        'else; '
        '  if occursin("://", src); Pkg.add(Pkg.PackageSpec(url=src)); '
        '  else; Pkg.add(src); end; '
        'end; '
        'Pkg.build("DECAES"); '
        'Pkg.precompile(); '
        'using DECAES; println("DECAES loaded OK");'
    )
    rc, out, err = _run([JULIA, "--project=@decaes", "-e", jl])
    if rc != 0:
        raise RuntimeError(f"DECAES インストール失敗\n--- STDOUT ---\n{out}\n--- STDERR ---\n{err}")


# ====== DICOM→NIfTI 変換 ======
def dcm2niix_convert(dicom_files: List[Path]) -> List[Dict[str, Any]]:
    """
    DICOM群をシリーズごとにNIfTIへ変換。
    """
    with TemporaryDirectory(prefix="dcm2niix_") as td:
        work = Path(td)
        src_dir = work / "dicom"; out_dir = work / "nifti"
        src_dir.mkdir(parents=True, exist_ok=True); out_dir.mkdir(parents=True, exist_ok=True)
        for i, p in enumerate(dicom_files):
            shutil.copy(str(p), str(src_dir / f"{i:06d}.dcm"))

        rc, out, err = _run([DCM2NIIX, "-z", "y", "-b", "y", "-f", "%p_%s", "-o", str(out_dir), str(src_dir)])
        if rc != 0:
            raise RuntimeError(f"dcm2niix 失敗\n--- STDOUT ---\n{out}\n--- STDERR ---\n{err}")

        results = []
        for nii in sorted(out_dir.glob("*.nii.gz")):
            # 永続化用の一時ファイルにコピー
            fd_n, fp_n = mkstemp(suffix=".nii.gz"); os.close(fd_n); shutil.copy(str(nii), fp_n)
            json_path = nii.with_suffix("").with_suffix(".json")
            if json_path.exists():
                fd_j, fp_j = mkstemp(suffix=".json"); os.close(fd_j); shutil.copy(str(json_path), fp_j)
                js = Path(fp_j)
            else:
                js = None
            results.append({"nifti": Path(fp_n), "json": js, "series_desc": nii.stem})
        if not results:
            raise RuntimeError("dcm2niix の出力が見つかりませんでした。アップロード内容をご確認ください。")
        return results


# ====== BIDS JSON → TE推定 ======
def parse_te_from_json_verbose(json_path: Path) -> Tuple[Optional[float], Dict[str, Any]]:
    """
    BIDS JSONを読み、EchoTimes (秒) または EchoTimeX 群から inter-echo 間隔を推定。
    戻り値: (採用TE[秒] or None, {"candidates": [...秒], "diffs": [...秒], "median_diff": 秒 or None})
    """
    info = {"candidates": [], "diffs": [], "median_diff": None}
    try:
        with open(json_path, "r") as f:
            meta = json.load(f)
    except Exception:
        return None, info

    candidates = []
    if "EchoTimes" in meta and isinstance(meta["EchoTimes"], list) and len(meta["EchoTimes"]) >= 2:
        try:
            candidates = sorted(float(x) for x in meta["EchoTimes"])
        except Exception:
            candidates = []
    else:
        keys = [k for k in meta.keys() if k.lower().startswith("echotime")]
        et_vals = []
        for k in keys:
            try:
                et_vals.append(float(meta[k]))
            except Exception:
                pass
        if len(et_vals) >= 2:
            candidates = sorted(set(et_vals))

    info["candidates"] = candidates
    if len(candidates) >= 2:
        diffs = [candidates[i+1] - candidates[i] for i in range(len(candidates)-1)]
        diffs = [d for d in diffs if d > 0]
        info["diffs"] = diffs
        if diffs:
            diffs.sort()
            info["median_diff"] = float(diffs[len(diffs)//2])
            return info["median_diff"], info
    return None, info


# ====== CSV → 4D NIfTI 変換 ======
def csv_to_nifti(csv_file: Path) -> Tuple[Path, Optional[float], Dict[str, Any]]:
    """
    CSV(TE[ms], Signal)を読み、(1,1,1,N) の NIfTI に変換。
    戻り値: (nifti_path, te_sec (近似; None可), info_dict)
      info_dict: {"nTE": N, "te_ms_list": [...], "interval_ms_median": x, "interval_ms_max_err": y}
    ※ TEが厳密等間隔でない場合、中央値差分を採用し、最大誤差を返す。
    """
    # pandasで柔軟に読み取り（カンマ/タブ/スペース対応）
    try:
        df = pd.read_csv(csv_file, header=None)
    except Exception:
        # セミコロン/タブなどの可能性
        try:
            df = pd.read_csv(csv_file, header=None, sep=None, engine="python")
        except Exception as e:
            raise RuntimeError(f"CSV読込に失敗: {e}")

    if df.shape[1] < 2:
        raise ValueError("CSVは少なくとも2列（1列目=TE[ms], 2列目=Signal）が必要です。")

    te_ms = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    sig = pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy()

    if np.isnan(te_ms).any() or np.isnan(sig).any():
        raise ValueError("CSVに数値化できない値が含まれています。1列目=TE[ms], 2列目=Signal を数値で与えてください。")

    if te_ms.ndim != 1 or sig.ndim != 1 or te_ms.size != sig.size or te_ms.size < 2:
        raise ValueError("CSVの行数が不正です（2行以上、列数一致が必要）。")

    # TE等間隔チェック（中央値差分を採用）
    te_sorted_idx = np.argsort(te_ms)
    te_ms = te_ms[te_sorted_idx]
    sig = sig[te_sorted_idx]

    diffs = np.diff(te_ms)
    med = float(np.median(diffs))
    max_err = float(np.max(np.abs(diffs - med))) if diffs.size else 0.0
    info = {
        "nTE": int(te_ms.size),
        "te_ms_list": te_ms.tolist(),
        "interval_ms_median": med,
        "interval_ms_max_err": max_err,
    }

    # 4D NIfTIへ（形状: 1x1x1xN）
    arr = sig.astype(np.float32).reshape((1, 1, 1, -1))
    img = nib.Nifti1Image(arr, affine=np.eye(4))
    fd, fp = mkstemp(suffix=".nii.gz"); os.close(fd)
    nib.save(img, fp)

    # TE（秒）を返す（中央値差分を近似値として）
    te_sec = med / 1000.0 if med > 0 else None
    return Path(fp), te_sec, info


# ====== .mat → CSV(gzip) ======
def _save_array_as_long_csv_gz(arr: np.ndarray, csv_gz_path: Path, max_rows: Optional[int] = None) -> int:
    if arr.size == 0:
        pd.DataFrame(columns=[*(f"index_{i}" for i in range(arr.ndim)), "value"]).to_csv(
            f"{csv_gz_path}.gz", index=False, compression="gzip"
        )
        return 0
    flat = arr.ravel(order="C")
    if max_rows is not None:
        flat = flat[:max_rows]
        idxs = np.unravel_index(np.arange(len(flat)), arr.shape)
    else:
        idxs = np.unravel_index(np.arange(arr.size), arr.shape)
    data = {f"index_{i}": idx for i, idx in enumerate(idxs)}
    data["value"] = flat
    df = pd.DataFrame(data)
    df.to_csv(f"{csv_gz_path}.gz", index=False, compression="gzip")
    return len(df)

def _save_array_slice_csv_gz(arr: np.ndarray, csv_gz_path: Path, axis: int, index: int) -> Tuple[int, str]:
    msg = ""
    if axis < 0 or axis >= arr.ndim:
        msg = f"axis {axis} が範囲外(0..{arr.ndim-1})"
        pd.DataFrame({"error":[msg]}).to_csv(f"{csv_gz_path}.gz", index=False, compression="gzip")
        return 0, msg
    if index < 0 or index >= arr.shape[axis]:
        msg = f"index {index} が軸{axis}の範囲外(0..{arr.shape[axis]-1})"
        pd.DataFrame({"error":[msg]}).to_csv(f"{csv_gz_path}.gz", index=False, compression="gzip")
        return 0, msg
    slicer = [slice(None)] * arr.ndim
    slicer[axis] = index
    sl = np.asarray(arr[tuple(slicer)])
    if sl.ndim == 2:
        pd.DataFrame(sl).to_csv(f"{csv_gz_path}.gz", index=False, compression="gzip")
        return int(sl.shape[0]), ""
    else:
        n = _save_array_as_long_csv_gz(sl, csv_gz_path, max_rows=None)
        return n, ""

def load_mat_any_numeric(mat_path: Path) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    try:
        with h5py.File(mat_path, "r") as f:
            def visit(name, obj):
                if isinstance(obj, h5py.Dataset):
                    try:
                        data = obj[()]
                        if isinstance(data, np.ndarray) and np.issubdtype(data.dtype, np.number):
                            arrays[name] = np.array(data)
                    except Exception:
                        pass
            f.visititems(visit)
        if arrays:
            return arrays
    except Exception:
        pass
    try:
        mdict = scipy_loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        for k, v in mdict.items():
            if k.startswith("__"):
                continue
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                arrays[k] = v
    except Exception:
        pass
    return arrays

def convert_mat_dir_to_csv_gz(
    mat_dir: Path, csv_dir: Path, mode: str = "long",
    slice_axis: int = 2, slice_index: int = 0,
    long_max_rows: Optional[int] = None,
    var_regex: Optional[re.Pattern] = None
) -> List[Path]:
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_paths: List[Path] = []

    if mode == "summary":
        rows = []
        for mat_file in sorted(mat_dir.glob("*.mat")):
            arrays = load_mat_any_numeric(mat_file)
            for name, arr in arrays.items():
                if var_regex and not var_regex.search(name):
                    continue
                arrf = np.array(arr, dtype=float)
                rows.append({
                    "mat_file": mat_file.name,
                    "var": name,
                    "ndim": arrf.ndim,
                    "shape": "x".join(map(str, arrf.shape)),
                    "size": int(arrf.size),
                    "nan_count": int(np.isnan(arrf).sum()),
                    "mean": float(np.nanmean(arrf)) if arrf.size else np.nan,
                    "std": float(np.nanstd(arrf)) if arrf.size else np.nan,
                    "min": float(np.nanmin(arrf)) if arrf.size else np.nan,
                    "max": float(np.nanmax(arrf)) if arrf.size else np.nan,
                    "sum": float(np.nansum(arrf)) if arrf.size else np.nan,
                })
        out = csv_dir / "summary.csv.gz"
        pd.DataFrame(rows).to_csv(out, index=False, compression="gzip")
        return [out]

    for mat_file in sorted(mat_dir.glob("*.mat")):
        arrays = load_mat_any_numeric(mat_file)
        for name, arr in arrays.items():
            if var_regex and not var_regex.search(name):
                continue
            safe = name.replace("/", "__")
            if mode == "long":
                out = csv_dir / f"{mat_file.stem}__{safe}__long.csv"
                _ = _save_array_as_long_csv_gz(np.asarray(arr), out, max_rows=long_max_rows)
                csv_paths.append(Path(f"{out}.gz"))
            elif mode == "slice":
                out = csv_dir / f"{mat_file.stem}__{safe}__slice_ax{slice_axis}_idx{slice_index}.csv"
                _n, _msg = _save_array_slice_csv_gz(np.asarray(arr), out, axis=slice_axis, index=slice_index)
                csv_paths.append(Path(f"{out}.gz"))
    return csv_paths


# ====== DECAES 実行 ======
def build_decaes_args(
    inputs: List[Path], masks: List[Path], out_dir: Path,
    do_t2map: bool, do_t2part: bool,
    te_sec: Optional[float], nT2: Optional[int],
    t2min_sec: Optional[float], t2max_sec: Optional[float],
    spmin_sec: Optional[float], spmax_sec: Optional[float],
    mpmin_sec: Optional[float], mpmax_sec: Optional[float],
    reg: str, reg_params: List[float],
    matrix_size: Optional[List[int]],
    nTE: Optional[int], threshold: Optional[float],
    chi2factor: Optional[float], t1: Optional[float], sigmoid: Optional[float],
    b1maps: List[Path], nRefAngles: Optional[int], nRefAnglesMin: Optional[int],
    minRefAngle: Optional[float], setFlipAngle: Optional[float], refConAngle: Optional[float],
    save_decay: bool, save_basis: bool, save_regparam: bool, save_resnorm: bool,
    use_bet: bool, betargs: List[str], betpath: str,
    extra_args: List[str],
) -> List[str]:
    args: List[str] = [str(p) for p in inputs]
    if masks: args += ["--mask"] + [str(m) for m in masks]
    args += ["--output", str(out_dir)]
    if do_t2map:  args.append("--T2map")
    if do_t2part: args.append("--T2part")
    if te_sec is not None:        args += ["--TE", str(te_sec)]
    if nT2 is not None:           args += ["--nT2", str(nT2)]
    if t2min_sec is not None and t2max_sec is not None: args += ["--T2Range", str(t2min_sec), str(t2max_sec)]
    if spmin_sec is not None and spmax_sec is not None: args += ["--SPWin", str(spmin_sec), str(spmax_sec)]
    if mpmin_sec is not None and mpmax_sec is not None: args += ["--MPWin", str(mpmin_sec), str(mpmax_sec)]
    args += ["--Reg", reg]
    if reg.lower() in ("chi2", "mdp") and reg_params: args += ["--RegParams"] + [str(x) for x in reg_params]
    if matrix_size and len(matrix_size) == 3: args += ["--MatrixSize", *map(str, matrix_size[:3])]
    if nTE is not None:          args += ["--nTE", str(nTE)]
    if threshold is not None:    args += ["--Threshold", str(threshold)]
    if chi2factor is not None:   args += ["--Chi2Factor", str(chi2factor)]
    if t1 is not None:           args += ["--T1", str(t1)]
    if sigmoid is not None:      args += ["--Sigmoid", str(sigmoid)]
    if b1maps:                   args += ["--B1map"] + [str(p) for p in b1maps]
    if nRefAngles is not None:   args += ["--nRefAngles", str(nRefAngles)]
    if nRefAnglesMin is not None:args += ["--nRefAnglesMin", str(nRefAnglesMin)]
    if minRefAngle is not None:  args += ["--MinRefAngle", str(minRefAngle)]
    if setFlipAngle is not None: args += ["--SetFlipAngle", str(setFlipAngle)]
    if refConAngle is not None:  args += ["--RefConAngle", str(refConAngle)]
    if save_decay:               args += ["--SaveDecayCurve"]
    if save_basis:               args += ["--SaveNNLSBasis"]
    if save_regparam:            args += ["--SaveRegParam"]
    if save_resnorm:             args += ["--SaveResidualNorm"]
    if use_bet:                  args += ["--bet"]
    if betpath.strip():          args += ["--betpath", betpath.strip()]
    if betargs:                  args += ["--betargs", " ".join(betargs)]
    if extra_args:               args += extra_args
    return args

def execute_decaes(args: List[str], out_dir: Path, timeout_sec: int) -> Tuple[bytes, str]:
    cmd = [JULIA, "--project=@decaes", "--threads=auto", "--compile=min", "-e", "using DECAES; main()", "--"] + args
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
    stdout, stderr = proc.stdout, proc.stderr
    (out_dir / "decaes_log.txt").write_text(stdout + ("\n\n[STDERR]\n" + stderr if stderr.strip() else ""), encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"[ERROR rc={proc.returncode}]\n--- STDOUT ---\n{stdout}\n--- STDERR ---\n{stderr}")

    # 出力一式をストリーミングZIP化
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in out_dir.rglob("*"):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(out_dir)))
    bio.seek(0)
    return bio.read(), stdout


# ====== UI ======
st.set_page_config(page_title="DECAES.jl Runner（改良版：CSV対応）", layout="wide")
st.title("DECAES.jl 実行UI（DICOM/NIfTI/CSV対応・任意パラメータ・CSV[gzip]・堅牢化）")

with st.expander("利用上の注意 / ライセンス", expanded=True):
    st.markdown(
        f"""
- **入力**: DICOM（1枚以上）/ NIfTI / **CSV(TE[ms], Signal)**。CSVは 1列目=TE[ms], 2列目=Signal を数値で。
- **単位**: TE/T2Range/SP/MP は **秒**（UIは ms 入力→内部で秒へ変換）。CSVは内部でTE間隔を推定。
- **PHI注意**: 医療画像は匿名化してアップロードしてください。

**ライセンス告知（必読）**
- DECAES.jl: MIT（ソース: <{DECAES_REPO_URL}>, LICENSE: <{DECAES_LICENSE_URL}>）
- dcm2niix: BSD-2-Clause（多くのソース）※一部 Public Domain / MIT（ソース: <{DCM2NIIX_REPO_URL}>, LICENSE: <{DCM2NIIX_LICENSE_URL}>）
- 本アプリは dcm2niix を**外部プログラム**として呼び出し、このリポジトリでバイナリ再配布は行いません。
- このアプリのソース: {("**" + APP_SOURCE_URL + "**") if APP_SOURCE_URL else "（環境変数 APP_SOURCE_URL で公開リポURLを設定可）"}
        """
    )
    if TPN_DCM2NIIX.exists():
        with open(TPN_DCM2NIIX, "r", encoding="utf-8") as f:
            st.text_area("dcm2niix LICENSE（同梱コピー）", f.read(), height=200)

# 依存CLI確認
assert_cli_available()

# Julia/DECAES セットアップ
ensure_decaes()

colL, colR = st.columns([1, 1])

# ====== 左: 入力・基本パラメータ ======
with colL:
    mode_in = st.radio("入力モード", ["DICOM（1枚以上）", "NIfTI（既存）", "CSV（TE[ms],Signal）"], index=0)

    dicoms = []
    nifti_inputs = []
    csv_inputs = []
    mask_paths = []

    if mode_in.startswith("DICOM"):
        dcm_files = st.file_uploader("DICOM（複数可）", type=None, accept_multiple_files=True)
        validate_uploads(dcm_files)
        if dcm_files:
            dicoms = [_save_upload_to_tempfile(f, suffix=f".{f.name.split('.')[-1]}") for f in dcm_files]
            st.success(f"DICOM一時保存: {len(dicoms)}")

    elif mode_in.startswith("NIfTI"):
        files = st.file_uploader("NIfTI（複数可）", type=["nii", "nii.gz"], accept_multiple_files=True)
        validate_uploads(files)
        if files:
            nifti_inputs = [_save_upload_to_tempfile(f, suffix=f".{f.name.split('.')[-1]}") for f in files]
            st.success(f"NIfTI一時保存: {len(nifti_inputs)}")

    else:  # CSV
        csv_files = st.file_uploader("CSV（複数可）: 1列目=TE[ms], 2列目=Signal", type=["csv", "txt"], accept_multiple_files=True)
        validate_uploads(csv_files)
        if csv_files:
            csv_inputs = [_save_upload_to_tempfile(f, suffix=f".csv") for f in csv_files]
            st.success(f"CSV一時保存: {len(csv_inputs)}")

    mask_files = None
    if not mode_in.startswith("CSV"):  # CSVは1ボクセル相当なのでMask不要
        mask_files = st.file_uploader("Mask（任意・NIfTI・未指定または入力と同数）", type=["nii", "nii.gz"], accept_multiple_files=True)
        validate_uploads(mask_files)
        if mask_files:
            mask_paths = [_save_upload_to_tempfile(f, suffix=f".{f.name.split('.')[-1]}") for f in mask_files]

    st.subheader("処理モード")
    do_t2map  = st.checkbox("T2map（T2分布推定）", value=True)
    do_t2part = st.checkbox("T2part（短/中T2帯解析・MWF等）", value=True)

    st.subheader("撮像/推定（ms入力→内部で秒）")
    te_ms     = st.number_input("TE [ms]（inter-echo spacing。CSV時は推定で上書き）", value=10.0, min_value=0.1, step=0.5)
    nT2       = st.number_input("nT2（T2ビン数）", value=60, min_value=10, max_value=200, step=5)
    t2min_ms  = st.number_input("T2Range 最小 [ms]", value=5.0,  min_value=0.1, step=0.5)
    t2max_ms  = st.number_input("T2Range 最大 [ms]", value=2000.0, min_value=1.0, step=10.0)
    spmin_ms  = st.number_input("SPWin 最小 [ms]", value=5.0,  min_value=0.1, step=0.5)
    spmax_ms  = st.number_input("SPWin 最大 [ms]", value=40.0, min_value=0.5, step=1.0)
    mpmin_ms  = st.number_input("MPWin 最小 [ms]", value=40.0, min_value=0.5, step=1.0)
    mpmax_ms  = st.number_input("MPWin 最大 [ms]", value=200.0, min_value=1.0, step=5.0)

    st.subheader("正則化")
    reg = st.selectbox("Reg", ["lcurve", "gcv", "chi2", "mdp", "none"], index=1)
    reg_params_txt = st.text_input("RegParams（chi2/mdp時・空白区切り）", value="")

# ====== 右: 拡張・CSV・実行 ======
with colR:
    st.subheader("拡張（CLIオプション）")
    with st.expander("B1 / 反転角・参照角（上級者向け）", expanded=False):
        b1_files = st.file_uploader("B1map（複数可）", type=["nii","nii.gz","mat","par","xml","rec"], accept_multiple_files=True)
        validate_uploads(b1_files)
        nRefAngles = st.number_input("nRefAngles", value=0, min_value=0, step=1)
        nRefAnglesMin = st.number_input("nRefAnglesMin", value=0, min_value=0, step=1)
        minRefAngle = st.number_input("MinRefAngle [deg]", value=0.0, step=1.0)
        setFlipAngle = st.number_input("SetFlipAngle [deg]", value=0.0, step=1.0)
        refConAngle = st.number_input("RefConAngle [deg]", value=0.0, step=1.0)

    with st.expander("T2map/T2part補助", expanded=False):
        matrix_size_txt = st.text_input("MatrixSize（例: 192 192 64）", value="")
        nTE_val = st.number_input("nTE（echo数；CSV時は自動設定）", value=0, min_value=0, step=1)
        threshold = st.number_input("Threshold（1st echo intensity cutoff）", value=0.0, step=0.1)
        chi2factor = st.number_input("Chi2Factor", value=0.0, step=0.1)
        t1_val = st.number_input("T1 [s]", value=0.0, step=0.1)
        sigmoid = st.number_input("Sigmoid", value=0.0, step=0.1)

    with st.expander("保存オプション", expanded=False):
        save_decay = st.checkbox("SaveDecayCurve", value=False)
        save_basis = st.checkbox("SaveNNLSBasis", value=False)
        save_regparam = st.checkbox("SaveRegParam", value=False)
        save_resnorm = st.checkbox("SaveResidualNorm", value=False)

    with st.expander("BET（自動mask生成）※FSL依存・非推奨", expanded=False):
        use_bet = st.checkbox("BETを使う（Mask未指定時のみ）", value=False)
        betpath = st.text_input("betpath（絶対パス指定用）", value="")
        betargs_txt = st.text_input("betargs（例: -m -n -f 0.25 -R）", value="")

    with st.expander("フリーフォーム追加引数", expanded=True):
        extra_txt = st.text_input("スペース区切りでCLIへ直渡し（例: --SaveRegParam --SaveResidualNorm）", value="")

    st.subheader("CSV出力（.mat → CSV.gz）")
    csv_mode = st.selectbox("CSVモード", ["long（index...,value）", "slice（指定2D）", "summary（要約統計）"], index=0)
    long_cap = st.number_input("long最大行数（0=無制限）", value=0, min_value=0, step=100000)
    slice_axis = st.number_input("slice軸（0起点）", value=2, min_value=0, step=1)
    slice_index = st.number_input("sliceインデックス", value=0, min_value=0, step=1)
    var_filter_regex = st.text_input("CSV対象 変数名フィルタ（正規表現, 空=全て）", value="")

    TIMEOUT_SEC = st.number_input("解析タイムアウト [秒]", value=3600, min_value=60, step=60)

    run_btn = st.button("解析実行", type="primary")
    log_area = st.empty()

# ====== 実行 ======
if run_btn:
    try:
        # 入力準備
        inputs_for_decaes: List[Path] = []
        te_ms_overwrite: Optional[float] = None
        te_evidence_msgs: List[str] = []

        if mode_in.startswith("DICOM"):
            if not dicoms:
                st.error("少なくとも1枚以上のDICOMをアップロードしてください。"); st.stop()
            st.info("dcm2niix で NIfTI 変換中…")
            series_list = dcm2niix_convert(dicoms)

            # シリーズ選択UI
            all_desc = [s["series_desc"] for s in series_list]
            picked = st.multiselect("解析するシリーズを選択（未選択=全選択）", all_desc, default=all_desc)
            selected = [s for s in series_list if (s["series_desc"] in picked or not picked)]

            for s in selected:
                inputs_for_decaes.append(s["nifti"])
                if te_ms_overwrite is None and s["json"] and s["json"].exists():
                    te_s, info = parse_te_from_json_verbose(s["json"])
                    if info["candidates"]:
                        ets_ms = [x * 1000.0 for x in info["candidates"]]
                        diffs_ms = [x * 1000.0 for x in info["diffs"]]
                        med_ms = info["median_diff"] * 1000.0 if info["median_diff"] else None
                        te_evidence_msgs.append(
                            f"[{s['series_desc']}] EchoTimes(ms)候補={np.round(ets_ms,3).tolist()} "
                            f"→ Δ(ms)={np.round(diffs_ms,3).tolist()} → 中央値≈ {med_ms:.3f} ms" if med_ms else
                            f"[{s['series_desc']}] EchoTimes(ms)候補={np.round(ets_ms,3).tolist()}（Δ推定不可）"
                        )
                    if te_s and te_s > 0:
                        te_ms_overwrite = te_s * 1000.0

            st.success(f"NIfTI変換: {len(inputs_for_decaes)}")
            if te_ms_overwrite is not None:
                st.info(f"DICOMから推定されたTE候補 ≈ {te_ms_overwrite:.3f} ms（必要に応じて修正可）")
                for m in te_evidence_msgs:
                    st.caption(m)
                te_ms = float(te_ms_overwrite)

        elif mode_in.startswith("NIfTI"):
            if not nifti_inputs:
                st.error("NIfTIを1つ以上アップロードしてください。"); st.stop()
            inputs_for_decaes = nifti_inputs

        else:  # CSV
            if not csv_inputs:
                st.error("CSVを1つ以上アップロードしてください。"); st.stop()
            st.info("CSV を 4D NIfTI（1x1x1xN）へ変換中…")
            csv_infos = []
            for i, p in enumerate(csv_inputs):
                nii_path, te_sec_est, info = csv_to_nifti(p)
                inputs_for_decaes.append(nii_path)
                csv_infos.append((p.name, te_sec_est, info))
            # TE推定（CSV由来）：最初のCSVでUI上書き
            first_name, te_sec_est, info = csv_infos[0]
            if te_sec_est and te_sec_est > 0:
                te_ms_overwrite = te_sec_est * 1000.0
                te_ms = float(te_ms_overwrite)
                # 誤差表示
                st.info(f"CSVから推定されたTE間隔 ≈ {te_ms:.3f} ms（{first_name}）")
                st.caption(
                    f"nTE={info['nTE']}, ΔTE中央値={info['interval_ms_median']:.6f} ms, "
                    f"最大偏差={info['interval_ms_max_err']:.6f} ms（非等間隔CSVは近似扱い）"
                )
            # nTEはCSVの行数に合わせる（UIのnTE入力とは独立）
            nTE_val = info["nTE"]

        if mask_paths and len(mask_paths) not in (0, len(inputs_for_decaes)):
            st.error("Maskは未指定か、入力と同数で指定してください。"); st.stop()

        # 単位変換 ms→s（0は未指定扱いに）
        def to_s_optional(ms: float) -> Optional[float]:
            return None if (ms is None or float(ms) == 0.0) else float(ms)/1000.0
        te_sec    = to_s_optional(te_ms)
        t2min_sec = to_s_optional(t2min_ms); t2max_sec = to_s_optional(t2max_ms)
        spmin_sec = to_s_optional(spmin_ms); spmax_sec = to_s_optional(spmax_ms)
        mpmin_sec = to_s_optional(mpmin_ms); mpmax_sec = to_s_optional(mpmax_ms)

        # nTE の扱い：CSVモードでは前段で自動設定済み。それ以外はUI値。
        if not mode_in.startswith("CSV"):
            nTE_val = int(nTE_val) if 'nTE_val' in locals() and nTE_val > 0 else (None if 'nTE_val' not in locals() else None)

        nT2_val = int(nT2) if nT2 > 0 else None

        # RegParams
        reg_params: List[float] = []
        if reg.lower() in ("chi2","mdp") and reg_params_txt.strip():
            reg_params = [float(x) for x in reg_params_txt.split()]

        # MatrixSize
        matrix_size = None
        if 'matrix_size_txt' in locals() and matrix_size_txt.strip():
            try:
                parts = [int(p) for p in matrix_size_txt.replace(",", " ").split()]
                matrix_size = parts[:3] if len(parts) >= 3 else None
            except Exception:
                matrix_size = None

        # B1map
        b1maps: List[Path] = []
        if 'b1_files' in locals() and b1_files:
            b1maps = [_save_upload_to_tempfile(f, suffix=f".{f.name.split('.')[-1]}") for f in b1_files]

        # 数値オプション（0→未指定）
        def none_if_zero(x: float) -> Optional[float]:
            return None if (x is None or float(x) == 0.0) else float(x)
        threshold_opt   = none_if_zero(threshold)
        chi2factor_opt  = none_if_zero(chi2factor)
        t1_opt          = none_if_zero(t1_val)
        sigmoid_opt     = none_if_zero(sigmoid)
        minRefAngle_opt = none_if_zero(minRefAngle)
        setFlipAngle_opt= none_if_zero(setFlipAngle)
        refConAngle_opt = none_if_zero(refConAngle)
        nRefAngles_opt  = int(nRefAngles) if nRefAngles > 0 else None
        nRefAnglesMin_opt = int(nRefAnglesMin) if nRefAnglesMin > 0 else None

        # 追加引数（サニタイズ）
        extra_args = parse_cli_extra(extra_txt)
        betargs = parse_cli_extra(betargs_txt)

        with TemporaryDirectory(prefix="decaes_out_") as td:
            out_dir = Path(td)
            # 引数構築
            args = build_decaes_args(
                inputs=inputs_for_decaes, masks=mask_paths, out_dir=out_dir,
                do_t2map=bool(do_t2map), do_t2part=bool(do_t2part),
                te_sec=te_sec, nT2=nT2_val,
                t2min_sec=t2min_sec, t2max_sec=t2max_sec,
                spmin_sec=spmin_sec, spmax_sec=spmax_sec,
                mpmin_sec=mpmin_sec, mpmax_sec=mpmax_sec,
                reg=str(reg), reg_params=reg_params,
                matrix_size=matrix_size, nTE=int(nTE_val) if 'nTE_val' in locals() and nTE_val else None,
                threshold=threshold_opt, chi2factor=chi2factor_opt, t1=t1_opt, sigmoid=sigmoid_opt,
                b1maps=b1maps, nRefAngles=nRefAngles_opt, nRefAnglesMin=nRefAnglesMin_opt,
                minRefAngle=minRefAngle_opt, setFlipAngle=setFlipAngle_opt, refConAngle=refConAngle_opt,
                save_decay=save_decay, save_basis=save_basis, save_regparam=save_regparam, save_resnorm=save_resnorm,
                use_bet=use_bet, betargs=betargs, betpath=betpath,
                extra_args=extra_args,
            )

            st.info("DECAES 実行中…（初回はプリコンパイルで時間を要する場合があります）")
            try:
                zip_bytes_mat, log = execute_decaes(args, out_dir, timeout_sec=int(TIMEOUT_SEC))
            except subprocess.TimeoutExpired:
                st.error(f"時間超過（>{TIMEOUT_SEC}s）。入力サイズやパラメータを調整してください。")
                st.stop()

            st.text_area("ログ（DECAES stdout/err）", value=log, height=280)

            # .mat → CSV.gz
            st.info("`.mat` → CSV.gz 変換中…")
            csv_dir = out_dir / "csv"
            max_rows = None if int(long_cap) == 0 else int(long_cap)
            var_pattern = re.compile(var_filter_regex) if var_filter_regex.strip() else None
            if csv_mode.startswith("long"):
                _ = convert_mat_dir_to_csv_gz(out_dir, csv_dir, mode="long", long_max_rows=max_rows, var_regex=var_pattern)
            elif csv_mode.startswith("slice"):
                _ = convert_mat_dir_to_csv_gz(out_dir, csv_dir, mode="slice",
                                              slice_axis=int(slice_axis), slice_index=int(slice_index),
                                              var_regex=var_pattern)
            else:
                _ = convert_mat_dir_to_csv_gz(out_dir, csv_dir, mode="summary", var_regex=var_pattern)

            # 総合ZIP（.mat + CSV.gz + ログ）
            final_bio = io.BytesIO()
            with zipfile.ZipFile(final_bio, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for p in out_dir.rglob("*"):
                    if p.is_file():
                        zf.write(p, arcname=str(p.relative_to(out_dir)))
            final_bio.seek(0)

            st.download_button(
                "出力一式（.mat + CSV.gz + ログ）をダウンロード（ZIP）",
                data=final_bio.read(),
                file_name="decaes_outputs_with_csv.zip",
                mime="application/zip"
            )

            st.success("完了。ZIPをダウンロードしてください。")

    except Exception as e:
        st.error(f"実行エラー: {e}")
