"""윈도잉·정규화 (전처리, T0.3).

흐름:
  1) 1D 신호 → 고정 길이 윈도우 (N, L)   [config: window.length / overlap]
  2) 정규화 통계(평균·표준편차)를 '정상 train'에서만 fit → models/scaler.pkl 저장
  3) 저장된 통계로 어떤 데이터든 동일하게 apply (누수 차단)

정규화 방식 — '전역(global)' z-score:
  정상 train 전체에서 평균·표준편차 '하나'를 구해 모든 윈도우에 동일 적용한다.
  윈도우마다 따로 정규화하면 진폭이 사라져 경로 A(RMS·peak·크레스트팩터)가 죽으므로,
  전역 통계로 스케일만 맞추고 윈도우 간 진폭 차이(=고장의 큰 진폭)는 보존한다.

실행(검증):
    .venv\\Scripts\\python.exe -m src.preprocessing
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path, set_seed


def make_windows(signal: np.ndarray, length: int, overlap: float) -> np.ndarray:
    """1D 신호를 고정 길이 윈도우로 자른다.

    length  : 한 윈도우 길이(포인트)
    overlap : 겹침 비율 (0.5 = 절반씩 겹침) → 이동 보폭 step = length*(1-overlap)
    반환: (N, length) 2D 배열. 신호가 length 보다 짧으면 (0, length).
    """
    signal = np.asarray(signal).ravel()
    step = max(1, int(round(length * (1.0 - overlap))))
    if len(signal) < length:
        return np.empty((0, length), dtype=np.float64)
    n = 1 + (len(signal) - length) // step
    idx = np.arange(length)[None, :] + (np.arange(n)[:, None] * step)
    return signal[idx].astype(np.float64)


def make_windows_from_signals(
    signals: dict[str, np.ndarray], length: int, overlap: float
) -> np.ndarray:
    """여러 신호(파일별)를 각각 윈도잉해 하나의 (N, L) 배열로 이어 붙인다."""
    chunks = [make_windows(sig, length, overlap) for sig in signals.values()]
    chunks = [c for c in chunks if len(c) > 0]
    if not chunks:
        return np.empty((0, length), dtype=np.float64)
    return np.concatenate(chunks, axis=0)


def fit_scaler(windows: np.ndarray, method: str = "zscore") -> dict:
    """정상 train 윈도우에서 전역 정규화 통계를 계산한다(여기서만 fit).

    반환: {'method','mean','std','n_windows','n_points'}
    """
    if method != "zscore":
        raise ValueError(f"지원하지 않는 정규화: {method}")
    mean = float(windows.mean())
    std = float(windows.std())
    if std == 0.0:
        std = 1.0  # 0 나눗셈 방지
    return {
        "method": method,
        "mean": mean,
        "std": std,
        "n_windows": int(windows.shape[0]),
        "n_points": int(windows.size),
    }


def apply_scaler(windows: np.ndarray, scaler: dict) -> np.ndarray:
    """저장된 통계로 z-score 정규화 적용: (x - mean) / std."""
    return (windows - scaler["mean"]) / scaler["std"]


def save_scaler(scaler: dict, path: str | Path) -> None:
    """정규화 통계를 pickle 로 저장 (배포 시 동일 전처리 보장)."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_scaler(path: str | Path) -> dict:
    """저장된 정규화 통계를 불러온다."""
    with open(resolve_path(path), "rb") as f:
        return pickle.load(f)


def _main() -> None:
    """T0.3 검증: 윈도우 (N, L) 생성 + 정규화 통계 저장, 단계별 shape 출력."""
    sys.stdout.reconfigure(encoding="utf-8")
    from src.data_loader import load_normal_signals

    cfg = load_config()
    set_seed(cfg["seed"])
    length = cfg["window"]["length"]
    overlap = cfg["window"]["overlap"]

    print("=== T0.3 윈도잉·전처리 ===")
    normal = load_normal_signals(cfg)
    print(f"정상 신호 {len(normal)}개 로드")
    for name, sig in normal.items():
        print(f"  - {name}: {len(sig):,} 포인트")

    # 1) 윈도잉
    windows = make_windows_from_signals(normal, length, overlap)
    print(f"\n[1] 윈도잉 → shape {windows.shape}  (length={length}, overlap={overlap})")
    if windows.shape[0] == 0:
        print("  ⚠️ 윈도우 0개 — data/raw/normal 에 .mat 데이터가 있는지 확인")
        return

    # 2) 정상 train 에서만 정규화 통계 fit → 저장
    scaler = fit_scaler(windows, cfg["normalize"]["method"])
    save_scaler(scaler, cfg["artifacts"]["scaler"])
    print(f"[2] 정규화 통계 fit (정상 train) → {cfg['artifacts']['scaler']}")
    print(f"    mean={scaler['mean']:.5f}, std={scaler['std']:.5f}, n_windows={scaler['n_windows']}")

    # 3) 적용 (정규화 후 평균≈0, 표준편차≈1 확인)
    norm = apply_scaler(windows, scaler)
    print(f"[3] 정규화 적용 → mean={norm.mean():.5f}, std={norm.std():.5f}")

    # 윈도우 배열 저장 (data/processed)
    out = resolve_path(cfg["data"]["processed"]) / "normal_windows.npy"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, windows)
    print(f"\n[저장] 윈도우 배열 → {out.relative_to(_ROOT)}  shape {windows.shape}")
    print("T0.3 OK — (N, L) 윈도우 + 정규화 통계(scaler.pkl) 생성됨.")


if __name__ == "__main__":
    _main()
