"""시간-주파수 변환 (스펙트로그램, T2.1) — 경로 B/C 입력 생성.


- STFT 로 윈도우를 시간-주파수 이미지로 바꾼다 (config 로 n_fft/hop 조절).
- 진폭은 로그(dB) 스케일 → 작은 변화도 살림.
- (N, L) 윈도우 배열 → (N, 1, H, W) 4D 배열 (오토인코더/CNN 입력 형태).
- 정규화 통계는 '정상 train'에서만 fit (apply 는 어디든) — 누수 차단.

CWT 는 임펄스가 약할 때 검토 대상 (config 로 선택, 현재는 STFT만 구현).
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.signal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path


def to_spectrogram(window: np.ndarray, cfg: dict) -> np.ndarray:
    """윈도우 1개 (L,) → 스펙트로그램 (H, W) = (주파수 bins, 시간 frames).

    H = n_fft/2 + 1,  W = 시간 프레임 수 (윈도우 길이·hop 으로 결정).
    log_scale=True 면 dB 스케일(20·log10).
    """
    sp = cfg["spectrogram"]
    if sp.get("transform", "stft") != "stft":
        raise NotImplementedError("현재 STFT만 지원 (CWT 는 추후 검토)")

    n_fft, hop = sp["n_fft"], sp["hop_length"]
    x = np.asarray(window, dtype=np.float64).ravel()
    # boundary=None, padded=False → 가장자리 패딩 없이 정확한 프레임만
    _, _, Zxx = scipy.signal.stft(
        x, nperseg=n_fft, noverlap=n_fft - hop, boundary=None, padded=False
    )
    mag = np.abs(Zxx)
    if sp.get("log_scale", True):
        return (20.0 * np.log10(mag + 1e-10)).astype(np.float64)  # dB
    return mag.astype(np.float64)


def to_spectrograms(windows: np.ndarray, cfg: dict) -> np.ndarray:
    """윈도우 배열 (N, L) → (N, 1, H, W) (채널축 1 추가, AE/CNN 입력 형태)."""
    specs = [to_spectrogram(w, cfg) for w in windows]
    if not specs:
        return np.empty((0, 1, 0, 0), dtype=np.float64)
    return np.stack(specs)[:, None, :, :]


def fit_spec_scaler(specs: np.ndarray) -> dict:
    """정상 train 스펙트로그램에서 전역 정규화 통계를 계산한다(여기서만 fit)."""
    arr = np.asarray(specs, dtype=np.float64)
    mean, std = float(arr.mean()), float(arr.std())
    return {"mean": mean, "std": std if std != 0.0 else 1.0}


def apply_spec_scaler(specs: np.ndarray, scaler: dict) -> np.ndarray:
    """저장된 통계로 스펙트로그램 정규화: (x - mean) / std."""
    return (specs - scaler["mean"]) / scaler["std"]


def save_spec_scaler(scaler: dict, path: str | Path) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)


def load_spec_scaler(path: str | Path) -> dict:
    with open(resolve_path(path), "rb") as f:
        return pickle.load(f)


def _main() -> None:
    """T2.1 검증: (N, L) → (N, 1, H, W) 변환, 단계별 shape 출력."""
    sys.stdout.reconfigure(encoding="utf-8")
    from src.data_loader import load_normal_signals
    from src.preprocessing import make_windows_from_signals

    cfg = load_config()
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    sp = cfg["spectrogram"]

    print("=== T2.1 스펙트로그램 변환 ===")
    print(f"설정: {sp['transform'].upper()}, n_fft={sp['n_fft']}, hop={sp['hop_length']}, log_scale={sp['log_scale']}")

    normal = load_normal_signals(cfg)
    windows = make_windows_from_signals(normal, length, overlap)
    print(f"[1] 윈도우 배열 (N, L) → {windows.shape}")

    specs = to_spectrograms(windows, cfg)
    print(f"[2] 스펙트로그램 (N, 1, H, W) → {specs.shape}")
    print(f"    H={specs.shape[2]} (주파수 bins), W={specs.shape[3]} (시간 frames)")
    print(f"    값 범위(dB): {specs.min():.1f} ~ {specs.max():.1f}")

    # 정규화 통계는 정상에서만 fit
    scaler = fit_spec_scaler(specs)
    normed = apply_spec_scaler(specs, scaler)
    print(f"[3] 정규화(정상 train 통계) → mean={scaler['mean']:.2f}, std={scaler['std']:.2f}")
    print(f"    정규화 후 mean={normed.mean():.4f}, std={normed.std():.4f}")

    print("\nT2.1 OK — (N, L) → (N, 1, H, W) 변환·shape 출력됨.")


if __name__ == "__main__":
    _main()
