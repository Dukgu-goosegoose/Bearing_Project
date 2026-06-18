"""경로 B — 디노이징 오토인코더 학습 실행 스크립트 (T2.3).

흐름:
  정상 신호 → 윈도잉 → 스펙트로그램 → 정규화(정상 train 통계, 저장)
  → 디노이징 학습(노이즈 입력→깨끗한 원본 복원) → 모델/스케일러 저장 → 손실곡선 그림

⚠️ 학습에는 '정상'만 쓴다(누수 차단). 고장은 평가(T2.4 이후)에서만 등장.

실행:
    .venv\\Scripts\\python.exe train_autoencoder.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"  # 한글 라벨 안 깨지게
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder_detector import (
    build_autoencoder,
    fit_threshold_b,
    recon_errors,
    save_model,
    save_threshold_b,
    train_ae,
)
from src.data_loader import load_normal_signals
from src.preprocessing import make_windows_from_signals
from src.spectrogram import apply_spec_scaler, fit_spec_scaler, save_spec_scaler, to_spectrograms
from src.utils import get_device, load_config, resolve_path, set_seed


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    set_seed(cfg["seed"])
    device = get_device(cfg)
    models_dir = cfg["artifacts"]["models"]
    ae_path = f"{models_dir}/autoencoder.pth"
    spec_scaler_path = f"{models_dir}/spec_scaler.pkl"

    print("=== T2.3 디노이징 오토인코더 학습 (정상만) ===")
    print(f"장치: {device}")

    # 1) 정상 → 스펙트로그램
    normal = load_normal_signals(cfg)
    windows = make_windows_from_signals(normal, cfg["window"]["length"], cfg["window"]["overlap"])
    specs = to_spectrograms(windows, cfg)              # (N, 1, H, W)
    print(f"정상 스펙트로그램: {specs.shape}")

    # 2) 정규화 통계는 정상 train 에서만 fit → 저장
    scaler = fit_spec_scaler(specs)
    save_spec_scaler(scaler, spec_scaler_path)
    specs_n = apply_spec_scaler(specs, scaler)

    # 3) train/val 분할 (시드 고정 셔플, 90/10) — 손실 곡선 모니터링용
    rng = np.random.default_rng(cfg["seed"])
    idx = rng.permutation(len(specs_n))
    n_val = max(1, len(idx) // 10)
    val_i, train_i = idx[:n_val], idx[n_val:]
    train_specs, val_specs = specs_n[train_i], specs_n[val_i]
    print(f"train {len(train_specs)} / val {len(val_specs)}")

    # 4) 모델 생성 + 학습
    model = build_autoencoder(cfg)
    history = train_ae(model, train_specs, cfg, device=device, val_specs=val_specs)

    # 5) 저장
    save_model(model, ae_path)
    print(f"\n모델 저장: {ae_path}")
    print(f"정규화 저장: {spec_scaler_path}")

    # 5.5) T2.4 — 정상 재구성오차로 임계값 산정·저장 (정상에서만, 누수 차단)
    n_err = recon_errors(model, specs_n, device=device)
    thr = fit_threshold_b(n_err, cfg)
    save_threshold_b(thr, cfg)
    print(f"\n[T2.4] 정상 재구성오차 평균 {thr['fit_mean']:.4f} (±{thr['fit_std']:.4f}) "
          f"→ 임계값 {thr['threshold']:.4f} ({thr['rule']}) → thresholds.json[path_b]")

    # 6) 손실 곡선 그림
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(history["train"], label="train", color="#2980b9")
    if history["val"]:
        ax.plot(history["val"], label="val", color="#e74c3c")
    ax.set_xlabel("epoch")
    ax.set_ylabel("재구성 손실 (MSE)")
    ax.set_title("경로 B 오토인코더 학습 손실 (정상만, 디노이징)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = resolve_path("outputs/figures/ae_training_loss.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"손실곡선: {out.relative_to(_ROOT)}")

    drop = history["train"][0] - history["train"][-1]
    print(f"\nT2.3 OK — 손실 {history['train'][0]:.5f} → {history['train'][-1]:.5f} (감소 {drop:.5f}), 모델 저장됨.")


if __name__ == "__main__":
    main()
