"""실험: 일반 오토인코더 vs 디노이징 오토인코더 — train/val 손실 비교.

같은 초기 가중치(시드 고정)·같은 정상 데이터·같은 train/val 분할로,
노이즈만 다르게(0.0 vs config) 학습해 손실 곡선을 나란히 본다.

실행:
    .venv\\Scripts\\python.exe compare_ae_denoising.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.autoencoder_detector import build_autoencoder, train_ae
from src.data_loader import load_normal_signals
from src.preprocessing import make_windows_from_signals
from src.spectrogram import apply_spec_scaler, fit_spec_scaler, to_spectrograms
from src.utils import get_device, load_config, resolve_path, set_seed


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    device = get_device(cfg)
    seed = cfg["seed"]

    # 정상 → 스펙트로그램 → 정규화 → 같은 train/val 분할
    set_seed(seed)
    normal = load_normal_signals(cfg)
    windows = make_windows_from_signals(normal, cfg["window"]["length"], cfg["window"]["overlap"])
    specs = apply_spec_scaler(to_spectrograms(windows, cfg), fit_spec_scaler(to_spectrograms(windows, cfg)))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(specs))
    n_val = max(1, len(idx) // 10)
    val_specs, train_specs = specs[idx[:n_val]], specs[idx[n_val:]]

    runs = {"일반 AE (noise=0.0)": 0.0, f"디노이징 AE (noise={cfg['ae']['noise_std']})": None}
    histories = {}
    for name, noise in runs.items():
        set_seed(seed)                       # 같은 초기 가중치
        model = build_autoencoder(cfg)
        print(f"\n--- {name} ---")
        histories[name] = train_ae(model, train_specs, cfg, device=device,
                                   val_specs=val_specs, noise_std=noise)

    # 그림: 좌(일반) 우(디노이징), 같은 y범위
    ymax = max(max(h["train"][1:] + h["val"][1:]) for h in histories.values()) * 1.1
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, (name, h) in zip(axes, histories.items()):
        ax.plot(h["train"], label="train", color="#2980b9")
        ax.plot(h["val"], label="val", color="#e74c3c")
        gap = h["val"][-1] - h["train"][-1]
        ax.set_title(f"{name}\ntrain={h['train'][-1]:.4f}  val={h['val'][-1]:.4f}  (gap {gap:+.4f})", fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylim(0, ymax)
        ax.grid(alpha=0.3)
        ax.legend()
    axes[0].set_ylabel("재구성 손실 (MSE)")
    fig.suptitle("일반 AE vs 디노이징 AE — train/val 손실 비교 (같은 초기값·데이터)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = resolve_path("outputs/figures/ae_denoising_compare.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)

    print("\n=== 최종 비교 ===")
    for name, h in histories.items():
        print(f"  {name}: train={h['train'][-1]:.4f}, val={h['val'][-1]:.4f}, "
              f"train-val gap={h['val'][-1]-h['train'][-1]:+.4f}")
    print(f"\n저장: {out.relative_to(_ROOT)}")


if __name__ == "__main__":
    main()
