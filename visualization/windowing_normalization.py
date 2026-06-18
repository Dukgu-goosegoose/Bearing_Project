"""[1] 윈도잉 + 정규화 파형 — 예시 신호별로 1장씩.

각 신호(정상 0~3, DE 고장 IR/OR/B, FE 고장 IR)에 대해:
  - 위: 앞 4개 윈도우(겹침 구조)
  - 아래: 윈도우 1개 원본 vs 정규화 후
신호가 무엇인지 제목에 자세히 표기한다.

독립 실행:
    .venv\\Scripts\\python.exe -m visualization.windowing_normalization
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from visualization._common import get_example_signals, plt, save_fig


def _plot_one(ex: dict, cfg: dict, scaler: dict) -> str:
    from src.preprocessing import apply_scaler, make_windows

    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    step = int(round(length * (1.0 - overlap)))
    sig = ex["signal"]
    windows = make_windows(sig, length, overlap)
    if len(windows) == 0:
        return ""
    n_win = windows.shape[0]

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    # [1] 앞 4개 윈도우 — 겹침 구조
    span = min(step * 3 + length, len(sig))
    axes[0].plot(np.arange(span), sig[:span], color="#333", lw=0.7)
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    for i in range(4):
        s = i * step
        if s + length > span:
            break
        axes[0].axvspan(s, s + length, color=colors[i], alpha=0.18)
        axes[0].axvline(s, color=colors[i], ls="--", lw=0.9)
    axes[0].set_xlim(0, span)
    axes[0].set_title(f"[1] 윈도잉 (앞 4개) — 길이 {length}, 보폭 {step}(겹침 {overlap})   "
                      f"※ 전체 {n_win}개 윈도우")
    axes[0].set_xlabel("샘플 인덱스")
    axes[0].set_ylabel("진폭")

    # [2] 정규화 효과 — 윈도우 1개
    w = windows[0]
    wn = apply_scaler(w, scaler)
    axes[1].plot(w, color="#e74c3c", lw=0.6, label=f"원본 (std={w.std():.4f})")
    axes[1].plot(wn, color="#2980b9", lw=0.6, alpha=0.8, label=f"정규화 후 (std={wn.std():.4f})")
    axes[1].set_title("[2] 정규화 효과: 윈도우 1개 — 모양 그대로, 스케일만 표준화")
    axes[1].set_xlabel(f"샘플 인덱스 (0~{length - 1})")
    axes[1].set_ylabel("진폭")
    axes[1].legend(loc="upper right", fontsize=9)

    fig.suptitle(ex["title"], fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = save_fig(fig, f"windowing_{ex['group']}_{ex['fname']}.png")
    return str(out.name)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.preprocessing import load_scaler
    from src.utils import load_config

    cfg = load_config()
    scaler = load_scaler(cfg["artifacts"]["scaler"])
    print("=== [1] 윈도잉 + 정규화 (예시 신호별) ===")
    for ex in get_example_signals(cfg):
        name = _plot_one(ex, cfg, scaler)
        if name:
            print(f"  저장: outputs/figures/{name}   <- {ex['title']}")


if __name__ == "__main__":
    main()
