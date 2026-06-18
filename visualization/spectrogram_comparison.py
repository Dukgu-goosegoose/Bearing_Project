"""[3] 스펙트로그램 비교 — 정상 vs 고장(DE IR/OR/B, FE IR).

각 신호의 '첫 윈도우' 스펙트로그램(가로=시간, 세로=주파수 kHz, 색=세기 dB).
색 스케일 통일. 어떤 신호인지 각 칸 제목에 자세히 표기.

독립 실행:
    .venv\\Scripts\\python.exe -m visualization.spectrogram_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization._common import get_example_signals, plt, save_fig


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.preprocessing import make_windows
    from src.spectrogram import to_spectrogram
    from src.utils import load_config

    cfg = load_config()
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    nyq_khz = cfg["data"]["target_rate"] / 2 / 1000.0

    exs = get_example_signals(cfg)
    rows = [exs[0]] + [e for e in exs if e["group"].startswith("fault")]  # 정상_0 + 고장 4종

    specs = []
    for e in rows:
        w = make_windows(e["signal"], length, overlap)
        if len(w) == 0:
            continue
        specs.append((e["title"], to_spectrogram(w[0], cfg)))

    vmin = min(s.min() for _, s in specs)
    vmax = max(s.max() for _, s in specs)

    n = len(specs)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 3.4 * nrow))
    axes = axes.ravel()
    for ax, (title, spec) in zip(axes, specs):
        im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                       vmin=vmin, vmax=vmax, extent=(0, spec.shape[1], 0, nyq_khz))
        # 제목이 길어서 두 줄로
        short = title.split("|")[0].strip()
        ax.set_title(short, fontsize=9)
        ax.set_xlabel("시간 프레임", fontsize=8)
        ax.set_ylabel("주파수 (kHz)", fontsize=8)
        fig.colorbar(im, ax=ax, label="dB")
    for ax in axes[len(specs):]:
        ax.axis("off")
    fig.suptitle("[3] 스펙트로그램 — 정상 vs 고장 (색=세기 dB, 12kHz)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out = save_fig(fig, "spectrogram_comparison.png")
    print(f"저장: {out}")
    for title, _ in specs:
        print(f"  - {title}")


if __name__ == "__main__":
    main()
