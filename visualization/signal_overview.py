"""[2] 신호 전체 길이 비교 — 정상 vs 고장(DE IR/OR/B, FE IR).

가로=시간(초), 세로=진폭(공통 스케일). 어떤 신호인지 각 행 제목에 자세히 표기.

독립 실행:
    .venv\\Scripts\\python.exe -m visualization.signal_overview
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from visualization._common import get_example_signals, plt, save_fig


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.utils import load_config

    cfg = load_config()
    exs = get_example_signals(cfg)
    # 대표 1개씩: 정상_0 + DE IR/OR/B + FE IR
    rows = [exs[0]] + [e for e in exs if e["group"].startswith("fault")]

    colors = {"normal": "#2c3e50", "fault_DE": "#e74c3c", "fault_FE": "#8e44ad"}
    ymax = max(float(np.abs(e["signal"]).max()) for e in rows) * 1.05

    fig, axes = plt.subplots(len(rows), 1, figsize=(12, 2.0 * len(rows)))
    for ax, e in zip(axes, rows):
        t = np.arange(len(e["signal"])) / e["sr"]
        ax.plot(t, e["signal"], color=colors.get(e["group"], "#333"), lw=0.4)
        ax.set_ylim(-ymax, ymax)
        ax.set_title(f"{e['title']}   —   {len(e['signal']):,}점 (약 {len(e['signal'])/e['sr']:.2f}초)",
                     fontsize=9)
        ax.set_ylabel("진폭")
    axes[-1].set_xlabel("시간 (초)")
    fig.suptitle("[2] 정상 vs 고장 신호 전체 길이 (공통 스케일, 12kHz)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))

    out = save_fig(fig, "signal_overview.png")
    print(f"저장: {out}")
    for e in rows:
        print(f"  - {e['title']}")


if __name__ == "__main__":
    main()
