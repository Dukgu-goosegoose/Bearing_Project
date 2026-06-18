"""[4] 경로 A 특징 분포 — 정상 vs 고장 (전체 데이터).

4개 특징(첨도·크레스트팩터·RMS·peak)마다 히스토그램(정상 파랑/고장 빨강)과
임계값선(점선). 사용한 데이터(정상 전체 vs 고장 전체)를 제목에 표기.

독립 실행:
    .venv\\Scripts\\python.exe -m visualization.feature_comparison
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from visualization._common import plt, save_fig


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.baseline_statistical import extract_features_batch, load_thresholds
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.preprocessing import make_windows_from_signals
    from src.utils import load_config

    cfg = load_config()
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    feats = cfg["stats"]["features"]

    normal = load_normal_signals(cfg)
    fault = load_fault_signals(cfg)
    nwins = make_windows_from_signals(normal, length, overlap)
    fwins = make_windows_from_signals({k: v["signal"] for k, v in fault.items()}, length, overlap)

    nfb = extract_features_batch(nwins, feats)
    ffb = extract_features_batch(fwins, feats)
    thr = load_thresholds(cfg)["path_a_statistical"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, f in zip(axes.ravel(), feats):
        ax.hist(nfb[f], bins=60, alpha=0.6, color="#2980b9", density=True, label="정상")
        ax.hist(ffb[f], bins=60, alpha=0.6, color="#e74c3c", density=True, label="고장")
        ax.axvline(thr[f]["threshold"], color="black", ls="--", lw=1.3,
                   label=f"임계값 {thr[f]['threshold']:.3f}")
        ax.set_title(f"{f}   (정상평균 {nfb[f].mean():.3f}  vs  고장평균 {ffb[f].mean():.3f})", fontsize=10)
        ax.set_xlabel("특징값")
        ax.set_ylabel("밀도")
        ax.legend(fontsize=8)

    data_note = (f"정상: normal 0~3 ({len(nwins)}윈도우, 12kHz)   |   "
                 f"고장: {len(fault)}파일 전체 ({len(fwins)}윈도우)  — 임계값은 정상에서만 산정")
    fig.suptitle("[4] 경로 A 특징 분포 — 정상(파랑) vs 고장(빨강) + 임계값(점선)\n" + data_note,
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = save_fig(fig, "feature_comparison.png")
    print(f"저장: {out}")
    print(f"  {data_note}")


if __name__ == "__main__":
    main()
