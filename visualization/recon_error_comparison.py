"""[B] 경로 B 재구성오차 분포 — 정상 vs 고장 + 임계값.

학습된 오토인코더로 정상/고장 스펙트로그램의 재구성오차를 구해,
'정상은 낮고 / 고장은 높아 임계값을 넘는지'(=이상 탐지되는지) 눈으로 확인한다.

독립 실행 (먼저 train_autoencoder.py 로 모델·임계값을 만들어둬야 함):
    .venv\\Scripts\\python.exe -m visualization.recon_error_comparison
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from visualization._common import plt, save_fig


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.autoencoder_detector import load_model, recon_errors
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.preprocessing import make_windows, make_windows_from_signals
    from src.spectrogram import apply_spec_scaler, load_spec_scaler, to_spectrograms
    from src.utils import get_device, load_config, resolve_path

    cfg = load_config()
    device = get_device(cfg)
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    models_dir = cfg["artifacts"]["models"]

    model = load_model(cfg, f"{models_dir}/autoencoder.pth", device)
    scaler = load_spec_scaler(f"{models_dir}/spec_scaler.pkl")
    thr = json.loads(resolve_path(cfg["artifacts"]["thresholds"]).read_text(encoding="utf-8"))["path_b"]
    T = thr["threshold"]

    # 정상 재구성오차
    normal = load_normal_signals(cfg)
    nspecs = apply_spec_scaler(to_spectrograms(make_windows_from_signals(normal, length, overlap), cfg), scaler)
    n_err = recon_errors(model, nspecs, device)

    # 고장 재구성오차 (파일별 처리, 위치 라벨 추적 — 메모리 절약)
    fault = load_fault_signals(cfg)
    f_err_by_loc: dict[str, list] = {"IR": [], "OR": [], "B": []}
    all_f = []
    for v in fault.values():
        w = make_windows(v["signal"], length, overlap)
        if len(w) == 0:
            continue
        e = recon_errors(model, apply_spec_scaler(to_spectrograms(w, cfg), scaler), device)
        loc = v["location"]
        if loc in f_err_by_loc:
            f_err_by_loc[loc].append(e)
        all_f.append(e)
    f_err = np.concatenate(all_f)

    # 탐지율
    fpr = float((n_err > T).mean())
    det = float((f_err > T).mean())
    print("=== 경로 B 재구성오차 — 정상 vs 고장 ===")
    print(f"임계값 T = {T:.4f}")
    print(f"[정상] 평균 {n_err.mean():.4f}  → 이상으로 잡힌 비율(오탐) {100*fpr:.1f}%")
    print(f"[고장] 평균 {f_err.mean():.4f}  → 이상으로 잡힌 비율(탐지) {100*det:.1f}%")
    for loc, lst in f_err_by_loc.items():
        if lst:
            e = np.concatenate(lst)
            print(f"   - {loc}: 평균 {e.mean():.4f}, 탐지율 {100*float((e>T).mean()):.1f}%")

    # 그림: 재구성오차 분포 (정상 vs 고장) + 임계값선
    fig, ax = plt.subplots(figsize=(11, 5))
    hi = np.percentile(np.concatenate([n_err, f_err]), 99)
    bins = np.linspace(0, hi, 80)
    ax.hist(n_err, bins=bins, alpha=0.6, color="#2980b9", density=True, label=f"정상 (평균 {n_err.mean():.3f})")
    ax.hist(f_err, bins=bins, alpha=0.6, color="#e74c3c", density=True, label=f"고장 (평균 {f_err.mean():.3f})")
    ax.axvline(T, color="black", ls="--", lw=1.5, label=f"임계값 {T:.3f} (평균+3σ)")
    ax.set_xlabel("재구성오차 (MSE)")
    ax.set_ylabel("밀도")
    ax.set_title(f"경로 B 재구성오차 — 정상(파랑) vs 고장(빨강)\n"
                 f"정상 오탐 {100*fpr:.1f}%  /  고장 탐지 {100*det:.1f}%", fontsize=11, fontweight="bold")
    ax.legend()
    fig.tight_layout()
    out = save_fig(fig, "recon_error_comparison.png")
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
