"""[C②] 경로 C CNN 평가 — 혼동행렬 + 위치별 정확도 + 한계 (T7.5).

학습된 CNN(fault_classifier.pth)으로 test 녹음을 직접 평가해:
  - 혼동행렬(어떤 위치를 무엇으로 헷갈리는지)
  - 위치별 재현율(맞힌 비율) — 볼 약점 강조
  - "알려진 유형만 분류 가능" 한계 명시
를 그림 1장으로 저장한다. 숫자는 손으로 적는 게 아니라 실제 모델에서 계산된다.

독립 실행 (먼저 train_classifier.py 로 모델·데이터셋을 만들어둬야 함):
    .venv\\Scripts\\python.exe -m visualization.pathc_evaluation
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from visualization._common import plt, save_fig

_LOC_KR = {"IR": "내륜", "OR": "외륜", "B": "볼"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    from src.cnn_fault_classifier import evaluate, load_model, make_loaders
    from src.schemas import FAULT_CLASSES
    from src.utils import get_device, load_config, resolve_path

    cfg = load_config()
    device = get_device(cfg)
    model_path = resolve_path(cfg["artifacts"]["models"]) / "fault_classifier.pth"
    model = load_model(model_path, num_classes=len(FAULT_CLASSES), device=device)

    loaders = make_loaders(cfg)
    m = evaluate(model, loaders["test"], device)         # 실제 모델로 직접 평가
    cm = np.array(m["confusion_matrix"], dtype=float)     # 행=정답, 열=예측
    acc = float(m["accuracy"])
    labels = m["labels"]                                  # ['IR','OR','B']
    disp = [f"{l}\n({_LOC_KR[l]})" for l in labels]
    recall = [float(m["per_class_recall"][l]) for l in labels]

    # ---- 콘솔 리포트 ----
    print("=== 경로 C② CNN 평가 (test 녹음, 실제 모델 직접 계산) ===")
    print(f"전체 정확도: {acc:.3f}")
    print("혼동행렬(행=정답, 열=예측):")
    for i, l in enumerate(labels):
        print(f"  {l}: {cm[i].astype(int).tolist()}   재현율 {recall[i]:.3f}")

    # ---- 그림 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # (1) 혼동행렬: 색=행 정규화(재현율), 셀=개수+%
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum > 0)
    im = ax1.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(disp)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(disp)
    ax1.set_xlabel("CNN 예측")
    ax1.set_ylabel("실제 정답")
    ax1.set_title(f"혼동행렬 (전체 정확도 {acc:.1%})", fontweight="bold")
    for i in range(len(labels)):
        for j in range(len(labels)):
            cnt, pct = int(cm[i, j]), cm_norm[i, j]
            ax1.text(j, i, f"{cnt}\n{pct:.0%}", ha="center", va="center",
                     color="white" if pct > 0.5 else "#333", fontsize=11)
    fig.colorbar(im, ax=ax1, fraction=0.046, label="행 기준 비율(재현율)")

    # (2) 위치별 재현율 막대 — 볼 빨강 강조 + 찍기선
    colors = ["#e74c3c" if l == "B" else "#2980b9" for l in labels]
    bars = ax2.bar(disp, recall, color=colors)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("재현율 (그 위치를 맞힌 비율)")
    ax2.set_title("위치별 분류 정확도 — 볼(빨강)이 약점", fontweight="bold")
    ax2.axhline(1 / 3, color="gray", ls=":", lw=1.2, label="무작위 찍기(33%)")
    for b, r in zip(bars, recall):
        ax2.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.0%}",
                 ha="center", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    note = ("※ 한계: CWRU에 있는 알려진 유형(내륜·외륜·볼)만 분류 가능 — 처음 보는 결함은 못 맞힘.\n"
            "   따라서 CNN은 '확정'이 아닌 '참고 보조'이며, 물리 진단과 교차검증으로 보완한다.")
    fig.suptitle("[경로 C②] CNN 고장위치 분류 — 정확도·혼동행렬·한계",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, 0.015, note, ha="center", fontsize=9.5, color="#444",
             bbox=dict(boxstyle="round", fc="#f7f7f7", ec="#cccccc"))
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))

    out = save_fig(fig, "pathc_cnn_evaluation.png")
    print(f"\n저장: {out}")
    print('한계 명시 완료 — "알려진 유형만 분류 가능"(참고 보조, 교차검증으로 보완).')


if __name__ == "__main__":
    main()
