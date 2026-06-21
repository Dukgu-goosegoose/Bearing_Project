"""평가 — AUC·pAUC·정상 FPR (탐지 A·B) + 혼동행렬·정확도 (진단 CNN) + 교차검증 일치율.

⚠️ 고장 데이터는 '여기서만' 등장한다(학습·임계값 설정엔 절대 사용 금지).
  - 비지도(A·B 융합): AUC·pAUC·정상 오탐률(FPR).  ← 라벨 없는 이상탐지의 표준 지표
  - 지도(C 분류)     : 정확도·혼동행렬.
  - 교차검증         : 물리(①)↔CNN(②) 일치율.
결과는 outputs/reports/eval_report.json 에 저장.

지표 읽는 법(비전공자용):
  AUC  = 정상과 고장을 점수로 얼마나 잘 가르나(0.5=찍기, 1.0=완벽).
  pAUC = "오탐이 적은 구간(FPR≤0.1)만 본" AUC. 현장에선 헛알람이 적은 게 중요해서 본다.
  FPR  = 정상인데 이상이라 잘못 외친 비율(낮을수록 좋음).

실행:
    .venv\\Scripts\\python.exe -m src.evaluator
    .venv\\Scripts\\python.exe -m src.evaluator selftest   # 지표 계산 로직만 합성으로 확인
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path


# =============================================================
#  1. 탐지 평가 (경로 A·B 융합) — AUC · pAUC · 정상 FPR
# =============================================================
def evaluate_detection(cfg: dict, art, include_48k_downsampled: bool = False) -> dict:
    """정상 vs 고장 윈도우의 융합 점수로 AUC·pAUC·FPR을 계산한다.

    윈도우 단위로 점수를 모은다(정상 윈도우 라벨 0, 고장 윈도우 라벨 1).
    A 단독·B 단독 AUC도 같이 내서 '융합이 나은가'를 비교(베이스라인 대조).
    """
    from sklearn.metrics import roc_auc_score

    from src.cnn_fault_classifier import collect_fault_recordings
    from src.data_loader import load_normal_signals
    from src.fusion import fuse_scores
    from src.inference import ab_scores

    sa_all: list[float] = []
    sb_all: list[float] = []
    sf_all: list[float] = []
    flags_all: list[bool] = []
    labels: list[int] = []

    def _accumulate(signal, label: int) -> None:
        sa, sb = ab_scores(signal, cfg, art)
        for a, b in zip(sa, sb):
            fused, is_anom = fuse_scores(float(a), float(b), cfg)
            sa_all.append(float(a))
            sb_all.append(float(b))
            sf_all.append(float(fused))
            flags_all.append(bool(is_anom))
            labels.append(label)

    normal = load_normal_signals(cfg)
    for sig in normal.values():
        _accumulate(sig, 0)

    recs = collect_fault_recordings(cfg, include_48k_downsampled)
    for rec in recs:
        _accumulate(rec.signal, 1)

    y = np.array(labels)
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return {"error": "정상·고장 둘 다 있어야 AUC 계산 가능", "n_pos": n_pos, "n_neg": n_neg}

    sf = np.array(sf_all)
    flags = np.array(flags_all)
    fpr = float(flags[y == 0].mean())

    return {
        "n_windows": int(len(y)),
        "n_normal_windows": n_neg,
        "n_fault_windows": n_pos,
        "auc_fused": round(float(roc_auc_score(y, sf)), 4),
        "pauc_fused_fpr10": round(float(roc_auc_score(y, sf, max_fpr=0.1)), 4),
        "auc_path_a_only": round(float(roc_auc_score(y, np.array(sa_all))), 4),
        "auc_path_b_only": round(float(roc_auc_score(y, np.array(sb_all))), 4),
        "normal_fpr": round(fpr, 4),
    }


# =============================================================
#  2. 진단 평가 (경로 C CNN) — 정확도 · 혼동행렬 · 교차검증 일치율
# =============================================================
def evaluate_classification(cfg: dict, model=None, split: str | None = "test") -> dict:
    """고장 녹음에 물리·CNN을 적용해 정확도·혼동행렬·교차검증 일치율을 낸다.

    split="test" 면 학습 때 안 본 test 녹음만으로 공정 평가(권장).
    manifest 가 없으면 자동으로 전체 고장으로 폴백한다.
    학습된 CNN(fault_classifier.pth)이 없으면 분류 평가는 건너뛴다.
    """
    from sklearn.metrics import confusion_matrix

    from src.physics_diagnosis import crosscheck_report

    model_path = resolve_path(cfg["artifacts"]["models"]) / "fault_classifier.pth"
    if model is None and not model_path.exists():
        return {"skipped": "fault_classifier.pth 없음 → CNN 학습 후 평가 가능"}

    classes = cfg["cnn"]["classes"]

    used_split = split
    try:
        rep = crosscheck_report(cfg, model=model, split=split)
    except (FileNotFoundError, KeyError):
        used_split = None
        rep = crosscheck_report(cfg, model=model, split=None)

    truth = [r["truth"] for r in rep["rows"] if r["cnn"] is not None]
    pred = [r["cnn"] for r in rep["rows"] if r["cnn"] is not None]
    cm = confusion_matrix(truth, pred, labels=classes).tolist() if truth else []

    per = rep["per_location"]
    cnn_hit = sum(per[c]["cnn_hit"] for c in classes)
    phys_hit = sum(per[c]["physics_hit"] for c in classes)
    n = rep["n_recordings"]

    return {
        "split_used": used_split or "all_fault(폴백)",
        "n_recordings": n,
        "cnn_accuracy": round(cnn_hit / n, 4) if n else 0.0,
        "physics_accuracy": round(phys_hit / n, 4) if n else 0.0,
        "agreement_rate": rep["agreement_rate"],
        "verdict_count": rep["verdict_count"],
        "confusion_matrix": {"labels": classes, "matrix": cm},
        "per_location": per,
    }


# =============================================================
#  3. 전체 실행 — 두 평가를 묶어 outputs/reports/ 에 저장
# =============================================================
def run_evaluation(cfg: dict, include_48k_downsampled: bool = False) -> dict:
    from src.inference import load_artifacts

    report: dict = {}

    try:
        art = load_artifacts(cfg)
        report["detection"] = evaluate_detection(cfg, art, include_48k_downsampled)
    except FileNotFoundError as e:
        report["detection"] = {"skipped": f"학습 산출물 없음 → {e}"}

    report["classification"] = evaluate_classification(cfg, split="test")

    out_dir = resolve_path(cfg["artifacts"]["outputs"]) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "eval_report.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    report["_saved_to"] = str(out_path.relative_to(_ROOT))
    return report


# =============================================================
#  4. 지표 계산 로직 자체검증 (합성 점수 — 데이터·모델 불필요)
# =============================================================
def _selftest() -> None:
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    normal = rng.normal(0.4, 0.15, 500)
    fault = rng.normal(1.4, 0.30, 500)
    y = np.r_[np.zeros(500), np.ones(500)]
    s = np.r_[normal, fault]
    print("=== 지표 계산 자체검증(합성) ===")
    print(f"  AUC          = {roc_auc_score(y, s):.4f}  (1.0에 가까울수록 잘 가름)")
    print(f"  pAUC(FPR<=.1)= {roc_auc_score(y, s, max_fpr=0.1):.4f}")
    fpr = float((normal > 1.0).mean())
    print(f"  정상 FPR(임계1.0)= {fpr:.4f}  (낮을수록 좋음)")
    print("  → 로직 OK. 실제 평가는 인자 없이 실행.")


def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        _selftest()
        return

    cfg = load_config()
    print("=== 평가 (탐지 AUC/pAUC/FPR + 진단 혼동행렬 + 교차검증) ===")
    rep = run_evaluation(cfg)

    d = rep.get("detection", {})
    print("\n[탐지 — 경로 A·B 융합]")
    if "skipped" in d:
        print(f"  건너뜀: {d['skipped']}")
    elif "error" in d:
        print(f"  오류: {d['error']}")
    else:
        print(f"  윈도우 {d['n_windows']}개 (정상 {d['n_normal_windows']} / 고장 {d['n_fault_windows']})")
        print(f"  AUC(융합)      = {d['auc_fused']}")
        print(f"  pAUC(FPR<=.1)  = {d['pauc_fused_fpr10']}")
        print(f"  AUC(A 단독)    = {d['auc_path_a_only']}   AUC(B 단독) = {d['auc_path_b_only']}")
        print(f"  정상 FPR       = {d['normal_fpr']}  (낮을수록 좋음)")

    c = rep.get("classification", {})
    print("\n[진단 — 경로 C 물리 vs CNN]")
    if "skipped" in c:
        print(f"  건너뜀: {c['skipped']}")
    else:
        print(f"  평가 대상: {c['split_used']} | 녹음 {c['n_recordings']}개")
        print(f"  CNN 정확도   = {c['cnn_accuracy']}   물리 정확도 = {c['physics_accuracy']}")
        print(f"  교차검증 일치율 = {c['agreement_rate']}  ({c['verdict_count']})")
        cm = c["confusion_matrix"]
        print(f"  혼동행렬 (행=진실, 열=예측) labels={cm['labels']}")
        for lbl, row in zip(cm["labels"], cm["matrix"]):
            print(f"    {lbl}: {row}")
        print("  위치별:")
        for loc, s in c["per_location"].items():
            print(f"    {loc}: 물리적중 {s['physics_hit']}/{s['n']} · "
                  f"CNN적중 {s['cnn_hit']}/{s['n']} · 일치 {s['agree']}/{s['n']}")

    print(f"\n저장: {rep.get('_saved_to')}")
    print("평가 완료 — AUC/pAUC↑·FPR↓ 면 탐지 양호, 교차검증 일치율↑ 면 진단 신뢰도 높음.")


if __name__ == "__main__":
    _main()