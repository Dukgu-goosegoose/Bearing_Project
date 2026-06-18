"""융합 + 디바운스 (T3.1) — 경로 A·B 점수를 합쳐 최종 판정.

A·B 점수는 둘 다 '값/임계값' 비율이라 1 초과면 이상(정규화돼 비교 가능).
  - OR 규칙      : 둘 중 하나라도 1 초과면 이상. 융합점수 = max(A, B)
  - weighted 규칙: 가중합(wa·A + wb·B)이 1 초과면 이상

디바운스: 한 윈도우 튄다고 알람 내지 않고, 연속 N개 윈도우가 이상일 때만 알람(오탐 억제).

config fusion: rule / weights / debounce_n
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import FusionResult
from src.utils import load_config


def fuse_scores(score_a: float, score_b: float, cfg: dict) -> tuple[float, bool]:
    """A·B 점수를 융합한다. 반환: (융합점수, 이상여부)."""
    f = cfg["fusion"]
    rule = f.get("rule", "or")
    if rule == "or":
        fused = max(float(score_a), float(score_b))
        is_anom = (score_a > 1.0) or (score_b > 1.0)
    elif rule == "weighted":
        wa, wb = f["weights"]["a"], f["weights"]["b"]
        fused = wa * float(score_a) + wb * float(score_b)
        is_anom = fused > 1.0
    else:
        raise ValueError(f"지원하지 않는 융합 규칙: {rule}")
    return fused, bool(is_anom)


class Debouncer:
    """연속 N개 윈도우가 이상일 때만 알람을 켠다(오탐 억제)."""

    def __init__(self, n: int):
        self.n = n
        self.count = 0

    def update(self, is_anomaly: bool) -> bool:
        """이번 윈도우 판정을 넣고 알람 여부를 반환. 정상이면 카운트 리셋."""
        self.count = self.count + 1 if is_anomaly else 0
        return self.count >= self.n

    def reset(self) -> None:
        self.count = 0


def fuse_stream(scores_a, scores_b, cfg: dict) -> list[FusionResult]:
    """윈도우 점수 시퀀스를 순서대로 융합+디바운스해 FusionResult 리스트로."""
    deb = Debouncer(cfg["fusion"]["debounce_n"])
    out: list[FusionResult] = []
    for sa, sb in zip(scores_a, scores_b):
        fused, is_anom = fuse_scores(sa, sb, cfg)
        alarm = deb.update(is_anom)
        out.append(FusionResult(fused_score=float(fused), is_anomaly=is_anom, alarm=alarm))
    return out


def _demo_debounce(cfg: dict) -> None:
    """디바운스 동작을 합성 시퀀스로 보여준다(개념 확인)."""
    n = cfg["fusion"]["debounce_n"]
    # 이상(1) 패턴: 단발 튐 vs 지속 이상
    seq = [0, 1, 0, 0, 1, 1, 1, 1, 0, 1]
    deb = Debouncer(n)
    alarms = [int(deb.update(bool(s))) for s in seq]
    print(f"  디바운스 N={n}")
    print(f"  이상판정 : {seq}")
    print(f"  알람     : {alarms}   (연속 {n}개부터 알람 ON)")


def _demo_real(cfg: dict) -> None:
    """실데이터: 정상 vs 고장 신호에 대해 A·B 융합 + 디바운스 결과."""
    import numpy as np

    from src.autoencoder_detector import anomaly_score_b, load_model, recon_errors
    from src.baseline_statistical import anomaly_score, extract_features, load_thresholds
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.preprocessing import make_windows
    from src.spectrogram import apply_spec_scaler, load_spec_scaler, to_spectrograms
    from src.utils import get_device

    device = get_device(cfg)
    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    feats = cfg["stats"]["features"]
    models_dir = cfg["artifacts"]["models"]

    thr = load_thresholds(cfg)
    thr_a, T_b = thr["path_a_statistical"], thr["path_b"]["threshold"]
    model = load_model(cfg, f"{models_dir}/autoencoder.pth", device)
    scaler = load_spec_scaler(f"{models_dir}/spec_scaler.pkl")

    def scores_for(sig):
        w = make_windows(sig, length, overlap)
        sa = np.array([anomaly_score(extract_features(x, feats), thr_a) for x in w])
        err = recon_errors(model, apply_spec_scaler(to_spectrograms(w, cfg), scaler), device)
        sb = anomaly_score_b(err, T_b)
        return sa, sb

    normal = load_normal_signals(cfg)
    fault = load_fault_signals(cfg)
    cases = [("정상 normal_0", next(iter(normal.values())))]
    ir = next((v["signal"] for v in fault.values() if v["location"] == "IR"), None)
    if ir is not None:
        cases.append(("고장 IR", ir))

    for name, sig in cases:
        sa, sb = scores_for(sig)
        results = fuse_stream(sa, sb, cfg)
        anom = sum(r.is_anomaly for r in results)
        alarms = sum(r.alarm for r in results)
        print(f"  [{name}] 윈도우 {len(results)}개 | A>1:{int((sa>1).sum())} B>1:{int((sb>1).sum())} "
              f"| 융합이상:{anom} | 알람:{alarms}")


def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    print("=== T3.1 융합 + 디바운스 ===")
    print(f"규칙: {cfg['fusion']['rule']}, 디바운스 N={cfg['fusion']['debounce_n']}")
    print("\n[1] 디바운스 개념 (합성 시퀀스)")
    _demo_debounce(cfg)
    print("\n[2] 실데이터 융합 (정상 vs 고장)")
    _demo_real(cfg)
    print("\nT3.1 OK — 융합점수가 A·B 반영, 디바운스로 연속 이상만 알람.")


if __name__ == "__main__":
    _main()
