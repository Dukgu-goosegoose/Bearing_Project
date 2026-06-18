"""경로 A — 통계 임펄스 탐지기 (T1.1 특징 추출).

윈도우에서 RMS·peak·crest factor(크레스트팩터)·kurtosis(첨도)를 뽑는다.
베어링 결함은 짧고 강한 '충격(임펄스)'을 만드는데, 이 충격성을 숫자로 잡는 게
crest factor 와 kurtosis 다 (정상보다 고장에서 커진다).

특징은 윈도잉된 신호(raw)에서 계산한다. 정규화(scaler)는 경로 B용이며,
crest/kurtosis 는 스케일 무관이라 정규화 여부에 영향받지 않는다.

이어지는 T1.2 에서 정상 분포 기반 임계값을 산정한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path


def extract_features(window: np.ndarray, features: list[str] | None = None) -> dict[str, float]:
    """윈도우 1개에서 통계 특징을 뽑아 딕셔너리로 반환한다.

    features 를 주면 그 목록만, 안 주면 4개 모두 반환.
    반환 예: {'rms':.., 'peak':.., 'crest_factor':.., 'kurtosis':..}
    """
    x = np.asarray(window, dtype=np.float64).ravel()

    rms = float(np.sqrt(np.mean(x**2)))            # 제곱평균제곱근 — 진동 에너지 크기
    peak = float(np.max(np.abs(x)))                # 최대 절댓값 — 가장 큰 충격
    crest_factor = float(peak / rms) if rms > 0 else 0.0   # peak/RMS — 충격성 정도
    # 첨도(Pearson, 정규분포=3). 뾰족한 충격이 많을수록 커짐 → 베어링 결함 핵심 지표
    kurtosis = float(scipy.stats.kurtosis(x, fisher=False))

    all_feats = {"rms": rms, "peak": peak, "crest_factor": crest_factor, "kurtosis": kurtosis}
    if features is None:
        return all_feats
    return {k: all_feats[k] for k in features if k in all_feats}


def extract_features_batch(
    windows: np.ndarray, features: list[str] | None = None
) -> dict[str, np.ndarray]:
    """여러 윈도우 (N, L) 에 대해 특징별 배열을 반환한다 (T1.2 임계값 산정용).

    반환 예: {'rms': array([..N..]), 'kurtosis': array([..N..]), ...}
    """
    rows = [extract_features(w, features) for w in windows]
    if not rows:
        keys = features or ["rms", "peak", "crest_factor", "kurtosis"]
        return {k: np.empty(0) for k in keys}
    keys = rows[0].keys()
    return {k: np.array([r[k] for r in rows]) for k in keys}


# =============================================================
#  T1.2 — 임계값 산정 (정상 분포 기반)
#  ⚠️ 임계값은 '정상 데이터에서만' 계산한다(고장은 보지 않는다, 누수 차단).
# =============================================================
def fit_thresholds(normal_windows: np.ndarray, cfg: dict) -> dict:
    """정상 윈도우들의 특징 분포에서 특징별 상단 임계값을 계산한다.

    규칙(config stats.threshold_rule):
      - percentile : 정상 분포 상위 p%  (예: 99 → 정상의 99% 지점)
      - mean_std   : 평균 + Nσ
    반환: {특징: {'threshold','rule','fit_mean','fit_std','n_windows'}}
    """
    feats = cfg["stats"]["features"]
    rule = cfg["stats"]["threshold_rule"]
    fb = extract_features_batch(normal_windows, feats)

    out: dict[str, dict] = {}
    for f in feats:
        vals = fb[f]
        mean, std = float(vals.mean()), float(vals.std())
        if rule == "percentile":
            p = cfg["stats"]["threshold_percentile"]
            thr = float(np.percentile(vals, p))
            rule_str = f"percentile_{p}"
        elif rule == "mean_std":
            sigma = cfg["stats"]["threshold_sigma"]
            thr = mean + sigma * std
            rule_str = f"mean+{sigma}std"
        else:
            raise ValueError(f"지원하지 않는 임계값 규칙: {rule}")
        out[f] = {
            "threshold": thr,
            "rule": rule_str,
            "fit_mean": mean,
            "fit_std": std,
            "n_windows": int(len(vals)),
        }
    return out


def save_thresholds(section: dict, cfg: dict, meta: dict | None = None) -> Path:
    """경로 A 임계값을 models/thresholds.json 의 'path_a_statistical' 에 저장한다.

    기존 파일이 있으면 병합한다(나중에 경로 B 가 같은 파일에 추가하므로).
    """
    path = resolve_path(cfg["artifacts"]["thresholds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
    data["path_a_statistical"] = section
    if meta:
        data.setdefault("meta", {}).update(meta)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_thresholds(cfg: dict) -> dict:
    """저장된 임계값 전체(JSON)를 불러온다."""
    return json.loads(resolve_path(cfg["artifacts"]["thresholds"]).read_text(encoding="utf-8"))


def is_anomaly(features: dict[str, float], thresholds: dict) -> tuple[bool, dict[str, bool]]:
    """특징이 임계값을 넘는지 판정한다. (한 특징이라도 초과하면 이상 — OR)

    반환: (이상여부, {특징: 초과여부})
    """
    flags = {f: float(features[f]) > t["threshold"] for f, t in thresholds.items() if f in features}
    return (any(flags.values()), flags)


def anomaly_score(features: dict[str, float], thresholds: dict) -> float:
    """경로 A 이상 점수 = 특징/임계값 비율의 최댓값.

    1.0 을 넘으면 어떤 특징이 임계값을 초과한 것(=이상). 융합(T3.1)에서 쓴다.
    """
    ratios = [
        float(features[f]) / t["threshold"]
        for f, t in thresholds.items()
        if f in features and t["threshold"] > 0
    ]
    return max(ratios) if ratios else 0.0


def _main() -> None:
    """T1.2 검증: 정상에서 임계값 산정·저장 후, 정상/고장 초과율 비교."""
    sys.stdout.reconfigure(encoding="utf-8")
    from src.data_loader import load_fault_signals, load_normal_signals
    from src.preprocessing import make_windows, make_windows_from_signals

    cfg = load_config()
    length = cfg["window"]["length"]
    overlap = cfg["window"]["overlap"]
    feats = cfg["stats"]["features"]

    # 1) 정상 윈도우 전체 → 임계값 산정 (고장은 안 봄)
    normal = load_normal_signals(cfg)
    nwins = make_windows_from_signals(normal, length, overlap)
    thresholds = fit_thresholds(nwins, cfg)
    out = save_thresholds(
        thresholds, cfg,
        meta={"fit_data": cfg["data"]["raw_normal"], "config_seed": cfg["seed"], "window_length": length},
    )
    print("=== T1.2 임계값 산정 (정상 train) ===")
    for f, t in thresholds.items():
        print(f"  {f:13s} 임계값 {t['threshold']:.4f}  ({t['rule']}, 평균 {t['fit_mean']:.4f})")
    print(f"  → 저장: {out.relative_to(_ROOT)}")

    # 2) 평가용 비교 — 정상/고장 윈도우의 '이상 판정' 비율 (고장은 여기서만 사용)
    n_flag = sum(is_anomaly(extract_features(w, feats), thresholds)[0] for w in nwins)
    print(f"\n[정상] 이상으로 잡힌 비율(오탐): {n_flag}/{len(nwins)} = {100*n_flag/len(nwins):.1f}%")

    fault = load_fault_signals(cfg)
    for loc in ("IR", "OR", "B"):
        v = next((x for x in fault.values() if x["location"] == loc), None)
        if v is None:
            continue
        fw = make_windows(v["signal"], length, overlap)
        f_flag = sum(is_anomaly(extract_features(w, feats), thresholds)[0] for w in fw)
        print(f"[고장 {loc}] 이상으로 잡힌 비율(탐지): {f_flag}/{len(fw)} = {100*f_flag/len(fw):.1f}%")

    print("\nT1.2 OK — 정상에서 임계값 자동 산정·저장. 정상 오탐 낮고 고장 탐지 높으면 성공.")


if __name__ == "__main__":
    _main()
