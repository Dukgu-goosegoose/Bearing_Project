"""T1.3 — 경로 A 검증 테스트 (합성 임펄스 주입).

정상 같은 신호엔 이상으로 거의 안 잡히고, 임펄스를 주입하면 임계값을 초과(이상)로
잡히는지 확인한다. 실데이터(.mat) 없이 합성 신호로 동작한다.
pytest 가 깔려 있으면 pytest 로도 돌고, 없으면 이 파일을 직접 실행하면 된다.

실행:
    .venv\\Scripts\\python.exe -m tests.test_baseline_statistical
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.baseline_statistical import (
    anomaly_score,
    extract_features,
    fit_thresholds,
    is_anomaly,
)
from src.utils import load_config

LENGTH = 1600


def _normal_window(rng: np.random.Generator) -> np.ndarray:
    """정상 같은 신호: 평범한 가우시안 노이즈 (충격 없음)."""
    return rng.normal(0.0, 1.0, LENGTH)


def _impulse_window(rng: np.random.Generator, n_impulses: int = 8, amp: float = 12.0) -> np.ndarray:
    """임펄스 주입 신호: 가우시안에 짧고 강한 충격 몇 개를 더한다 (고장 모사)."""
    w = rng.normal(0.0, 1.0, LENGTH)
    idx = rng.integers(0, LENGTH, n_impulses)
    w[idx] += amp * np.sign(rng.normal(size=n_impulses))
    return w


def _fit_thresholds_on_synthetic():
    """합성 정상 신호 300개로 임계값을 산정한다 (실데이터 불필요)."""
    cfg = load_config()
    rng = np.random.default_rng(0)
    normal = np.stack([_normal_window(rng) for _ in range(300)])
    return cfg, fit_thresholds(normal, cfg)


def test_clean_window_not_anomaly():
    """정상 신호는 대부분 이상으로 안 잡힌다(오탐 10% 미만)."""
    cfg, thr = _fit_thresholds_on_synthetic()
    rng = np.random.default_rng(123)
    flagged = sum(
        is_anomaly(extract_features(_normal_window(rng), cfg["stats"]["features"]), thr)[0]
        for _ in range(100)
    )
    assert flagged < 10, f"정상 오탐이 너무 많음: {flagged}/100"


def test_impulse_exceeds_threshold():
    """임펄스를 주입하면 전부 이상으로 잡혀야 한다."""
    cfg, thr = _fit_thresholds_on_synthetic()
    rng = np.random.default_rng(7)
    detected = sum(
        is_anomaly(extract_features(_impulse_window(rng), cfg["stats"]["features"]), thr)[0]
        for _ in range(20)
    )
    assert detected == 20, f"임펄스 탐지 실패: {detected}/20"


def test_score_increases_with_impulse():
    """임펄스 점수는 1.0(임계값)을 넘고, 정상 점수보다 높아야 한다."""
    cfg, thr = _fit_thresholds_on_synthetic()
    rng = np.random.default_rng(9)
    feats = cfg["stats"]["features"]
    s_clean = anomaly_score(extract_features(_normal_window(rng), feats), thr)
    s_impulse = anomaly_score(extract_features(_impulse_window(rng), feats), thr)
    assert s_impulse > 1.0, f"임펄스 점수가 임계값 미만: {s_impulse:.2f}"
    assert s_impulse > s_clean, f"임펄스({s_impulse:.2f})가 정상({s_clean:.2f})보다 높아야 함"


def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    tests = [
        test_clean_window_not_anomaly,
        test_impulse_exceeds_threshold,
        test_score_increases_with_impulse,
    ]
    print("=== T1.3 경로 A 검증 테스트 ===")
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} 통과")


if __name__ == "__main__":
    _main()
