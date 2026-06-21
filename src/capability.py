"""경로별 가용성 판정기 — 새 기계(.mat)를 받았을 때 각 경로가 어떤 상태인지 결정.

상태 3종:
  🟢 READY    = 즉시 작동(공식/통계는 계산만, 딥러닝은 학습된 가중치 있음)
  🟡 RETRAIN  = 재학습 필요(가중치 없음 + 정상 데이터 있음). 라벨 0개여도 됨(정상만).
  ⚪ NO_INFO  = 필수 입력이 빠져서 못 켬(라벨/베어링 사양/rpm 없음). 능력 아닌 '재료' 문제.

설계 의도:
  물리(C①)·통계(A)는 "공식/통계"라 데이터만 있으면 🟢. 새 기계에 즉시 통함(상용화 면역).
  오토인코더(B)·CNN(C②)은 "외운 가중치"라, 가중치 없으면 🟡(재학습) 또는 ⚪(라벨 없음).

핵심 분리:
  gather_facts(cfg)  → 사실 수집(파일 존재·정상데이터·베어링·라벨 등) — 부수효과(파일 I/O)
  resolve(facts)     → 순수 판정 로직(부수효과 없음) — 그래서 단위 테스트가 쉽다

실행(자체 검증, 데이터·모델 불필요):
    .venv\\Scripts\\python.exe -m src.capability
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path


class Status(str, Enum):
    READY = "ready"        # 🟢 즉시
    RETRAIN = "retrain"    # 🟡 재학습 필요
    NO_INFO = "no_info"    # ⚪ 정보 없음


ICON = {Status.READY: "🟢", Status.RETRAIN: "🟡", Status.NO_INFO: "⚪"}


@dataclass
class PathStatus:
    path: str        # "A" / "B" / "C_physics" / "C_cnn"
    status: Status
    reason: str      # 사람이 읽을 짧은 설명

    def line(self) -> str:
        return f"{ICON[self.status]} 경로 {self.path}: {self.status.value} — {self.reason}"


# =============================================================
#  1. 사실 수집 — 파일/설정을 훑어 불리언 사실들을 모은다
# =============================================================
def _has_mat_files(folder: Path) -> bool:
    return folder.exists() and any(folder.glob("*.mat"))


def _bearing_valid(cfg: dict) -> bool:
    """베어링 기하가 결함주파수를 계산할 만큼 채워졌는가(0/누락이면 무효)."""
    b = cfg.get("bearing") or {}
    try:
        return b["N"] > 0 and b["Bd_mm"] > 0 and b["Pd_mm"] > 0
    except (KeyError, TypeError):
        return False


def gather_facts(cfg: dict, models_dir: str | Path | None = None) -> dict:
    """판정에 필요한 사실들을 수집한다(파일 존재 여부 등).

    반환 키: normal_data, ae_weights, cnn_weights, thr_a, thr_b,
            bearing_valid, has_rpm, has_labels
    """
    import json

    mdir = resolve_path(models_dir or cfg["artifacts"]["models"])
    thr_path = resolve_path(cfg["artifacts"]["thresholds"])

    thr_a = thr_b = False
    if thr_path.exists():
        try:
            thr = json.loads(thr_path.read_text(encoding="utf-8"))
            thr_a = "path_a_statistical" in thr
            thr_b = "path_b" in thr
        except (ValueError, OSError):
            pass

    return {
        "normal_data": _has_mat_files(resolve_path(cfg["data"]["raw_normal"])),
        "ae_weights": (mdir / "autoencoder.pth").exists(),
        "cnn_weights": (mdir / "fault_classifier.pth").exists(),
        "thr_a": thr_a,
        "thr_b": thr_b,
        "bearing_valid": _bearing_valid(cfg),
        "has_rpm": bool(cfg.get("domain", {}).get("rpm")),
        "has_labels": bool(cfg.get("labels", {}).get("has_fault_labels", False)),
    }


# =============================================================
#  2. 순수 판정 — 사실 → 경로별 상태 (부수효과 없음 → 테스트 쉬움)
# =============================================================
def resolve(facts: dict) -> dict[str, PathStatus]:
    """사실 dict 를 받아 경로별 PathStatus 를 돌려준다(순수 함수)."""

    # --- 경로 A (통계): 공식·통계라 학습 불필요. 정상 데이터만 있으면 즉시. ---
    if facts["thr_a"]:
        a = PathStatus("A", Status.READY, "임계값 산정 완료(저장된 μ,σ 사용)")
    elif facts["normal_data"]:
        a = PathStatus("A", Status.READY, "정상 데이터로 μ,σ 즉시 산출(학습 불필요)")
    else:
        a = PathStatus("A", Status.NO_INFO, "정상 데이터 없음 → 기준선 못 만듦")

    # --- 경로 C 물리(①): 베어링 사양 + rpm 만 있으면 즉시(공식). ---
    if facts["bearing_valid"] and facts["has_rpm"]:
        cp = PathStatus("C_physics", Status.READY, "사양·rpm 입력됨 → 결함주파수 즉시 계산")
    elif not facts["bearing_valid"]:
        cp = PathStatus("C_physics", Status.NO_INFO, "베어링 사양 미입력 → 사양 필요")
    else:
        cp = PathStatus("C_physics", Status.NO_INFO, "rpm 미입력 → 회전수 필요")

    # --- 경로 B (오토인코더): 가중치 있으면 즉시, 없으면 정상 데이터로 재학습. ---
    if facts["ae_weights"] and facts["thr_b"]:
        b = PathStatus("B", Status.READY, "학습된 오토인코더 사용")
    elif facts["normal_data"]:
        b = PathStatus("B", Status.RETRAIN, "이 기계용 가중치 없음 → 정상 데이터로 재학습(라벨 0)")
    else:
        b = PathStatus("B", Status.NO_INFO, "가중치·정상 데이터 모두 없음")

    # --- 경로 C CNN(②): 라벨 없으면 아예 못 켬. ---
    if not facts["has_labels"]:
        cc = PathStatus("C_cnn", Status.NO_INFO, "고장 라벨 없음 → CNN 미적용(물리 단독)")
    elif facts["cnn_weights"]:
        cc = PathStatus("C_cnn", Status.READY, "학습된 CNN 사용")
    else:
        cc = PathStatus("C_cnn", Status.RETRAIN, "라벨 있음·가중치 없음 → 고장 라벨로 재학습")

    return {"A": a, "B": b, "C_physics": cp, "C_cnn": cc}


def resolve_capability(cfg: dict, models_dir: str | Path | None = None) -> dict[str, PathStatus]:
    """cfg 를 받아 (사실 수집 → 판정)을 한 번에. 웹/오케스트레이터가 이걸 호출한다."""
    return resolve(gather_facts(cfg, models_dir))


# =============================================================
#  3. 자체 검증 — 가상 시나리오로 판정 로직만 확인(데이터·모델 불필요)
# =============================================================
def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    print("=== 1) 순수 판정 로직 — 가상 시나리오 ===")
    scenarios = {
        "CWRU 풀세트(전부 학습됨)": dict(
            normal_data=True, ae_weights=True, cnn_weights=True, thr_a=True, thr_b=True,
            bearing_valid=True, has_rpm=True, has_labels=True),
        "새 기계(정상만 있음, 라벨 없음, 사양 입력됨)": dict(
            normal_data=True, ae_weights=False, cnn_weights=False, thr_a=False, thr_b=False,
            bearing_valid=True, has_rpm=True, has_labels=False),
        "새 기계(사양 모름, rpm만 있음)": dict(
            normal_data=True, ae_weights=False, cnn_weights=False, thr_a=False, thr_b=False,
            bearing_valid=False, has_rpm=True, has_labels=False),
        "rpm조차 없음": dict(
            normal_data=True, ae_weights=False, cnn_weights=False, thr_a=False, thr_b=False,
            bearing_valid=False, has_rpm=False, has_labels=False),
    }
    for name, facts in scenarios.items():
        print(f"\n[{name}]")
        for st in resolve(facts).values():
            print("   " + st.line())

    print("\n=== 2) 실제 config 기준(현재 폴더 상태) ===")
    try:
        cfg = load_config()
        print(f"  기계 프로필: {cfg.get('machine') or 'cwru(기본)'}")
        for st in resolve_capability(cfg).values():
            print("   " + st.line())
    except Exception as e:  # noqa: BLE001
        print(f"  (config 로드 실패: {e})")


if __name__ == "__main__":
    _main()
