"""CWRU .mat 로더 — 정상/고장 분리, 멀티 샘플링 소스 지원.

핵심 원칙(데이터 누수 차단):
- 정상(normal)만 비지도 경로(A·B) 학습·임계값 설정에 쓴다.
- 고장(fault)은 load_fault_signals() 라는 별도 함수로만 접근한다.
  → A·B 학습 코드는 load_normal_signals() 만 호출(고장에 손 못 댐).
  → 단, 경로 C(지도 분류)는 고장 라벨이 필요하므로 load_fault_signals() 를 쓴다
    (대신 C 내부에서 train/test 분할 필수).

지원 소스(config.yaml 의 data.fault_sources):
- data/raw/normal        : 정상 (학습용, 48k, DE_time)
- data/raw/fault_48k     : 고장 48kHz (평가 전용, DE_time)
- data/raw/fault_12k     : 고장 12kHz (경로 C 학습용, DE_time)
- data/raw/fan_end_fault : Fan End 고장 (평가 전용, FE_time)

실행(개수 확인):
    .venv\\Scripts\\python.exe -m src.data_loader
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

# 직접 실행(python src/data_loader.py)해도 src 패키지를 찾도록 루트를 경로에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path

# 파일이름에서 결함 위치·크기를 뽑는 정규식.
#   IR007 -> ('IR','007'),  OR021@6 -> ('OR','021'),  B014_2 -> ('B','014')
_LABEL_RE = re.compile(r"(IR|OR|B)(\d{3})", re.IGNORECASE)


def _extract_signal(mat_path: Path, key_suffix: str) -> np.ndarray:
    """.mat 파일 하나에서 진동 신호(1차원)를 꺼낸다.

    변수명이 파일마다 X097_DE_time, X108_DE_time 처럼 숫자가 달라서
    '_DE_time'(또는 '_FE_time')으로 *끝나는* 키를 찾아 매칭한다.
    """
    data = sio.loadmat(mat_path)
    keys = [k for k in data if k.endswith(key_suffix)]
    if not keys:
        raise KeyError(f"{mat_path.name} 에 '{key_suffix}' 로 끝나는 변수가 없음")
    # squeeze: (N,1) 을 (N,) 1차원으로 펴 줌
    return data[keys[0]].squeeze().astype(np.float64)


def parse_fault_label(filename: str) -> tuple[str | None, str | None]:
    """파일이름에서 (위치, 손상크기)를 뽑는다. 9클래스 = 위치 × 크기.

    IR007  -> ('IR', '0.007')   # 내륜
    OR021  -> ('OR', '0.021')   # 외륜 (위치 3·6·12시는 합침)
    B014   -> ('B',  '0.014')   # 볼
    매칭 안 되면 (None, None).
    """
    m = _LABEL_RE.search(filename.upper())
    if not m:
        return None, None
    location = m.group(1)
    size = "0." + m.group(2)  # '007' -> '0.007'
    return location, size


def load_normal_signals(cfg: dict) -> dict[str, np.ndarray]:
    """정상 신호들을 로드한다. (비지도 A·B 학습에 쓰는 유일한 데이터)

    반환: {파일이름: 신호배열}  예) {'normal_0': array([...]), ...}
    """
    folder = resolve_path(cfg["data"]["raw_normal"])
    suffix = cfg["data"]["normal_mat_key"]
    signals: dict[str, np.ndarray] = {}
    for mat in sorted(folder.glob("*.mat")):
        signals[mat.stem] = _extract_signal(mat, suffix)
    return signals


def load_fault_signals(cfg: dict) -> dict[str, dict]:
    """고장 신호들을 멀티 소스로 로드한다.

    ⚠️ A·B(비지도) 학습·임계값 설정에는 절대 쓰지 말 것(평가 전용).
       경로 C(지도 분류)만 학습에 사용하며, 그때도 train/test 분할 필수.

    반환: {'소스/파일이름': {
              'signal': 신호배열, 'location': 'IR'/'OR'/'B', 'size': '0.007'...,
              'label': 'IR_0.007', 'source': 'fault_12k', 'sampling_rate': 12000 }}
    """
    out: dict[str, dict] = {}
    for src in cfg["data"]["fault_sources"]:
        folder = resolve_path(src["path"])
        suffix = src["mat_key"]
        for mat in sorted(folder.glob("*.mat")):
            loc, size = parse_fault_label(mat.stem)
            out[f"{src['name']}/{mat.stem}"] = {
                "signal": _extract_signal(mat, suffix),
                "location": loc,
                "size": size,
                "label": f"{loc}_{size}" if loc else "UNKNOWN",
                "source": src["name"],
                "sampling_rate": src["sampling_rate"],
            }
    return out


def _main() -> None:
    """T0.2 검증: 정상·고장 개수를 출력한다 (데이터가 폴더에 있어야 보임)."""
    sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 콘솔 한글 깨짐 방지
    cfg = load_config()

    print("=== T0.2 데이터 로드 ===")
    normal = load_normal_signals(cfg)
    print(f"정상 파일 {len(normal)}개 (A·B 학습용)")
    for name, sig in normal.items():
        print(f"  - {name}: {len(sig):,} 포인트")

    fault = load_fault_signals(cfg)
    print(f"\n고장 파일 {len(fault)}개 (평가/경로C 전용 — A·B 학습 금지)")

    by_source: dict[str, int] = {}
    by_label: dict[str, int] = {}
    for info in fault.values():
        by_source[info["source"]] = by_source.get(info["source"], 0) + 1
        by_label[info["label"]] = by_label.get(info["label"], 0) + 1
    print("  [소스별]")
    for s, c in sorted(by_source.items()):
        print(f"    - {s}: {c}개")
    print("  [라벨별 9클래스]")
    for lab, c in sorted(by_label.items()):
        print(f"    - {lab}: {c}개")

    print("\nT0.2 OK — 정상/고장 개수 출력됨. 고장은 load_fault_signals() 로만 접근(A·B 격리).")


if __name__ == "__main__":
    _main()
