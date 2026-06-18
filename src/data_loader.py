"""CWRU .mat 로더 — 정상/고장 분리, 12k 통일(다운샘플) + 멀티 로드.

핵심 원칙(데이터 누수 차단):
- 정상(normal)만 비지도 경로(A·B) 학습·임계값 설정에 쓴다.
- 고장(fault)은 load_fault_signals() 라는 별도 함수로만 접근한다.
  → A·B 학습 코드는 load_normal_signals() 만 호출(고장에 손 못 댐).
  → 단, 경로 C(지도 분류)는 고장 라벨이 필요하므로 load_fault_signals() 를 쓴다.

12k 통일:
- 모든 신호를 target_rate(=12kHz)로 맞춘다. 48k 소스는 다운샘플(안티앨리어싱 포함).
- 한 가지 rate로 파이프라인을 단일화 → B→C 변환 다리 불필요, 코드 단순.

라벨:
- 위치(IR/OR/B)만 분류 대상(팀 합의). 크기는 참고용. 로드(0~3)는 멀티조건 평가용.

실행(개수 확인):
    .venv\\Scripts\\python.exe -m src.data_loader
"""
from __future__ import annotations

import re
import sys
from math import gcd
from pathlib import Path

import numpy as np
import scipy.io as sio
import scipy.signal

# 직접 실행(python src/data_loader.py)해도 src 패키지를 찾도록 루트를 경로에 추가
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path

# 파일이름에서 결함 위치·크기를 뽑는 정규식.
#   IR007 -> ('IR','007'),  OR021@6 -> ('OR','021'),  B028_2 -> ('B','028')
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


def _resample_to_target(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """신호를 target_sr 로 맞춘다. 같으면 그대로, 다르면 다운/리샘플(안티앨리어싱)."""
    if orig_sr == target_sr:
        return signal
    if orig_sr % target_sr == 0:
        # 정수배 다운샘플: decimate 가 저역통과 필터까지 해줌 (앨리어싱 방지)
        factor = orig_sr // target_sr
        return scipy.signal.decimate(signal, factor, ftype="iir", zero_phase=True).astype(np.float64)
    # 비정수 비율: 다항식 리샘플
    g = gcd(orig_sr, target_sr)
    return scipy.signal.resample_poly(signal, target_sr // g, orig_sr // g).astype(np.float64)


def parse_fault_label(filename: str) -> tuple[str | None, str | None]:
    """파일이름에서 (위치, 손상크기)를 뽑는다.

    IR007  -> ('IR', '0.007')   # 내륜
    OR021  -> ('OR', '0.021')   # 외륜 (위치 3·6·12시는 합침)
    B028   -> ('B',  '0.028')   # 볼
    위치(IR/OR/B)가 분류 대상, 크기는 참고용. 매칭 안 되면 (None, None).
    """
    m = _LABEL_RE.search(filename.upper())
    if not m:
        return None, None
    return m.group(1), "0." + m.group(2)  # '007' -> '0.007'


def parse_load(filename: str) -> int | None:
    """파일이름 끝의 로드 번호(0~3)를 뽑는다.

    IR007_2 -> 2,  OR007_6_3 -> 3(위치6 뒤의 로드),  B028_0 -> 0
    """
    last = filename.split("_")[-1]
    return int(last) if last.isdigit() else None


def load_normal_signals(cfg: dict) -> dict[str, np.ndarray]:
    """정상 신호들을 로드한다 (원본 48k → target_rate 12k 다운샘플).

    이게 비지도 A·B 학습에 쓰는 유일한 데이터다.
    반환: {파일이름: 신호배열(12k)}
    """
    folder = resolve_path(cfg["data"]["raw_normal"])
    suffix = cfg["data"]["normal_mat_key"]
    orig_sr = cfg["data"]["normal_sampling_rate"]
    target = cfg["data"]["target_rate"]
    signals: dict[str, np.ndarray] = {}
    for mat in sorted(folder.glob("*.mat")):
        sig = _extract_signal(mat, suffix)
        signals[mat.stem] = _resample_to_target(sig, orig_sr, target)
    return signals


def load_fault_signals(cfg: dict) -> dict[str, dict]:
    """고장 신호들을 멀티 소스로 로드한다 (전부 target_rate 12k 로 맞춤).

    ⚠️ A·B(비지도) 학습·임계값 설정에는 절대 쓰지 말 것(평가 전용).
       경로 C(지도 분류)만 학습에 사용하며, 그때도 train/test 분할 필수.

    반환: {'소스/파일이름': {
              'signal': 신호배열(12k), 'location': 'IR'/'OR'/'B', 'size': '0.007'...,
              'load': 0~3, 'label': 'IR'(=위치), 'source': 'fault_12k',
              'sampling_rate': 12000 }}
    """
    target = cfg["data"]["target_rate"]
    out: dict[str, dict] = {}
    for src in cfg["data"]["fault_sources"]:
        folder = resolve_path(src["path"])
        suffix = src["mat_key"]
        orig_sr = src["sampling_rate"]
        for mat in sorted(folder.glob("*.mat")):
            loc, size = parse_fault_label(mat.stem)
            sig = _resample_to_target(_extract_signal(mat, suffix), orig_sr, target)
            out[f"{src['name']}/{mat.stem}"] = {
                "signal": sig,
                "location": loc,        # 분류 대상 (IR/OR/B)
                "size": size,           # 참고용
                "load": parse_load(mat.stem),
                "label": loc or "UNKNOWN",  # 위치 = 분류 라벨
                "source": src["name"],
                "sampling_rate": target,
            }
    return out


def _main() -> None:
    """검증: 정상·고장 개수를 출력한다 (12k 통일 + 로드 파싱 확인)."""
    sys.stdout.reconfigure(encoding="utf-8")  # 윈도우 콘솔 한글 깨짐 방지
    cfg = load_config()
    target = cfg["data"]["target_rate"]

    print(f"=== 데이터 로드 (전부 {target}Hz 로 통일) ===")
    normal = load_normal_signals(cfg)
    print(f"정상 파일 {len(normal)}개 (A·B 학습용, 48k→12k 다운샘플)")
    for name, sig in normal.items():
        print(f"  - {name}: {len(sig):,} 포인트")

    fault = load_fault_signals(cfg)
    print(f"\n고장 파일 {len(fault)}개 (평가/경로C 전용 — A·B 학습 금지)")

    by_source: dict[str, int] = {}
    by_loc: dict[str, int] = {}
    by_load: dict[int, int] = {}
    for info in fault.values():
        by_source[info["source"]] = by_source.get(info["source"], 0) + 1
        by_loc[info["label"]] = by_loc.get(info["label"], 0) + 1
        by_load[info["load"]] = by_load.get(info["load"], 0) + 1
    print("  [소스별]", {s: c for s, c in sorted(by_source.items())})
    print("  [위치별 3클래스]", {s: c for s, c in sorted(by_loc.items())})
    print("  [로드별]", {s: c for s, c in sorted(by_load.items())})

    print("\nOK — 정상/고장 로드됨(12k 통일). 고장은 load_fault_signals() 로만 접근(A·B 격리).")


if __name__ == "__main__":
    _main()
