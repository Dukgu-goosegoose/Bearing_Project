"""시각화 공통 — 한글 폰트, 경로, '예시 신호 + 자세한 라벨' 선택.

각 그래프 스크립트가 import 해서 같은 예시 신호·라벨을 공유한다.
(그래프 그리는 코드 자체는 각 스크립트에 따로 있다.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 화면 없이 파일 저장
import matplotlib.pyplot as plt  # noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# 한글 라벨 안 깨지게
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

FIG_DIR = "outputs/figures"
_LOC_KR = {"IR": "내륜", "OR": "외륜", "B": "볼"}


def save_fig(fig, name: str) -> Path:
    """outputs/figures/<name> 으로 저장하고 경로 반환."""
    from src.utils import resolve_path

    out = resolve_path(f"{FIG_DIR}/{name}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def get_example_signals(cfg: dict) -> list[dict]:
    """모든 그래프가 공유하는 대표 예시 신호 + 자세한 라벨.

    - 정상: normal_0~3 (HP0~3, 원본 48k→12k)
    - DE 고장: fault_48k → 12k 다운샘플, HP0 의 IR/OR(6시)/B (크기 0.007")
    - FE 고장: fan_end_fault(12k), HP0 의 IR (크기 0.007")
    각 항목: {group, fname, signal, sr, location, size, load, title}
    """
    from src.data_loader import load_fault_signals, load_normal_signals

    sr = cfg["data"]["target_rate"]
    khz = sr // 1000
    out: list[dict] = []

    # --- 정상 0~3 ---
    normal = load_normal_signals(cfg)
    for i, (name, sig) in enumerate(sorted(normal.items())):
        out.append(dict(
            group="normal", fname=name, signal=sig, sr=sr,
            location=None, size=None, load=i,
            title=f"정상 {name}  |  HP{i}  |  {khz}kHz (원본 48k→12k 다운샘플)",
        ))

    # --- 고장 (HP0, 크기 0.007") ---
    fault = load_fault_signals(cfg)

    def find(source, loc, key_contains=None):
        cands = [(k, v) for k, v in fault.items()
                 if v["source"] == source and v["location"] == loc
                 and v["load"] == 0 and v["size"] == "0.007"]
        if key_contains:
            for k, v in cands:
                if key_contains in k:
                    return k, v
        return cands[0] if cands else (None, None)

    plan = [
        ("fault_48k",     "IR", None,   "Drive-End", "48k→12k 다운샘플"),
        ("fault_48k",     "OR", "_6_",  "Drive-End", "48k→12k 다운샘플"),  # 6시(하중영역)
        ("fault_48k",     "B",  None,   "Drive-End", "48k→12k 다운샘플"),
        ("fan_end_fault", "IR", None,   "Fan-End",   "12k 원본"),
    ]
    for source, loc, kc, end, srcnote in plan:
        k, v = find(source, loc, kc)
        if v is None:
            continue
        fname = k.split("/")[-1]
        pos = " 6시(하중영역)" if kc == "_6_" else ""
        out.append(dict(
            group=f"fault_{'DE' if end == 'Drive-End' else 'FE'}",
            fname=fname, signal=v["signal"], sr=sr,
            location=loc, size=v["size"], load=0,
            title=f"{end} 고장 {loc}({_LOC_KR[loc]}){pos}  |  크기 {v['size']}\"  |  HP0  |  {khz}kHz ({srcnote})  [{fname}]",
        ))
    return out
