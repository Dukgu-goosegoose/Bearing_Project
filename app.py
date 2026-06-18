"""베어링 이상 진단 대시보드 (Streamlit).

신호 1개를 골라 전체 파이프라인을 돌리고 결과를 3구역으로 보여준다.
  ① 입력·신호   : 원신호 파형 + 스펙트로그램
  ② 이상 판정   : 정상/이상/검토필요 신호등 + 경로 A·B 점수
  ③ 진단(이상 시): 물리 vs CNN 위치 + 교차검증 + 신뢰도 + 엔벨로프 스펙트럼

실행:
    .venv\\Scripts\\streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
import matplotlib.pyplot as plt
import streamlit as st

from src.schemas import CrossCheck
from src.utils import load_config

matplotlib.rcParams["font.family"] = "Malgun Gothic"   # 윈도우 한글
matplotlib.rcParams["axes.unicode_minus"] = False

_TARGET_SR = 12000
_LOC_KR = {"IR": "내륜(IR)", "OR": "외륜(OR)", "B": "볼(B)"}


# =============================================================
#  데이터·결과물 로드 (캐시)
# =============================================================
@st.cache_resource
def get_context():
    """config + A·B·C 결과물 로드. 결과물 없으면 err 에 사유."""
    cfg = load_config()
    try:
        from src.inference import load_artifacts
        return cfg, load_artifacts(cfg), None
    except Exception as e:  # noqa: BLE001
        return cfg, None, str(e)


@st.cache_data
def list_recordings(_cfg) -> dict:
    """선택 가능한 신호 목록 {표시이름: (신호, 정답위치|None)}."""
    from src.data_loader import load_fault_signals, load_normal_signals

    items: dict = {}
    for name, sig in list(load_normal_signals(_cfg).items())[:4]:
        items[f"🟢 정상 — {name}"] = (np.asarray(sig), None)
    for key, info in load_fault_signals(_cfg).items():
        if info["location"] is None or info["source"] == "fault_48k":
            continue
        items[f"🔴 고장 {info['location']} — {key.split('/')[-1]}"] = (
            np.asarray(info["signal"]), info["location"])
    return items


# =============================================================
#  그림 헬퍼
# =============================================================
def fig_waveform(signal: np.ndarray, fs: int):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(9, 2.4))
    ax.plot(t, signal, lw=0.4, color="#2c3e50")
    ax.set_xlabel("시간 (초)")
    ax.set_ylabel("진폭")
    ax.set_title("원신호 파형")
    fig.tight_layout()
    return fig


def fig_spectrogram(signal: np.ndarray, cfg: dict):
    from src.preprocessing import make_windows
    from src.spectrogram import to_spectrogram

    w = make_windows(signal, cfg["window"]["length"], cfg["window"]["overlap"])
    fig, ax = plt.subplots(figsize=(9, 2.6))
    if len(w):
        spec = to_spectrogram(w[0], cfg)
        nyq = cfg["data"]["target_rate"] / 2 / 1000
        im = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma",
                       extent=(0, spec.shape[1], 0, nyq))
        ax.set_ylabel("주파수 (kHz)")
        ax.set_xlabel("시간 프레임")
        fig.colorbar(im, ax=ax, label="dB")
    ax.set_title("스펙트로그램 (첫 윈도우)")
    fig.tight_layout()
    return fig


def fig_ab_scores(sa: np.ndarray, sb: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.plot(sa, lw=0.8, color="#16a085", label="경로 A (통계)")
    ax.plot(sb, lw=0.8, color="#8e44ad", label="경로 B (오토인코더)")
    ax.axhline(1.0, color="red", ls="--", lw=1, label="이상 임계(1.0)")
    ax.set_xlabel("윈도우 번호")
    ax.set_ylabel("이상 점수")
    ax.set_title("경로 A·B 윈도우별 점수 (1.0 초과 = 이상)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def fig_envelope(signal: np.ndarray, cfg: dict, rpm: float):
    from src.physics_diagnosis import defect_frequencies, envelope_spectrum

    freqs, spec = envelope_spectrum(signal, _TARGET_SR, band=(2500, 5500))
    fd = defect_frequencies(rpm)
    fig, ax = plt.subplots(figsize=(9, 2.8))
    mask = freqs <= 400
    ax.plot(freqs[mask], spec[mask], lw=0.8, color="#2c3e50")
    colors = {"IR": "#e74c3c", "OR": "#2980b9", "B": "#27ae60"}
    for loc, f in fd.items():
        ax.axvline(f, color=colors[loc], ls="--", lw=1.2, label=f"{_LOC_KR[loc]} {f:.0f}Hz")
    ax.set_xlabel("주파수 (Hz)")
    ax.set_ylabel("엔벨로프 세기")
    ax.set_title("엔벨로프 스펙트럼 — 결함 주파수 위치(점선)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# =============================================================
#  메인 UI
# =============================================================
def main() -> None:
    st.set_page_config(page_title="베어링 이상 진단", layout="wide")
    st.title("🔧 베어링 이상 진단 대시보드")
    st.caption("정상만 학습한 비지도 탐지(A·B) + 물리·CNN 교차검증 진단(C) — 게이트형 파이프라인")

    cfg, art, err = get_context()
    recordings = list_recordings(cfg)

    if not recordings:
        st.error("⚠️ 분석할 신호가 없습니다 — `data/raw/` 에 .mat 데이터가 없어요.\n\n"
                 "이 폴더의 **`data/raw/normal`** 과 **`data/raw/fault_12k`** 에 "
                 "CWRU .mat 파일을 넣은 뒤 새로고침하세요.")
        st.stop()

    # ----- 사이드바: 신호 선택 -----
    with st.sidebar:
        st.header("신호 선택")
        choice = st.selectbox("분석할 신호", list(recordings.keys()))
        rpm = st.number_input("회전수 (rpm)", value=int(cfg["domain"]["rpm"]), step=10)
        run = st.button("▶ 분석 실행", type="primary", use_container_width=True)
        if err:
            st.warning("A·B·C 결과물이 아직 없어 ②·③은 제한됩니다.\n"
                       "(먼저 baseline_statistical / train_autoencoder / train_classifier 실행)")

    signal, truth = recordings[choice]

    # ================= ① 입력·신호 =================
    st.subheader("① 입력 · 신호")
    c1, c2 = st.columns(2)
    c1.pyplot(fig_waveform(signal, _TARGET_SR))
    c2.pyplot(fig_spectrogram(signal, cfg))
    st.caption(f"선택: {choice}  |  길이 {len(signal):,}점 (약 {len(signal)/_TARGET_SR:.2f}초, 12kHz)"
               + (f"  |  정답 위치: {_LOC_KR.get(truth, truth)}" if truth else "  |  정답: 정상"))

    if not run:
        st.info("◀ 왼쪽에서 신호를 고르고 **분석 실행**을 누르세요.")
        return
    if art is None:
        st.error(f"결과물이 없어 분석을 못 합니다: {err}")
        return

    # 파이프라인 실행
    from src.inference import ab_scores, run_recording
    res = run_recording(signal, cfg, art, rpm=rpm)
    sa, sb = ab_scores(signal, cfg, art)

    # ================= ② 이상 판정 =================
    st.subheader("② 이상 판정 (경로 A·B)")
    if not res.is_anomaly:
        status, color = "정상", "🟢"
    elif res.pathc and res.pathc.cross_check == CrossCheck.DISAGREE:
        status, color = "이상 · 검토 필요", "🟠"
    else:
        status, color = "이상 · 진단됨", "🔴"
    m1, m2, m3 = st.columns(3)
    m1.metric("상태", f"{color} {status}")
    m2.metric("이상 윈도우", f"{res.n_anomaly_windows} / {res.n_windows}")
    m3.metric("알람(디바운스)", f"{res.n_alarm_windows}")
    st.pyplot(fig_ab_scores(sa, sb))

    # ================= ③ 진단 (이상일 때만 = 게이트) =================
    st.subheader("③ 진단 (경로 C — 이상일 때만 호출)")
    if not res.is_anomaly:
        st.success("정상으로 판정 → 경로 C 미호출 (게이트 OFF). 진단할 결함 없음.")
        return
    pc = res.pathc
    d1, d2, d3 = st.columns(3)
    d1.metric("① 물리 진단", _LOC_KR.get(pc.physics.location.value, "검출 못함")
              if pc.physics.location else "검출 못함")
    d2.metric("② CNN 진단", _LOC_KR.get(pc.cnn.location.value) if pc.cnn else "—")
    verdict_kr = {"agree": "✅ 일치", "disagree": "⚠️ 검토필요",
                  "physics_only": "물리만", "cnn_only": "CNN만(물리 약함)"}
    d3.metric("교차검증", verdict_kr.get(pc.cross_check.value, pc.cross_check.value))

    final = _LOC_KR.get(pc.final_location.value, "미정") if pc.final_location else "미정 (검토 필요)"
    st.markdown(f"### 최종 진단: **{final}**  ·  신뢰도 **{pc.final_confidence:.0%}**")
    if truth:
        ok = pc.final_location and pc.final_location.value == truth
        st.caption(f"(정답 {_LOC_KR.get(truth)} — {'일치 ✅' if ok else '불일치'})")
    st.pyplot(fig_envelope(signal, cfg, rpm))
    st.caption("물리(공식)와 CNN(학습)이 같은 위치를 가리키면 신뢰도↑. 볼은 물리가 약해 CNN이 보완.")


if __name__ == "__main__":
    main()
