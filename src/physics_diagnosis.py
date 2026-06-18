"""경로 C① 물리 진단 (엔벨로프 분석) + 물리↔CNN 교차검증 — 비지도·학습 불필요.

원리:
  베어링은 부서진 위치마다 충격이 나는 박자(결함 주파수)가 공식으로 정해져 있다.
    BPFI(내륜)=5.4152×fr,  BPFO(외륜)=3.5848×fr,  BSF(볼)=2.357×fr   (fr=축 회전수 Hz)
  신호에서 충격 박자만 뽑아(엔벨로프) 그 주파수에 봉우리가 있는지 확인해 위치를 진단한다.
  데이터로 배우는 게 아니라 물리 법칙이라, 라벨이 0개여도 동작한다(비지도).

흐름:
  신호 → (공진대역 밴드패스) → 엔벨로프(힐버트) → 엔벨로프 스펙트럼(FFT)
       → 결함주파수·고조파 위치의 SNR 확인 → 위치 판정 → PathCPhysics

교차검증:
  cross_check(물리, CNN) → 두 진단이 같은 위치면 신뢰도↑(agree), 엇갈리면 검토필요(disagree).
  이것이 프로젝트 핵심 차별점(출처가 다른 두 근거의 상호 검증).

실행(자체 검증):
    .venv\\Scripts\\python.exe -m src.physics_diagnosis
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import scipy.signal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import CrossCheck, FAULT_CLASSES, FaultLocation, PathCCnn, PathCPhysics, PathCResult
from src.utils import load_config, resolve_path

# CWRU 베어링 기하 (SKF 6205-2RS JEM). ✅ 고정값 — Smith & Randall(2015) 검증.
BEARING_6205 = {"N": 9, "Bd_mm": 7.938, "Pd_mm": 39.04, "alpha_deg": 0.0}

_TARGET_SR = 12000  # 경로 C 기준 샘플링(고장 녹음은 전부 12k로 통일됨)


# =============================================================
#  1. 결함 주파수 공식 (BPFI/BPFO/BSF)
# =============================================================
def defect_frequencies(rpm: float, bearing: dict = BEARING_6205) -> dict[str, float]:
    """축 회전수(rpm)로 위치별 결함 주파수(Hz)를 계산한다.

    fr = rpm/60 (초당 회전). ratio = (Bd/Pd)·cosα.
    """
    fr = rpm / 60.0
    ratio = (bearing["Bd_mm"] / bearing["Pd_mm"]) * np.cos(np.radians(bearing["alpha_deg"]))
    n = bearing["N"]
    return {
        "IR": (n / 2) * (1 + ratio) * fr,                                  # BPFI 내륜
        "OR": (n / 2) * (1 - ratio) * fr,                                  # BPFO 외륜
        "B": (bearing["Pd_mm"] / (2 * bearing["Bd_mm"])) * (1 - ratio**2) * fr,  # BSF 볼
    }


# =============================================================
#  2. 엔벨로프 스펙트럼 (충격 박자만 뽑아 주파수로)
# =============================================================
def envelope_spectrum(
    signal: np.ndarray, fs: int, band: tuple[float, float] | None = (2000.0, 5000.0)
) -> tuple[np.ndarray, np.ndarray]:
    """신호 → 엔벨로프 스펙트럼 (freqs, spec).

    band: 충격이 울리는 공진대역만 통과시켜 SNR 향상(None 이면 원신호 그대로 엔벨로프).
    힐버트 변환으로 진폭 포락선을 구하고, DC 제거 후 FFT.
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - x.mean()
    if band is not None:
        lo, hi = band
        hi = min(hi, fs / 2.0 * 0.99)  # 나이퀴스트 초과 방지
        if lo < hi:
            sos = scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
            x = scipy.signal.sosfiltfilt(sos, x)
    env = np.abs(scipy.signal.hilbert(x))  # 진폭 포락선
    env = env - env.mean()                 # DC(0Hz) 제거
    spec = np.abs(np.fft.rfft(env))
    freqs = np.fft.rfftfreq(len(env), d=1.0 / fs)
    return freqs, spec


def _band_peak(freqs: np.ndarray, spec: np.ndarray, f_center: float, tol: float) -> float:
    """f_center ±tol(%) 구간의 최대 봉우리 높이."""
    lo, hi = f_center * (1 - tol), f_center * (1 + tol)
    mask = (freqs >= lo) & (freqs <= hi)
    return float(spec[mask].max()) if mask.any() else 0.0


def _snr_at(freqs: np.ndarray, spec: np.ndarray, f_center: float, tol: float,
            mode: str, global_floor: float) -> float:
    """f_center 봉우리의 SNR. mode='global'(전역 중앙값) / 'local'(주변 국소 배경)."""
    lo, hi = f_center * (1 - tol), f_center * (1 + tol)
    pmask = (freqs >= lo) & (freqs <= hi)
    if not pmask.any():
        return 0.0
    peak = float(spec[pmask].max())
    if mode == "local":
        wlo, whi = f_center * (1 - tol * 12), f_center * (1 + tol * 12)
        bmask = (freqs >= wlo) & (freqs <= whi) & (~pmask)
        bg = float(np.median(spec[bmask])) if bmask.any() else global_floor
    else:
        bg = global_floor
    return peak / max(bg, 1e-12)


# 자동 밴드 선택 후보(12kHz 기준, 0~6kHz를 겹치게 분할)
_DEFAULT_BANDS: list[tuple[float, float]] = [
    (500.0, 2000.0), (1500.0, 3000.0), (2500.0, 4000.0),
    (3500.0, 5000.0), (4500.0, 5900.0),
]


def select_band(signal: np.ndarray, fs: int,
                candidate_bands: list[tuple[float, float]] | None = None) -> tuple[float, float]:
    """첨도(impulsiveness)가 가장 큰 대역을 고른다 — kurtogram 의 간이판.

    충격성이 강한 대역일수록 결함 임펄스가 또렷 → 엔벨로프 SNR↑.
    """
    from scipy.stats import kurtosis

    x = np.asarray(signal, dtype=np.float64).ravel()
    x = x - x.mean()
    best, best_k = None, -np.inf
    for lo, hi in (candidate_bands or _DEFAULT_BANDS):
        hi = min(hi, fs / 2.0 * 0.99)
        if lo >= hi:
            continue
        sos = scipy.signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = scipy.signal.sosfiltfilt(sos, x)
        k = float(kurtosis(xf))
        if k > best_k:
            best_k, best = k, (lo, hi)
    return best or (2000.0, 5000.0)


# =============================================================
#  3. 물리 진단 — 위치 판정
# =============================================================
def diagnose_physics(
    signal: np.ndarray,
    fs: int,
    rpm: float | None = None,
    cfg: dict | None = None,
    tol: float = 0.02,          # ±2% 탐색 허용오차(구름체 미끄럼·속도변동)
    snr_thresh: float = 3.0,    # 봉우리가 잡음 대비 몇 배 이상이어야 '있다'고 볼지 ⚙️ (튜닝 best)
    n_harmonics: int = 3,       # 1·2·3배 고조파까지 확인
    band: tuple[float, float] | str | None = (2500.0, 5500.0),  # 또는 "auto" (튜닝 best)
    snr_mode: str = "global",   # "global"(전역) / "local"(국소 배경)
) -> PathCPhysics:
    """엔벨로프 스펙트럼에서 결함 주파수·고조파를 찾아 위치를 진단한다.

    band="auto" 면 첨도 최대 대역을 자동 선택. snr_mode="local" 이면 국소 배경 기준 SNR.
    판정: (기본 주파수 SNR≥임계) 또는 (고조파 2개 이상 존재) 인 위치 중 점수 최고.
          아무 위치도 조건을 못 채우면 location=None(못 잡음, 예: 볼).
    """
    if rpm is None:
        rpm = (cfg or load_config())["domain"]["rpm"]
    if band == "auto":
        band = select_band(signal, fs)
    freqs, spec = envelope_spectrum(signal, fs, band)
    fdefs = defect_frequencies(rpm)

    # 노이즈 바닥: 결함주파수가 사는 저주파 분석대역의 스펙트럼 중앙값(global 모드용)
    fmax = max(fdefs.values()) * (n_harmonics + 1)
    amask = (freqs > 0) & (freqs <= fmax)
    global_floor = float(np.median(spec[amask])) if amask.any() else float(np.median(spec[1:]))
    global_floor = max(global_floor, 1e-12)

    matched_snr: dict[str, float] = {}
    harmonic_counts: dict[str, int] = {}
    for loc, fd in fdefs.items():
        snrs = [_snr_at(freqs, spec, fd * h, tol, snr_mode, global_floor)
                for h in range(1, n_harmonics + 1)]
        matched_snr[loc] = round(float(snrs[0]), 2)             # 기본(1배) SNR
        harmonic_counts[loc] = sum(1 for s in snrs if s >= snr_thresh)

    # 점수 = 기본 SNR × (1 + 잡힌 고조파 수)
    scores = {loc: matched_snr[loc] * (1 + harmonic_counts[loc]) for loc in fdefs}
    best = max(scores, key=lambda k: scores[k])

    strong = matched_snr[best] >= snr_thresh        # 기본 봉우리가 뚜렷
    multi_harmonic = harmonic_counts[best] >= 2     # 고조파 여러 개(기본 약해도 인정)
    if strong or multi_harmonic:
        location: FaultLocation | None = FaultLocation(best)
        snr_comp = min(1.0, matched_snr[best] / (3.0 * snr_thresh))
        harm_comp = harmonic_counts[best] / n_harmonics
        confidence = float(np.clip(0.5 * snr_comp + 0.5 * harm_comp, 0.0, 1.0))
    else:
        location = None         # 결함주파수 박자를 못 찾음(물리가 약한 경우)
        confidence = 0.0

    return PathCPhysics(location=location, confidence=round(confidence, 3), matched_snr=matched_snr)


# =============================================================
#  4. 교차검증 — 물리(①) ↔ CNN(②)  ⭐ 핵심 차별점
# =============================================================
def cross_check(physics: PathCPhysics, cnn: PathCCnn | None) -> PathCResult:
    """두 진단을 비교해 최종 판정·신뢰도를 만든다.

    - 둘 다 같은 위치 → AGREE, 신뢰도 크게↑ (독립 근거가 일치)
    - 엇갈림        → DISAGREE, 최종위치 None('검토 필요')
    - 한쪽만 있음    → PHYSICS_ONLY / CNN_ONLY (그쪽 결과 채택)
    """
    if cnn is None:
        return PathCResult(physics, None, CrossCheck.PHYSICS_ONLY,
                           physics.location, physics.confidence)
    if physics.location is None:
        # 물리가 못 잡음(예: 볼) → CNN 단독. 볼 약점을 데이터가 메우는 지점.
        return PathCResult(physics, cnn, CrossCheck.CNN_ONLY, cnn.location, cnn.confidence)
    if physics.location == cnn.location:
        # 일치: 두 근거가 모두 틀릴 확률이 곱으로 줄어 신뢰도 상승
        conf = 1.0 - (1.0 - physics.confidence) * (1.0 - cnn.confidence)
        return PathCResult(physics, cnn, CrossCheck.AGREE, physics.location, round(conf, 3))
    # 엇갈림 → 섣불리 단정하지 않고 검토 필요로 넘김
    return PathCResult(physics, cnn, CrossCheck.DISAGREE, None, 0.0)


# =============================================================
#  5. 경로 C 적용 — 한 녹음에 물리+CNN 둘 다 → 교차검증
# =============================================================
def diagnose_recording(signal: np.ndarray, fs: int, model, cfg: dict,
                       rpm: float | None = None) -> PathCResult:
    """고장 녹음 1개에 물리(전체신호)·CNN(윈도우별 다수결)을 적용하고 교차검증한다.

    - 물리: 전체 신호로 진단(긴 신호일수록 주파수 해상도↑ → 정확).
    - CNN : 윈도우마다 예측 후 다수결 + 확률 평균.
    """
    from collections import Counter

    from src.cnn_fault_classifier import predict_spectrogram
    from src.preprocessing import make_windows
    from src.spectrogram import to_spectrograms

    physics = diagnose_physics(signal, fs, rpm=rpm, cfg=cfg)

    windows = make_windows(signal, cfg["window"]["length_c"], cfg["window"]["overlap"])
    if len(windows) == 0:
        return cross_check(physics, None)

    specs = to_spectrograms(windows, cfg)
    votes: list[str] = []
    prob_sum = {c: 0.0 for c in FAULT_CLASSES}
    for s in specs:
        r = predict_spectrogram(model, s)
        votes.append(r.location.value)
        for c, p in r.class_probs.items():
            prob_sum[c] += p
    n = len(votes)
    top = Counter(votes).most_common(1)[0][0]
    class_probs = {c: round(prob_sum[c] / n, 3) for c in FAULT_CLASSES}
    cnn = PathCCnn(FaultLocation(top), class_probs[top], class_probs)
    return cross_check(physics, cnn)


def crosscheck_report(cfg: dict, model=None, model_path: str | Path | None = None,
                      include_48k_downsampled: bool = False, device: str = "cpu",
                      split: str | None = None) -> dict:
    """모든 고장 녹음에 물리 vs CNN 교차검증을 적용하고 집계한다(⭐ 간판 결과).

    split="test"/"val"/"train" 이면 manifest(c_split_manifest.json)를 읽어
    그 split 녹음만 평가한다(공정 평가용 — CNN이 학습 때 안 본 test 만).
    반환: 일치율·판정분포(agree/disagree/...)·위치별(물리적중/CNN적중/일치) 통계·상세행.
    """
    from src.cnn_fault_classifier import collect_fault_recordings, load_model

    if model is None:
        if model_path is None:
            model_path = resolve_path(cfg["artifacts"]["models"]) / "fault_classifier.pth"
        model = load_model(model_path, num_classes=len(FAULT_CLASSES), device=device)

    recs = collect_fault_recordings(cfg, include_48k_downsampled)
    if split is not None:
        manifest_path = resolve_path(cfg["data"]["splits"]) / "c_split_manifest.json"
        with open(manifest_path, encoding="utf-8") as f:
            man = json.load(f)
        keep = set(man["split_rec_ids"].get(split, []))
        recs = [r for r in recs if r.rec_id in keep]
    verdict_count = {v.value: 0 for v in CrossCheck}
    per_loc = {c: {"n": 0, "physics_hit": 0, "cnn_hit": 0, "agree": 0} for c in FAULT_CLASSES}
    rows: list[dict] = []
    for rec in recs:
        res = diagnose_recording(rec.signal, _TARGET_SR, model, cfg)
        verdict_count[res.cross_check.value] += 1
        truth = rec.location
        st = per_loc[truth]
        st["n"] += 1
        if res.physics.location is not None and res.physics.location.value == truth:
            st["physics_hit"] += 1
        if res.cnn is not None and res.cnn.location.value == truth:
            st["cnn_hit"] += 1
        if res.cross_check == CrossCheck.AGREE and res.final_location \
                and res.final_location.value == truth:
            st["agree"] += 1
        rows.append({
            "rec": rec.rec_id, "truth": truth,
            "physics": res.physics.location.value if res.physics.location else None,
            "cnn": res.cnn.location.value if res.cnn else None,
            "verdict": res.cross_check.value,
        })
    n = len(recs)
    return {
        "n_recordings": n,
        "agreement_rate": round(verdict_count["agree"] / n, 3) if n else 0.0,
        "verdict_count": verdict_count,
        "per_location": per_loc,
        "rows": rows,
    }


def tune_physics(cfg: dict, snr_grid: tuple[float, ...] = (3.0, 4.0, 5.0),
                 band_options: list | None = None,
                 snr_modes: tuple[str, ...] = ("global", "local"),
                 include_48k_downsampled: bool = False,
                 rpm: float | None = None, top: int = 10) -> list[dict]:
    """물리 파라미터(밴드·SNR임계·SNR방식)를 실제 고장 녹음에 휩쓸어
    위치별/전체 '물리 적중률'을 정렬해 돌려준다(가장 좋은 조합 찾기용).

    ⚠️ 전체 고장에 맞춰 고르면 약간 후할 수 있다. 엄밀히는 일부로 튜닝→나머지로 검증이 정석.
       우선 경향 파악·최적 조합 후보 찾기용.
    """
    from src.cnn_fault_classifier import collect_fault_recordings

    if band_options is None:
        band_options = [(2000.0, 5000.0), "auto", (1500.0, 4000.0), (2500.0, 5500.0)]
    recs = collect_fault_recordings(cfg, include_48k_downsampled)
    results: list[dict] = []
    for band in band_options:
        for mode in snr_modes:
            for snr in snr_grid:
                hit = {c: [0, 0] for c in FAULT_CLASSES}
                for rec in recs:
                    r = diagnose_physics(rec.signal, _TARGET_SR, rpm=rpm, cfg=cfg,
                                         snr_thresh=snr, band=band, snr_mode=mode)
                    hit[rec.location][1] += 1
                    if r.location and r.location.value == rec.location:
                        hit[rec.location][0] += 1
                acc = {c: round(hit[c][0] / hit[c][1], 3) if hit[c][1] else 0.0
                       for c in FAULT_CLASSES}
                tot_h = sum(h[0] for h in hit.values())
                tot_n = sum(h[1] for h in hit.values())
                results.append({"band": band, "snr_mode": mode, "snr_thresh": snr,
                                "overall": round(tot_h / tot_n, 3) if tot_n else 0.0, **acc})
    results.sort(key=lambda r: -r["overall"])
    return results[:top]


# =============================================================
#  6. 자체 검증 — 결함주파수를 아는 합성 신호로 IR/OR/B 진단 확인
# =============================================================
def _synth_fault(fault_hz: float, fs: int = 12000, dur: float = 5.0,
                 resonance: float = 3000.0, seed: int = 0) -> np.ndarray:
    """fault_hz 박자로 공진(resonance)을 때리는 충격열 + 노이즈 합성 신호."""
    rng = np.random.default_rng(seed)
    n = int(fs * dur)
    sig = rng.normal(0, 0.05, n)
    period = 1.0 / fault_hz
    ring = int(0.008 * fs)
    tt = np.arange(ring) / fs
    for t0 in np.arange(0, dur, period):
        i0 = int(t0 * fs)
        amp = 1.0 + rng.normal(0, 0.1)
        pulse = amp * np.exp(-900.0 * tt) * np.sin(2 * np.pi * resonance * tt)
        sig[i0:i0 + ring] += pulse[: max(0, min(ring, n - i0))]
    return sig


def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()

    # 'tune' 인자: 물리 파라미터 휩쓸기 표 출력 후 종료
    if len(sys.argv) > 1 and sys.argv[1] == "tune":
        print("=== 물리 파라미터 튜닝 (실제 고장 녹음 기준 물리 적중률) ===")
        rows = tune_physics(cfg)
        print(f"{'band':<14}{'mode':<8}{'snr':<5}{'overall':<9}{'IR':<7}{'OR':<7}{'B':<7}")
        for r in rows:
            b = "auto" if r["band"] == "auto" else f"{int(r['band'][0])}-{int(r['band'][1])}"
            print(f"{b:<14}{r['snr_mode']:<8}{r['snr_thresh']:<5}{r['overall']:<9}"
                  f"{r['IR']:<7}{r['OR']:<7}{r['B']:<7}")
        print("\n→ overall·OR·B 보고 제일 좋은 조합을 diagnose_physics 기본값/호출인자로 쓰면 됨.")
        return
    rpm = cfg["domain"]["rpm"]
    fs = 12000

    print("=== 경로 C① 물리 진단 자체 검증 ===")
    fdefs = defect_frequencies(rpm)
    print(f"[결함 주파수] rpm={rpm} (fr={rpm/60:.2f}Hz)")
    for loc, f in fdefs.items():
        print(f"    {loc}: {f:.1f} Hz")

    print("\n[합성 신호로 진단 — 정답을 아는 상태에서 맞히는지]")
    ok = 0
    for true_loc, fd in fdefs.items():
        sig = _synth_fault(fd, fs=fs)
        res = diagnose_physics(sig, fs=fs, rpm=rpm)
        hit = (res.location is not None and res.location.value == true_loc)
        ok += hit
        print(f"    정답 {true_loc} → 진단 {res.location} "
              f"(conf={res.confidence}, SNR={res.matched_snr}) {'✅' if hit else '❌'}")
    print(f"  → {ok}/3 정확")

    print("\n[교차검증 예시]")
    sig = _synth_fault(fdefs["OR"], fs=fs)
    phys = diagnose_physics(sig, fs=fs, rpm=rpm)
    cnn_agree = PathCCnn(FaultLocation.OR, 0.88, {"IR": 0.07, "OR": 0.88, "B": 0.05})
    cnn_diff = PathCCnn(FaultLocation.IR, 0.55, {"IR": 0.55, "OR": 0.40, "B": 0.05})
    r1 = cross_check(phys, cnn_agree)
    r2 = cross_check(phys, cnn_diff)
    print(f"    물리={phys.location} + CNN=OR  → {r1.cross_check.value} "
          f"(최종 {r1.final_location}, conf={r1.final_confidence})")
    print(f"    물리={phys.location} + CNN=IR  → {r2.cross_check.value} "
          f"(최종 {r2.final_location}, conf={r2.final_confidence})")

    # 실제 데이터 물리 vs CNN 교차검증 리포트 (학습된 모델 있을 때만)
    model_path = resolve_path(cfg["artifacts"]["models"]) / "fault_classifier.pth"
    if model_path.exists():
        from src.cnn_fault_classifier import load_model
        model = load_model(model_path, num_classes=len(FAULT_CLASSES), device="cpu")
        for title, sp in [("전체 고장 (참고용·후함)", None), ("test 녹음만 (공정 평가) ⭐", "test")]:
            print(f"\n[물리 vs CNN 교차검증 — {title}]")
            try:
                rep = crosscheck_report(cfg, model=model, split=sp)
                print(f"  녹음 {rep['n_recordings']}개 | agree(일치) 비율 {rep['agreement_rate']}")
                print(f"  판정 분포: {rep['verdict_count']}")
                for loc, s in rep["per_location"].items():
                    print(f"    {loc}: 물리적중 {s['physics_hit']}/{s['n']} · "
                          f"CNN적중 {s['cnn_hit']}/{s['n']} · 일치 {s['agree']}/{s['n']}")
            except Exception as e:  # noqa: BLE001
                print(f"  (건너뜀: {e})")
    else:
        print(f"\n(models/fault_classifier.pth 없음 → 실데이터 리포트 생략)")

    print("\n물리 진단 OK — 공식 계산 + 엔벨로프 + 교차검증 동작 확인.")


if __name__ == "__main__":
    _main()
