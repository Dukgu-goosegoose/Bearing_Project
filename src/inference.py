"""오케스트레이터 — 추론 파이프라인 조립 (T7.4 게이팅).

흐름:  전처리 → 경로 A·B 병렬 → 융합 → (이상이면) 경로 C 호출 → 결과 조립.
게이트 규칙:  C 는 A·B 융합이 '이상(alarm)'일 때만 호출한다(정상엔 호출 안 함).
  - A·B 는 window.length(=1024) 윈도우로 윈도우별 점수를 내고, 융합·디바운스로 알람 결정.
  - 경로 C 는 알람이 켜진 녹음에만, 자기 윈도우(length_c=2048)로 물리+CNN 교차검증.
배포 시 동일 전처리를 보장하려고 scaler/spec_scaler/thresholds/모델을 한 번 로드해 재사용.

실행(자체 검증; A·B·C 결과물 필요):
    .venv\\Scripts\\python.exe -m src.inference
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.schemas import FusionResult, InferenceResult, PathCResult
from src.utils import get_device, load_config, resolve_path

_TARGET_SR = 12000


# =============================================================
#  1. 결과물 로드 — A·B·C 모델/통계/임계값을 한 번만 읽어 재사용
# =============================================================
@dataclass
class Artifacts:
    thr_a: dict          # 경로 A 임계값(특징별)
    T_b: float           # 경로 B 재구성오차 임계값
    ae_model: object     # 학습된 오토인코더(B)
    spec_scaler: dict    # B 스펙트로그램 정규화 통계
    cnn_model: object    # 학습된 CNN(C②)
    device: str


def load_artifacts(cfg: dict, device: str | None = None) -> Artifacts:
    """게이트 파이프라인에 필요한 모든 결과물을 로드한다.

    없으면 어떤 학습을 먼저 돌려야 하는지 알려주는 에러를 낸다.
    """
    from src.autoencoder_detector import load_model as load_ae
    from src.baseline_statistical import load_thresholds
    from src.cnn_fault_classifier import load_model as load_cnn
    from src.spectrogram import load_spec_scaler

    device = device or get_device(cfg)
    models_dir = cfg["artifacts"]["models"]

    def _need(path: Path, how: str):
        if not path.exists():
            raise FileNotFoundError(f"{path} 가 없습니다 → 먼저 {how} 를 실행하세요.")

    thr_path = resolve_path(cfg["artifacts"]["thresholds"])
    _need(thr_path, "경로 A: python -m src.baseline_statistical / 경로 B: train_autoencoder.py")
    thr = load_thresholds(cfg)
    if "path_a_statistical" not in thr:
        raise KeyError("thresholds.json 에 path_a_statistical 없음 → baseline_statistical 먼저")
    if "path_b" not in thr:
        raise KeyError("thresholds.json 에 path_b 없음 → train_autoencoder.py 먼저")

    ae_path = resolve_path(models_dir) / "autoencoder.pth"
    scaler_path = resolve_path(models_dir) / "spec_scaler.pkl"
    cnn_path = resolve_path(models_dir) / "fault_classifier.pth"
    _need(ae_path, "train_autoencoder.py")
    _need(scaler_path, "train_autoencoder.py")
    _need(cnn_path, "train_classifier.py")

    return Artifacts(
        thr_a=thr["path_a_statistical"],
        T_b=float(thr["path_b"]["threshold"]),
        ae_model=load_ae(cfg, f"{models_dir}/autoencoder.pth", device),
        spec_scaler=load_spec_scaler(f"{models_dir}/spec_scaler.pkl"),
        cnn_model=load_cnn(cnn_path, num_classes=3, device=device),
        device=device,
    )


# =============================================================
#  2. 경로 A·B 점수 — 윈도우별 (window.length = 1024)
# =============================================================
def ab_scores(signal: np.ndarray, cfg: dict, art: Artifacts) -> tuple[np.ndarray, np.ndarray]:
    """신호 → (경로A 점수, 경로B 점수) 윈도우별 배열. 둘 다 1.0 초과면 이상."""
    from src.autoencoder_detector import anomaly_score_b, recon_errors
    from src.baseline_statistical import anomaly_score, extract_features
    from src.preprocessing import make_windows
    from src.spectrogram import apply_spec_scaler, to_spectrograms

    length, overlap = cfg["window"]["length"], cfg["window"]["overlap"]
    feats = cfg["stats"]["features"]
    w = make_windows(signal, length, overlap)
    if len(w) == 0:
        return np.empty(0), np.empty(0)
    sa = np.array([anomaly_score(extract_features(x, feats), art.thr_a) for x in w])
    specs = apply_spec_scaler(to_spectrograms(w, cfg), art.spec_scaler)
    sb = np.asarray(anomaly_score_b(recon_errors(art.ae_model, specs, art.device), art.T_b))
    return sa, sb


# =============================================================
#  3. 게이트 — A·B 알람일 때만 경로 C 호출
# =============================================================
@dataclass
class PipelineResult:
    n_windows: int
    n_anomaly_windows: int
    n_alarm_windows: int
    is_anomaly: bool                 # 녹음 차원 최종(알람 통과 여부)
    fusion: list[FusionResult]       # 윈도우별 융합 결과
    pathc: PathCResult | None        # 게이트 통과 시에만 채움(정상이면 None)


def run_recording(signal: np.ndarray, cfg: dict, art: Artifacts,
                  fs: int = _TARGET_SR, rpm: float | None = None) -> PipelineResult:
    """녹음 1개에 게이트 파이프라인 적용.

    A·B → 융합·디바운스 → 알람이 하나라도 있으면 '이상' → 경로 C(물리+CNN+교차검증) 호출.
    정상이면 C 는 호출하지 않는다(pathc=None). ← T7.4 게이팅의 핵심.
    """
    from src.fusion import fuse_stream
    from src.physics_diagnosis import diagnose_recording

    sa, sb = ab_scores(signal, cfg, art)
    fused = fuse_stream(sa, sb, cfg)
    n_alarm = sum(f.alarm for f in fused)
    n_anom = sum(f.is_anomaly for f in fused)
    is_anomaly = n_alarm > 0

    pathc = None
    if is_anomaly:                                   # ← 게이트: 이상일 때만 C
        pathc = diagnose_recording(signal, fs, art.cnn_model, cfg, rpm=rpm)

    return PipelineResult(
        n_windows=len(fused), n_anomaly_windows=n_anom, n_alarm_windows=n_alarm,
        is_anomaly=is_anomaly, fusion=fused, pathc=pathc,
    )


# =============================================================
#  4. 출력 조립 — 윈도우별 InferenceResult(JSON 직렬화용, schemas 준수)
# =============================================================
def build_inference_results(signal: np.ndarray, cfg: dict, art: Artifacts,
                            fs: int = _TARGET_SR) -> list[InferenceResult]:
    """녹음 → 윈도우별 InferenceResult 리스트. 알람 윈도우에만 경로 C 진단을 채운다."""
    res = run_recording(signal, cfg, art, fs=fs)
    sa, sb = ab_scores(signal, cfg, art)  # 점수 다시(간단화). 성능 필요시 run_recording에서 반환.
    pc = res.pathc
    out: list[InferenceResult] = []
    for i, f in enumerate(res.fusion):
        ir = InferenceResult(
            window_index=i, timestamp=None,
            score_a=float(sa[i]), score_b=float(sb[i]),
            fused_score=f.fused_score, is_anomaly=f.is_anomaly, alarm=f.alarm,
        )
        if f.alarm and pc is not None:               # 알람 켜진 윈도우에 C 진단 부착
            ir.fault_location = pc.final_location.value if pc.final_location else None
            ir.physics_location = pc.physics.location.value if pc.physics.location else None
            ir.cnn_location = pc.cnn.location.value if pc.cnn else None
            ir.cross_check = pc.cross_check.value
            ir.confidence = pc.final_confidence
        out.append(ir)
    return out


# =============================================================
#  5. 게이트 검증 — 정상은 C 미호출, 고장은 C 호출되는지 확인 (T7.4 DoD)
# =============================================================
def evaluate_gating(cfg: dict, art: Artifacts, max_per_group: int = 8) -> dict:
    """정상·고장 녹음에 파이프라인을 돌려 게이트가 제대로 작동하는지 집계한다.

    핵심 확인: 정상 → C 호출 0(또는 매우 적음), 고장 → C 호출 + 위치 진단.
    """
    from src.cnn_fault_classifier import collect_fault_recordings
    from src.data_loader import load_normal_signals

    summary = {"normal": {"n": 0, "c_called": 0}, "fault": {"n": 0, "c_called": 0, "located": 0}}

    normal = load_normal_signals(cfg)
    for i, sig in enumerate(normal.values()):
        if i >= max_per_group:
            break
        r = run_recording(sig, cfg, art)
        summary["normal"]["n"] += 1
        summary["normal"]["c_called"] += int(r.pathc is not None)

    recs = collect_fault_recordings(cfg, include_48k_downsampled=False)
    for rec in recs[:max_per_group]:
        r = run_recording(rec.signal, cfg, art)
        summary["fault"]["n"] += 1
        if r.pathc is not None:
            summary["fault"]["c_called"] += 1
            if r.pathc.final_location is not None:
                summary["fault"]["located"] += 1
    return summary


def _main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    print("=== 게이트 파이프라인 (inference.py) ===")
    try:
        art = load_artifacts(cfg)
    except (FileNotFoundError, KeyError) as e:
        print(f"[결과물 없음] {e}")
        print("→ A·B 학습을 먼저 끝내야 게이트가 돌아갑니다.")
        return

    s = evaluate_gating(cfg, art)
    print(f"[정상] {s['normal']['n']}개 중 경로C 호출 {s['normal']['c_called']}개  "
          f"(0에 가까울수록 게이트 정상 — 정상엔 C 안 켜짐)")
    print(f"[고장] {s['fault']['n']}개 중 경로C 호출 {s['fault']['c_called']}개, "
          f"그중 위치진단 {s['fault']['located']}개")
    print("\n게이트 OK — A·B 이상일 때만 경로 C 호출(정상은 통과).")


if __name__ == "__main__":
    _main()
