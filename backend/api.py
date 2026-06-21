"""FastAPI 백엔드 — 진동 신호를 받아 게이트 파이프라인(A·B→융합→C)을 돌려 JSON 응답.

프론트엔드(frontend/index.html)가 호출하는 엔드포인트:
  GET  /                 → 서버 상태 + 기계 프로필 + 클래스 (연결 확인용)
  GET  /examples         → 데모용 예시 신호 목록(정상/고장)
  POST /diagnose_example → {name} 예시 1개 진단
  POST /diagnose         → {signal:[...]} 원시 신호 배열 진단
  POST /diagnose_mat     → .mat 파일 업로드 진단 (새 파일 분석 — 범용 .mat 분석기)

실행:
    .venv\\Scripts\\python.exe -m uvicorn backend.api:app --reload
필요 패키지: fastapi, uvicorn, python-multipart  (requirements 에 없으면 pip install)
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path

app = FastAPI(title="베어링 고장 진단 API")
# 프론트가 file:// 로 열려 127.0.0.1:8000 을 부르므로 CORS 허용
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---- 전역 상태(서버 시작 시 1회 로드) ----
_CFG = load_config()
_ART = None                         # 학습 산출물(지연 로드)
_EXAMPLES: dict[str, dict] = {}     # {name: {"signal":array, "type":..., "truth":...}}


def _get_art():
    """학습 산출물(AE·scaler·임계값·CNN)을 1회만 로드. 없으면 503."""
    global _ART
    if _ART is None:
        from src.inference import load_artifacts
        try:
            _ART = load_artifacts(_CFG)
        except FileNotFoundError as e:
            raise HTTPException(503, f"학습 산출물 없음 — 먼저 학습을 돌리세요: {e}")
    return _ART


def _load_examples() -> None:
    """예시 신호 캐시(정상 전부 + 위치별 고장 일부)를 1회 구성."""
    global _EXAMPLES
    if _EXAMPLES:
        return
    from src.cnn_fault_classifier import collect_fault_recordings
    from src.data_loader import load_normal_signals

    for name, sig in load_normal_signals(_CFG).items():
        _EXAMPLES[name] = {"signal": np.asarray(sig), "type": "normal", "truth": "정상"}

    # 위치별로 몇 개만 노출(데모가 너무 길어지지 않게)
    per_loc_cap, seen = 3, {}
    for rec in collect_fault_recordings(_CFG, include_48k_downsampled=False):
        loc = rec.location
        if seen.get(loc, 0) >= per_loc_cap:
            continue
        seen[loc] = seen.get(loc, 0) + 1
        _EXAMPLES[rec.rec_id] = {"signal": np.asarray(rec.signal), "type": "fault", "truth": loc}


def _preview(signal: np.ndarray, n: int = 600) -> list[float]:
    """파형 캔버스용으로 신호를 n점 이하로 솎아 반환."""
    x = np.asarray(signal, dtype=float).ravel()
    if len(x) <= n:
        return x.tolist()
    idx = np.linspace(0, len(x) - 1, n).astype(int)
    return x[idx].tolist()


def _result_json(res, name: str, signal: np.ndarray, truth: str | None) -> dict:
    """PipelineResult → 프론트가 기대하는 JSON 모양으로 변환."""
    diagnosis = None
    pc = res.pathc
    if pc is not None:
        diagnosis = {
            "fault_location": pc.final_location.value if pc.final_location else None,
            "confidence": pc.final_confidence,
            "cross_check": pc.cross_check.value,
            "physics_location": pc.physics.location.value if pc.physics.location else None,
            "cnn_location": pc.cnn.location.value if pc.cnn else None,
        }
    return {
        "name": name,
        "truth": truth,
        "is_anomaly": res.is_anomaly,
        "n_windows": res.n_windows,
        "n_alarm": res.n_alarm_windows,
        "diagnosis": diagnosis,
        "preview": _preview(signal),
    }


def _run(signal: np.ndarray, name: str, truth: str | None) -> dict:
    from src.inference import run_recording
    res = run_recording(np.asarray(signal, dtype=float), _CFG, _get_art())
    return _result_json(res, name, signal, truth)


# =============================================================
#  엔드포인트
# =============================================================
@app.get("/")
def root() -> dict:
    return {
        "status": "ok",
        "machine": _CFG.get("machine") or "CWRU(기본)",
        "classes": _CFG["cnn"]["classes"],
    }


@app.get("/examples")
def examples() -> dict:
    _load_examples()
    return {"examples": [
        {"name": n, "type": v["type"], "truth": v["truth"]}
        for n, v in _EXAMPLES.items()
    ]}


class ExampleReq(BaseModel):
    name: str


@app.post("/diagnose_example")
def diagnose_example(req: ExampleReq) -> dict:
    _load_examples()
    ex = _EXAMPLES.get(req.name)
    if ex is None:
        raise HTTPException(404, f"예시 없음: {req.name}")
    return _run(ex["signal"], req.name, ex["truth"])


class SignalReq(BaseModel):
    signal: list[float]
    name: str | None = None


@app.post("/diagnose")
def diagnose(req: SignalReq) -> dict:
    if not req.signal:
        raise HTTPException(400, "빈 신호")
    return _run(np.array(req.signal), req.name or "업로드 신호", None)


@app.post("/diagnose_mat")
async def diagnose_mat(file: UploadFile = File(...)) -> dict:
    """업로드된 .mat 에서 가속도 신호를 뽑아 진단(범용 .mat 분석)."""
    import scipy.io as sio

    if not file.filename.endswith(".mat"):
        raise HTTPException(400, ".mat 파일만 업로드 가능")
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    data = sio.loadmat(tmp_path)
    # 설정된 가속도 키(예: DE_time)로 끝나는 변수 우선, 없으면 가장 긴 1D 수치배열
    suffix = _CFG["data"]["normal_mat_key"]
    keys = [k for k in data if not k.startswith("__") and k.endswith(suffix)]
    if not keys:
        cands = [(k, v) for k, v in data.items()
                 if not k.startswith("__") and isinstance(v, np.ndarray) and v.size > 100]
        if not cands:
            raise HTTPException(400, "가속도 신호를 .mat 에서 찾지 못했습니다")
        key = max(cands, key=lambda kv: kv[1].size)[0]
    else:
        key = keys[0]
    sig = np.asarray(data[key], dtype=float).ravel()
    return _run(sig, file.filename, None)
