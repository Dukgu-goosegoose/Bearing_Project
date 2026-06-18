"""공통 함수 — config 로드, device 선택, 시드 고정, 경로 처리.


다른 모듈에서 공통으로 쓰는 헬퍼를 한 곳에 모은다.
    from src.utils import load_config, get_device, set_seed, resolve_path
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

# 이 파일(src/utils.py) 기준으로 프로젝트 최상위 폴더를 찾는다.
#   src/utils.py -> .parent = src -> .parent = 프로젝트 루트
# 어느 위치에서 실행하든 경로가 틀어지지 않게 한다.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """configs/config.yaml 을 읽어 dict 로 반환한다.

    path 를 안 주면 기본 위치(configs/config.yaml)를 읽는다.
    """
    if path is None:
        path = PROJECT_ROOT / "configs" / "config.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(cfg: dict) -> str:
    """config 의 device 설정을 실제 장치 문자열로 바꾼다.

    'auto' 면 GPU가 있으면 'cuda', 없으면 'cpu'. 그 외엔 설정값 그대로.
    """
    import torch

    setting = cfg.get("device", "auto")
    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


def set_seed(seed: int) -> None:
    """난수 시드를 고정한다(재현성). numpy·random·torch 모두 적용."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        # torch 가 아직 없어도 numpy/random 시드는 고정된다
        pass


def resolve_path(p: str | Path) -> Path:
    """config 의 상대경로(data/raw/normal 등)를 프로젝트 루트 기준 절대경로로."""
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p
