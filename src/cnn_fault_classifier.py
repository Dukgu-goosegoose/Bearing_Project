"""경로 C — 고장 유형(위치) CNN 분류기 (지도학습, 참고 보조).

이 파일은 경로 C 전체를 담는다:
  (1) 데이터셋 빌드 + PyTorch Dataset   ← 현재 구현됨(아래)
  (2) MobileNetV2 분류 모델 + 학습/평가 ← 다음 단계에서 이 파일 아래에 추가

데이터 흐름:
  12k 고장 로드 → 위치 라벨(IR/OR/B, 3클래스) → length_c(2048) 윈도우
    → 스펙트로그램 → 녹음 단위 train/val/test 분할 → .npy 저장
  학습 시 FaultSpectrogramDataset 가 스펙트로그램을 224×224·3채널 텐서로 변환.

핵심 원칙:
- 위치 3클래스만 분류한다(크기 0.007/0.014/0.021/0.028 무시).
  → 물리 진단(엔벨로프)도 '위치'만 판별하므로, 교차검증 단위를 위치로 맞춘다.
- 녹음(파일) 단위로 분할한다. 같은 녹음의 조각이 train/test 양쪽에 섞이면
  모델이 '패턴'이 아니라 '그 녹음'을 외워 점수가 가짜로 높아진다(데이터 누수).
- 12k 네이티브(DriveEnd_fault_12k)만 우선 사용. 48k→12k 다운샘플은 옵션(증강용).
- Fan End 고장(FE 센서)은 경로 C 학습에서 제외한다(eval 전용·센서 다름).

실행(데이터셋 빌드 검증):
    .venv\\Scripts\\python.exe -m src.cnn_fault_classifier
"""
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import scipy.signal

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_fault_signals
from src.preprocessing import make_windows
from src.schemas import FAULT_CLASSES, FaultLocation, PathCCnn  # 위치 3클래스 + 추론 결과 타입
from src.spectrogram import to_spectrograms
from src.utils import load_config, resolve_path, set_seed

# 위치 문자열 → 정수 라벨 (CNN 출력 인덱스). schemas 의 순서를 그대로 따른다.
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(FAULT_CLASSES)}

# ImageNet 사전학습 MobileNetV2 입력 정규화 상수(전이학습 관례).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_TARGET_SR = 12000  # 경로 C 기준 샘플링


# =============================================================
#  하나의 '녹음' = .mat 파일 1개 (분할의 최소 단위)
# =============================================================
@dataclass
class Recording:
    rec_id: str          # 고유 이름 예) 'fault_12k/IR007_0'
    signal: np.ndarray   # (L,) 12kHz 신호
    location: str        # 'IR' / 'OR' / 'B'


def _downsample_to_12k(signal: np.ndarray, orig_sr: int) -> np.ndarray:
    """48kHz 등 고샘플링 신호를 12kHz로 안티앨리어싱 다운샘플한다.

    그냥 솎으면(슬라이싱) 가짜 주파수(앨리어싱)가 생기므로 decimate(FIR)를 쓴다.
    """
    if orig_sr == _TARGET_SR:
        return np.asarray(signal, dtype=np.float64)
    q = orig_sr // _TARGET_SR  # 48000 // 12000 = 4
    if orig_sr % _TARGET_SR != 0:
        raise ValueError(f"{orig_sr} 는 {_TARGET_SR} 의 정수배가 아님 — decimate 불가")
    return scipy.signal.decimate(
        np.asarray(signal, dtype=np.float64), q, ftype="fir", zero_phase=True
    )


# =============================================================
#  1. 고장 녹음 수집 (12k 네이티브 + 옵션으로 48k→12k)
# =============================================================
def collect_fault_recordings(
    cfg: dict, include_48k_downsampled: bool = False
) -> list[Recording]:
    """경로 C 학습용 고장 녹음들을 모은다(전부 12kHz 로 통일).

    - fault_12k     : 그대로 사용
    - fault_48k     : include_48k_downsampled=True 일 때만, 12k 로 내려서 사용
    - fan_end_fault : 제외(FE 센서·eval 전용)
    - 위치(location) 없는 파일(UNKNOWN)도 제외
    """
    fault = load_fault_signals(cfg)
    recordings: list[Recording] = []
    for key, info in fault.items():
        loc = info["location"]
        src = info["source"]
        sr = info["sampling_rate"]
        if loc is None:
            continue  # 라벨 파싱 실패(UNKNOWN) → 학습 제외
        if src == "fan_end_fault":
            continue  # FE 센서 → 경로 C 학습 제외
        if src == "fault_12k":
            sig = np.asarray(info["signal"], dtype=np.float64)
        elif src == "fault_48k":
            if not include_48k_downsampled:
                continue
            sig = _downsample_to_12k(info["signal"], sr)
        else:
            continue  # 정의되지 않은 소스는 안전하게 건너뜀
        recordings.append(Recording(rec_id=key, signal=sig, location=loc))
    return recordings


# =============================================================
#  2. 녹음 단위 train/val/test 분할 (층화 = 클래스 비율 유지)
# =============================================================
def split_recordings(
    recordings: list[Recording],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """녹음(rec_id)을 위치별로 묶어 train/val/test 로 나눈다.

    - 위치별로 따로 나눠 각 split 에 IR/OR/B 가 모두 들어가게 한다(층화).
    - 한 녹음은 한 split 에만 속한다 → 조각 단위 누수 차단.
    반환: {'train': [rec_id,...], 'val': [...], 'test': [...]}
    """
    by_loc: dict[str, list[str]] = {}
    for rec in recordings:
        by_loc.setdefault(rec.location, []).append(rec.rec_id)

    rng = random.Random(seed)
    split: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for loc, ids in sorted(by_loc.items()):
        ids = sorted(ids)       # 재현성 위해 먼저 정렬
        rng.shuffle(ids)
        n = len(ids)
        if n < 3:
            # 녹음이 너무 적으면 전부 train (val/test 못 뗌) — 경고
            print(f"  ⚠️ '{loc}' 녹음 {n}개뿐 → 전부 train (val/test 분할 불가)")
            split["train"].extend(ids)
            continue
        n_test = max(1, round(n * test_ratio))
        n_val = max(1, round(n * val_ratio))
        # train 이 최소 1개는 남도록 보정
        if n_test + n_val >= n:
            n_test, n_val = 1, 1
        split["test"].extend(ids[:n_test])
        split["val"].extend(ids[n_test:n_test + n_val])
        split["train"].extend(ids[n_test + n_val:])
    return split


# =============================================================
#  3. 녹음 → 윈도우 → 스펙트로그램 (split 별)
# =============================================================
def _windows_and_labels(
    recordings: list[Recording],
    rec_ids: list[str],
    length: int,
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    """주어진 rec_id 들의 신호를 윈도잉하고, 윈도우마다 위치 라벨을 붙인다.

    반환: (windows (N, L),  labels (N,) int)
    """
    id_set = set(rec_ids)
    win_chunks: list[np.ndarray] = []
    lab_chunks: list[np.ndarray] = []
    for rec in recordings:
        if rec.rec_id not in id_set:
            continue
        w = make_windows(rec.signal, length, overlap)  # (n_i, L)
        if len(w) == 0:
            print(f"  ⚠️ {rec.rec_id}: 신호가 윈도우({length})보다 짧아 0개")
            continue
        win_chunks.append(w)
        lab_chunks.append(np.full(len(w), CLASS_TO_IDX[rec.location], dtype=np.int64))
    if not win_chunks:
        return np.empty((0, length), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.concatenate(win_chunks, axis=0), np.concatenate(lab_chunks, axis=0)


def build_dataset(
    cfg: dict,
    include_48k_downsampled: bool = False,
    save: bool = True,
) -> dict:
    """경로 C 데이터셋을 만든다: 수집 → 분할 → 윈도우 → 스펙트로그램 → 저장.

    반환: 요약 dict (split 별 shape·클래스 분포·저장 경로 등).
    """
    seed = cfg["seed"]
    set_seed(seed)
    length = cfg["window"]["length_c"]      # 2048 (12kHz, ≈5회전)
    overlap = cfg["window"]["overlap"]
    val_ratio = cfg["cnn"].get("val_ratio", 0.2)
    test_ratio = cfg["cnn"].get("test_ratio", 0.15)

    # (1) 고장 녹음 수집
    recordings = collect_fault_recordings(cfg, include_48k_downsampled)
    if not recordings:
        raise RuntimeError("고장 녹음 0개 — data/raw/fault_12k 경로·파일명을 확인하세요")

    # (2) 녹음 단위 분할
    split = split_recordings(recordings, val_ratio, test_ratio, seed)

    # (3) split 별 윈도우 → 스펙트로그램 → 저장
    feat_dir = resolve_path(cfg["data"]["features"])
    split_dir = resolve_path(cfg["data"]["splits"])
    if save:
        feat_dir.mkdir(parents=True, exist_ok=True)
        split_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {"splits": {}, "class_order": FAULT_CLASSES}
    for name in ("train", "val", "test"):
        windows, labels = _windows_and_labels(recordings, split[name], length, overlap)
        specs = to_spectrograms(windows, cfg).astype(np.float32)  # (N,1,H,W)
        # 클래스 분포
        dist = {FAULT_CLASSES[i]: int((labels == i).sum()) for i in range(len(FAULT_CLASSES))}
        summary["splits"][name] = {
            "n_windows": int(specs.shape[0]),
            "spec_shape": tuple(specs.shape[1:]) if specs.shape[0] else None,
            "class_dist": dist,
            "n_recordings": len(split[name]),
        }
        if save and specs.shape[0] > 0:
            np.save(feat_dir / f"c_X_{name}.npy", specs)
            np.save(feat_dir / f"c_y_{name}.npy", labels)

    # (4) 분할 명세·클래스 가중치 저장(재현성·불균형 대응)
    summary["class_weights"] = _class_weights_from_summary(summary)
    if save:
        manifest = {
            "seed": seed,
            "length_c": length,
            "overlap": overlap,
            "include_48k_downsampled": include_48k_downsampled,
            "class_order": FAULT_CLASSES,
            "split_rec_ids": split,
            "summary": summary["splits"],
            "class_weights": summary["class_weights"],
        }
        with open(split_dir / "c_split_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        summary["manifest_path"] = str((split_dir / "c_split_manifest.json").relative_to(_ROOT))
    return summary


def _class_weights_from_summary(summary: dict) -> dict[str, float]:
    """train 클래스 분포로 역빈도 가중치를 계산한다(불균형 대응, 학습에서 사용).

    가중치 = 전체수 / (클래스수 × 해당클래스 표본수). 평균 1 근처로 정규화.
    """
    dist = summary["splits"].get("train", {}).get("class_dist", {})
    total = sum(dist.values())
    k = len([v for v in dist.values() if v > 0])
    if total == 0 or k == 0:
        return {c: 1.0 for c in FAULT_CLASSES}
    weights = {}
    for c in FAULT_CLASSES:
        n = dist.get(c, 0)
        weights[c] = (total / (k * n)) if n > 0 else 0.0
    return weights


# =============================================================
#  4. 스펙트로그램 → CNN 입력 이미지 (224×224·3채널)
#     순수 numpy + OpenCV. (torch 없이도 테스트 가능)
# =============================================================
def spectrogram_to_image(
    spec: np.ndarray, size: int = 224, imagenet_norm: bool = True
) -> np.ndarray:
    """스펙트로그램 1장 (1,H,W) 또는 (H,W) → (3, size, size) float32 이미지.

    단계: 이미지별 min-max [0,1] → OpenCV 리사이즈 → 1ch 복제 3ch → (ImageNet 정규화).
    이미지별 정규화는 누수 걱정 없이 밝기 스케일만 맞춘다(위치 패턴은 보존).
    """
    s = np.asarray(spec, dtype=np.float32).squeeze()  # (H,W)
    mn, mx = float(s.min()), float(s.max())
    s = (s - mn) / (mx - mn + 1e-8)                    # [0,1]
    img = cv2.resize(s, (size, size), interpolation=cv2.INTER_LINEAR)  # (size,size)
    img = np.stack([img, img, img], axis=0)            # (3,size,size)
    if imagenet_norm:
        img = (img - _IMAGENET_MEAN[:, None, None]) / _IMAGENET_STD[:, None, None]
    return img.astype(np.float32)


# torch 가 없어도 이 모듈의 빌드 함수는 import·실행되도록 안전하게 감싼다.
try:
    import torch
    from torch.utils.data import Dataset as _TorchDataset
    _HAS_TORCH = True
except ImportError:  # 학습 단계에서만 torch 필요
    _TorchDataset = object  # type: ignore
    _HAS_TORCH = False


class FaultSpectrogramDataset(_TorchDataset):
    """저장된 스펙트로그램(.npy)을 받아 224×224·3채널 텐서를 내주는 Dataset.

    X: (N,1,H,W) 스펙트로그램,  y: (N,) 정수 라벨
    __getitem__ → (torch.FloatTensor (3,224,224),  int 라벨)
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        size: int = 224,
        imagenet_norm: bool = True,
    ) -> None:
        if not _HAS_TORCH:
            raise ImportError("FaultSpectrogramDataset 는 torch 가 필요합니다(학습 단계).")
        assert len(X) == len(y), "X, y 길이 불일치"
        self.X = X
        self.y = np.asarray(y, dtype=np.int64)
        self.size = size
        self.imagenet_norm = imagenet_norm

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        img = spectrogram_to_image(self.X[i], self.size, self.imagenet_norm)
        return torch.from_numpy(img), int(self.y[i])


# =============================================================
#  5. 모델 — MobileNetV2 전이학습 (출력 3클래스 IR/OR/B)
# =============================================================
def build_model(
    num_classes: int = 3,
    pretrained: bool = True,
    freeze_backbone: bool = True,
    device: str = "cpu",
):
    """사전학습 MobileNetV2 를 불러와 분류 머리를 num_classes 로 교체한다.

    - pretrained=True : ImageNet 사전학습 가중치. 녹음 수가 적어 전이학습이 유리(권장).
    - freeze_backbone=True : 특징추출부(backbone)는 얼리고 머리만 학습(가벼운 미세조정).
      데이터가 적어 통째로 학습하면 과적합하므로 기본 True.
    """
    if not _HAS_TORCH:
        raise ImportError("build_model 은 torch/torchvision 이 필요합니다(학습 단계).")
    from torchvision import models

    if pretrained:
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
        model = models.mobilenet_v2(weights=weights)
    else:
        model = models.mobilenet_v2(weights=None)  # 가중치 다운로드 없음(테스트/오프라인용)

    if freeze_backbone:
        for p in model.features.parameters():
            p.requires_grad = False  # backbone 동결 → 분류 머리만 학습

    in_features = model.classifier[1].in_features            # 1280
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)  # 1000 → 3
    return model.to(device)


def save_model(model, path: str | Path) -> None:
    """학습된 가중치를 저장한다 (models/fault_classifier.pth)."""
    p = resolve_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), p)


def load_model(path: str | Path, num_classes: int = 3, device: str = "cpu"):
    """저장된 분류기를 불러온다(추론·배포용)."""
    model = build_model(num_classes, pretrained=False, freeze_backbone=False, device=device)
    model.load_state_dict(torch.load(resolve_path(path), map_location=device))
    model.eval()
    return model


# =============================================================
#  6. 데이터로더 — 저장된 .npy → DataLoader
# =============================================================
def _load_split(cfg: dict, name: str) -> tuple[np.ndarray, np.ndarray]:
    feat = resolve_path(cfg["data"]["features"])
    X = np.load(feat / f"c_X_{name}.npy")
    y = np.load(feat / f"c_y_{name}.npy")
    return X, y


def make_loaders(cfg: dict, batch_size: int | None = None) -> dict:
    """train/val/test DataLoader 를 만든다(스펙트로그램→224×224·3채널은 Dataset 이 처리)."""
    if not _HAS_TORCH:
        raise ImportError("make_loaders 는 torch 가 필요합니다(학습 단계).")
    from torch.utils.data import DataLoader

    bs = batch_size or cfg["cnn"]["batch_size"]
    loaders: dict = {}
    for name, shuffle in (("train", True), ("val", False), ("test", False)):
        X, y = _load_split(cfg, name)
        ds = FaultSpectrogramDataset(X, y)
        loaders[name] = DataLoader(ds, batch_size=bs, shuffle=shuffle)
    return loaders


# =============================================================
#  7. 학습 — 클래스 가중치 손실 + 조기종료
# =============================================================
def _run_epoch(model, loader, criterion, optimizer, device: str, train: bool,
               desc: str = "") -> tuple[float, float]:
    """1 에폭 실행. 반환: (평균 손실, 정확도). desc 주면 tqdm 진행바 표시."""
    from tqdm import tqdm

    model.train() if train else model.eval()
    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    bar = tqdm(loader, desc=desc, leave=False, unit="batch")  # 배치 진행바
    with ctx:
        for xb, yb in bar:
            xb, yb = xb.to(device), yb.to(device)
            if train:
                optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(yb)
            correct += int((out.argmax(1) == yb).sum().item())
            n += len(yb)
            bar.set_postfix(loss=f"{total_loss/n:.3f}", acc=f"{correct/n:.3f}")  # 실시간 표시
    return (total_loss / n, correct / n) if n else (0.0, 0.0)


def train_model(
    model, loaders: dict, cfg: dict, class_weights: dict[str, float],
    device: str = "cpu", patience: int = 5,
) -> tuple[object, list[dict], float]:
    """전이학습 학습 루프. val 정확도 기준 조기종료 + 최상 가중치 복원."""
    epochs = cfg["cnn"]["epochs"]
    lr = cfg["cnn"]["lr"]
    # 클래스 불균형 보정: schemas 순서대로 가중치 벡터 구성
    w = torch.tensor([class_weights[c] for c in FAULT_CLASSES], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=w)
    params = [p for p in model.parameters() if p.requires_grad]  # 머리만 학습
    optimizer = torch.optim.Adam(params, lr=lr)

    best_acc, best_state, wait = -1.0, None, 0
    history: list[dict] = []
    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, loaders["train"], criterion, optimizer, device, True,
                                     desc=f"epoch {ep:2d}/{epochs} [train]")
        va_loss, va_acc = _run_epoch(model, loaders["val"], criterion, optimizer, device, False,
                                     desc=f"epoch {ep:2d}/{epochs} [val]  ")
        history.append({"epoch": ep, "train_acc": tr_acc, "val_acc": va_acc,
                        "train_loss": tr_loss, "val_loss": va_loss})
        print(f"  epoch {ep:2d}: train acc {tr_acc:.3f} | val acc {va_acc:.3f}")
        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"  조기종료(val {patience}회 개선 없음)")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_acc


# =============================================================
#  8. 평가 — 정확도 + 혼동행렬
# =============================================================
def evaluate(model, loader, device: str = "cpu") -> dict:
    """테스트셋 평가. 정확도·혼동행렬·클래스별 재현율 반환."""
    model.eval()
    k = len(FAULT_CLASSES)
    cm = np.zeros((k, k), dtype=int)  # cm[정답][예측]
    with torch.no_grad():
        for xb, yb in loader:
            pred = model(xb.to(device)).argmax(1).cpu().numpy()
            for t, p in zip(yb.numpy(), pred):
                cm[t, p] += 1
    acc = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    recall = {FAULT_CLASSES[i]: (float(cm[i, i] / cm[i].sum()) if cm[i].sum() else 0.0)
              for i in range(k)}
    return {"accuracy": acc, "confusion_matrix": cm.tolist(),
            "per_class_recall": recall, "labels": FAULT_CLASSES}


# =============================================================
#  9. 추론 — 스펙트로그램 1장 → PathCCnn (물리 교차검증 입력)
# =============================================================
def predict_spectrogram(model, spec: np.ndarray, device: str = "cpu",
                        imagenet_norm: bool = True) -> PathCCnn:
    """스펙트로그램 (1,H,W) → 경로 C② CNN 진단 결과(PathCCnn).

    schemas.PathCCnn 으로 반환하므로, 이후 물리 진단(PathCPhysics)과 바로 교차검증 가능.
    """
    model.eval()
    img = spectrogram_to_image(spec, imagenet_norm=imagenet_norm)  # (3,224,224)
    x = torch.from_numpy(img)[None].to(device)                     # (1,3,224,224)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
    class_probs = {FAULT_CLASSES[i]: float(probs[i]) for i in range(len(FAULT_CLASSES))}
    top = int(probs.argmax())
    return PathCCnn(
        location=FaultLocation(FAULT_CLASSES[top]),
        confidence=float(probs[top]),
        class_probs=class_probs,
    )


# =============================================================
#  10. 오케스트레이션 — 데이터셋→학습→평가→저장 (train_classifier.py 가 호출)
# =============================================================
def run_training(cfg: dict, include_48k_downsampled: bool = False,
                 device: str | None = None) -> tuple[object, dict, list[dict]]:
    """경로 C 전체 학습 파이프라인. 반환: (model, test_metrics, history)."""
    if not _HAS_TORCH:
        raise ImportError("run_training 은 torch/torchvision 이 필요합니다.")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    summary = build_dataset(cfg, include_48k_downsampled, save=True)  # 1) 데이터셋
    loaders = make_loaders(cfg)                                       # 2) 로더
    model = build_model(len(FAULT_CLASSES), pretrained=True,          # 3) 모델
                        freeze_backbone=True, device=device)
    model, history, best_val = train_model(                          # 4) 학습
        model, loaders, cfg, summary["class_weights"], device)
    metrics = evaluate(model, loaders["test"], device)               # 5) 평가
    model_path = resolve_path(cfg["artifacts"]["models"]) / "fault_classifier.pth"
    save_model(model, model_path)                                    # 6) 저장
    print(f"\n[저장] {model_path.relative_to(_ROOT)}  | best val acc={best_val:.3f}")
    print(f"[test] 정확도={metrics['accuracy']:.3f}  클래스별 재현율={metrics['per_class_recall']}")
    return model, metrics, history


def _main() -> None:
    """T7 검증: 데이터셋 빌드 + 단계별 shape·클래스 분포·누수 차단 확인."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()

    print("=== T7 경로 C 데이터셋 빌더 ===")
    print(f"설정: length_c={cfg['window']['length_c']}, overlap={cfg['window']['overlap']}, "
          f"클래스={FAULT_CLASSES}")

    recordings = collect_fault_recordings(cfg, include_48k_downsampled=False)
    print(f"\n[1] 고장 녹음 수집(12k) → {len(recordings)}개")
    loc_count: dict[str, int] = {}
    for r in recordings:
        loc_count[r.location] = loc_count.get(r.location, 0) + 1
    for loc in FAULT_CLASSES:
        print(f"    - {loc}: {loc_count.get(loc, 0)}개 녹음")

    summary = build_dataset(cfg, include_48k_downsampled=False, save=True)

    print("\n[2] split 별 결과 (녹음 단위 분할)")
    for name in ("train", "val", "test"):
        s = summary["splits"][name]
        print(f"    - {name:5s}: 윈도우 {s['n_windows']:5d}개  "
              f"shape={s['spec_shape']}  녹음 {s['n_recordings']}개  분포={s['class_dist']}")

    print(f"\n[3] 클래스 가중치(불균형 대응): {summary['class_weights']}")

    # 누수 차단 검증: split 간 녹음 교집합이 0이어야 함
    with open(resolve_path(cfg["data"]["splits"]) / "c_split_manifest.json", encoding="utf-8") as f:
        man = json.load(f)
    tr = set(man["split_rec_ids"]["train"])
    va = set(man["split_rec_ids"]["val"])
    te = set(man["split_rec_ids"]["test"])
    overlap_ok = not (tr & va) and not (tr & te) and not (va & te)
    print(f"\n[4] 누수 차단 검증(녹음 교집합 0): {'✅ 통과' if overlap_ok else '❌ 실패 — 겹침!'}")

    # CNN 입력 변환 확인(224×224·3채널)
    if summary["splits"]["train"]["n_windows"] > 0:
        X = np.load(resolve_path(cfg["data"]["features"]) / "c_X_train.npy")
        img = spectrogram_to_image(X[0])
        print(f"[5] CNN 입력 변환: {X[0].shape} → {img.shape}  (값범위 {img.min():.2f}~{img.max():.2f})")

    print("\nT7 OK — 녹음단위 분할 + 스펙트로그램 .npy 저장 + 224×224 변환 확인됨.")


if __name__ == "__main__":
    _main()
