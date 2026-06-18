"""경로 B — 디노이징 Conv 오토인코더 (T2.2 모델 정의).

정상 스펙트로그램만 학습해서 '정상 구조'를 압축·복원하도록 만든다.
고장 스펙트로그램은 학습 때 안 본 패턴이라 복원이 잘 안 됨 → 재구성 오차가 큼 → 이상.

구조:
  입력 (1, H, W)
   → Conv 3단(stride2)으로 공간 압축
   → Flatten → Linear → 잠재벡터(latent_dim)   ← 병목(강제 압축)
   → Linear → reshape → ConvTranspose 3단으로 복원
   → 마지막에 입력 크기로 정확히 맞춤(F.interpolate)
   → 출력 (1, H, W)

이번 단계(T2.2)는 모델 뼈대 + forward(출력 shape=입력 shape) 점검까지.
디노이징 노이즈 주입·학습은 T2.3, 점수·임계값은 T2.4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils import load_config, resolve_path


class ConvAutoencoder(nn.Module):
    """스펙트로그램용 Conv 오토인코더 (병목 있는 인코더-디코더)."""

    def __init__(self, in_ch: int = 1, input_hw: tuple[int, int] = (129, 25),
                 base: int = 16, latent_dim: int = 128):
        super().__init__()
        self.input_hw = tuple(input_hw)

        # --- 인코더 (Conv stride2 ×3) ---
        self.enc_conv = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        # conv 출력 크기를 더미로 자동 계산 (입력 크기 바뀌어도 적응)
        with torch.no_grad():
            enc = self.enc_conv(torch.zeros(1, in_ch, *self.input_hw))
        self._enc_shape = tuple(enc.shape[1:])          # (C, H', W')
        flat = int(np.prod(self._enc_shape))

        # --- 병목 (Flatten → 잠재벡터 → 복원) ---
        self.enc_fc = nn.Linear(flat, latent_dim)
        self.dec_fc = nn.Linear(latent_dim, flat)

        # --- 디코더 (ConvTranspose stride2 ×3) ---
        self.dec_conv = nn.Sequential(
            nn.ConvTranspose2d(base * 4, base * 2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base * 2, base, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base, in_ch, 3, stride=2, padding=1, output_padding=1),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """입력 → 잠재벡터 (N, latent_dim)."""
        z = self.enc_conv(x)
        return self.enc_fc(z.flatten(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        z = self.encode(x)
        d = self.dec_fc(z).view(-1, *self._enc_shape)
        out = self.dec_conv(d)
        # 홀수 차원 등으로 1~2픽셀 어긋날 수 있어 입력 크기로 정확히 맞춤
        return F.interpolate(out, size=size, mode="bilinear", align_corners=False)


def build_autoencoder(cfg: dict) -> ConvAutoencoder:
    """config 기반으로 모델 생성. 스펙트로그램 크기(H,W)를 자동 산출."""
    from src.spectrogram import to_spectrogram

    length = cfg["window"]["length"]
    spec = to_spectrogram(np.zeros(length, dtype=np.float64), cfg)  # (H, W)
    latent = int(cfg.get("ae", {}).get("latent_dim", 128))
    return ConvAutoencoder(in_ch=1, input_hw=spec.shape, latent_dim=latent)


def reconstruction_error(model: ConvAutoencoder, x: torch.Tensor) -> torch.Tensor:
    """샘플별 재구성 오차(MSE) → (N,) 텐서. 클수록 이상."""
    model.eval()
    with torch.no_grad():
        recon = model(x)
        return ((recon - x) ** 2).mean(dim=(1, 2, 3))


# =============================================================
#  T2.3 — 디노이징 학습 (정상 스펙트로그램만)
# =============================================================
def add_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """디노이징용 가우시안 노이즈 주입. 학습 입력에만 쓴다(타깃은 깨끗한 원본)."""
    return x + std * torch.randn_like(x)


def train_ae(
    model: ConvAutoencoder, train_specs: np.ndarray, cfg: dict,
    device: str = "cpu", val_specs: np.ndarray | None = None,
    noise_std: float | None = None,
) -> dict[str, list[float]]:
    """정상 스펙트로그램으로 (디노이징) 학습.

    입력=노이즈 섞은 스펙트로그램, 타깃=깨끗한 원본 → 노이즈를 걷어내며 '정상 구조'를 배움.
    noise_std=0 이면 일반 오토인코더(노이즈 없이 그대로 복원).
    noise_std 를 주면 config 값 대신 그 값을 쓴다(실험용).
    반환: {'train': [에폭별 손실], 'val': [...]}
    """
    ae = cfg["ae"]
    epochs, bs, lr = ae["epochs"], ae["batch_size"], ae["lr"]
    noise = ae["noise_std"] if noise_std is None else noise_std

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    xt = torch.tensor(np.asarray(train_specs), dtype=torch.float32)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xt), batch_size=bs, shuffle=True
    )
    xv = (torch.tensor(np.asarray(val_specs), dtype=torch.float32, device=device)
          if val_specs is not None and len(val_specs) else None)

    history: dict[str, list[float]] = {"train": [], "val": []}
    for ep in range(epochs):
        model.train()
        tot, n = 0.0, 0
        for (xb,) in loader:
            xb = xb.to(device)
            recon = model(add_noise(xb, noise))   # 노이즈 입력 → 깨끗한 원본 복원
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        history["train"].append(tot / n)

        if xv is not None:
            model.eval()
            with torch.no_grad():
                history["val"].append(loss_fn(model(xv), xv).item())  # val은 노이즈 없이

        if ep == 0 or (ep + 1) % 5 == 0 or ep == epochs - 1:
            msg = f"  epoch {ep+1:>3}/{epochs}  train={history['train'][-1]:.5f}"
            if xv is not None:
                msg += f"  val={history['val'][-1]:.5f}"
            print(msg)
    return history


def save_model(model: ConvAutoencoder, path: str | Path) -> None:
    """모델 가중치 저장 (models/autoencoder.pth)."""
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(cfg: dict, path: str | Path, device: str = "cpu") -> ConvAutoencoder:
    """저장된 가중치로 모델 복원."""
    model = build_autoencoder(cfg)
    model.load_state_dict(torch.load(resolve_path(path), map_location=device))
    model.to(device)
    return model


# =============================================================
#  T2.4 — 이상 점수·임계값 (정상 재구성오차 분포 기반)
# =============================================================
def recon_errors(model: ConvAutoencoder, specs: np.ndarray, device: str = "cpu",
                 batch: int = 256) -> np.ndarray:
    """스펙트로그램 배열 (N,1,H,W) → 샘플별 재구성오차 (N,). 배치로 처리."""
    model.eval().to(device)
    x = torch.tensor(np.asarray(specs), dtype=torch.float32)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            xb = x[i:i + batch].to(device)
            out.append(((model(xb) - xb) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)


def fit_threshold_b(normal_errors: np.ndarray, cfg: dict) -> dict:
    """정상 재구성오차 분포에서 임계값을 산정한다(정상에서만 — 누수 차단).

    config ae.threshold_rule: mean_std(평균+Nσ) / percentile.
    반환: {'threshold','rule','fit_mean','fit_std','n_windows'}
    """
    ae = cfg["ae"]
    mean, std = float(normal_errors.mean()), float(normal_errors.std())
    rule = ae.get("threshold_rule", "mean_std")
    if rule == "mean_std":
        sigma = ae["threshold_sigma"]
        thr, rule_str = mean + sigma * std, f"mean+{sigma}std"
    elif rule == "percentile":
        p = ae.get("threshold_percentile", 99)
        thr, rule_str = float(np.percentile(normal_errors, p)), f"percentile_{p}"
    else:
        raise ValueError(f"지원하지 않는 임계값 규칙: {rule}")
    return {"threshold": thr, "rule": rule_str, "fit_mean": mean,
            "fit_std": std, "n_windows": int(len(normal_errors))}


def save_threshold_b(section: dict, cfg: dict) -> Path:
    """경로 B 임계값을 thresholds.json 의 'path_b' 에 저장(병합)."""
    path = resolve_path(cfg["artifacts"]["thresholds"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    data["path_b"] = section
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def anomaly_score_b(error: float | np.ndarray, threshold: float) -> float | np.ndarray:
    """경로 B 이상 점수 = 재구성오차 / 임계값. 1 초과면 이상. (융합 T3.1에서 사용)"""
    return error / threshold if threshold > 0 else error


def _main() -> None:
    """T2.2 검증: 더미 입력 → 출력 shape = 입력 shape 확인."""
    sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config()
    model = build_autoencoder(cfg)

    H, W = model.input_hw
    x = torch.randn(4, 1, H, W)   # 더미 입력 (배치 4)
    out = model(x)
    z = model.encode(x)
    n_params = sum(p.numel() for p in model.parameters())

    print("=== T2.2 Conv 오토인코더 모델 ===")
    print(f"입력 shape       : {tuple(x.shape)}")
    print(f"잠재벡터 shape   : {tuple(z.shape)}  (병목 {z.shape[1]}차원)")
    print(f"출력 shape       : {tuple(out.shape)}")
    print(f"인코더 conv 출력 : {model._enc_shape}")
    print(f"파라미터 수      : {n_params:,}")
    err = reconstruction_error(model, x)
    print(f"재구성오차(예시) : {err.tolist()}")

    assert out.shape == x.shape, "출력 shape 가 입력과 다름!"
    print("\nT2.2 OK — 출력 shape = 입력 shape. (학습은 T2.3)")


if __name__ == "__main__":
    _main()
