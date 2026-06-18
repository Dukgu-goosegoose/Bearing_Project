"""경로 C 학습 실행 — 데이터셋 빌드 → MobileNetV2 전이학습 → 평가 → 저장.

실행(프로젝트 맨 위 폴더에서, 가상환경 켠 상태):
    python train_classifier.py

산출물:
    data/features/c_X_*.npy, c_y_*.npy   (스펙트로그램 데이터셋)
    data/splits/c_split_manifest.json    (녹음 단위 분할 기록)
    models/fault_classifier.pth          (학습된 분류기)
"""
from src.cnn_fault_classifier import run_training
from src.utils import load_config


def main() -> None:
    cfg = load_config()
    # 48k 고장을 12k로 다운샘플해 데이터를 늘리고 싶으면 True 로.
    model, metrics, history = run_training(cfg, include_48k_downsampled=True)

    print("\n=== 학습 완료 요약 ===")
    print(f"테스트 정확도: {metrics['accuracy']:.3f}")
    print("클래스별 재현율:", {k: round(v, 3) for k, v in metrics["per_class_recall"].items()})
    print("혼동행렬(행=정답, 열=예측):", metrics["labels"])
    for label, row in zip(metrics["labels"], metrics["confusion_matrix"]):
        print(f"  {label}: {row}")


if __name__ == "__main__":
    main()