"""최종 추론 실행 스크립트 (streaming + inference 연결).

스트리밍 시뮬레이터로 신호를 윈도우 단위로 흘려보내며,
inference 오케스트레이터(전처리→A·B→융합→게이트 C)로 실시간 판정한다.
저장된 scaler.pkl / thresholds.json / 모델을 로드해 동일 전처리를 보장한다.

실행:
    .venv\\Scripts\\python.exe run_inference.py
"""
