"""FastAPI 백엔드 — 추론 결과를 JSON 으로 제공.

윈도우(또는 스트림)를 받아 inference 오케스트레이터로 판정 후
점수·판정·(이상 시) 유형 추정을 JSON 으로 응답한다.
프론트엔드 대시보드가 이 API 를 호출한다.

실행:
    .venv\\Scripts\\python.exe -m uvicorn backend.api:app --reload
"""
