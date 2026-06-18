"""실시간 스트리밍 시뮬레이터 + 슬라이딩 윈도우 버퍼.

파일에 저장된 신호를 실시간처럼 한 점씩 흘려보내며, 링버퍼가 가득 차면
한 윈도우를 잘라 파이프라인(inference)으로 넘긴다.

TODO: StreamSimulator(signal, window_length, overlap) -> 윈도우 제너레이터.
"""
