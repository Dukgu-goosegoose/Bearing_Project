"""평가 — AUC·pAUC·정상 FPR·confusion matrix.

고장 데이터는 '여기서만' 등장한다(학습·임계값 설정에는 절대 사용 금지).
- 비지도(A·B 융합): AUC·pAUC·정상 오탐률(FPR).
- 지도(C 분류): 정확도·confusion matrix.
결과는 outputs/reports/ 에 저장.

TODO: evaluate_detection(scores, labels), evaluate_classification(pred, true).
"""
