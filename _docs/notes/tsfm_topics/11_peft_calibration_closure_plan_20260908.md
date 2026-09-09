# C 후보의 단순 보정 폐쇄 진단

2026-09-08. [반복 탐색 계약](11_peft_topic_search_protocol_20260908.md)의 첫 실행이다. GPU 재학습 없이 10번의 기존 val/eval 예측을 CPU에서 분석한다. 10번 평가를 이미 봤으므로 개발용 탐색이며 새로운 확인 실험이 아니다. 이번 결과만으로 C 전체를 기각하지 않는다.

## 질문과 비교

관측은 ALIGN_ATTN의 점수 개선과 80% coverage 하락이다. 대립 설명은 (a) 간단한 분위수 위치 보정으로 해결되는 잔차 분포 오차, (b) 보정 뒤에도 점수·coverage·폭의 어려운 교환 관계가 남는 오차다. (a)가 맞으면 현상 자체를 위한 새 PEFT는 필요성이 약하다.

10번에서 corpus0 validation으로 이미 선택한 F0/ATTN/ALIGN_F0/ALIGN_ATTN, 세 corpus 모두를 사용한다. LoRA LR나 checkpoint를 이번 평가로 바꾸지 않는다. 각 절차에 두 처리를 동일하게 적용한다.

```text
SORT  : 각 예측의 분위수 축을 오름차순 정렬
QCAL  : SORT 예측의 q별 val 잔차 y-p_q에서 empirical q-quantile을 추정,
        해당 고정 offset을 새 예측 p_q에 더하고 분위수 축 재정렬
```

21개 offset은 채널 Y 하나에서 validation128episode×16horizon을 함께 사용한다. Fit API는 val prediction/target/quantile만 받으며 eval 배열을 받지 않는다. Train 추가 fitting, oracle, 추가 candidate grid, GPU, adapter scaling은 없다. QCAL은 같은 validation을 checkpoint/LR 선택과 보정에 재사용하므로 선택 편향이 가능하다. 이를 conformal finite-sample coverage 보장으로 부르지 않는다. 향후 실제 원천에서는 calibration 구간을 별도 분리해야 한다.

ORACLE/RAW는 이전 결과의 진단 기준으로만 표시하며 QCAL 파라미터 선택에 쓰지 않는다. 새로운 source 파일은 `experiments/peft_calibration_closure_v1/`, 출력은 `results/peft_calibration_closure_v1/`, 실행로그는 `runs/peft_calibration_closure_v1/`에 둔다. 기존 10번 source/result/contract hash를 검증·보존한다.

## 판정과 해석

주진단은 ALIGN_ATTN QCAL과 SORT의 비교다. score는 같은 raw mean21quantile2-pinball, coverage80·width80·medianMSE·crossing을 함께 계산한다. 세 corpus 평균 coverage가78~82% 안이고, 세 corpus 모두 score 악화가 F0 score의1% 이하이면 이 관측의 단순 보정 가능성을 지지한다. 평균 외에 각 corpus 값을 공개한다. Score 개선만으로 보정 성공을 선언하지 않는다. 반대로 기준 미충족도 새 PEFT 필요성의 증명이 아니다.

512개 shared eval episode를4000회 paired bootstrap(seed2026090811)해 score 변화/F0 및 coverage 변화를 기술적95%구간으로 보고한다. Interval은 세 fitted repetition과 이전 선택에 조건부이며 사후 개발 진단이다. 8개방법×3corpus의 모든 행을 남기고 유리한 subset만 보고하지 않는다.

CPU 검사는 q-axis정렬/원본불변, val-only offset재현, evaltarget변경시offset불변, loss독립재계산, paired episode대응·F0분모재계산을 다룬다. 이 계획의 한 번 실행 후 결과와 자기 평가를 기록하고 실제 원천 후보로 넘어간다. 같은 평가에 더 많은 보정식을 탐색하지 않는다.
