---
title: "PEFT B — 적은 데이터에서 적응 범위 선택의 신뢰성"
working_title: "When Is PEFT Selection Reliable for Time-Series Adaptation?"
created: 2026-09-07
status: "2026-09-08 저장 후보 진단 완료 / 현재 B 분기 종료 / 새 방법 미확보"
priority_within_peft: 2
---

# 1. 핵심 질문

> [가설·미검증] 시간적으로 겹친 window는 많지만 독립적인 기간·target 사건은 적을 때, adapter 위치와 rank를 고르는 과정이 과도하게 불안정해져 고정된 작은 PEFT보다 미래 성능이 나빠지는가?

목표는 **적응 범위 선택에 충분한 증거가 있는지 판단하는 것**이다. 선택한 module이 자주 바뀐다는 사실만으로는 문제가 아니다. 다른 선택들이 같은 성능을 내는 비식별성일 수 있으므로 독립 미래의 selection regret가 핵심이다.

# 2. 분리해야 할 설명

- 같은 raw 기간에서 window 수만 늘어남: 새로운 정보가 아니라 중복 sampling·반복 가중치의 변화일 수 있다.
- 관측 기간·entity가 늘어남: 정보량뿐 아니라 다양성·최근성·분포도 바뀔 수 있다.
- Adaptive selection의 악화: 유효 정보 부족 외에도 LR·업데이트 횟수·validation 반복 사용이 원인일 수 있다.

자기상관 하나로 유효 표본수를 확정하지 않는다. 고유 target timestamp, forecast origin, 독립 entity·기간, 예측 잔차의 의존성을 구분해 기록한다.

# 3. 최소 판별 비교

1. 원시 기간을 고정하고 stride·복제를 바꾸되 optimizer update와 target 노출 가중치를 통제한다.
2. 별도로 기간·독립 entity를 늘려 중복 window 증가와 구분한다.
3. Head/LN, fixed LoRA, TRACE/AdaLoRA류 선택을 비교한다. 선택 비용도 HPO 예산에 포함한다.
4. Block을 바꿔 선택했을 때의 module/rank 변동과 독립 test 손실을 함께 본다. 사후 최적 모델은 평가 상한이며 실제 선택에 쓰지 않는다.

# 4. 반증과 후속 방법의 조건

일반적인 시간 분리 validation·early stopping으로 문제가 사라지거나 선택의 변동이 미래 regret와 무관하면, 새 선택 안정화 방법의 필요성은 약하다. 관찰된 실패가 반복될 때만 선택의 불확실성에 따라 범위를 제한하는 단순 규칙을 검토한다.

경쟁 문헌은 TRACE, TS-PET, AdaLoRA/EVA, Time-PEFT다. 공개 TSFM-PEFT-Bench artifact도 설계가 가깝지만 원문·코드를 확보한 증거와 구별한다. [원문·한계 모음](../../reference/time_series_peft_research_dossier_20260907.md).

# 5. 현재 상태

09-07에는 후보를 보존했다. 09-08 [A](04_peft_adaptation_scope.md)의 현재 분기를 닫은 뒤 [14번 진단](14_peft_selection_regret_results_20260908.md)을 완료했다. 미선택 28추론/추가학습0회로 저장 후보40개의 미래 점수를 완성했다. Bike13의 선택 손실은 +3.079%F0/98.75%CI[0.414,5.838]%였지만 사전1% 하한과 두 원천 반복 조건을 통과하지 못했다. 고정 LR1e-5가 두 사전 정책의 최저점수 대비 네 cell 모두1% 이내인 단순 규칙 veto를 만족했다. 사전 계획에 따라 현재 B 분기를 닫는다. 이는 전체 후보의 사후 최적 손실이 없거나 PEFT 선택 연구 전체가 불필요하다는 뜻은 아니다. 새 방법은 확보하지 못했다.
