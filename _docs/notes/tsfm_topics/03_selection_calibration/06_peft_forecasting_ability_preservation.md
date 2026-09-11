---
title: "PEFT C — Target 적응과 필요한 예측 능력의 보존"
working_title: "Preserving Conditional Forecasting Ability during Target-Specific PEFT"
created: 2026-09-07
status: "합성 조건의 단순 보정 진단 완료 / 새 PEFT 방법 필요성 미입증"
priority_within_peft: 3
---

# 1. 핵심 질문

> [가설·미검증] Target에 PEFT하면 같은 조건의 점수는 개선하면서, 실제 배포에서 필요한 확률예측 또는 새로운 기간·series·context 구성의 성능을 손상시키는가?

Base weight가 frozen이어도 adapter-on 예측 함수는 달라진다. Adapter를 끄면 원모델로 돌아간다는 사실은 active adapter의 일반화 보존을 입증하지 않는다.

# 2. 먼저 고정할 사용 조건

확률 calibration 보존 또는 group/context 변화에 대한 일반화 중 **하나**를 1차 문제로 고른다. 하나의 adapter를 다른 series·regime에서 계속 사용할 필요가 없는 상황이라면, 그 조건의 손실은 허용된 specialization일 수 있다. 이전 적응 task의 망각, target의 다음 기간 성능, 원래 zero-shot 능력을 한 지표로 섞지 않는다.

# 3. 최소 판별 비교

- Chronos-2의 native quantile loss를 사용하는 일반 LoRA를 주 비교군으로 둔다. MSE로 바꿔 quantile 악화를 유도한 결과만으로 문제를 입증하지 않는다.
- Head-only calibration, validation early stopping, adapter 강도 축소, frozen fallback을 먼저 비교한다.
- Target validation에서 선택한 adapter를 고정한 뒤 실제 deployment에 필요한 다음 기간·유보 series·context 구성을 평가한다.
- Proper score와 coverage·width를 함께 보고, 임의 가중합으로 tradeoff를 숨기지 않는다. 유한 quantile grid와 공동 경로분포 평가를 구분한다.

# 4. 이미 가까운 연구와 반증

MixFT/replay, ORTCL, SFF/LP-FT, WiSE-FT, building-energy LoRA, Chronos-2 Gated-LoRA가 비교 대상이다. 특히 상태별 adapter 강도는 이미 선행연구가 있으므로 constant/shuffled-state 대조를 두어야 한다. Replay를 사용하면 양쪽 정보 조건도 맞춘다. [출처와 확인 범위](../../../reference/time_series_peft_research_dossier_20260907.md).

간단한 early stopping·calibration·shrinkage로 손실이 해소되면 새 PEFT 보존 기법을 만들 근거는 약하다. 극단적인 합성 shift에만 나타나거나 실제 사용하지 않을 조건에서만 악화하는 결과도 주제의 동기로는 부족하다.

# 5. 현재 상태

2026-09-07 최초 작성 때에는 후보로 보존했고 학습·방법 구현을 시작하지 않았다. [A](../02_adaptation_scope/04_peft_adaptation_scope.md)의 단순 음성 결과가 C의 증거가 되지는 않는다.

2026-09-08에는 A의 합성 후속에서 관측한 undercoverage를 대상으로 [단순 보정 진단11](11_peft_calibration_closure_results_20260908.md)을 완료했다. 별도 새 GPU 학습 없이 empirical quantile offset을 적용하자 ALIGN_ATTN의80% coverage가69.97%에서79.82%로 바뀌었고 proper score도 개선됐다. 이미 본 평가 구간과 선택 validation 재사용이므로 외부 확증은 아니다. 이 조건의 undercoverage만을 위한 새 PEFT 방법 필요성은 약하다고 판정했다. C 전체를 기각하지 않으며 실제 원천에서 일반화 손상이 관측되면 새로운 계약에서 검토한다.
