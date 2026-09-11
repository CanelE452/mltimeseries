---
title: "PEFT A — 변화의 종류와 필요한 적응 위치"
working_title: "Where Should a Multivariate Time-Series Foundation Model Adapt?"
created: 2026-09-07
status: "2026-09-08 현재 배포·예산의 A 분기 종료 / PEFT 전체 반증 아님"
priority_within_peft: 1
---

# 1. 핵심 질문

**2026-09-08 최신 상태:** S1, 합성 진단, native 모듈 삭제, train-only 지연 대조 및 실제 원천 두 시간 블록을 실행했다. [13번 결과](13_peft_temporal_replication_results_20260908.md)에서 전체 실용 진입 기준이 다시 충족되지 않아 현재 A 분기를 사전 규칙대로 닫는다. Bike의 큰 H 대 LoRA 차이는 F0 초과 개선을 뜻하지 않았다. 새 내부 PEFT의 일반적 필요성·위치별 실패 원인의 인과식별은 확보하지 못했다. 아래는 09-07 최초 가설과 설계 근거를 보존한 내용이다.

> [가설·미검증] Target에서 관찰되는 변화의 종류에 따라 출력 보정으로 충분한 경우와, 시계열 FM의 시간 처리 또는 변수 관계 처리 부분을 갱신해야 하는 경우가 구분되는가?

목표는 새로운 adapter 이름을 만드는 것이 아니라 **같은 target 정보와 적응 예산에서 필요한 갱신 범위를 찾는 것**이다. 현재 확인한 현상이 아니라 실험으로 존재부터 확인할 후보다. 소비처는 연구 주제 선택과 첫 판별 실험의 설계다.

# 2. 먼저 구분할 세 설명

- **출력 불일치:** 유용한 정보는 이미 있지만 출력의 수준·폭 또는 읽는 방식이 target에 맞지 않는다.
- **시간 처리의 불일치:** 자체 시간 의존성을 이용하는 방식의 변경이 추가 이득을 준다.
- **변수 관계 처리의 불일치:** 다른 series의 정보를 결합하는 방식의 변경이 추가 이득을 준다.

이 설명들은 배타적인 원인이 아니다. Time module의 변경도 이후 group interaction을 바꾸며 group module도 시간 정보가 들어간 표현을 받는다. 모듈별 성능 차이는 우선 **적응 위치의 효과**다. 별도의 판별 증거 없이 실패 원인의 인과식별로 부르지 않는다.

# 3. 어떤 결과가 가설을 지지하는가

같은 정보·checkpoint에서 출력 보정 및 동결 표현 probe를 넘어서는 내부 갱신의 이득이 존재하고, 그 이득의 위치가 관측 가능한 변화 유형에 따라 반복적으로 달라져야 한다. 이 관계가 독립 domain에서도 유지되고 단순 validation 선택보다 유용할 때 선택 규칙을 연구한다.

반대로 강한 출력/probe가 차이를 설명하거나, head·rank·LR를 맞춘 뒤 위치별 차이가 사라지면 A의 방법 개발 근거는 약해진다. Full FT가 잘된다는 사실과 역할별 PEFT가 유용하다는 사실도 별개다.

# 4. 직접 경쟁하는 선행연구

- **Time-PEFT:** complexity 진단, frequency/channel adapter. 초록·공개 코드만 확인한 범위와 본문 미확보를 구분한다.
- **TRACE, AdaLoRA, EVA:** 중요도·activation에 따른 위치/rank 배분. 단순 위치 선택 자체는 새 기여가 아니다.
- **두 CoRA, UniCA, ChronosX:** 공변량·관계 적응. 정보 추가와 weight update의 효과를 구분해야 한다.
- **SFF:** full FT의 최적화 조건을 강하게 두어야 하는 이유다.

원문과 확인 한계: [PEFT dossier §4–6](../../../reference/time_series_peft_research_dossier_20260907.md), [카탈로그](../../../reference/time_series_peft_paper_catalog_20260907.json).

# 5. 진행 방향

첫 backbone 후보는 Chronos-2다. 공식 fit과 구분된 time/group module이 있어 실제 비교를 정의할 수 있기 때문이다. 아직 LoRA backward와 비용을 실측하지 않았으며, 로컬 `peft` 미설치 및 full FT fallback을 실행 전 해소해야 한다.

실험의 중심은 frozen → 출력/probe → 일반 LoRA/full FT의 적응 여지 확인 → 동일 예산 time/group 비교 순서다. 합성 데이터는 역할을 판별하고, 실제 데이터는 발견의 적용 범위를 확인한다. 자세한 비교군·시간 계약·예산·반증 조건은 [A 실험 설계](04_peft_adaptation_scope_experiment_plan.md)에 둔다.

# 6. B·C와의 관계

A는 우선 설계할 후보다. 적응 범위 선택이 데이터에 따라 불안정하면 [B](../03_selection_calibration/05_peft_selection_reliability.md)를 검토한다. 적응 후 실제로 유지해야 하는 예측 능력의 손실이 보이면 [C](../03_selection_calibration/06_peft_forecasting_ability_preservation.md)를 검토한다. 세 가지를 동시에 구현하는 계획은 아니다.

상태: 2026-09-07 사용자 요청으로 노트 보존과 실험 설계만 진행했다. 가설의 성공·실패나 novelty는 미판정이다.
