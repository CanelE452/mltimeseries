# 18. PEFT 문제 중심 후보와 다음 진입 판단

2026-09-08. [사전조사 본문](../../../reference/time_series_peft_problem_first_dossier_20260908.md)과 [출처·검색 카탈로그](../../../reference/time_series_peft_problem_first_catalog_20260908.json)의 작업용 결정 기록이다. **새 실험 결과·새 방법·확정된 실행 계약은 아니다.** 기존 A/B/C와 구분해 R1–R3로 부른다.

## 현재 판단

- **R1 — 수정되는 정답에 대한 적응 교정:** 다음 자료 진입 검사 우선. 실제 revision 자료와 날짜가 있어 입력 수정과 학습 경로의 영향을 분리할 수 있다. PEFT 필요성·신규성·성능은 미확정.
- **R2 — 집계 정답 + 소량 fine 정답:** 조건부 보류. Source 개별 정답과 target 집계 정답을 함께 쓰는 hybrid LLP가 이미 존재한다. 현재17의 coarse-only 실패를 새 구조의 필요성으로 해석하지 않는다.
- **R3 — 채널·측정 방식 변화:** 실제 forecasting 정보 계약을 확보할 때까지 보류. Native 다변량 FM·기존 공변량 적응을 먼저 이겨야 한다.

R1의 우선순위는 “현재 가장 높은 성능이 기대된다”가 아니라 **실제 조건을 관측하고 강한 반례로 빨리 판단할 수 있다**는 근거에 따른다.

## R1에서 확인할 문장

> [가설·미검증] 잠정 정답으로 진행한 내부 적응의 유해한 영향이 정정 자료 도착 후에도 남으며, 같은 정보·비용의 현재-vintage head 및 표준 LoRA replay로 충분히 제거되지 않는다.

이 가설이 반증되면 해당 새 PEFT 안을 닫는다. 성능이 낮은 stale-label arm만 이기는 것은 목표가 아니다.

가까운 선행은 [MacroCast](https://arxiv.org/html/2606.28670v1), [ORCA](https://arxiv.org/html/2606.14222v1), [AdapTS](https://arxiv.org/html/2502.12920v2), [Evolving Observations](https://proceedings.mlr.press/v272/bar-on25a.html), corrective unlearning이다. Vintage 사용·replay·gradient 수정 자체를 최초라고 주장하지 않는다.

## 다음 1회차의 범위

자료만 검사한다. PAYEMS와 다른 revision 생성 과정을 가진 보조 계열을 검토하고 다음을 고정할 수 있는지 판단한다.

- 원 관측 기간, 값의 발표·수정 날짜, 각 vintage 값.
- 실제 revision 빈도·크기·시차와 vintage coverage.
- 예측 시점에 사용 가능한 값의 집합과 같은 날 발표 처리 규칙.
- 미래 정답을 평가할 고정 성숙 시점과 그 값의 이용 가능 날짜.
- 단위·계열 정의·변환·결측 처리와 과거 정의 변경.
- FM checkpoint의 사전학습 범위. 모르면 역사적 평가의 무누수를 주장하지 않는다.

[ALFRED](https://alfred.stlouisfed.org/help)는 과거 vintage를 제공하지만 intraday 수신 시각까지 보증하지 않는다. 공식 가용성을 확인한 상태이며 데이터 payload·전체 이력의 품질 검사는 아직 없다. API key가 없는 상황을 우회했다고 주장하지 않고 공식 웹 다운로드 경로의 실재를 확인한다. 표본 수나 변환을 보고 임의로 유리한 계열만 선택하지 않는다.

## 자료 통과 후의 검사 순서

1. **CPU 기존 해법:** raw feature AR/ridge, 현재 자료 배치 적합, 정답 revision을 반영한 충분통계 갱신. 마지막 둘은 동일한 고정 feature·λ·가중치라면 같은 해여야 한다. 발표 차수·경과시간별 평균 revision 또는 작은 회귀 보정도, 이미 성숙한 과거 사례만으로 학습해 비교한다.
2. **Frozen FM와 head:** 먼저 과거 학습 context/feature 버전과 현재 예측 입력을 고정한 label-only revision 진단을 한다. 이후 context까지 정정하는 실제 운영 조건을 별도로 평가한다. Context 수정에 따른 feature 재계산 시간도 기록한다.
3. **표준 PEFT 필요성:** 같은 base·자료·선택 조건의 corrected replay, shrinkage/forecast mixture, reset/refit과 강한 head/ORCA를 비교한다.
4. **새 방법 설계:** 앞 단계에서 남은 실용적인 오차·회복 비용만 대상으로 한다. 단순 과적합·빈약한 replay 예산을 고의로 만들어 유리한 문제를 만들지 않는다.

실제 단계 1–3을 시작하기 전에 series·기간·origin·모델 revision·연산 예산·주지표·선택 규칙·중단 기준을 별도 계획에 고정한다. 본 노트는 수치 gate까지 확정한 preregistration이 아니다.

## 중단과 진행 기준의 원칙

현재 입력 교체, 정확한 head 재적합, 표준 corrected replay 또는 reset/refit으로 같은 비용 안에서 충분하면 새 PEFT 필요성이 약하다. 그 해결책을 결과로 기록하고 종료한다. 자료 coverage 미확보는 알고리즘 실패와 구분한다.

진행하려면 단순 대조가 회수하지 못하는 미래 손실이나 실제 누적 비용 차이가 있고, 그것을 설명하는 관측 가능한 기전이 있어야 한다. 새 방법을 고른 뒤 다른 기간·원천·backbone으로 확인한다. Macro series 여러 개를 서로 독립인 여러 domain으로 세지 않는다.

실용적인 개선 기준과 불확실성 평가를 실행 전에 정한다. 모든 dataset에서 무조건 승리할 것을 일반 요건으로 두지 않으며, 주장하는 적용 범위의 평균 이득과 나쁜 경우의 손실을 함께 보고한다. 이미 고정된 이전 실험의 판정은 변경하지 않는다.

## 기존17과의 관계

17번 source·결과·report는 동결 상태로 유지한다. `p0 + mean(p − p0)`의 단순 패턴 보존 대조는 향후 별도 **사후 원인 진단**으로만 수행할 수 있다. 이번에는 실행하지 않았다. 이를 새18번 방법이나17번 사전 arm으로 추가하지 않는다.

전체 연구 목표는 여전히 새롭고 검증 가능한 PEFT 방법의 확보이며 미완료다. 이번 완료 범위는 사전조사·후보 정리다. 후속 학습도 기존 공유 guard 아래 GPU 한 작업씩 실행하고 시스템 설정은 변경하지 않는다.
