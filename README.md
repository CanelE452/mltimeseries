# mltimeseries

**완료: [36번 순서 의존성 PEFT Phase 0 게이트](_docs/notes/tsfm_topics/07_research_direction/36_dependence_phase0_results_20260911.md).** 현재 전/역 3상태 Markov DGP는 GPU 진입 전 중단. 주변분포·ACF/PSD·Bayes risk 통제는 성립하지만 부호 재라벨링과 정확히 동치이고, train-only 전이표가 oracle 대비 평균0.0032% excess risk에 그쳐 내부 PEFT 필요성을 검증하지 못함. GPU 0회, adapter/외부검증 미진입.

**완료: [35번 head 학습량·성분 개입 검증](_docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md).** 180→720step 대조 완료. LoRA는 동일 용량 head보다 평균8.804%F0 좋지만 F0보다4.305% 나빠 G1 미통과. head가 따라잡은 결과는 아니며,12개 선택 모델 모두 train/V 개선과 D 악화. 조건부 head재학습·warm-start 미실행.75 GPU 실행·독립 검산 완료,학습 종료.

**완료: [34번 초기 LoRA 이득 진입 검사](_docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md).** 초기 LoRA 이득 G1은 4조합 중3조합 통과, controller 여지 G2는 +0.0408/+0.0756%F0로 미통과. 평균 동일 용량 head 대비 +2.428%F0, F0 대비 +1.418%F0. 일부 head의 학습 상한 선택으로 수렴·표현 원인은 미확정. 123 GPU 실행·독립 검산·5개 그림 완료, 학습 종료.

**완료: [33번 시기 전이·신호 제거·실제 비용 검증](_docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md).** 177 GPU 실행·독립 검산 완료, 세 기준 미통과. 후보는12/12 STOP과 같은 모델을 반환했고 조기 종료보다20.1~33.7% 느렸다. 동일 용량 head 대비 평균 LoRA 이득+.944%F0는 Jena/BMRA에서 부호가 다르다. 4개 그래프·전체 결과·실패 원인과 다음 연구 순서 정리; 이번 학습 종료.

**완료: [32번 미래 LoRA 업데이트 가치 진단](_docs/notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md).** 12조건·24분기와 독립 검산 완료. 현재 기여가 양수여도 미래 업데이트가 불리한 사례16개, checkpoint 선택 후21/24 같은 결과. 새 방법의 우월성은 미확보; 결과·그래프·논문 요건 정리.

시계열 ML 연구 저장소. 2025년 말–2026년 시계열 예측 문헌을 정리하고, 그 위에서
후속 연구 주제를 고르는 것이 목적이다.

상태 갱신: 2026-09-11

## 현재 상태

**2026-09-11 완료: [31번 단일 LoRA 적응량 검증](_docs/notes/tsfm_topics/07_research_direction/31_contribution_freeze_20260911.md).** 48학습+50평가+S0+시간재측정13,총112 GPU실행정상. 기여 기반 동결은 보완 시간 절감7.42/3.77%로 두seed 기준실패;고정동결19.92/21.46%절감·평균성능우세. 기여와 미래학습가치의 불일치 반례를 기록했고, 새방법주장은 하지 않는다. 그래프·전체기록·독립검산·후속방향 정리. 아래는 이전 이력이다.

**최신: 동일 파라미터 head·짧은 반응 측정 완료.** [30번 결과와 그래프](_docs/notes/tsfm_topics/07_research_direction/30_capacity_probe_20260910.md): 새60학습/24평가/1smoke 모두 정상 종료. 학습 파라미터1,768,949개를 맞춰도 P1 Bike FULL90 LoRA 이득은 +4.779%F0이고 P1 12/12비교가 양수였다. 짧은 반응의6cell 순위 상관은0.543이지만, 이진 선택은 항상LoRA보다 정확하지 않고 시간도6.4~14.3% 늘었다. 통제 근거는 추가됐으나 새 PEFT 방법은 미확보. 해당 학습은 종료됐다. 아래는 이전 이력이다.

**1→2→3 검증 완료.** [28번 최적화 대조](_docs/notes/tsfm_topics/07_research_direction/28_optimization_control_20260910.md)와 [29번 새 시기 결과·그래프](_docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md): 본학습72개/예측74개 완료. 새 시기 같은-head 구조 대비 LoRA 이득은12개 대응 비교 모두 양수였다. 그러나 중복 비율로 LoRA를 생략하는 규칙은 시간11.54~12.63% 절감에도 손실0.503~0.674%F0로 사전 정확도 기준을 실패했다. 두 원천 모두 SPREAD30/RECENT30 효용 순위가 반전했다. 학습형 predictor·새 adapter·독립 원천 일반화는 아직 없으며 현재 실행 중인 학습은 없다. 아래는 이전 실행 이력이다.

**최신: R1 후속 검증 완료.** [학습 구간 구성·잔차 보정 결과](_docs/notes/tsfm_topics/07_research_direction/27_r1_coverage_residual_validation_20260910.md): 새16 fits/16 평가. Bike의 같은-head LoRA 추가 이득은 전체90개 +4.171%F0, 분산30개 −1.019%F0, 최근30개 +1.267%F0였다. 원점 수만으로는 설명되지 않지만 최적화와 시간 구성이 혼입돼 원인은 미확정이다. 단순 편향 보정은 약 .497%만 개선했다. 독립 수치 감사 완료, 학습 프로세스 없음. 아래는 직전 R1–R3 결과다.

**2026-09-10 R1–R3 통제 실험·독립 감사 완료.** [최신 결과와 그래프](_docs/notes/tsfm_topics/07_research_direction/26_mechanism_diagnostics_20260910.md)를 먼저 읽는다. 새 36 fits와 24 평가에서 같은 head 대비 내부 LoRA 추가 이득은 Bike 4.171%F0, Household .418%F0였다. 특정 층 선택은 같은 예산의 전체 저 rank 대조를 넘지 못했다. Hospital 개인화의 미래 손해는 월 제외 선택 불안정만으로 설명되지 않았다. 새 방법이나 독립 test 확증은 아직 없다.

이번 통제 실행은 약 33분, 63 guard jobs 모두 정상 종료, 안전 중단 0회였다. 진행 중인 학습은 없다. [Hospital 본결과](_docs/notes/tsfm_topics/08_hospital_shared_strength/24_hospital_results_20260910.md)와 [전체 과정의 이전 스냅샷](_docs/notes/tsfm_topics/research_review_20260910/README.md), [주제별 8개 폴더](_docs/notes/tsfm_topics/README.md)에서 앞선 기록을 찾을 수 있다.

### 앞선 실험의 요약

문헌 조사와 여러 PEFT 실험을 진행했다. 현재 A/B/C의 제한된 진단에서는 새 방법의 필요성을 뒷받침하는 사전 기준을 통과하지 못했다. [최신 실행 기록](_docs/PROJECT_LOG.md)과 [후보별 결과](_docs/notes/tsfm_topics)에서 근거와 적용 범위를 확인할 수 있다. 아직 새로운 ML 방법이나 독립 확증 결과를 확보한 것은 아니다.

최근 [15번 목적함수 정렬 진단](_docs/notes/tsfm_topics/04_objective_observation/15_peft_objective_alignment_results_20260908.md)의 6fit·6forecast를 완료했다. 동일한 LoRA에서 normalized loss와 raw loss를 비교했으나 사전 실용 기준을 통과하지 못해 현재 변경을 닫았다. 이어 [16번 관측 연산 진입 검사](_docs/notes/tsfm_topics/04_objective_observation/16_observation_operator_entry_results_20260908.md)는 CPU에서 기존 Gaussian 계산의 충분성을 확인했다. USCRN 두 제품의 마지막5분 평균은 일치했지만 시간 평균 재집계에는0.125°C 예외 한 건이 있어 단순 산술 항등성을 가정하지 않는다. 새 PEFT 방법은 아직 확보하지 못했다. 구현은 [experiments/](experiments), 실행 기록은 `runs/`, 분석 산출물은 `results/`에 둔다.

최신 [17번 월합 감독 실험](_docs/notes/tsfm_topics/04_objective_observation/17_coarse_supervision_results_20260908.md)도 완료했다. 월합으로 적합한 head는 Eagle의 월평균 오차를 조금 줄였지만 시간별 MSE는 동결 모델의 약 28배로 악화됐다. 그 head를 고정한 LoRA는 손상을 회복하지 못해 현재 조건을 종료한다. 월합 감독으로 통제되지 않는 패턴 변경이라는 구체 현상은 확인했지만, 새 PEFT 방법의 해결 성과는 아직 없다. CPU68검사와 독립 수치 검산을 통과했고 9개 guard 모두 정상 종료했다.

## 무엇이 들어 있나

- [`_docs/PROJECT_LOG.md`](_docs/PROJECT_LOG.md) — 진입점. 자료 목록, 해석 경계,
  연구 후보, 실행 결과와 다음 단계.
- [`_docs/reference/time_series_research_dossier_20260906.md`](_docs/reference/time_series_research_dossier_20260906.md)
  — 48개 외부 연구·모델·벤치마크 항목의 사전조사. 항목마다 확인 깊이와 해석 경계를
  함께 적었다.
- [`_docs/reference/paper_catalog.json`](_docs/reference/paper_catalog.json)
  — 같은 48개 항목의 기계 판독용 정리. `venue_status` / `mechanism` / `inspection` /
  `interpretation_limit` / `primary_sources`.
- [`_docs/history/`](_docs/history) — 날짜별 작업 기록.

## 이 자료를 읽을 때의 전제

아래는 최초 문헌 사전조사 자료에 적용되는 제약이다. 이후 로컬 실험 수치는 각 결과 보고서의 실행 근거와 한계를 따른다.

- 대부분 항목은 **채택 상태와 공식 초록 수준의 스크리닝**이다. 전수 독해나 재현이 아니다.
- 성능 수치는 출처가 보고한 값이며, 이 조사에서 모델을 실행해 확인한 값이 아니다.
- 메인 학회 채택 / workshop / arXiv 공개본 / 기업 공개를 구분한다.
  `arXiv 공개본; 정식 채택 미확인`은 심사가 진행 중이라는 뜻이 아니다.
- 특정 검색어와 관심사에 따른 사전조사이므로 체계적 문헌고찰이 아니다.
  분야 전체의 빈도나 비율을 추정하는 데 쓰지 않는다.
- **novelty 확정이나 새 실험 실행 승인의 근거로 쓰지 않는다.**

## 관련 저장소

[`covariate-trust-pilot`](https://github.com/CanelE452/covariate-trust-pilot)의
`ts-idea-tournament-v1` 브랜치가 이 카탈로그의 두 항목(P32 FAF/AdaRho, P33 MTLinear)을
이미 실험으로 다뤘다. 같은 주제를 다시 시작하기 전에 그쪽 결과를 먼저 본다.
