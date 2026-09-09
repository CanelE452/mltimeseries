# mltimeseries

시계열 ML 연구 저장소. 2025년 말–2026년 시계열 예측 문헌을 정리하고, 그 위에서
후속 연구 주제를 고르는 것이 목적이다.

상태 갱신: 2026-09-08

## 현재 상태

문헌 조사와 여러 PEFT 실험을 진행했다. 현재 A/B/C의 제한된 진단에서는 새 방법의 필요성을 뒷받침하는 사전 기준을 통과하지 못했다. [최신 실행 기록](_docs/PROJECT_LOG.md)과 [후보별 결과](_docs/notes/tsfm_topics/)에서 근거와 적용 범위를 확인할 수 있다. 아직 새로운 ML 방법이나 독립 확증 결과를 확보한 것은 아니다.

최근 [15번 목적함수 정렬 진단](_docs/notes/tsfm_topics/15_peft_objective_alignment_results_20260908.md)의 6fit·6forecast를 완료했다. 동일한 LoRA에서 normalized loss와 raw loss를 비교했으나 사전 실용 기준을 통과하지 못해 현재 변경을 닫았다. 이어 [16번 관측 연산 진입 검사](_docs/notes/tsfm_topics/16_observation_operator_entry_results_20260908.md)는 CPU에서 기존 Gaussian 계산의 충분성을 확인했다. USCRN 두 제품의 마지막5분 평균은 일치했지만 시간 평균 재집계에는0.125°C 예외 한 건이 있어 단순 산술 항등성을 가정하지 않는다. 새 PEFT 방법은 아직 확보하지 못했다. 구현은 [experiments/](experiments/), 실행 기록은 `runs/`, 분석 산출물은 `results/`에 둔다.

최신 [17번 월합 감독 실험](_docs/notes/tsfm_topics/17_coarse_supervision_results_20260908.md)도 완료했다. 월합으로 적합한 head는 Eagle의 월평균 오차를 조금 줄였지만 시간별 MSE는 동결 모델의 약 28배로 악화됐다. 그 head를 고정한 LoRA는 손상을 회복하지 못해 현재 조건을 종료한다. 월합 감독으로 통제되지 않는 패턴 변경이라는 구체 현상은 확인했지만, 새 PEFT 방법의 해결 성과는 아직 없다. CPU68검사와 독립 수치 검산을 통과했고 9개 guard 모두 정상 종료했다.

## 무엇이 들어 있나

- [`_docs/PROJECT_LOG.md`](_docs/PROJECT_LOG.md) — 진입점. 자료 목록, 해석 경계,
  연구 후보, 실행 결과와 다음 단계.
- [`_docs/reference/time_series_research_dossier_20260906.md`](_docs/reference/time_series_research_dossier_20260906.md)
  — 48개 외부 연구·모델·벤치마크 항목의 사전조사. 항목마다 확인 깊이와 해석 경계를
  함께 적었다.
- [`_docs/reference/paper_catalog.json`](_docs/reference/paper_catalog.json)
  — 같은 48개 항목의 기계 판독용 정리. `venue_status` / `mechanism` / `inspection` /
  `interpretation_limit` / `primary_sources`.
- [`_docs/history/`](_docs/history/) — 날짜별 작업 기록.

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
