# mltimeseries

시계열 ML 연구 저장소. 2025년 말–2026년 시계열 예측 문헌을 정리하고, 그 위에서
후속 연구 주제를 고르는 것이 목적이다.

기준일: 2026-09-06

## 현재 상태

문헌 사전조사에서 나온 연구 주제 노트 3건 중 2건의 1차 파일럿을 실행했다 —
`HQ-TOKEN-PILOT-v1`(예측 목표 조건부 토큰화)과 `OA-RESOLUTION-PILOT-v1`(관측 방식
인지형 해상도 전이), 둘 다 판정은 `INCONCLUSIVE`다. 상세는
[`_docs/PROJECT_LOG.md`](_docs/PROJECT_LOG.md)의 계열 현황표를 본다.

## 무엇이 들어 있나

- [`_docs/PROJECT_LOG.md`](_docs/PROJECT_LOG.md) — 진입점. 계열 현황표, 자료 목록,
  해석 경계, 아직 확정하지 않은 후보 연구군 5개.
- [`_docs/notes/tsfm_topics/`](_docs/notes/tsfm_topics/) — 연구 주제 노트 3건. 01·02는
  1차 파일럿 결과까지, 03은 아직 착수 전 제안만 담고 있다.
- [`_docs/reference/time_series_research_dossier_20260906.md`](_docs/reference/time_series_research_dossier_20260906.md)
  — 48개 외부 연구·모델·벤치마크 항목의 사전조사. 항목마다 확인 깊이와 해석 경계를
  함께 적었다.
- [`_docs/reference/paper_catalog.json`](_docs/reference/paper_catalog.json)
  — 같은 48개 항목의 기계 판독용 정리. `venue_status` / `mechanism` / `inspection` /
  `interpretation_limit` / `primary_sources`.
- [`_docs/history/`](_docs/history/) — 날짜별 작업 기록.

## 이 자료를 읽을 때의 전제

조사 자체가 명시한 제약이며, 그대로 지킨다.

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
