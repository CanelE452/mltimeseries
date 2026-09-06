# mltimeseries

시계열 ML 연구 프로젝트. `covariate-trust-pilot` 과 저장소·기록을 분리한다.

시작: 2026-09-06

## 현재 상태

문헌 사전조사 자료 두 건(아래 "자료")에 더해, 2026-09-06에 `_docs/notes/tsfm_topics/`
아래 연구 주제 노트 3건이 들어왔다. 그중 1번(예측 목표 조건부 토큰화)과 2번(관측 방식
인지형 해상도 전이)의 1차 파일럿을 각각 실행했고, 둘 다 `INCONCLUSIVE`로 끝났다.
계열별 상태는 바로 아래 "계열 현황" 표에서 본다.

## 계열 현황

| 계열 | 묻는 것 | 상태 | 최신 결과 | 문서 |
|---|---|:--:|---|---|
| 01_forecast_query_tokenization | 같은 토큰 예산에서 예측 기간(horizon)을 함께 보는 압축이 입력만 보는 학습형 압축보다 정확한가 | 🟡 | 27/27 fit 완료, 판정 결론 보류(`INCONCLUSIVE`) — 주요 대조 macro -0.037% [-0.128%, +0.033%] (사전등록 문턱 +1.0%), 8개 조건 중 3개 통과 (09-06) | [notes](notes/tsfm_topics/01_forecast_query_tokenization.md) |
| 02_observation_aware_resolution | 관측 간격뿐 아니라 순간값·구간평균·구간합계 같은 관측 의미까지 모델에 알려주면 해상도가 바뀌어도 더 정확한가 | 🟡 | 12/12 fit 완료, 판정 `INCONCLUSIVE` — 사전등록 주 대조(미학습 보간, O vs M) macro -0.049%, bootstrap 95% CI [-0.279%, +0.175%] (0 포함), Go/No-Go 6개 중 4개 실패. Phase-2(pretrained model 이식) 제안 안 함, 이 계열 확대 중단 제안 (09-06). 사후 감사(AUDIT-CLOSURE-v1, model fit 0회)로 주 산술·bootstrap 재현 확인, FlowState native END_BIN 의미론 오류와 r=12 표 셀 매핑 오류 정정(집계·판정 무영향). 현재 구현은 Phase-2 미승격 | [notes](notes/tsfm_topics/02_observation_aware_resolution.md) |
| 03_uncertain_future_covariates | 미래 보조변수가 확정값이 아니라 예보일 때, 그 분포를 작은 표현으로 받는 모델이 점 입력 방식보다 나은가 | ⚪ | 노트만 보존, 데이터 계약 미확정 · 이번 회차는 실행하지 않기로 명시적으로 정함 | [notes](notes/tsfm_topics/03_uncertain_future_covariates.md) |

결론 난 계열이 아직 없으므로(1·2번 모두 `INCONCLUSIVE`로 열려 있음) 표를 나누지 않는다.

## 자료

| 파일 | 내용 |
| --- | --- |
| [`reference/time_series_research_dossier_20260906.md`](reference/time_series_research_dossier_20260906.md) | 2025년 말–2026년 시계열 논문·파운데이션 모델 사전조사. 업로드 포스터 4개 재독해, 48개 외부 항목, 5개 후보 연구군, 비교 시 필수 구분 |
| [`reference/paper_catalog.json`](reference/paper_catalog.json) | 위 조사의 48개 항목을 기계가 읽을 수 있게 정리 (`id` / `venue_status` / `mechanism` / `inspection` / `interpretation_limit` / `primary_sources`) |

두 파일은 업로드 원본을 그대로 옮긴 것이다 (배치 시 sha256 일치 확인). 이후 dossier 의
포스터 섹션 헤더 4개에서 발표자 실명·원본 PDF 파일명·학회 포스터 번호를 지우고
등장 순서대로 `포스터 1`~`포스터 4` 로 바꿨다 (2026-09-06, 개인정보 제거).
그 4줄 외에는 내용을 고치지 않는다.

## 이 자료를 어떻게 쓰고 어떻게 쓰지 않는가

dossier 자신이 경계를 명시하고 있고, 그대로 따른다.

- 대부분 항목은 **채택 상태와 공식 초록 수준의 스크리닝**이다. 전수 독해·재현·검산이 아니다.
- 성능 수치는 출처가 보고한 값이며 이 조사에서 실행해 확인한 값이 아니다.
- 메인 학회 채택 / workshop / arXiv 공개본 / 기업 공개를 구분한다.
  `arXiv 공개본; 정식 채택 미확인` 은 심사 중이라는 뜻이 아니다.
- 특정 검색어와 관심사에 따른 사전조사이지 체계적 문헌고찰이 아니므로,
  분야 전체의 빈도·비율을 추정하는 데 쓰지 않는다.
- **novelty 확정이나 새 실험 실행 승인의 근거로 쓰지 않는다.** 대표 논문을 깊게 볼 때는
  초록의 성능 주장을 옮기지 말고 target·input·split·checkpoint·compute·metric·
  baseline·ablation 을 추출한다.

## 아직 확정하지 않은 5개 후보 연구군

dossier 에 있는 그대로 옮긴다. 연구 기회가 남아 있다고 증명된 것이 아니라,
기능적으로 가장 가까운 선행연구를 더 읽기 위한 후보군이다.

| 연구군 | 대표 비교 문헌 | 다음 독해에서 답할 질문 |
| --- | --- | --- |
| Fine-tuning/adapter | SFF, Time-PEFT, CoRA, UniCA | LoRA/Full FT 보다 나아지는 원리가 무엇이며 최신 native multivariate 파운데이션 모델에도 남는 문제인가? |
| Recurrent/continuous forecasting | TiRex-2, FlowState, Toto 2.0 | state/context/해상도/horizon 을 어떤 구조로 처리하며 공개 코드가 실제 지원하는 범위는 어디까지인가? |
| Generative output/training loss | Sundial, MMPD, DBLoss, DistDF | 개선 대상이 point mean, quantile, joint trajectory 중 무엇이며 어떤 강한 baseline 을 넘어서는가? |
| Multimodal/related-context transfer | UniCA, VisionTS, TimeOmni-VL, In-Context Fine-Tuning | 추가 문맥 정보가 수치 예측을 실제로 개선하는가, 정보량·계산량 통제 후에도 효과가 남는가? |
| General representation | TSPulse, GTM, Zeus, CauKer | forecasting 외 분류/복원/이상탐지까지 범위를 넓힐 가치가 있는가, task-specific tuning 조건은 무엇인가? |

## 옆 저장소와의 접점

`covariate-trust-pilot` 의 `ts-idea-tournament-v1` 브랜치가 이 카탈로그의 두 항목을
이미 실험으로 건드렸다. 같은 주제를 다시 시작하기 전에 그쪽 결과를 먼저 본다.

- **P32 (FAF/AdaRho, AISTATS 2026)** — 그 실험의 Track F 가 AdaRho 를 논문 기준 로컬
  구현으로 재현해 비교했다. 동일 20% 예산에서 AdaRho 가 정상 regime 변화의 72–79% 를
  같이 버렸다. 단 원 논문의 online selection 을 완전히 재현한 것은 아니다.
- **P33 (MTLinear, AISTATS 2025)** — 그 실험의 Track G 가 PCGrad·norm-balanced 대조군을
  두었으나 MTLinear 자체는 read-only 참조로만 두었다.

두 결과 모두 `covariate-trust-pilot` 저장소 `ts-idea-tournament-v1` 브랜치의
`_docs/notes/ts-idea-tournament-v1.md` 에 있다.

## 다음에 할 일

1번 계열(예측 목표 조건부 토큰화)의 1차 파일럿이 `INCONCLUSIVE`로 끝나
[노트의 §12.8 다음 행동 후보](notes/tsfm_topics/01_forecast_query_tokenization.md)에
선택지가 정리돼 있다 — 이 계열을 계속할지 접을지는 아직 정해지지 않았다.

2번 계열(관측 방식 인지형 해상도 전이)의 1차 파일럿도 `INCONCLUSIVE`로 끝났다. 이쪽은
사전등록 주 대조가 명확히 실패했으므로(Go/No-Go 6개 중 4개 실패) 노트의
["다음 결정"](notes/tsfm_topics/02_observation_aware_resolution.md)에서 이 방법의
확대를 중단하고 다른 후보 주제로 이동할 것을 제안한다 — Phase-2(pretrained model 이식)는
제안하지 않는다.

3번 계열이나 dossier 의 5개 후보 연구군으로 넘어가는 경우, 연구군을 고르기 전에
dossier 의 "비교의 필수 구분" 절을 먼저 읽는다.
