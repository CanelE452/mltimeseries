# mltimeseries

시계열 ML 연구 프로젝트. `covariate-trust-pilot` 과 저장소·기록을 분리한다.

시작: 2026-09-06

## 현재 상태

아직 코드도 실험도 없다. 문헌 사전조사 자료 두 건만 들어와 있다.

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

정해진 것 없음. 연구군을 고르기 전에 dossier 의 "비교의 필수 구분" 절을 먼저 읽는다.
