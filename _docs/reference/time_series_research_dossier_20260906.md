# 시계열 ML 사전조사 — 2025년 말·2026년 논문과 파운데이션 모델

기준일: **2026-09-06**

목적: 새 forecasting/representation/학습 방법을 찾기 위한 문헌 지도다. 기존 G/X/F 또는 Point–Hurdle 실험을 재실행하는 지시문이 아니다.

## 조사 범위와 증거 수준

처음 업로드한 포스터 4개는 본문 전체와 페이지 이미지를 다시 읽었다. 아래 외부 자료는 공식 proceedings, 학회 프로그램, 저자 논문, 공식 프로젝트 자료에서 수집했다. 대다수는 **채택 상태와 공식 초록 수준의 스크리닝**이며, TiRex-2는 구조·학습·평가 관련 본문/부록도 일부 검토했다. 모든 논문을 전체 재현·검산한 것으로 읽으면 안 된다.

- 메인 학회 채택과 workshop·arXiv·기업 공개를 구분한다.
- `arXiv 공개본; 정식 채택 미확인`은 심사가 진행 중이라는 뜻이 아니다.
- CoRA 및 일부 OpenReview 페이지는 브라우저 확인 절차에 막혔다. 실제 리뷰 점수·리뷰어 발언은 확보한 것으로 보고하지 않는다.
- 성능·데이터 규모는 출처가 보고한 내용이며 본 조사에서 모델을 실행해 확인한 값이 아니다.
- 이 목록은 특정 검색어와 관심사에 따른 사전조사이며 분야 전체의 빈도나 비율을 추정하는 체계적 문헌고찰은 아니다.

## 포스터를 다시 읽고 남긴 구분

### 포스터 1 — 변수별 신뢰도 기반 gradient 조정

포스터의 중심은 단일 backward로 만든 변수별 gradient proxy가 실제 gradient를 얼마나 잘 재구성하는지 검사하고, 신뢰 가능한 layer에서만 방향을 조정하는 것이다. 여기서 reliability는 독립 미래 구간의 성능 예측 정확도가 아니라 **gradient proxy 복원 정확도**다. G 토너먼트의 probe-harm 분석과 동일하지 않다. 7 datasets × 5 backbones에서 다수 개선을 포스터가 보고하지만, 한 장의 자료만으로 모든 seed·학습예산·유의성 조건까지 확인한 것은 아니다.

### 포스터 2 — Retrieval-Guided Residual Correction

유사 과거의 미래값 대신 기본 모델이 그때 냈던 residual vector를 검색하고 bounded correction으로 출력에 더한다. 정답이 관측된 후 key–residual memory를 갱신한다. 표의 실제 비교 backbone은 iTransformer이며, 다양한 다른 backbone으로의 확장은 future-work 문맥이다. 기존 expert-loss memory는 모델 가중치를 선택하는 문제였으므로 같은 방법의 반증이 아니다.

### 포스터 3 — Adaptive Reference Memory

재구성 기반 이상탐지의 over-generalization과 정상분포 변화 오경보를 다룬다. K-means/farthest-point 초기화, memory-matched reconstruction·kNN 거리, EMA scale 정렬, anomaly gate/diversity/eviction을 결합한다. 안정된 분포에서 이득이 제한된다는 한계도 포스터에 있다. forecasting MSE/SOTA가 아니라 anomaly detection 연구다.

### 포스터 4 — Hurdle Decomposition

주변분포를 고정하고 발생 간격·크기의 순서를 조작해 direct mean과 factorized mean을 비교한 연구다. 여기서 유지할 자산은 공정한 비교와 주장에 맞춘 실험이며, 이후 연구까지 간헐수요·rho·head routing에 묶을 필요는 없다.

## 조사에서 반복적으로 확인된 방향 [문헌 종합에 따른 해석]

1. **Native multivariate/covariate modeling:** 최신 FM은 단변량 모델 위에 외부 회귀를 덧붙이는 것만이 아니라, 사전학습 구조에서 변수와 공변량을 직접 섞는다.
2. **구조·학습 목표·데이터의 결합:** TiRex 계열의 masking/state tracking, Sundial/MMPD의 생성 목적함수, FlowState의 연속시간 decoder처럼 한 구성요소의 목적이 명확하다.
3. **사전학습과 downstream adaptation은 별도 연구 문제:** CoRA/UniCA/SFF/Time-PEFT처럼 무엇을 업데이트하고 어떻게 기존 표현을 유지할지가 방법의 중심이다.
4. **고정 모델 위의 추론도 연구 대상:** TATO/COSA/ICF/RAFT/GTR은 입력 변환, 문맥, 잔차 보정 등 서로 다른 위치에 개입한다.
5. **거대 모델만 존재하는 것은 아님:** TSPulse·SEMPO·FACT 등 작은 모델/모듈/목적함수도 메인 학회 연구다. 작은 파라미터 수가 자동으로 저렴한 사전학습을 뜻하지는 않는다.
6. **forecaster와 representation model을 구분:** 분류/복원/이상탐지/멀티모달 reasoning의 성과를 GIFT-Eval forecasting SOTA와 혼합하지 않는다.

## TiRex-2를 고려할 때의 핵심

`tirez2`는 문맥상 **TiRex-2**를 가리키는 것으로 해석했다.

- TiRex 원 논문은 NeurIPS 2025 메인; TiRex-2의 현재 확인된 문헌은 2026-07 arXiv 공개본이다.
- 시간 정보를 처리하는 xLSTM과 변수 간 정보를 결합하는 attention을 나누고, future-known covariates를 처리하되 target의 시간 인과성을 유지하는 비대칭 연결을 사용한다.
- 기존 단변량 corpus를 서로 coupling하여 다변량 학습 예제로 만든다. ‘합성 데이터’라는 이름은 같아도 기존 matched-marginal 검증 실험과 목적이 다르다.
- 공식 README는 단변량 모드 38.4M 활성 파라미터, 다변량 모드에서 추가 44.1M을 명시한다. 다변량 수치는 합계 약 82.5M으로 읽어야 한다.
- 논문은 사전학습 700,000 steps, H100 두 장, 약 50시간을 보고한다. 이는 해당 환경의 저자 수치이며 4070/4090 예상 시간으로 바로 환산할 수 없다.
- 논문의 streaming 설계와 OSS 제품 기능을 분리해야 한다. 공식 introduction은 공개 `forecast`가 호출마다 전달된 전체 context를 다시 계산하며, incremental state-carrying API는 공개 릴리스에 없다고 명시한다. 현재 README는 incremental streaming, fine-tuning 등을 TiRex-2 Pro 확장으로 소개한다. 연구자가 별도 구현을 할 수 없다는 뜻도 아니다. 출처: <https://nx-ai.github.io/tirex-2/introduction/>
- GIFT-Eval/fev용으로 사전학습 corpus를 다르게 제외한 checkpoint를 사용했다고 서술하므로, 실제 재현에서는 이름만 같은 모델이 아니라 checkpoint와 corpus 규칙까지 맞춰야 한다.

## 아직 주제로 확정하지 않고 함께 검토할 5개 연구군

| 연구군 | 대표 비교 문헌 | 다음 독해에서 답할 질문 |
|---|---|---|
| Fine-tuning/adapter | SFF, Time-PEFT, CoRA, UniCA | LoRA/Full FT보다 나아지는 원리가 무엇이며 최신 native multivariate FM에도 남는 문제인가? |
| Recurrent/continuous forecasting | TiRex-2, FlowState, Toto 2.0 | state/context/해상도/forecast horizon을 어떤 구조로 처리하며 공개 코드가 실제 지원하는 범위는 어디까지인가? |
| Generative output/training loss | Sundial, MMPD, DBLoss, DistDF | 개선 대상이 point mean, quantile, joint trajectory 중 무엇이며 어떤 강한 baseline을 넘어서는가? |
| Multimodal/related-context transfer | UniCA, VisionTS, TimeOmni-VL, In-Context Fine-Tuning | 추가 문맥 정보가 수치 예측을 실제로 개선하는가, 정보량·계산량 통제 후에도 효과가 남는가? |
| General representation | TSPulse, GTM, Zeus, CauKer | forecasting 외 분류/복원/이상탐지까지 범위를 넓힐 가치가 있는가, task-specific tuning 조건은 무엇인가? |

위 항목은 연구 기회가 남아 있다고 증명된 새 아이디어가 아니다. 기능적으로 가장 가까운 선행연구를 더 읽기 위한 후보군이다. 이번 단계에서는 새 loss·데이터 생성기·실험 게이트를 설계하지 않는다.

## 비교의 필수 구분

- zero-shot, target-data fine-tuning, online feedback adaptation은 같은 예산이 아니다.
- univariate, multivariate target, 과거 공변량, 미래에 실제 알려진 공변량은 다른 정보 조건이다.
- 파라미터 수, 활성 파라미터 수, 사전학습 비용, 추론 지연은 별도 값이다.
- GIFT/fev/TIME 순위는 모델 제출 범위·버전·metric·aggregation·leakage policy에 의존한다.
- 저자의 최고 성능 주장은 발표 당시 comparison set의 결과다. 이 문서는 현재 실시간 종합 leaderboard 1위를 인증하지 않는다.
- `code available`은 곧 `모든 training/evaluation code 실행 검증`이 아니다.

## 수집 자료 목록

각 항목의 ‘확인 깊이’와 ‘해석 경계’를 함께 읽는다.

총 **48개 외부 연구/모델/벤치마크 항목**. 업로드 포스터 4개는 이 숫자에 포함하지 않았다.

### 파운데이션 모델·구조

#### P01 · TiRex: Zero-Shot Forecasting Across Long and Short Horizons with Enhanced In-Context Learning

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** xLSTM의 상태 추적과 contiguous patch masking을 결합한 zero-shot forecasting.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 발표 당시 비교에서의 우위이지 2026년 모든 모델 대비 현재 1위를 뜻하지 않는다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/5356603f9c47399adfd372f77a677057-Abstract-Conference.html>  

#### P02 · TiRex-2: Generalizing TiRex to Multivariate Data and Streaming

**연도·상태:** 2026 / 2026-07 arXiv 공개본; 정식 채택 미확인  
**핵심:** 시간 mixer와 비대칭 변수 attention, 합성 다변량 coupling, recurrent state 재사용.  
**확인 깊이:** 본문 §3–4·부록 B/D/E 일부와 공식 README 확인  
**해석 경계:** 논문상 streaming/finetuning과 OSS 제품 API 지원은 다르다. 공식 README는 streaming·fine-tuning 등을 Pro 확장으로 소개한다.

출처 1: <https://arxiv.org/html/2607.01204v1>  
출처 2: <https://github.com/NX-AI/tirex-2>  

#### P03 · Chronos-2: From Univariate to Universal Forecasting

**연도·상태:** 2025 / 2025-10 arXiv 기술보고서; 정식 채택 미확인  
**핵심:** 시간 attention과 group attention으로 다변량·관련 계열·공변량을 통합; 합성 다변량 학습.  
**확인 깊이:** 공개본 초록·Amazon Science 설명·공식 README 확인  
**해석 경계:** 원래 Chronos의 TMLR 2024 채택과 Chronos-2의 상태를 혼동하지 않는다.

출처 1: <https://arxiv.org/abs/2510.15821>  
출처 2: <https://www.amazon.science/blog/introducing-chronos-2-from-univariate-to-universal-forecasting>  
출처 3: <https://github.com/amazon-science/chronos-forecasting>  

#### P04 · TimesFM 3.0

**연도·상태:** 2026 / 2026-08 기업 공식 모델 공개; 별도 메인 채택 미확인  
**핵심:** 다변량 target과 past/future-known covariates를 직접 처리하는 최신 TimesFM 세대.  
**확인 깊이:** 저자 공식 자료·README 확인; 학습 실행 없음  
**해석 경계:** 소스 코드와 가중치 라이선스를 분리한다. 현재 저장소는 3.0 가중치를 비상업·비프로덕션용으로 명시한다. 순위는 저자 보고.

출처 1: <https://github.com/google-research/timesfm>  
출처 2: <https://research.google/blog/timesfm-3-a-zero-shot-foundation-model-for-multivariate-forecasting/>  

#### P05 · Toto 2.0: Time Series Forecasting Enters the Scaling Era

**연도·상태:** 2026 / 2026-05 arXiv 기술보고서; 정식 채택 미확인  
**핵심:** 4M–2.5B 모델군, 공통 학습 recipe와 u-muP hyperparameter transfer, 한 번/블록 단위 예측.  
**확인 깊이:** 기술보고서 초록·Datadog 공식 공개 설명·모델 카드 확인  
**해석 경계:** 스케일링·리더보드 결과는 해당 데이터와 저자 실험 범위다. 현재 보편적 최고라고 단정하지 않는다.

출처 1: <https://arxiv.org/abs/2605.20119>  
출처 2: <https://www.datadoghq.com/blog/ai/toto-2/>  

#### P06 · FlowState: Sampling-Rate-Equivariant Time-Series Forecasting

**연도·상태:** 2026 / ICML 2026 메인  
**핵심:** State-space encoder와 functional basis decoder로 관측 해상도·예측 길이를 유연하게 처리.  
**확인 깊이:** ICML 공식 목록·IBM publication/event의 Conference paper 확인  
**해석 경계:** NeurIPS 2025 선행 버전은 workshop paper다. 2026 버전과 구분.

출처 1: <https://research.ibm.com/publications/flowstate-sampling-rate-equivariant-time-series-forecasting>  
출처 2: <https://research.ibm.com/events/icml-2026>  
출처 3: <https://icml.cc/Downloads/2026>  

#### P08 · Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts

**연도·상태:** 2025 / ICML 2025 메인  
**핵심:** 주기별 수동 모델 분할 대신 sparse MoE를 이용한 token-level 전문화.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** MoE는 이미 존재한다. 단순 expert 추가가 새 기여는 아니다.

출처 1: <https://proceedings.mlr.press/v267/liu25an.html>  

### 생성 예측·목적함수

#### P07 · Sundial: A Family of Highly Capable Time Series Foundation Models

**연도·상태:** 2025 / ICML 2025 메인  
**핵심:** TimeFlow loss로 다음 patch의 연속 예측분포를 생성.  
**확인 깊이:** 공식 ICML 초록·저자 프로젝트 확인  
**해석 경계:** 포인트 예측과 분포 예측, 샘플 수/추론비용을 분리해서 비교해야 한다.

출처 1: <https://icml.cc/virtual/2025/poster/45591>  
출처 2: <https://github.com/thuml/Sundial>  

#### P23 · DBLoss: Decomposition-based Loss Function for Time Series Forecasting

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** 정답·예측을 trend/seasonal 성분으로 분해해 학습 손실을 구성.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 새 분포 family를 추가하는 것과 다른 학습 objective 연구다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/2823c4f838727564c108c85d7e6e8507-Abstract-Conference.html>  

#### P24 · MMPD: Diverse Time Series Forecasting via Multi-Mode Patch Diffusion Loss

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 미래 잠재 patch에 조건화한 diffusion과 다중 mode 표현으로 여러 가능한 미래를 모델링.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** point 평균오차와 predictive diversity를 별도 평가해야 한다. MSE가 언제나 Gaussian 가정을 필요로 한다고 일반화하지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/be7b70477c8fca697f14b1dbb1c086d1-Abstract-Conference.html>  

#### P25 · DistDF: Time-Series Forecasting Needs Joint-Distribution Wasserstein Alignment

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 예측과 정답의 joint-distribution discrepancy로 학습하는 미분 가능한 Wasserstein 정렬.  
**확인 깊이:** 공식 학회 초록·arXiv 초록 확인  
**해석 경계:** 초록의 MSE/likelihood 비판을 모든 회귀문제에 대한 보편적 정리로 확대하지 않는다.

출처 1: <https://iclr.cc/virtual/2026/poster/10009095>  
출처 2: <https://arxiv.org/abs/2510.24574>  

### 경량 모델·표현

#### P09 · SEMPO: Lightweight Foundation Models for Time Series Forecasting

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** 고·저 에너지 주파수 정보를 반영한 mixture-of-prompts 기반 경량 사전학습.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 경량 모델이라는 사실과 사전학습 데이터·시간 비용은 별개다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/ecfb69ce6be017deb5a926c2718f6bc1-Abstract-Conference.html>  

#### P10 · GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series data

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 주파수 attention, reconstruction/autoregressive masking 등으로 일반적인 시계열 표현을 학습.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 예측뿐 아니라 표현·분류 등 과제별 설정을 분리해야 한다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/0c6639f49f01a8578675303ce0030233-Abstract-Conference.html>  

#### P11 · TSPulse: Tiny Pre-Trained Models with Disentangled Representations for Rapid Time-Series Analysis

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 약 1M 파라미터 모델에서 시간·주파수·의미 표현을 분리하고 작은 task 모듈로 재사용.  
**확인 깊이:** 공식 학회 초록 확인  
**해석 경계:** 이상탐지·검색·복원·분류 성능을 forecasting SOTA로 바꾸어 말하지 않는다.

출처 1: <https://iclr.cc/virtual/2026/poster/10010081>  

#### P12 · Zeus: Towards Tuning-Free Foundation Model for Time Series Analysis

**연도·상태:** 2026 / ICML 2026 메인  
**핵심:** point-wise token, U자형 multi-scale 구조, multi-objective temporal masking으로 다과제 재사용.  
**확인 깊이:** 공식 학회 목록 및 저자 공개본 초록 확인  
**해석 경계:** tuning-free라는 명칭도 실제 task별 입력·사전학습/평가 설정을 확인해야 한다.

출처 1: <https://arxiv.org/abs/2607.01918>  
출처 2: <https://icml.cc/Downloads/2026>  

#### P26 · MoFo: Empowering Long-term Time Series Forecasting with Periodic Pattern Modeling

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** 주기에 맞춘 patch 구성과 modulation으로 장기 패턴을 표현.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 데이터셋별 학습 모델의 성능과 zero-shot FM 성능은 별개다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/7a99ad21706dec5b28f9ad715e12197f-Abstract-Conference.html>  

#### P27 · FACT: Fine-grained Across-variable Convolution for Multivariate Time Series Forecasting

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 시간·주파수의 세밀한 변수 상호작용을 depth-wise/dilated 2D convolution으로 모델링.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 변수 간 모델링이 반드시 attention이어야 한다는 가정을 깨는 구조 연구.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/090d636f951ac0bb18fd5610c95c6a76-Abstract-Conference.html>  

### 적응·미세조정

#### P13 · CoRA: Boosting Time Series Foundation Models for Multivariate Forecasting through Correlation-aware Adapter

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 시변/불변 상관 구조를 저랭크 모듈과 contrastive supervision으로 파운데이션 모델에 반영.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 현재 native multivariate 모델까지 baseline에 넣어야 한다. 개별 OpenReview 리뷰 본문 접근은 차단됐다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/ae3e173398d6e43fea63cbcc16fbaa98-Abstract-Conference.html>  
출처 2: <https://openreview.net/forum?id=JRlNrcTllN>  

#### P14 · UniCA: Unified Covariate Adaptation for Time Series Foundation Model

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 수치 외의 범주·이미지·텍스트 공변량까지 시계열 표현으로 정렬해 통합.  
**확인 깊이:** 공식 ICLR 초록 및 프로그램 확인  
**해석 경계:** 추가 modality가 실제 예측 시점에 이용 가능한지부터 구분해야 한다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/0b5eb45a22ff33956c043dd271f244ea-Abstract-Conference.html>  

#### P15 · Lost in the Non-convex Loss Landscape: How to Fine-tune the Large Time Series Model?

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** Smoothed Full Fine-tuning: 사전학습 가중치와 무작위 초기화 모델 가중치의 보간 후 전체 미세조정.  
**확인 깊이:** 공식 학회 초록 확인  
**해석 경계:** 일반적인 fine-tuning과 비교할 때 동일 backbone/예산/검증 선택을 유지해야 한다.

출처 1: <https://iclr.cc/virtual/2026/poster/10011175>  

#### P16 · Time-PEFT: Temporal and Multichannel Complexity-Based Fine-Tuning for Time-Series Foundation Models

**연도·상태:** 2026 / ICML 2026 메인  
**핵심:** 시간·다채널 복잡도를 진단하고 주파수 adapter와 채널 adapter를 결합.  
**확인 깊이:** ICML 공식 목록과 저자 README 확인  
**해석 경계:** MOMENT 기반 보고를 TiRex-2·Chronos-2에서도 검증된 것으로 확대하지 않는다.

출처 1: <https://github.com/kaist-dmlab/TimePEFT>  
출처 2: <https://icml.cc/Downloads/2026>  

### 추론·온라인 적응

#### P17 · Context-aware Output-Space Adapter for Test-Time Adaptation in Time Series Forecasting

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** COSA: frozen base의 출력에 문맥 조건부 residual adapter를 적용하고 관측된 feedback으로 갱신.  
**확인 깊이:** 공식 학회 초록 확인  
**해석 경계:** 이미 도착한 정답을 쓰는 online adaptation과 정답 없는 test-time adaptation을 구분해야 한다.

출처 1: <https://iclr.cc/virtual/2026/poster/10010061>  

#### P18 · Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** TATO: 모델보다 입력의 context/scale/outlier 변환을 조정하여 frozen 모델의 성능을 개선.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 이전 공개 초록과 최종 proceedings의 효과 수치가 달라 숫자 인용에는 버전 고정이 필요하다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/48c5226582f41254026748c7e35d4ac2-Abstract-Conference.html>  

#### P19 · In-Context Fine-Tuning for Time-Series Foundation Models

**연도·상태:** 2025 / ICML 2025 메인  
**핵심:** 관련 시계열 예제를 문맥으로 이용하도록 훈련해 추론 시 gradient update 없이 적응.  
**확인 깊이:** 공식 학회 초록 확인  
**해석 경계:** 제목의 fine-tuning과 실제 배포 시 gradient-free adaptation을 구분.

출처 1: <https://icml.cc/virtual/2025/poster/43707>  

#### P20 · Retrieval Augmented Time Series Forecasting

**연도·상태:** 2025 / ICML 2025 메인  
**핵심:** RAFT: 유사 과거 구간과 그 이후 값을 훈련 데이터에서 검색해 예측 입력에 보강.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 업로드한 residual-memory 포스터는 미래값 대신 잔차를 검색한다는 차이가 있다.

출처 1: <https://proceedings.mlr.press/v267/han25d.html>  

#### P21 · Enhancing Multivariate Time Series Forecasting with Global Temporal Retrieval

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** GTR: 짧은 lookback 밖의 전역 주기를 temporal embedding·retrieval·잔차 융합으로 반영.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 모든 retrieval이 raw 사례 검색은 아니다. 전역 주기 표현과 실제 memory bank를 구분.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/7a3284ce53f620d244fca0c79d94c478-Abstract-Conference.html>  

#### P22 · Lightweight Online Adaption for Time Series Foundation Model Forecasts

**연도·상태:** 2025 / ICML 2025 메인  
**핵심:** ELF: 고정 FM과 경량 online forecaster를 feedback 기반 weighter로 결합.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 기존 Point/Hurdle-only loss weighting 실험이 이 방법을 직접 반증하지 않는다.

출처 1: <https://proceedings.mlr.press/v267/lee25ag.html>  

### 사전학습 데이터

#### P28 · Synthetic Series-Symbol Data Generation for Time Series Foundation Models

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** SymTime: 시계열과 생성 규칙의 기호 표현을 연결하는 합성 학습 데이터.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 통제실험용 matched-marginal 생성과 표현 사전학습용 생성은 목적이 다르다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/0f45ce821121c459398fd03a0ecc6b60-Abstract-Conference.html>  

#### P29 · Forging Time Series with Language: A Large Language Model Approach to Synthetic Data Generation

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** SDForger: 시계열 임베딩과 언어모델을 연결한 합성 시계열 생성.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** synthetic fidelity와 실제 downstream forecasting 향상을 별도로 검증해야 한다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/70fb27f33beefef137e09bbd875c28ad-Abstract-Conference.html>  

#### P30 · CauKer: Classification Time Series Foundation Models Can Be Pretrained on Synthetic Data

**연도·상태:** 2026 / ICLR 2026 메인·Oral 확인  
**핵심:** GP kernel 조합과 구조적 인과모델을 이용해 분류 FM용 합성 시계열을 생성.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 주과제는 classification이다. forecasting leaderboard 성능으로 바꿔 읽지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/702b67152ec4435795f681865b67999c-Abstract-Conference.html>  
출처 2: <https://iclr.cc/virtual/2026/events/oral>  

### 학습 신호·정규화

#### P31 · DropoutTS: Sample-Adaptive Dropout for Robust Time Series Forecasting

**연도·상태:** 2026 / ICML 2026 메인  
**핵심:** 주파수 재구성 잔차로 sample noise를 추정하고 dropout 강도를 조정.  
**확인 깊이:** 공식 학회 목록 및 저자 공개본 초록 확인  
**해석 경계:** 표본을 제거하는 선별법과 다르다. 공식 ICML 목록에서 채택을 교차 확인.

출처 1: <https://arxiv.org/abs/2601.21726>  
출처 2: <https://icml.cc/Downloads/2026>  

#### P32 · Filter, Augment, Forecast: Online Data Selection for Robust Time Series Forecasting

**연도·상태:** 2026 / AISTATS 2026 메인  
**핵심:** reference model 기반 online data selection과 augmentation; shift에 맞춰 reference 갱신.  
**확인 깊이:** 공식 proceedings 초록 검색 확인  
**해석 경계:** 정적 1-pass selection-frequency filter를 원 논문의 완전한 재현이라고 부르지 않는다.

출처 1: <https://proceedings.mlr.press/v300/taga26a.html>  

#### P33 · A Multi-Task Learning Approach to Linear Multivariate Forecasting

**연도·상태:** 2025 / AISTATS 2025 메인  
**핵심:** MTLinear: 변수 관련성에 따른 task grouping과 prediction-error 기반 gradient scaling.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** generic PCGrad 또는 업로드된 PV-surgery와 동일한 방법은 아니다.

출처 1: <https://proceedings.mlr.press/v258/nochumsohn25a.html>  

#### P34 · TimeDistill: Efficient Long-Term Time Series Forecasting with MLPs via Cross-Architecture Distillation

**연도·상태:** 2026 / 저자 공식 저장소가 KDD 2026으로 명시; publisher proceedings 교차 확인 미완료  
**핵심:** Transformer/CNN의 multi-scale·multi-period 표현을 가벼운 MLP에 증류.  
**확인 깊이:** 저자 공식 자료·README 확인; 학습 실행 없음  
**해석 경계:** 공식 저장소의 학회 표기 수준으로 기록. 이전 확률 head 증류 실험과 전달 대상이 다르다.

출처 1: <https://github.com/LingFengGold/TimeDistill>  
출처 2: <https://arxiv.org/abs/2502.15016>  

### 멀티모달·추론

#### P35 · TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 시계열 인식·인과·사건 조건 예측 등 reasoning 과제와 보상 기반 학습.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 자연어 reasoning 점수의 향상을 일반 forecasting MSE SOTA와 같게 보지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/f7227d037ee7202106f511f78e9ecb80-Abstract-Conference.html>  

#### P36 · TimeOmni-VL: Unified Models for Time Series Understanding and Generation

**연도·상태:** 2026 / ICML 2026 메인  
**핵심:** 시계열↔이미지 양방향 매핑과 이해 결과를 이용한 수치 생성.  
**확인 깊이:** 공식 학회 목록 및 저자 공개본 초록 확인  
**해석 경계:** near-lossless mapping 및 의미 이해·수치 생성 각각의 실험 조건을 분리해서 봐야 한다.

출처 1: <https://arxiv.org/abs/2602.17149>  
출처 2: <https://icml.cc/Downloads/2026>  

#### P37 · VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters

**연도·상태:** 2025 / ICML 2025 메인  
**핵심:** 이미지 사전학습 masked autoencoder의 복원 능력을 시계열 예측으로 전이.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 이미지로 바꾸기 자체가 새 아이디어는 아니다. 2026 모델과 다시 비교가 필요하다.

출처 1: <https://proceedings.mlr.press/v267/chen25be.html>  

### 분류·이상탐지·의료 표현

#### P38 · MIRA: Medical Time Series Foundation Model for Real-World Health Data

**연도·상태:** 2025 / NeurIPS 2025 메인  
**핵심:** 불규칙 의료 관측을 위한 연속시간 위치 표현과 frequency expert 등 도메인 표현.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 의료 데이터 접근성과 task 특성이 일반 시계열 benchmark와 다르다.

출처 1: <https://proceedings.neurips.cc/paper_files/paper/2025/hash/8e12ba543adc673da5b89c9311fcf72c-Abstract-Conference.html>  

#### P39 · Repurposing Foundation Model for Generalizable Medical Time Series Classification

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** FORMED: task/channel·label query를 이용한 의료 시계열 분류 전이.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** forecasting과 분류 SOTA를 구분한다. 새 task query의 학습량도 공개해야 한다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/8707924df5e207fa496f729f49069446-Abstract-Conference.html>  

#### P40 · When Foundation Models are One-Liners: Limitations and Future Directions for Time Series Anomaly Detection

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** FM 기반 이상 점수를 moving variance 등 매우 단순한 baseline과 비교.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 일부 점수화 방식·benchmark의 한계를 모든 TSFM의 무용성으로 확대하지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/cf70320e93c08b39b1b29a348097a376-Abstract-Conference.html>  

#### P41 · Complexity- and Statistics-Guided Anomaly Detection in Time Series Foundation Models

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 복잡도에 맞춘 예측·통계 정보로 over-generalization과 정규화 손실을 보완.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 이상 점수 개선을 평균 예측성능 개선과 섞지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/d06768735ce7d11ade5baef7099b4e05-Abstract-Conference.html>  

#### P42 · Adaptive Conformal Anomaly Detection with Time Series Foundation Models for Signal Monitoring

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** FM 예측오차에 adaptive conformal anomaly scoring을 결합.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** coverage 보장은 가정과 online protocol을 확인해야 하며 arbitrary shift에 무조건 보장되지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/f54c9fc57aa6e1c72400cc127917fcf8-Abstract-Conference.html>  

### 벤치마크·평가

#### P43 · Beyond Accuracy: Are Time Series Foundation Models Well-Calibrated?

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 정확도 외에 output head·horizon·AR decoding에 따른 uncertainty calibration을 분석.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 진단 논문은 방법 연구의 비교 기준으로 활용하며 사용자 연구를 자동으로 분석 논문으로 돌리지 않는다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/9af2b1d6acf561af9c4cf70d52c7a49d-Abstract-Conference.html>  

#### P44 · Context parroting: A simple but tough-to-beat baseline for foundation models in scientific machine learning

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 동역학 예측에서 context 복사 baseline과 FM을 비교.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 해당 과제군에서의 관찰이지 모든 시계열에서 복사가 최선이라는 뜻은 아니다.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/5fbdcd4d2190682b913c6b39b58e95f0-Abstract-Conference.html>  

#### P45 · TimeRecipe: A Time-Series Forecasting Recipe via Benchmarking Module Level Effectiveness

**연도·상태:** 2026 / ICLR 2026 메인  
**핵심:** 정규화·분해·임베딩·모델 등의 구성요소를 1만 회 이상 조합 실험으로 분석.  
**확인 깊이:** 정식 메인 논문집·초록 확인  
**해석 경계:** 초기 작은 연구에 동일 실험량을 요구하는 것이 아니다. 강한 조합 baseline을 찾는 참고 자료.

출처 1: <https://proceedings.iclr.cc/paper_files/paper/2026/hash/f46ddea413df86832418c5e04e59644f-Abstract-Conference.html>  

#### P46 · GIFT-Eval: A Benchmark for General Time Series Forecasting Model Evaluation

**연도·상태:** 2024 / 공개 benchmark; NeurIPS 2024 TSALM workshop/ICLR 2025 제출본 확인, 메인 채택은 이번 조사 미확인  
**핵심:** 다양한 빈도·도메인·horizon에 대한 일반 시계열 예측 평가와 비누출 pretraining corpus.  
**확인 깊이:** 논문 초록·OpenReview 검색 결과·공식 benchmark 자료 확인  
**해석 경계:** 다른 공개 버전에서 dataset 수가 다르다. 과제 수·모델 수·train policy를 version pin해야 한다.

출처 1: <https://arxiv.org/abs/2410.10393>  
출처 2: <https://openreview.net/forum?id=9EBSEkFSje>  
출처 3: <https://github.com/SalesforceAIResearch/gift-eval>  

#### P47 · fev-bench: A Realistic Benchmark for Time Series Forecasting

**연도·상태:** 2025 / 2025-09 arXiv 공개본; 별도 메인 채택 미확인  
**핵심:** 공변량 포함 과제와 bootstrap win rate/skill score를 제공하는 100-task benchmark.  
**확인 깊이:** 공개 초록 확인  
**해석 경계:** 현재 코드 버전과 paper version을 고정하고 zero-shot/FT 조건을 구분한다.

출처 1: <https://arxiv.org/abs/2509.26468>  
출처 2: <https://github.com/autogluon/fev>  

#### P48 · It's TIME: Towards the Next Generation of Time Series Forecasting Benchmarks

**연도·상태:** 2026 / ICML 2026 메인  
**핵심:** 새로운 50개 데이터·98개 과제와 task-centric zero-shot 평가.  
**확인 깊이:** 공식 학회 목록 및 저자 공개본 초록 확인  
**해석 경계:** 저자들이 contamination-resistant 설계를 주장한다. 모든 사전학습 corpus의 절대적 비누출을 독립 검증한 것은 아니다.

출처 1: <https://arxiv.org/abs/2602.12147>  
출처 2: <https://icml.cc/Downloads/2026>  

## 접근 제한 및 미완료 독해

CoRA, GIFT-Eval 등의 OpenReview forum을 열었으나 일부는 브라우저 검증 화면으로 전환됐다. 따라서 이번 조사에는 실제 리뷰어 점수, rebuttal 후 점수 변화, 거절 사유를 재구성한 분석이 포함되지 않는다. 공식 메인 채택 여부와 공개 리뷰 본문 열람 여부는 다른 항목이다.

ICML 2026 일부 개별 poster 페이지는 cache miss였지만 공식 Downloads 목록과 저자 공개본으로 상태를 교차 확인했다. KDD 2026 TimeDistill은 저자 저장소의 명시를 확인했으며 publisher proceedings 교차 확인은 미완료로 표시했다.

대표 논문을 더 깊게 볼 때는 초록의 성능 주장만 옮기지 말고, 실험의 target·input·split·checkpoint·compute·metric·baseline·ablation을 추출해야 한다. 그 작업이 끝나기 전에는 이 자료를 novelty 확정이나 새 실험 실행 승인으로 사용하지 않는다.
