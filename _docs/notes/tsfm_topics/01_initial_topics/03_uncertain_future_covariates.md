---
title: "불확실한 미래 보조변수의 분포 조건부 예측"
working_title: "Distribution-Conditioned Forecasting with Uncertain Future Covariates"
created: 2026-09-06
status: "연구 후보 / 데이터 계약 미확정 / 이번에는 실행하지 않음"
priority: 3
suggested_repository_path: "_docs/notes/tsfm_topics/01_initial_topics/03_uncertain_future_covariates.md"
---

# 1. 핵심 질문

> [가설·미검증] 미래 보조변수가 정확히 알려진 값이 아니라 예보일 때, 그 분포와 경로 간 의존성을 작은 표현으로 받아들이는 모델이 점 입력 방식보다 정확하고 계산 효율적인 예측을 할 수 있는가?

목표는 운영 최적화나 재고 정책이 아니라 **불확실한 입력을 처리하는 forecasting 모델**이다. 소비처는 실제 시점에 이용 가능했던 외부 예보를 입력받는 예측 시스템이다.

Known-future calendar와 uncertain-future weather는 같은 정보 조건이 아니다. Calendar는 미래 값이 알려질 수 있지만, 미래 날씨 예보는 나중의 실측값과 다를 수 있다.

# 2. 평균 입력만 넣으면 놓칠 수 있는 것

보조변수 경로 C의 예측분포를 Q_C라고 하면:

\[
p(Y\mid X,Q_C)=\int p(Y\mid X,C)\,dQ_C(C).
\]

일반적으로 비선형 예측기 f에서는:

\[
f(\mathbb E[C])\ne\mathbb E[f(C)].
\]

예시 Y=C²에서는 E[Y]=(E[C])²+Var(C)다. 즉 불확실한 입력을 처리하는 문제는 예측구간에만 관계되는 것이 아니라, 비선형 관계에서 평균 예측에도 연결될 수 있다.

다만 **이 수학과 입력 불확실성 전파 자체는 새로운 연구가 아니다.**

# 3. 가까운 선행연구와 현재 모델

| 연구·모델 | 확인된 역할 | 신규성 한계 |
|---|---|---|
| Gaussian Process Priors with Uncertain Inputs — Application to Multiple-Step Ahead Time Series Forecasting | NeurIPS 2002 학회 논문; proceedings 출판은 2003으로 기록되는 출처도 있음. 중간 입력 불확실성을 전파한다. [S1] | ‘불확실성을 전파한다’는 원리 자체는 오래됨 |
| Using weather ensemble predictions in electricity demand forecasting(2003/International Journal of Forecasting) | 여러 날씨 시나리오를 수요 예측에 넣고 평균·분포를 평가한다. [S2] | Monte Carlo 시나리오를 여러 번 넣는 것만으로 새 기여가 아님 |
| Chronos-2 | 미래 보조변수와 다변량 예측을 지원하는 공식 모델·구현. [S3] | 보조변수를 추가하는 것만으로 차별화되지 않음 |
| TiRex-2 | 과거·미래에 알려진 보조변수를 지원하는 다변량 모델. [S4] | 새 방법은 단순한 covariate 사용을 넘어야 함 |

이 후보의 novelty는 아직 미확정이다. 확률 입력 encoder, conditional stochastic process, amortized integration, ensemble distillation과 기능적으로 겹치는 연구를 추가 확인해야 한다.

# 4. 방법 후보

[설계·미검증] 여러 미래 보조변수 경로를 그대로 K번 별도 추론하지 않고, permutation-invariant set encoder 또는 경로 의존성을 보존하는 분포 encoder로 작은 문맥 표현을 만든다.

\[
z_C=E_\phi(\{C^{(1)},\ldots,C^{(K)}\},\text{forecast metadata}),
\]
\[
p_\theta(Y\mid X,z_C).
\]

경로별 시간 상관을 없애고 각 시점의 평균·표준편차만 쌓는 것이 충분한지는 별도로 비교해야 한다. Ensemble member 순서를 바꿔도 같은 분포 조건이므로 결과가 불필요하게 바뀌어서는 안 된다.

이 모델의 핵심 주장은 두 가지 중 하나여야 한다.

- 같은 시나리오·같은 계산비용에서 더 좋은 예측.
- 같은 수준의 분포 정확도를 훨씬 적은 추론비용으로 달성.

두 번째만 성립한다면 정확도 SOTA라고 포장하지 않고, 효율적인 불확실성 전파 방법으로 주장 범위를 잡는다.

# 5. 가장 강한 baseline

1. 보조변수 없이 target history만 사용하는 강한 모델.
2. 보조변수 ensemble mean을 입력한 같은 모델.
3. 같은 K개의 시나리오 각각으로 추론하고 결과를 정확히 결합한 모델.
4. 평균·분산 요약만 입력한 같은 용량 모델.
5. 제안한 경로 분포 encoder.
6. 미래 실측 보조변수 입력은 **배포 불가능한 diagnostic oracle**로만 보고.

미래 실측값을 주는 baseline을 일반적인 ‘known-future covariate’ 모델로 표시하면 누출이다.

**예측분포를 혼합할 때 CDF 평균과 quantile 평균을 혼동하지 않는다.**

\[
F_{mix}(y)=\frac1K\sum_kF_k(y)
\]

의 quantile은 F_mix를 역산해야 한다. `mean_k Q_k(tau)`는 일반적으로 다른 분포다. Point prediction의 경우에도 mean인지 median인지 명시하고 같은 estimand를 비교한다.

# 6. 데이터에서 먼저 확보해야 하는 것

이 주제는 모델 구현보다 **historical forecast vintage** 확보가 먼저다.

| 필요한 의미 | 검증 조건 |
|---|---|
| 예보 발행 시각 | 예측 요청 시각 이전에 실제로 공개됐는가? |
| 예보 유효 시각 | target의 어느 미래 시점과 대응되는가? |
| ensemble member 또는 predictive distribution | 점 예보 하나만 있는지, 불확실성 입력이 있는지 |
| 예보 revision | 나중에 수정된 예보를 과거 시점의 입력으로 쓰지 않았는가? |
| 실제 target과 보조변수 관측 | 평가 기준으로만 올바르게 연결됐는가? |
| 공개·이용 조건 | 연구용 저장·배포가 가능한가? |

위 항목은 앞으로 데이터 계약에 필요한 **논리적 필드**다. 아직 확인하지 않은 파일의 실제 컬럼명은 아니다.

수요·에너지·날씨 예보 결합 데이터가 후보지만, 이번 노트에서 특정 archive의 완전한 다운로드·정렬·라이선스 감사를 끝낸 것은 아니다. 데이터가 없으면 `DATA_CONTRACT_UNRESOLVED`로 보류한다.

# 7. 인공 잡음 실험의 위치

실측 미래 보조변수에 인공 잡음을 더해 smoke test를 만들 수는 있다. 그러나 그것은 실제 과거 예보의 오차 구조와 같지 않다.

인공 실험은 다음 검증에만 우선 쓴다.

- 불확실성이 0이면 점 입력 방식으로 일치하는가?
- ensemble member permutation에 불변인가?
- 시나리오 수를 늘리면 결과가 수치적으로 안정되는가?
- 동일 시나리오의 explicit integration과 작은 예에서 맞는가?

그 결과만으로 실제 forecaster의 성능 개선을 주장하지 않는다.

# 8. 평가와 예상 결과

Point mean forecasting이면 동일 mean에 대한 MSE/MAE를, predictive distribution이면 CRPS·pinball loss·coverage·구간 폭을 평가한다. 여러 미래 시점의 공동 경로를 주장한다면 joint score도 필요하다. 특정 목적과 무관한 지표를 늘리지 않는다.

[예상·미검증] 입력 불확실성이 커지고 target 관계가 비선형인 조건에서 이득이 커질 수 있다. 그러나 효과가 단순 scenario averaging으로 전부 설명되면 새 encoder의 필요성은 약하다.

| 결과 | 해석 |
|---|---|
| 점 입력보다 좋지만 동일 K scenario integration과 같고 비용도 큼 | 새로운 방법 기여가 부족 |
| 동일 비용에서 더 좋거나 적은 비용으로 같은 분포 정확도 | 후속 방법 연구 근거 |
| 실제 forecast vintage에서는 이득이 사라짐 | 인공 잡음 조건에 한정 |
| 실제 보조변수 oracle만 크게 좋음 | 불확실성 처리보다 입력 정보 품질이 병목일 수 있음 |
| variance 요약과 전체 경로 encoder가 같음 | 복잡한 경로 모델 필요성이 약함 |

# 9. 중단·보류 기준

- 과거 시점별 예보 자료를 확보하지 못하면 모델 학습에 착수하지 않는다.
- 강한 scenario-integration baseline이 충분하면 새 모듈을 덧붙여 주제를 살리지 않는다.
- 효과가 inference sample 수 증가에 의존하면 같은 비용 비교로 다시 해석한다.
- 실측 미래 입력을 사용한 결과만 있으면 deployment performance 주장을 하지 않는다.

3순위인 이유는 구현 친숙도가 아니라 **입력 정보 계약과 baseline의 강도**다. 의미 있는 데이터와 비교를 확보하면 주제 우선순위가 바뀔 수 있다.

# 10. 출처

[S1] **Gaussian Process Priors with Uncertain Inputs — Application to Multiple-Step Ahead Time Series Forecasting(2002/NeurIPS; proceedings 출판 2003)**. 저자 기관 자료: <https://www.microsoft.com/en-us/research/publication/gaussian-process-priors-with-uncertain-inputs-application-to-multiple-step-ahead-time-series-forecasting/> · <https://eprints.gla.ac.uk/3117/>

[S2] **Using weather ensemble predictions in electricity demand forecasting(2003/International Journal of Forecasting)**. <https://www.sciencedirect.com/science/article/abs/pii/S0169207001001236> · 저자 기관 archive: <https://ora.ox.ac.uk/objects/uuid%3Ac2e05834-3255-46b6-944d-3a1d39c492aa>

[S3] **Chronos-2: From Univariate to Universal Forecasting(2025/arXiv 기술보고서; 메인 채택 미확인)**. <https://github.com/amazon-science/chronos-forecasting>

[S4] **TiRex-2: Generalizing TiRex to Multivariate Data and Streaming(2026/arXiv 공개본; 메인 채택 미확인)**. <https://github.com/NX-AI/tirex-2>

확인일: 2026-09-06. 기존 자료의 요약과 이 문서의 미검증 설계를 구분한다. 실제 데이터 정렬·모델 학습은 수행하지 않았다.
