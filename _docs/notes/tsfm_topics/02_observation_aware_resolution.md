---
title: "관측 방식 인지형 해상도 전이"
working_title: "Observation-Aware Time Series Models Across Resolutions"
created: 2026-09-06
status: "연구 후보 / 이번에는 실행하지 않음"
priority: 2
suggested_repository_path: "_docs/notes/tsfm_topics/02_observation_aware_resolution.md"
---

# 1. 핵심 질문과 목표

> [가설·미검증] 관측 간격뿐 아니라 각 값이 순간값·구간 평균·구간 합계 중 무엇인지 모델에 알려주면, 서로 다른 시간 해상도로 옮겨갈 때 더 정확하게 예측할 수 있는가?

목표는 새 집계 지표가 아니라 **관측 의미를 이해하는 입력 표현과 예측 decoder**를 만드는 것이다. 평가 소비처는 다른 sampling rate·aggregation task에서도 같은 모델을 재사용하려는 연구자다.

이 주제는 1번의 후속 branch가 아니다. 1번이 실패해도 그 사실만으로 이 주제가 실패하는 것은 아니다. 이번 작업은 후보를 문서로 보존하는 데까지만 한다.

# 2. 문제를 정확히 정의하기

각 관측을 잠재적인 시간 함수 f에 작용하는 연산으로 쓴다.

\[
y_k=\mathcal A_k[f]+\epsilon_k.
\]

예를 들어:

\[
\mathcal A_{\mathrm{point},t}[f]=f(t),
\]
\[
\mathcal A_{\mathrm{mean},[a,b]}[f]=\frac1{b-a}\int_a^b f(u)du,
\]
\[
\mathcal A_{\mathrm{integral},[a,b]}[f]=\int_a^b f(u)du.
\]

이산 데이터의 단순 합계라면 그 정의를 그대로 별도로 적는다. 연속 적분과 샘플 합을 단위·sampling interval 없이 같은 것으로 두지 않는다.

‘같은 1시간 간격 데이터’여도 정각의 온도와 지난 1시간 평균 온도는 동일한 관측이 아니다. 반면 합과 평균은 관측 구간 길이·샘플 수를 알면 서로 변환 가능한 경우가 많다. **단위 변환만으로 해결되는 문제를 새 모델의 장점으로 포장하지 않는다.**

이 문제는 결측값 복원과도 다르다. coarse 평균 관측만으로 사라진 fine-scale 정보를 완전히 복원할 수 있다고 주장하지 않는다.

# 3. 가까운 선행연구

| 논문 | 이미 다루는 것 | 이 후보가 별도로 보여줘야 할 것 |
|---|---|---|
| FlowState: Sampling-Rate-Equivariant Time-Series Forecasting(2026/ICML) [S1] | SSM encoder·함수형 basis decoder로 sampling rate와 출력 시간 격자에 대응 | point/mean/sum 같은 관측 연산을 **입력부터** 반영하는 것이 추가로 유효한가? |
| Coherent Probabilistic Forecasting of Temporal Hierarchies(2023/AISTATS) [S2] | 여러 시간 집계 수준의 표현과 coherent probabilistic forecast | 사후 reconciliation 또는 hierarchy embedding만으로는 부족한 관측 전이가 있는가? |

연속시간 decoder, 시간 해상도 전이, 합계 일관성은 이미 존재하는 연구다. 이들을 다시 구현한 것만으로 새롭다고 하지 않는다. 선행연구의 기능적 동등성 검사는 아직 완료되지 않았다.

# 4. 방법 후보

[설계·미검증] 입력 token에 다음 의미를 담는 encoder를 검토한다.

- 관측값과 단위 정규화 결과.
- 관측 시작·끝 시각 또는 순간 관측 시각.
- 연산 종류(point / mean / discrete sum / integral).
- 결측 여부와 유효 관측 수.

이 목록은 **앞으로 정의할 데이터 의미**이며, 실제 데이터 파일에 이런 이름의 컬럼이 있다고 주장하는 것이 아니다.

출력 query는 ‘다음 K개 점’만이 아니라 ‘이 구간의 평균 또는 합계’를 지정한다. Decoder가 잠재적인 연속 함수 또는 일관된 basis representation을 만들고, 요청한 관측 연산을 적용해 예측한다.

예를 들어 동일 정보 시점에서 구간 합계는:

\[
\hat S(a,c)=\hat S(a,b)+\hat S(b,c)
\]

를 만족하도록 만들 수 있다. 그러나 **일관된 오답도 가능하므로 주지표는 실제 target 예측오차**다. 일관성 오차는 보조지표다.

# 5. 데이터를 확보할 때 필요한 조건

가장 먼저 확인할 것은 source의 관측 의미다.

| 필수 정보 | 없을 때 생기는 문제 |
|---|---|
| 각 값이 instantaneous/mean/sum 중 무엇인지 | 다른 관측을 잘못 정규화할 수 있음 |
| interval endpoint와 단위 | 합·평균·적분 변환이 잘못됨 |
| 원본 고해상도 signal과 timestamp | 관측별 변형을 같은 잠재 경로에서 만들기 어려움 |
| split과 각 집계 window의 support | train/test 경계를 넘는 평균으로 누출 가능 |
| 실제 다른 해상도·다른 source | 합성 집계에서만 작동한 것을 실제 전이로 확대하기 어려움 |

후보 도메인은 전력의 power/energy 측정, 환경 센서의 point/average 기록 등이다. **특정 공개 데이터셋이 위 의미를 모두 제공한다고 이 문서에서 검증한 것은 아니다.** source metadata가 불분명하면 수치 배열을 보고 연산 종류를 추측하지 않는다.

초기 통제 검증을 위해 하나의 fine-resolution 원본에서 여러 관측을 만들 수 있지만, 이는 ‘구성된 관측 전이 실험’으로 표시한다. 실제 deployment와 동일하다고 부르지 않는다.

# 6. 반드시 포함할 baseline

1. 연산 의미를 이용한 올바른 단위 변환·재표본화 후 강한 forecasting 모델.
2. FlowState 또는 동일 역할의 연속 decoder에 같은 정보를 제공한 모델.
3. Fine-scale 예측 후 올바른 집계 연산을 적용한 모델.
4. Temporal hierarchy / reconciliation baseline.
5. 관측 연산 정보를 받지 않는 같은 용량 encoder.
6. 제안한 observation-aware encoder·decoder.

No-metadata baseline에 단위조차 잘못 준 뒤 이기는 비교는 금지한다. 모든 방법에 허용된 metadata를 같게 준 비교도 있어야 한다.

# 7. 최소 검증의 논리

[설계·미검증] 완전한 실험 계획 합의 전에는 실행하지 않는다. 먼저 다음 순서의 검증 가능성을 확인한다.

**동일 latent history·동일 정보량 비교:** 관측 방식만 바꿨을 때 성능을 비교한다. Fine observation을 쓰는 방법이 coarse observation만 쓰는 방법보다 좋다고 표현의 우위를 주장하지 않는다.

**보지 않은 해상도:** 학습한 관측 연산과 다른 간격을 평가하되, 스케일·구간·단위의 분포까지 나란히 보고한다.

**보지 않은 연산:** 연산 타입 일반화와 해상도 일반화는 별개다. 후자는 통과하고 전자는 실패할 수 있다.

**실제 source 전이:** metadata가 신뢰할 만한 별도 source에서 반복되는지 확인한다.

# 8. 예상 결과와 중단 이유

| 예상 가능한 결과 | 판정 |
|---|---|
| 단위 변환·resampling으로 같은 성능 | 새 모델을 만들 이유가 약함 |
| 일관성만 좋아지고 target 오차는 같음 | 예측 성능 중심 목표에는 불충분 |
| 합성 관측에서만 개선 | 증거 범위를 통제 실험으로 제한 |
| 같은 정보량에서도 unseen resolution 성능 개선 | 방법 연구로 발전시킬 근거 |
| coarse 입력에서 예측분포가 넓어짐 | 반드시 실패는 아님. 사라진 정보를 반영하는 정직한 불확실성일 수 있음 |

반증 조건은 ‘metadata를 정확히 준 단순 baseline과 강한 연속시간 모델이 이미 충분한가’다. 이 주제를 살리려고 불리한 단위·비정상 집계·잘못된 baseline을 만들지 않는다.

# 9. 주제로 선택할 때의 장점과 위험

**장점:** 입력과 출력의 의미를 동시에 바꾸는 모델 기여가 명확하다. Sampling rate와 measurement semantics를 분리해 실험할 수 있다.

**위험:** 데이터 의미 확인이 어렵고, 단순 변환으로 해결되는 문제일 수 있다. 고정 해상도 benchmark의 종합 MSE 1위를 곧바로 겨냥하는 주제와는 다소 다르다.

1번보다 후순위인 이유는 익숙함이나 이전 코드 재사용이 아니라, **실험을 정당화할 데이터 계약을 확보해야 하는 부담** 때문이다.

# 10. 출처

[S1] **FlowState: Sampling-Rate-Equivariant Time-Series Forecasting(2026/ICML)**. IBM 공식 연구 페이지: <https://research.ibm.com/publications/flowstate-sampling-rate-equivariant-time-series-forecasting>

[S2] **Coherent Probabilistic Forecasting of Temporal Hierarchies(2023/AISTATS)**. <https://proceedings.mlr.press/v206/rangapuram23a.html>

확인일: 2026-09-06. 이 노트의 모델 설계·예상 결과는 출처 논문의 결과가 아니라 새 가설이다. 실제 모델 학습은 수행하지 않았다.
