# PEFT 학습 공간과 평가 목적의 차이: 고정 6-fit 진단

2026-09-08. [14번 B 종료](14_peft_selection_regret_results_20260908.md) 이후, 새 손실로 학습하거나 그 미래 예측을 읽기 전에 고정한다. 사용자의 [반복 탐색 승인](11_peft_topic_search_protocol_20260908.md)을 따른다. 목적은 ML 방법론 주제를 찾는 것이며 이번 단계는 관측된 구현 차이가 실제 적응 결과를 설명하는지 확인하는 제한된 진단이다. **새 방법을 제안하거나 신규성을 입증하는 실험이 아니다.**

## 관측, 대안, 선택 이유

[기존 학습](../../../experiments/peft_adaptation_scope_v1/modeling.py:49)은 context별 표준화·asinh 공간에서 quantile loss를 계산한다. [선택·평가](../../../experiments/peft_external_gap_v1/train.py:147)는 SORT한 raw 예측을 train-global 표준편차로 나눈 loss를 사용한다. 두 목표는 자유로운 조건부 quantile 함수에서 단조변환을 통해 같은 최적해를 갖지만, 작은 공유 LoRA와 유한 업데이트에서는 gradient 가중치·방향이 달라질 수 있다. Native loss가 improper하거나 이전 PEFT 실패의 원인이라고 단정하지 않는다.

[Darts Chronos2Model](https://github.com/unit8co/darts/blob/master/darts/models/forecasting/chronos2_model.py)은 이미 역정규화한 loss로 fine-tuning하는 구현이다. [Gneiting의 scoring 논의](https://arxiv.org/abs/0912.0902)와 [generalized quantile scores](https://arxiv.org/abs/1503.08195)도 loss 선택과 estimand를 구분한다. Raw loss 자체나 그 미분을 새 방법이라고 부르지 않는다.

다른 후보인 정규화 통계 복원은 직접 선행이 있고, 경로 공변량은 기존 UCP에서 이미 음성이며, marginal-preserving copula head에도 직접 선행이 있다. 이들에 새 이름을 붙이는 대신, 실제 코드에서 확인한 목표 차이를 이번 한 번의 대조로 판단한다. 유사한 현상을 미분기하의 pullback 가중치로 보는 해석은 가능하지만 새 가중치 학습 규칙을 이번에 추가하지 않는다.

## 고정한 자료·모델·예산

Study12의 Bike와 Household 두 개발 블록을 그대로 재사용한다. Study13/14의 새 블록을 합치거나 더 좋은 기간을 선택하지 않는다. 원 raw source·전처리·train63/V13/C13/E83 origins·target2·L336/H48·stride24·native quantile21을 유지한다. 이미 평가한 개발 원천이므로 독립 확증이 아니다. Fit archive와 C/E archive를 분리하고 모든 fit·checkpoint 선택을 동결한 뒤 새 C/E 예측을 만든다.

백본은 기존 로컬 amazon/chronos-2 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`다. 모든 arm은 같은 OFF_LORA module map, rank8/alpha16, trainable1,206,912개, seed12000이다. 모델을 바꾸면 loss 이외의 설명이 늘어나므로 유지한다. LR은 이전 고정 grid의 가운데 값 **3e-5** 하나다. 200 updates, val_every40, AdamW weight_decay0/foreachFalse, gradient clip1, 8origin 유효 batch(4+4), BF16·FP32weights·autocast cacheFalse·TF32False·dropout0·CPUthreads2를 유지한다. 추가 LR나 다른 seed로 이번 판정을 구제하지 않는다. 같은 seed와 sampler가 같다는 것을 실제 hash로 확인한다.

총 **6개 새 fit(두 원천×세 arm), 6개 새 C/E forecast**, 기존 F0 두 forecast 재사용이다. NATIVE도 새로 실행해 기존 LR3e-5 trajectory와 대조한다. 원 cache는 초기 예측 동일성 검사에 읽기만 사용하며 새 cache를 생성하거나 원 manifest를 수정하지 않는다. OFF_LORA optimizer와 validation은 원 direct numerical helper를 재사용한다.

## 세 목적함수와 공정성의 범위

1. **NATIVE:** 기존 `native_pinball` 그대로. Horizon 평균·quantile 합·비표적의 masked zero를 포함한 전체 channel 평균이다. 예측 정렬도 추가하지 않는다.
2. **NORM_ALIGNED:** asinh-normalized 공간에서 quantile 예측을 SORT하고 표적별 train 전체 유효 관측 수로 나눈 뒤 두 target을 동일 가중 평균한다. Quantile은 평균한다.
3. **RAW_ALIGNED:** 2와 동일한 SORT·표적·분모·Q평균을 사용하되, raw 역변환 예측과 raw target의 pinball을 기존 target train-global std로 나눈다.

주대비는 2↔3이다. 1↔2는 SORT와 reduction을 함께 바꾸므로 reduction 단독 대조라고 해석하지 않는다. C는 QCAL 보정에만 쓰고 checkpoint나 objective 선택에 쓰지 않는다. QCAL은 보조 진단으로 유지한다.

표적 j의 분모 `D_j`는 **전체 train origin×48 lead의 유효 target 셀** 수다. 같은 timestamp가 다른 origin에서 반복되면 별도 예측 셀로 센다. 한 microbatch B=4에서 `N_train/B × mean_j[sum_(b,q,h) loss_bjqh / (Q*D_j)]`를 계산한다. 두 microbatch의 loss를 각각1/2배해 누적하면 uniform-origin sampling에 대한 전체 목표의 불편 gradient 추정량이다. Microbatch의 유효 개수나 unique timestamp 수로 바꾸지 않는다. 결측·비표적은 gradient0이고 inverse/SORT 경로의 부동소수점·mask 처리를 CPU에서 검사한다.

단순 상수 배율의 차이를 줄이기 위해 초기 모델에서 원천별 **train의 균등 간격 8origin**(`np.linspace(0,N_train-1,8).astype(int)`)을 4+4로 계산한다. 세 목적 각각 누적 gradient 벡터의 L2 norm을 얻고, `multiplier_arm=norm_native/norm_arm`을 고정 상수로 저장한다. NATIVE는 정확히1이다. Norm의 평균을 쓰지 않는다. 0/비유한 gradient나 multiplier면 중단한다. 측정은 optimizer update 없이 하며 parameter/RNG 불변, 마지막 gradient 제거를 확인한다. 계수에 gradient를 통과시키지 않는다.

이 보정은 초기 norm만 맞춘다. AdamW의 좌표별 적응·epsilon·후반 clipping을 모두 통제한 것은 아니므로 양성 결과를 손실 기하의 단독 인과로 주장하지 않는다. 초기 목적별 norm/cosine, quantile crossing, target의 역변환 Jacobian `context_scale*cosh(norm_prediction)/train_std` 범위, 200step clipping 횟수·gradient norm·multiplier를 기록한다.

## 선택, 분석, 중단 기준

원 SORT validation score로 step0/40/80/120/160/200 중 best checkpoint를 strict `<` 규칙으로 선택한다. 원12의 NATIVE LR3e-5 fit과 초기 tensor hash·sampler·전체 V history·best step·복원 tensor hash·V예측을 비교한다. 파일 serialization hash 대신 실제 parameter tensor hash와 예측을 실행 간 동등성에 사용한다. 예상치 않은 replay 차이는 새 효과를 읽기 전에 조사한다. 이를 설명하지 못하면 과학적 결과 분석으로 넘어가지 않는다.

두 주효과는 `100 × (SORT_NORM_ALIGNED − SORT_RAW_ALIGNED) / SORT_F0`다. 양수는 raw 적응의 이득이다. 원천별7일 moving block4,000회, seed2026090815, 두 비교 각각97.5% CI를 사용한다. 저장 checkpoint/한 optimizer seed에 조건부인 시간 불확실성이며 HPO·훈련의 전체 불확실성이 아니다. 3/14일 block, odd/even 비중복 target window, 각 target과 QCAL을 기술적 보조로 남긴다. Odd/even stride2일에서 요청7일은4origin(실제targetspan8일)으로 계산한다.

두 원천 모두 주효과 CI 하한이 **+1%F0**를 넘으면 `RAW_ALIGNMENT_PRACTICALLY_RELEVANT`로 기록한다. 이때도 기존 raw-loss 적응이 이 제한된 조건에 유용했다는 진단이며 새로운 PEFT의 성공이 아니다. NATIVE↔NORM_ALIGNED와 NATIVE↔RAW_ALIGNED, clipping 차이를 함께 해석한다. 이를 통과하지 못하면 `CLOSE_CURRENT_OBJECTIVE_ALIGNMENT_SCREEN`: 이번 두 개발 블록·LR3e-5·200updates에서 이 변경이 사전 실용 기준을 충족하지 못했다고 기록한다. 변환 목표가 일반적으로 무관하다는 뜻은 아니다. 평가 기간·LR·threshold를 늘리거나 바꾸어 양성을 찾지 않는다.

## CPU·S0·실행 안전

새 namespace `experiments/peft_objective_alignment_v1/`, `runs/peft_objective_alignment_v1/`, `results/peft_objective_alignment_v1/`를 사용한다. 12–14의 소스·모델·cache·결과·계약·실패 기록을 보호하며 코드나 원 선택 gate를 완화하지 않는다. 모든 핵심 새 소스와 이 계획을 동결한 뒤 S0를 시작한다.

CPU에서 전체 목적과 microbatch 누적 gradient 일치, target-macro 결측 가중치, masked gradient0, 정렬/역변환, Jacobian 공식과 단조 quantile 이론의 제한함수 반례를 검사한다. S0는 두 원천×세 arm의 train-only5updates다. 원12 S0와 같은 LR1e-5를 사용하고 train8/pseudoV4 분리를 유지한다. 이 예외는 배관·정확 replay 감사용이며 production LR 선택을 위한 성능 탐색이 아니다. NATIVE S0는 원12 S0의 trajectory/복원/예측과 비교한다. 모든arm의 초기동일성·finitegradient·checkpoint복원·동결tensor보존을 통과해야 production으로 간다.

Root만 GPU child를 한 개씩 기존 shared guard로 실행한다. RAM5GiB/commit6GiB/childRSS8GiB/Git32/GPU10500MiB·85°C 한도를 유지한다. 실패 시 해당 시도 로그를 보존하고 해결 가능한 원인을 수정한 뒤 필요한 작업만 재개하며 모든 시도 비용을 센다. CPU분석도 별도 guard로 실행한다. 시스템 설정·무관 프로세스 변경·commit/push는 없다.
