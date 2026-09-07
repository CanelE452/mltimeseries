# TSFM benchmark gap discovery v1

주제축 문서. 시간축 기록은 `_docs/history/2026-09-07.md`.

## 1. 제안

**가설/질문.** 2025~2026 공개 시계열 파운데이션 모델(TSFM)들이 fev-bench 의 어떤
조건에서 반복적으로 상대 성능을 잃는가, 그 손실 중 실제로 회복 가능한 부분은 얼마이며,
단순한 배포 가능 처리로 사라지지 않는가.

**소비처.** 다음 논문 주제 결정. 이 study 안에서는 새 method 를 만들지 않는다.

**방법.** fev-bench (commit `eadb28ed` 고정, 공식 `fev` evaluator) 위에서
서로 다른 architecture family 3종을 동일 정보 조건으로 비교한다.

| 슬롯 | 모델 | family |
|---|---|---|
| primary | `amazon/chronos-2` | T5형 encoder |
| primary | `NX-AI/TiRex-2` | xLSTM recurrent |
| primary | `google/timesfm-3.0-pytorch` | patch/variate transformer |
| 진단 | `autogluon/chronos-2-synth` | 합성 데이터만 학습, 오염 통제 앵커 |
| baseline | SeasonalNaive (B0), 선형 자기회귀 특화모델 (B1) | — |

정보 조건은 섞지 않는다. TRACK U(공통 단변량, 공변량 없음)가 1차 탐색,
TRACK M(네이티브 다변량)·TRACK C(미래 기지 공변량)는 보조 증거.

**판정 지표 (착수 전 고정, `configs/study.yaml`).**

| 게이트 | 기준 |
|---|---|
| failure gate | discovery task ≥3, 영향 family ≥2, 조건 gap ≥5%, 최강 배포모델 대비 regret ≥8% |
| headroom screen | oracle headroom ≥5%, 단순 처리 회수율 <0.50, 잔여 ≥3%, family ≥2 |
| simple-fix stop | 회수율 ≥0.70 또는 잔여 ≤1.5% → `SIMPLE_BASELINE_SOLVES` |
| confirmation | 방향 재현 + family ≥2 + 심각도 ≥3% + oracle ≥3% + 잔여 ≥2% |

**예상 실패 모드.**
- 조건별 차이가 task 1~2개에서만 나오는 `CASE_ONLY`.
- 오염 위험이 큰 checkpoint 때문에 절대 순위 해석 불가.
- 단순 앙상블이 oracle headroom 대부분을 먹어 method 가치 소멸.

**중단 기준.** failure gate 를 통과하는 조건이 없으면 `NO_STRONG_GAP_FOUND` 로 종료하고
oracle probe 를 억지로 만들지 않는다.

## 2. 실행 중 확정된 사실

### 2.1 TiRex-2 의 fev-bench 탈오염 checkpoint 는 flagship 과 동일

`NX-AI/TiRex-2` 모델 카드가 `NX-AI/TiRex-2-fevbench` 를 "fev-bench eval dataset 전부
제외" 로 광고하지만, 두 저장소의 `model.ckpt` 는 같은 git-LFS oid
(`5596fb4bd1ecc4fbf93c5d7d3c9c68bd4e8492b3`, 380,613,375 B)와 같은 config oid 를 가진다.
GIFT-Eval 변종 2종은 실제로 다르다(330,093,831 B, 서로 다른 oid). 즉 이 벤더의 탈오염
파이프라인은 산출물을 실제로 분리하지만, fev-bench 변종만은 예외다.

결과: recurrent family 슬롯을 flagship `NX-AI/TiRex-2` 로 채우고, fev-bench 오염 상태를
`CLEAN_BY_OFFICIAL_EXCLUSION` 이 아니라 `OVERLAP_RISK_UNKNOWN` 으로 기록한다.
근거·판단·시점은 `results/tsfm_benchmark_gap_discovery_v1/MODEL_SUBSTITUTION.md`.

fev-bench 기준 오염 상태는 primary 3개 중 2개가 미해결이다. 이는 failure 탐색에는
쓸 수 있으나 절대 순위 주장에는 하향 적용해야 하고, 합성 전용 앵커
(`chronos-2-synth`)의 가치를 올린다.

### 2.2 task 선택 규칙을 실행 전에 한 번 교체

1차안((frequency, horizon) 계층 + task_uid lexical)은 discovery 12개 중 ETT 4개·SZ_TAXI 2개를
뽑아 7개 domain 중 4개만 덮었다. 같은 상위 데이터셋에서 온 task 는 독립 증거가 아니므로
"cross-task 일관성"을 부풀린다. 모델을 한 번도 돌리기 전에 규칙을
`(domain, frequency)` 계층 round-robin + **dataset family 당 1개** 로 바꿨다.
결과 discovery = 12 task / 12 family / 7 domain / horizon 버킷 4·4·4.
`selected_tasks.sha256` = `f10d6a20…`.

### 2.3 per-origin 분해가 공식 지표와 일치

oracle probe 와 task 내부 재표집을 공식 지표와 같은 양 위에서 수행하기 위해 fev 의
`_quantile_loss` / `_abs_seasonal_error_per_item` 을 그대로 써서 SQL 을 origin 단위로
분해했다. smoke 에서 공식 집계와의 최대 상대 오차는 2.6e-16.

## 3. 결과 — `NO_STRONG_GAP_FOUND`

정본은 `results/tsfm_benchmark_gap_discovery_v1/STATUS.md`. 실행 규모는 estimator 6 × task 18 ×
정보조건 3 = 174 cell, 평가 origin 363,180개, 측정된 추론 시간 0.85 GPU-hour. 실패 cell 0.
무결성 검사 20/20 통과.

### 3.1 failure gate 를 통과한 조건이 없다

조건별 효과 자체는 크다 — 미래 공변량 없는 task 가 있는 task 보다 56% 나쁘고(9 task),
긴 horizon 비율이 39%(4 task), 약한 계절성이 38%(7 task) 나쁘다. 그리고 이 악화는 세 family 에
**동시에** 나타난다. 그런데 등록된 게이트의 regret 조건(최강 배포 가능 추정기 대비 ≥8%)에서
전부 걸린다. TRACK U discovery 중앙값 regret 은 timesfm-3.0 0%, tirex-2 2.4%, chronos-2 4.5%,
선형 특화모델 36.8%, SeasonalNaive 77.6% 다.

즉 **어떤 조건에서도 세 모델이 더 나은 대안에 지지 않는다.** 조건은 진짜로 어렵고, 이 모델들이
그 조건에서 현재 가장 나은 것이다. 이것이 method 후보가 없는 이유다.

### 3.2 oracle headroom 은 실체가 없다 — 잡음 수확

discovery 전체를 한 집합으로 본 특성화: H_oracle 6.75%
[부트스트랩 5.76–8.58],
단순 처리 회수율 R_simple -0.16(앙상블이 오히려 나쁨), 잔여 7.83%.
표면만 보면 headroom screen 을 통과하고, 지시문 §25 의 "cross-model complementarity" 범주에
해당하는 것처럼 보인다.

세 모델이 통계적으로 구별 불가능해도 origin 마다 최소값을 고르면 항상 이득이 난다. 그래서
승자가 시계열의 성질인지, 매 window 다시 뽑히는지를 두 검정으로 갈랐다.

| | discovery (11 task) | confirmation (6 task) |
|---|---|---|
| 평균 oracle headroom | 5.60% | 6.48% |
| 과거만 쓰는 per-series router | -1.85% | -1.46% |
| headroom 중 router 회수 비율 | -0.33 | -0.23 |
| 승자 지속성 (관측) | 0.3716 | 0.3847 |
| 승자 지속성 (우연) | 0.3657 | 0.3577 |
| 우연 대비 초과 | 0.0059 | 0.0270 |
| 비교쌍 수 | 6,300 | 15,644 |

승자는 이전 window 와 우연 수준으로만 일치하고, 과거 승자를 따라가는 router 는 최강 단일 모델보다
**나쁘다**. 두 split 이 같은 답을 준다. 이 6% 는 method 가 잡을 수 있는 구조가 아니다.

예외 하나는 평균에 묻지 않고 적어 둔다 — `redset_1H`(cloud, hourly)는 지속성이 우연 대비 +10.8%p,
router 회수 +1.5% 로 유일하게 양수다. task 하나이므로 결과가 아니라 단서다.

### 3.3 단순 처리들의 실제 성적

`headroom_characterisation_per_task.csv`. affine calibration 은 크게 악화시키고
(ecdc_ili 1.92→3.85, uci_air_quality 0.81→2.52), 특화모델은 전 task 악화, 앙상블은 근소 악화.
유일하게 개선하는 것은 공변량 공급(P5)이다 — uci_air_quality 0.815→0.698(−14.3%),
entsoe 0.348→0.331(−5.0%), hermes 0.609→0.599(−1.7%). 다만 이건 세 모델이 이미 갖춘 기능이지
gap 이 아니다.

### 3.4 다음 탐색

STATUS §17 이 정본. 요지는 "task 를 더 모으는 것"이 아니라 **이기는 비교 대상**을 만드는 것이다.
가장 큰 조건 효과들은 실재하지만 공유되므로, 그 조건에서 실제로 이기는 강한 task-specific 모델이
하나 있어야 gap 이 측정 가능해진다. 그것이 안 되면 그 조건은 그냥 어려운 것이다.
