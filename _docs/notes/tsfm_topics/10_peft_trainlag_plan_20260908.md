# Train-only 지연 추정 후속 실행 계약

2026-09-08. 사용자가 [09 결과의 후속 제안](09_peft_module_ablation_results_20260908.md)에 “그렇게 해줘”로 승인했다. 목적은 ML 방법론 주제 판단 → 기존 단순 풀이의 효용 확인 → 참 지연을 알려주지 않아도 같은 관측 정보로 공변량 신호를 회수하는지 검사하는 것이다. 새 데이터 생성·학습 결과를 보기 전에 이 계약을 저장한다.

## 선택과 근거

기존 Chronos-2/native quantile 출력부와 attention LoRA(r8/alpha16,96 time/group qkvo,1,179,648params)를 유지한다. 출력 projection LoRA와 새 head는 넣지 않는다. 09에서 이 경로가 BOTH 성능을 유지했기 때문이다. 새 모듈을 설계하는 실험이 아니다.

후보는 (a) 채널별 최대 절대상관 지연 추정, (b) 조건부 forward selection, (c) 지연을 고정하지 않고 모든 후보의 회귀 예측을 섞는 방식이다. (a)는 추가 탐색 순서·규제를 줄이고 같은 추정 지연을 FM과 RAW에 주기 쉬워 채택한다. (b)는 proxy lag에 더 강할 수 있지만 순서·정지 기준이 추가된다. (c)는 지연 불안정성에 유리할 수 있지만 이번 정보 사전 제거 질문에 비해 용량·선택 범위가 커진다. 이번 결과를 본 뒤 (b)/(c)로 구제하지 않는다.

[LIFT, ICLR2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b52b07a239a7afa155ca25cf17a55074-Paper-Conference.pdf)는 과거에서 lead 추정·정렬·refinement와 frozen backbone을 다룬다. 여기서는 한 train corpus에서 고정한 채널별 지연과 native 미래 공변량 경로를 사용하므로 LIFT 직접 재현은 아니다. [LTSF-Linear, AAAI2023](https://ojs.aaai.org/index.php/AAAI/article/download/26317/26089)는 단순 선형 대조를 점검할 근거지만, 여기의 Y/U/V 공동 지연 회귀와 동일한 모델이 아니다. 지연 정렬이나 선형 비교 자체를 신규성으로 주장하지 않는다.

## 데이터와 정보 범위

기존 Q00와 같은 Gaussian 생성 분포를 새 BASE_SEED=2026090810으로 생성한다. 이는 새로운 난수 표본이며 다른 실제 데이터 원천은 아니다. Train64개 독립 episode ×corpus0/1/2, validation128/evaluation512는 corpus 간 공유한다. L256/H16, Y/U/V past → Y future이다. 원 실행의 seed20260908 및 episode ID/hash와 분리한다. 생성식·oracle 값은 데이터 QC와 최종 oracle 진단에서만 사용한다.

Lag selector는 `context_train,target_train` 배열만 받는다. 각 채널 Y/U/V에 대해 사전 지정 정수 범위16…128(H…L/2)의 `context[n,c,L+h-lag]`와 `target[n,h]` 사이 절대 Pearson 상관을 계산하고 최댓값을 선택한다. 동률은 작은 lag를 택한다. 전체 past Y를 추가 label로 쓰지 않고, FM과 같은64×16=1024개 train 미래 label만 쓴다. Lag는 채널별로 다르게 선택할 수 있다. Y 자기지연도 추정한다. 작은 정답 사전[32,48,64]는 사용하지 않는다.

이 범위는 모든 가능한 지연이 아니며, 모든 horizon에서 원 과거로 관측된 값을 쓰기 위한 하한H와 사전 제한 상한L/2를 둔다. Val/eval/oracle/생성식의 참 지연은 selector API에 전달하지 않는다. 추정 lag와 상관 전수 점수, 입력 hash를 저장한다. 참 lag를 맞혔는지는 진단이며 실패·재생성 조건이 아니다. QC는 shape/분할/seed 독립/finite/정상 분포·잔차 조건을 검사하며 추정기 성공에 맞춰 표본을 고르지 않는다.

## 비교 절차

```text
보고 이름      입력 변환                         FM 학습
F0             원 과거                           없음
ATTN           원 과거                           attention LoRA
ALIGN_F0       train 추정 U/V 지연으로 정렬       없음
ALIGN_ATTN     같은 추정 U/V 지연으로 정렬        attention LoRA
RAW            같은 추정 Y/U/V 지연의3개feature  선형 회귀
ORACLE         생성식의 참 조건부 분위수          진단 전용
```

정렬은 Y 과거를 유지하고 U/V 각각을 선택 lag만큼 이동한다. 시작 부분은 NaN padding으로 두며, 재시각화된 미래 U/V는 원 과거의 `L+h-lag` 값으로 채운다. Y 미래 입력은 마스킹한다. 추가 원본 미래 관측은0이다. 정렬·초기 결측·정규화·유효 문맥 길이·native 미래 공변량 경로가 함께 변하므로 어느 하나의 원인을 식별하는 대조는 아니다. U/V별로 다른 지연을 선택하면 이름별 전처리가 교환 대칭을 바꿀 수도 있다. 같은 추정 lag를 쓰는 RAW와 함께 보고하고 순수한 attention 기전으로 해석하지 않는다. ALIGN_F0도 target label에서 지연을 학습한 절차이므로 zero-shot이라고 부르지 않는다.

RAW는 intercept+추정 lag Y/U/V3개 feature에 train OLS를 적합하고, 같은 train 잔차의 empirical21 quantiles를 더한다. Ridge/HPO/OOF를 사용하지 않는다. 적합 잔차는 실제 새 표본보다 작을 수 있다는 한계를 명시하고 coverage와 oracle 평균 오차도 보고한다. 단순함을 위해 선택한 고정 baseline이며 최상의 회귀 절차라는 주장이 아니다. 추정 lag를 먼저 전체 train에서 고정한 회귀만 OOF로 돌리는 불완전 OOF는 만들지 않는다. Val/eval은 RAW 적합에서 쓰지 않는다.

## 학습·선택·실행 예산

새 FM LoRA12fit =raw/aligned ×LR{3e-5,1e-4} ×corpus3, 새 F0/cache6개(raw/aligned ×corpus3)이다. 모든 LR 반복 비용을 포함한다. Optimizer seed8100+c, 같은 retained module 초기값/같은 minibatch sampler를 raw/aligned에 사용한다. 200updates, val0/40/80/120/160/200, effective8/micro4 groups, native normalized Y-only quantile loss, AdamW wd0/clip1이다. Corpus0 validation으로 각 입력 절차의 LR를 선택하고 다른 corpus에도 고정한다. Lag 선택은 이 validation과 독립인 train-only다. 더 큰 LR·긴 학습·추가 generator를 결과에 맞춰 실행하지 않는다.

주분석은 corpus0 선택 LR 절차를 사용하고, LR3e-5 고정 비교는 기술적 sensitivity로 남긴다. Runner는 evaluation score를 LR 선택에 사용하거나 진행 출력에 표시하지 않는다. 같은 코드·계약으로18개 GPU trial과 CPU RAW가 완료된 뒤 evaluation 결과를 모아 분석한다.

S0는 F0/ATTN/ALIGN_F0/ALIGN_ATTN 각8episode, LoRA5updates로 검사한다. 동결/checkpoint/초기A·B/모듈map/target·group격리/정렬 원과거인덱스/LoRA gradient/정렬 F0-cache와 LoRA step0 일치를 확인한다. Raw 경로는 원 encode와 수치 동일성을 검사한다. 미래가 실제 관측되는 aligned 상태에 기존 “미래 전부mask0이어도 동일” 검사를 적용하지 않는다. Y미래 마스킹과 다른 group 격리를 검사한다. S0를 통과한 source/plan/data/lag/model 계약에서만 본실행한다.

## 지표·판정·중단

두 주효과는 양수가 개선이 되도록 고정한다.

```text
입력 정렬 대 원 attention 적응  (S_ATTN − S_ALIGN_F0) / S_F0
정렬 후 attention 추가 효용    (S_ALIGN_F0 − S_ALIGN_ATTN) / S_F0
```

512 evaluation episode를4000회 paired bootstrap하고 세 corpus 평균 후 같은 가중치로 분자/분모를 재계산한다. 주가족2대비에 각97.5% CI, 운영 문턱δ=.01을 적용한다. 하한>.01 및 세 corpus 모두양수면 반복적 실용 개선, CI전체가±.01안이면 세 fitted repetition에 조건부인 평균 효과의 실용 동등, 반대 큰 변화/나머지는 악화·미결정으로 구분한다. CI는 학습·validation 선택의 전체 불확실성을 포함하지 않는다.

별도 탐색적 RAW 진단1개는 `(S_RAW−S_ORACLE)/S_F0`의95% paired CI를 계산한다. 상한<.01이면 이 범위에서 oracle에 실용적으로 근접한 신호로 본다. RAW median과 참 조건부 평균의 MSE, MSE/참 평균의 분산, 80% coverage/width, crossing을 함께 기록한다. 이 별도 진단까지 합친 전체 family-wise 오류율을 통제했다고 부르지 않는다. RAW의 실패는 정보 부재를 증명하지 않는다.

RAW가 이 기준으로 oracle에 근접하면 **이 Gaussian 조건에서 공변량 정보 회수를 위한 새 PEFT 설계 필요성은 약하다**고 판정하고 해당 조건의 새 adapter 탐색을 중단한다. Alignment 또는 ALIGN_ATTN이 개선돼도 단순 회귀가 해결하는 문제를 FM 내부 상대 성능만으로 새 방법의 필요성이라고 부르지 않는다. 반대 결과도 자동으로 새 알고리즘을 정당화하지 않는다. 외부 일반화는 실제 유보 원천과 다른 backbone이 필요한 별도 단계다.

## 안전·보존·산출물

12:52:43 KST 사전조회: RAM여유16.45GiB/commit16.11GiB, Git0, GPU1244MiB/46°C, 공유guard lock없음. 기존 `.venv-peft`와 단일 GPU guard를 유지한다. 경계 RAM5GiB/commit6GiB/childRSS8GiB/Git32/GPU10500MiB·85°C, CPU2threads/interop1/BF16 cache-disabled/TF32off/CUDA cap.67이다. GPU child는 root가 한 개씩만 실행한다. 오류는 로그를 보존하고 다음 trial을 중단한다. 소유하지 않은 프로세스·시스템 설정은 변경하지 않는다.

새 소스 `experiments/peft_trainlag_v1/`, 큰 artifact `runs/peft_trainlag_v1/`, 작은 결과 `results/peft_trainlag_v1/`를 사용한다. 기존 학습 소스·계획·결과를 보존한다. 완료 후 실제 예측 점수 재계산·선택·lag/input/source hash·18trial/선택/RAW·cache·checkpoint·자원·Windows 로그를 검증하고10번 결과 보고서/그림/기록을 남긴다. Commit/push는 하지 않는다.
