# 실제 원천의 출력 적응 대비 내부 PEFT 효용 screen

2026-09-08. [반복 탐색 계약](11_peft_topic_search_protocol_20260908.md)에 따른 다음 실행이다. 목적은 기존 합성 조건에서 벗어나 실제 수요·전력 예측에서 강한 출력 적응과 단순 회귀를 넘어서는 내부 적응의 효용이 있는지 확인하는 것이다. 이번 비교가 양성이어도 표준 LoRA의 효용이지 새로운 PEFT 방법의 완성은 아니다.

## 원천·배포 조건·선택 이유

[UCI Bike Sharing](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset)은 hourly 자전거 대여 수요다. Target은 casual/registered 두 개, 과거 입력은 두 target과 temp/hum/windspeed의 총5개다. 같은 정보의 총합 cnt는 중복 입력에서 제외한다. 공개된 정규화 상수를 쓰는 열은 그대로 사용하고 train-only 단위 통계를 따로 저장한다. 미래 날씨·calendar feature는 이번 비교에 제공하지 않는다.

[UCI Household Power](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)는 한 가정의 분 단위 전력 측정이다. Target은 Global_active_power/Global_reactive_power 두 개, 과거 입력은 두 target과 Voltage/Global_intensity의 총4개다. 분 측정을 시각별로 평균해 hourly series로 만들며 열별로 해당 hour의 finite 분 관측이45개 미만이면 그 열의 hour를NaN으로 둔다. 전력 kW와 전류·전압의 평균이며 energy sum과 혼동하지 않는다. 미래 측정이나 submetering의 보조 loss는 제공하지 않는다.

두 원천은 수요와 가정 전력이라는 다른 예측 대상이어서 선택한다. 공식 metadata 및 시간·결측 QC를 사용하며 성능을 본 뒤 원천/기간/열을 교체하지 않는다. 기존 프로젝트에서 소비한 family와 겹침 여부를 기록한다. 기존 FM의 pretraining 포함 여부는 모르면 unknown이며, 프로젝트의 새 데이터라는 이유로 contamination-free라 부르지 않는다.

모델 실행 전 source audit에 따른 수정: 최초 작성 중 Air Quality를 후보로 두었으나, dataagent가 이전fev discovery의 `uci_air_quality_1H` 평가 노출을 확인했다. 따라서 새로운 원천이라고 부를 수 없어 main screen에서 제외하고 이미 로컬에 있는 Household Power로 교체했다. Air Quality archive/QC는 보존하며 새 성능은 계산하지 않았다. Household의 기존 다운로드 자체와 모델 평가 노출은 구분한다. Bike의 기존평가노출은 찾지못했고 Household도검색범위에서모델결과노출을찾지못했다. 이 변경은모델점수나새예측을열기전에고정한다.

Static adapter를64일의 과거 target로 적합하고, 선택·보정을 마친 뒤84일간 재학습 없이 사용한다. 이후 실제 관측은 다음 origin의 과거 문맥에만 들어간다. 이 배포 조건에서 시간 일반화는 필요한 능력이며, 모든 배포가 이렇게 작동한다는 주장은 아니다.

## 데이터 계약

Hourly L336(14일)/H48(2일): 일·주 패턴을 포함하고 native16-point output patch와 맞춘다. 각 파일의 첫 완전한 날짜부터 pre-context14일 → train64일 → V_select14일 → C_cal14일 → evaluation84일을 사용한다. 각 split의 future target이 자신의 구간을 벗어나지 않게 origin을24시간 간격으로 둔다. 예측창길이는48시간이며인접창의중복은24시간이다. 이를 block 분석에 반영한다. 짧은 train 때문에 더 많은 중복 window를 독립 표본수로 세지 않는다.

Timestamp를 정렬하고 중복 hourly timestamp는 같은 열의 finite 값 평균으로 합친다. 비어 있는 전체 hour를 hourly grid에 NaN으로 넣고0으로 단정하지 않는다. Target NaN은 학습·평가에서 마스킹하고 채워 넣지 않는다. 과거 context만 시간순 forward-fill하며 처음부터 관측이 없는 leading gap은 train 관측 median으로 채운다. 원본과 missing mask를 유지하고 이 전처리가 target imputation이 아님을 검사한다. Train 통계는 train 기간의 관측값에서만 계산한다. 비-target 미래 열은 loss에서 모두 마스킹한다.

단순 QC는 파일/열/날짜 범위/중복/finite 수/분할 경계/fit std>0이다. 어떤 target의 split 관측 비율이70% 미만이면 그대로 결측 문제로 보고하고 해당 원천의 실행을 중단한다. 다른 기간을 찾아 바꾸지 않는다. 이 경우 새 원천 선택은 별도 실패 평가·새 계획으로 처리한다. Eval 값의 크기·분포·모델 점수로 표본을 고르지 않는다.

Prepared API는 `context_values[T,C]`, `target_values[T,C]`(원 missing 유지), `timestamps[T]`, `channels[C]`, `target_indices[2]`, `fit_mean/std/median[C]`, `train/val/cal/eval_origins`, `context=336`, `horizon=48`을 포함한다. 선택 이전 학습 subprocess에는 train/val 파일만 제공하고 cal/eval target을 넘기지 않는다. Cal/eval prepared 파일과 hash는 먼저 저장하되 결과는 모든 선택이 고정된 다음에 읽는다.

## 모델·학습·예산

기존 검증된 Chronos-2 snapshot29ec3766d36d6f73f0696f85560a422f50e8498c와 native21quantile loss를 유지한다. 다른 backbone은 새로운 기전 후보가 생긴 후의 확인 단계로 남긴다. F0, H_MLP(동결 hidden의 residual head), H_FULL(native output head), OFF_LORA(공식97projection r8/alpha16,1,206,912params)를 비교한다. 10번 attention-only 결론을 다른 실제 원천의 최적 모듈 지도라고 가정하지 않고 표준 map으로 되돌아간다.

Seed0 개발 후보는 방법별 세 LR다: H_MLP{1e-4,3e-4,1e-3}, H_FULL{3e-5,1e-4,3e-4}, OFF_LORA{1e-5,3e-5,1e-4}. 원천별 V_select로 H_MLP/H_FULL의6후보 중 H를, LoRA3후보 중 LoRA를 선택한다. 같은 주어진 정보·update 예산이며 H의 family 선택 비용은 더 크다. 이를 동일 HPO 예산이라고 부르지 않는다. 넓은 H 후보군은 더 강한 출력 적응을 찾을 기회를 주지만 짧은validation에서 선택 과적합도 커질 수 있으므로 무조건 H에 유리하다고 단정하지 않는다. 동점은 (method이름,LR) 오름차순으로 정한다. 각 방법의 추가seed1/2에는 고른 family/LR를 고정하고 자신의 V_select checkpoint만 선택한다.

따라서 두 원천 완료 시 새 adaptation fit26개(개발18+추가8)와F0/cache2개다. Optimizer seeds12000/12001/12002,200updates,validation0/40/80/120/160/200, effective8/micro4 groups, AdamW weight_decay0/clip1이다. 같은 seed의 method간 sampler를 맞춘다. H_MLP 구조와 native 출력부 초기값은 기존 S1을 유지한다. 모든 방법의 validation/evaluation quantile을 공통으로 오름차순 정렬한다. Native 훈련 loss에는 사후 정렬을 삽입하지 않는다.

Cache와 native 직접 평가 모두4개의 isolated groups로 고정하고 마지막 부족 batch는 마지막 origin을 별도 group으로 복제해 pad한 뒤 버린다. 동결 cache/step0 수치 차이를 S0에서 직접 검사한다. BF16 cache-disabled, float32 weights, TF32off, CPU2threads/interop1, CUDA cap.67이다. 새 head/LoRA의 동결·초기값·gradient·checkpoint 복구를 확인한다.

S0는 두 원천×F0/H_MLP/H_FULL/OFF_LORA의8개,5updates로 실행한다. 모든 S0 입력과 label은 train origin 부분집합만 사용한다. 유보 eval label을 S0에서 읽지 않는다. S0 후 고정한 source/plan/data/native checkpoint 계약에서만 본학습을 한다. 더 긴 학습·추가 LR의 사후 구제는 이번 screen에 없다.

## 단순 풀이와 calibration

RAW는 동일 과거336×C를 펼친 direct multi-horizon ridge다. X는 관측 train 통계로 표준화하고 feature 수의 제곱근으로 나눈다. 목적식은 관측 train행의 `sum squared error + lambda * ||beta||^2`이며 SSE를표본수로나누지않는다. Intercept는 penalty에서 제외하고 각target/horizon관측subset에서X와y를중심화해해결한다. Target별 horizon별 관측 train label만으로 적합하고, lambda{0.1,10,1000} 중 V_select scaled2-pinball이 가장 좋은 값을 선택한다. 동점은작은lambda다. Val 분포 예측에는 train 적합 잔차의 empirical21quantiles를 target별로 horizon pooling해 사용한다. 적합 잔차의 낙관성을 공개한다. Cal/eval은 ridge 계수 적합에 사용하지 않는다.

선택이 고정된 F0/H/LoRA/RAW 각각에 동일 QCAL을 적용한다: 각 target와 quantile별로 C_cal의 관측 잔차 y-p_q의 empirical q-quantile을 추정하고 고정 offset을 더한 뒤 정렬한다. Horizon은 pooling한다. 별도 calibrator family/HPO는 없다. C_cal은 checkpoint·LR·method 선택에 쓰지 않는다. 이는 시계열의 conformal coverage 보장이나 CQR80과 동일한 방법이 아니다. 무보정 SORT 결과도 모두 보존한다. SORT로선택후QCAL로평가하는고정절차의비교이며, calibration까지공동최적화한최상의H를이겼다는주장이아니다.

## 평가·주제 진입 판정

주점수는 target별 관측 cell의 mean2-pinball을 해당 train std로 나눈 뒤 두 target을 동일 가중 평균한다. 방법별 관측 mask와origin은 동일하다. Coverage80·train std 단위 interval width·crossing·medianMSE와 horizon별 결과를 함께 공개한다. 불규칙 결측이면 관측된 조건의 결과라는 범위를 유지한다.

주효과 `(S_H_QCAL−S_LORA_QCAL)/S_F0_QCAL`는 양수가 LoRA 개선이다. 두 원천에 각각97.5% paired CI,1%F0 운영 문턱을 적용한다. 날짜 moving block7일을4000회(seed2026090812) 재표집하고 동일 draw를 모든 방법·seed에 사용한다. Target별 valid loss 합·수를 다시 집계하며 F0분모도 재계산한다. Block3/14일은 기술적 민감도다. CI는 세 fittedseed와개발선택에 조건부이고 전체훈련/HPO불확실성을포함하지않는다.

두 원천 모두 주CI하한>1%이며각seed양수이고단순RAW가LoRA보다좋지않을때, 내부적응효용을후속모듈가설의진입근거로본다. RAW veto는 `S_RAW_QCAL < mean_seed S_LORA_QCAL`의점추정에기반한보수적진입중단규칙이며 RAW우월성의통계검정이아니다. 이는새방법신규성의증명이아니다. 한원천만양성이면관측조건을좁히되그원천선택자체는탐색으로기록하고새유보원천에서확인한다. RAW가더좋거나강한head로충분하면그조건의새내부PEFT필요성은약하다. CI가넓으면실패와동등을구분하고훈련seed증가만으로해결하지않는다.

## 구현·안전·보존

새 소스 `experiments/peft_external_gap_v1/`, 큰artifact `runs/peft_external_gap_v1/`, 작은결과 `results/peft_external_gap_v1/`를 사용한다. 기존소스와실험결과는수정하지않는다. GPU작업은root만공유guard로직렬실행하고,오류/안전중단시로그를보존한다. 새로운package설치·시스템설정변경·commit/push는하지않는다. Root가runner/통합, dataagent가source/data/RAW, trainingagent가native실행기, safetyagent가분석을책임진다.
