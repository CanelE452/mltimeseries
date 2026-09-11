# Study37 — PEFT 논문 종료를 위한 목적·claim·gate 계약

[판정] 2026-09-11 작업 방향은 **A: CHARACTERIZATION / ANALYSIS**로 고정한다. 새 adapter 개발은 최상위 목표가 아니다. 아래 기준은 사용자의 이번 요청을 옮긴 운영 계약이며, 선행연구·노출·수치 감사 통과를 실험 진입 조건으로 삼는다. 최종 진행 상태는 [PAPER_READINESS](../../../../results/peft_paper_closure_v1/PAPER_READINESS.md)를 따른다.

## 1수 — 최상위 목표와 소비처

[판정] **TSFM에서 internal PEFT가 capacity-matched output adaptation을 넘어서는 실제 가치를 갖는 조건과, 그 adaptation utility의 chronological transfer 안정성을 규명한다.**

[판정] 소비처는 시계열 Foundation Model 연구자와 target-domain adaptation 사용자다. 독자는 head만 학습하면 되는지, 내부 LoRA가 필요한지, validation에서 좋아진 adaptation을 미래에도 믿을 수 있는지, 현상만으로 논문이 충분한지, 이를 해결할 새 PEFT가 필요한지를 판단해야 한다.

[판정] 목적 위계는 논문 한 편의 검증 가능한 결론 → internal value / chronological utility의 분리 → 기존 근거·데이터 노출·novelty 감사 → 자격을 갖춘 fresh replication이다. 그림이나 실행 횟수는 목표가 아니다.

## 2수 — 각 행동의 목적 후보와 가지치기

- [판정] F0/HEAD/WIDE/JOINT 비교: (1) head capacity와 내부 적응을 분리한다. (2) adaptation 자체가 F0보다 유용한지 확인한다. 같은 파라미터 수는 같은 함수공간·최적화 난이도를 뜻하지 않는다.
- [판정] fresh chronological replication: (1) 이미 반복 확인한 개발자료와 독립 근거를 분리한다. (2) 시간 전이 주장의 재현성을 확인한다. 나중에 고른 과거 기간은 FM 배포 이후의 전향적 검증과 다르다.
- [판정] fresh-head representation control: (1) backbone 변화와 공동 최적화 경로를 분리한다. (2) 기본 head 부족 설명을 더 강하게 검사한다. native output까지 바뀌면 순수 hidden representation 실험이라고 부르지 않는다.
- [판정] rolling characterization: (1) local gain과 future gain의 차이를 시간축으로 측정한다. (2) selector 개발 전에 안정적 이득·불안정·불필요의 분포를 확인한다.
- [판정] 새 PEFT method: (1) 분석으로 확인된 반복 가능한 실패를 완화한다. (2) 비용 또는 미래 일반화를 개선한다. 반복성·예측성·강한 baseline 우월성이 없으면 만들지 않는다.
- [판정] novelty audit: (1) 이미 알려진 online adaptation/negative transfer와 겹치는 부분을 제외한다. (2) 현재 기여와 미래 update utility라는 좁은 estimand 차이가 남는지 확인한다.
- [판정] 각 행동은 목적 후보 둘 이상을 유지한다. 하나만 남으면 해당 행동을 `사후 정당화`로 표시하고 자동 실행하지 않는다.

## 3수 — Phase 2 필수 설계 조건

- [판정] F0를 항상 포함.
- [판정] Head-only와 LoRA 비교만으로 결론내리지 않음.
- [판정] LoRA와 동일 trainable parameter 수의 WIDE head 포함.
- [판정] HEAD/WIDE/JOINT의 학습 budget 및 선택 기회 공정화.
- [판정] 같은 output format/metric/data windows 사용.
- [판정] validation에서 모델을 선택한 뒤 chronological next period에서 평가.
- [판정] 개발 중 이미 확인한 E/D를 final test라고 부르지 않음.
- [판정] 최종 test는 방법과 규칙을 freeze한 뒤 처음 열기.
- [판정] overlapping rolling origin을 독립 표본으로 세지 않음.
- [판정] 같은 target을 공유하는 seed를 독립 dataset으로 세지 않음.
- [판정] pretraining contamination/overlap이 UNKNOWN이면 UNKNOWN으로 유지.
- [판정] 결과를 보고 threshold, arm, period를 변경하지 않음.

## 4수 — 역주행 예상

- [판정] A: JOINT > WIDE이고 JOINT > F0. 예상은 유용한 내부 적응의 추가 이득이다. 목적 지지=지지, 최상위 도달=부분적으로 닿음(시간 전이는 별도 미검증). 리뷰어 첫 질문: 동일 head·동일 예산·독립 기간에서도 재현되는가?
- [판정] B: JOINT > WIDE지만 JOINT < F0. 내부 procedure의 상대 차이는 있으나 adaptation 자체가 해롭다. 목적 지지=부분, 최상위 도달=유용성 주장에는 안 닿음. 리뷰어 첫 질문: F0를 유지하면 되는데 새 adapter가 왜 필요한가? temporal generalization 문제를 먼저 다룬다.
- [판정] C: HEAD/WIDE ≈ JOINT. 내부 PEFT 필요성 지지=불가, 최상위 도달=필요성을 배제하는 조건부 characterization에는 닿음. 리뷰어 첫 질문: 차이가 없다는 것이 등가성인가, 표본 부족인가? 관측된 작은 차이를 통계적 등가성으로 바꾸지 않고 method 개발을 중단한다.
- [판정] D: V에서는 JOINT가 좋지만 chronological D에서 부호 반전. 시간적 불안정 후보 지지=부분, 최상위 도달=독립 재현·불확실성 조건 충족 시 닿음. 리뷰어 첫 질문: 한 기간/seed의 우연인가, 선택 과적합인가, 반복 가능한 시간 전이 문제인가?

## 기존 근거에서 출발하는 위치

[확인] [독립 재계산](../../../../results/peft_paper_closure_v1/evidence_reconstruction.md)에서 Study20 Bike LoRA 추가 이득은 +3.953447%F0, Study26 동일 residual MLP 뒤 추가 이득은 +4.170706%F0, Study30 P1 Bike FULL90 WIDE 대비는 +4.778926%F0다. 원시 예측을 새 NumPy pinball 계산으로 대조했다. 이는 이미 노출된 개발자료이며 세 study를 독립 원천 세 개로 세지 않는다.

[확인] Study35 L720에서 JOINT는 WIDE보다 평균8.804323%F0 좋으나 F0보다4.304809%F0 나빴다. 두 원천·두 seed의 네 cell 모두 F0 손해다. [판정] 따라서 목적 B 시나리오의 역사적 사례이며 새로운 Stage A 결과가 아니다. 내부 적응의 유한 procedure 비교는 남지만 현재 기간에서 적응 가치·representation 필요성·method 필요성은 지지되지 않는다.

[확인] Study32 현재 contribution 양수와 future continuation utility 음수가 함께 있는 반례가 저장된 분기에서 확인됐다. [판정] 이 관측을 미래 utility predictor의 성능으로 바꾸지 않는다. [확인] Study36의 sign conjugacy와 latest-state quadratic Bayes predictor도 CPU 모집단 계산으로 재확인됐다. 현재 DGP GPU는 중단 상태를 유지한다.

## 관측할 estimand

[판정] 모든 loss는 작을수록 좋다. 기간 t의 F0 loss를 L0,t라 두고, adaptation gain G(a,t)=100×(L0,t−La,t)/L0,t, internal increment I(t)=100×(LWIDE,t−LJOINT,t)/L0,t로 고정한다. V와 D는 각자의 F0 분모를 사용하며 G(JOINT,V)−G(JOINT,D)를 시간 전이 감소량으로 보고한다. 상대 baseline 대비 %와 %F0를 혼용하지 않는다.

[판정] 현재 adapter contribution C는 동일 checkpoint에서 adapter on/off 차이다. 미래 update utility U는 동일 prefix의 모델·optimizer 상태에서 JOINT continuation과 HEAD-only continuation을 분기시킨 뒤 미래 평가 loss 차이로 측정한다. final-step U와 V-selected U를 별도로 보고하며 STOP을 포함한다. C, G(V), G(D), U를 동의어로 쓰지 않는다. on/off ablation은 counterfactual 재학습과도 다르다.

## A/B/C 후보 비교와 primary

- [판정] **A — primary:** “Internal LoRA can outperform capacity-matched output adaptation, but local adaptation gains need not transfer chronologically; current adapter contribution and future update utility are distinct.” 필요한 증거는 F0, matched head, 공정한 선택 기회, fresh replication, future utility 분해, 강한 단순 대조다. 현재는 검증할 claim 후보이며 보편적 사실 선언이 아니다.
- [판정] **B — conditional:** 과거 temporal stability signal로 F0/LoRA/freeze 결정을 개선한다. A의 반복 재현과 train/V-only 신호의 개발 원천 밖 예측성이 필요하다. 단지 새 adapter라는 이유로 contribution을 만들지 않는다.
- [판정] **C — secondary:** validation gain과 chronological adaptation/update utility를 별도 보고하는 평가 protocol이다. 새 방법 실패와 관계없이 여러 모델·데이터에서 기존 결론을 실제 바꾸는지를 보여야 독립 contribution이 된다. 현상 재현이 약한 상태에서는 protocol 문서만으로 논문 준비 완료라고 하지 않는다.
- [판정] 가장 조건부인 C도 서로 다른 목적의 score를 한 개로 압축하는 평가 관행을 바꿀 수 있다는 장점이 있다. 그러나 일반적인 chronological evaluation 원칙과의 차이를 입증해야 하므로 지금은 A를 지원하는 평가 구성으로 둔다. primary 변경은 이유·새 근거·시각을 별도 기록하고 final 결과 뒤 변경하지 않는다.

## 상류 게이트와 Stage A

[판정] GPU 진입에는 저장소 감사, 목적·story map, 기존 핵심 값 재계산 PASS, 데이터 노출 및 fresh unit 확정, novelty 진입 판단, A/B/C 고정이 모두 필요하다. 부동소수점/문서 반올림을 넘는 실제 불일치는 `EVIDENCE_MISMATCH`로 실험 중단한다. 반올림 허용 폭은 비교 전 정하고 원문 정밀도·실제 차이를 함께 남긴다.

[판정] 이미 본 기간은 개발용으로만 사용한다. 원천·target·전체 train/V/D의 노출을 실제 metadata로 검사하고, 알 수 없는 노출은 UNKNOWN이다. 새로운 source/완전히 새로운 기간 둘 이상을 확보할 수 없으면 `NO_CLEAN_HOLDOUT_AVAILABLE`로 범위를 축소한다. 외부 후보가 있을 가능성과 실행 가능한 clean manifest 확보를 구분한다. final reserve는 Stage A 개발자료와 분리한다.

[판정] Stage A는 새 adapter/selector 없이 F0/HEAD/WIDE/JOINT, seed 2개, 동일 train/V/D, native output/metric, 동일 optimizer·checkpoint 선택 기회를 쓴다. 새 기간은 target 값을 열기 전에 metadata와 ledger만으로 고정한다. HEAD/JOINT 기본 head는 동일, WIDE/JOINT 파라미터 수는 같게 맞추고 오차가 있으면 사전 명시한다. 기존 제한 LR recipe를 재사용하고 D 결과 뒤 확장하지 않는다. seed prediction ensemble은 금지한다. Full FT 실행 여부는 Stage A manifest에서 미리 정한다.

[판정] G1: 하나 이상의 fresh source-period에서 JOINT−WIDE 이득이 두 seed 모두 양수, 평균 ≥0.5%F0이며 같은 조건에서 JOINT가 F0도 이기면 PASS. F0 비교도 두 seed 각각 보고하며 보수적으로 두 seed 모두 양수를 요구한다. WIDE만 이기면 `INTERNAL_EFFECT_WITHOUT_ADAPTATION_VALUE`; G1 미통과는 adapter/selector STOP이다.

[판정] G2: (A) 최소 두 독립 source-period에서 V의 JOINT gain ≥0.5%F0가 다음 D에서 ≤0으로 바뀌거나, (B) paired temporal gain 감소 평균 ≥0.5%F0와 time-block bootstrap interval의 0 제외가 성립해야 현상 지원이다. 같은 원천의 인접/겹친 기간은 독립이라고 자동 인정하지 않는다. interval 수준·block 길이·gap·분모 재계산·seed 공동 재표집은 데이터 길이만 확인해 manifest에 고정한다.

[판정] G3: 전부 JOINT 우세면 stable LoRA value, 상황별 부호 차이면 unstable utility, 전부 F0 우세면 no PEFT need로 분류한다. 전부 F0 우세일 때 internal method 개발을 중단한다. 운영 threshold는 통계적 유의성이나 학회 기준이 아니다.

## Stage B/C와 방법 진입

[판정] Stage B는 G1 통과 source-period에서만 별도 preregistration 후 실행한다. 원 backbone B0와 JOINT-adapted backbone B1을 완전 freeze하고 동일 architecture·fresh initialization seed·optimizer·budget·train/V/D·checkpoint 선택을 적용한다. 두 seed 모두 B1 fresh head가 ≥0.5%F0 좋으면 “adapted representation supports better downstream readout” 근거가 강화된다. 차이가 작고 JOINT만 좋으면 joint optimization/path effect 후보, 둘 다 F0보다 나쁘면 temporal failure가 상위 문제다. missing periodic information 회복이나 모든 head의 불가능성을 주장하지 않는다.

[판정] Stage C는 Stage A 현상이 재현될 때만 별도 preregistration 후 실행한다. 같은 source에서 과거 선택 block→다음 평가 block을 시간순 이동한다. 경계·stride·gap·개수는 데이터 길이에 의해 먼저 고정하며 결과 뒤 조정하지 않는다. F0/HEAD/WIDE/JOINT loss, V gain, next gain, selected step/recipe, 실제 wall-clock, F0 output shift, 가능한 train-only 통계를 모두 저장한다. C/G/U 구분을 재현하려면 rolling gain 외에 동일-prefix continuation 대조가 별도로 필요하다.

[판정] A~C 후 matched-head를 넘는 유용성, chronological instability, current/validation gain과 future utility 불일치가 여러 source/period에서 재현되면 새 방법 없이 분석 논문을 준비한다. 구조가 다른 두 번째 backbone이 없으면 그 한계를 공개한다.

[판정] B 진입은 추가로 (1) best fixed action 대비 oracle headroom ≥0.5%F0, (2) train/V-only simple rule이 development-held-out source에서 oracle headroom ≥30% 회수, (3) 신호 추가 wall-clock ≤full adaptation의 5%가 모두 필요하다. 하나라도 실패하면 `METHOD_BRANCH_STOP`. 과거 V gain 평균·분산·최악·기울기, selected-step stability, F0 deviation, 기존 training 통계만 먼저 검사한다. oracle은 상한이며 실행 가능한 정책이 아니다.

[판정] 방법은 문제→측정 signal→adaptation decision→미래 utility의 경로를 갖춰야 한다. F0 유지/standard LoRA/고정/계속 update/budget 축소로 해결할 수 없는 근거가 있어야 rank나 adapter 구조를 추가한다. baseline은 F0, HEAD, WIDE, standard LoRA, standard early stopping, fixed freeze, AFLoRA 또는 가장 가까운 freezing, AdaLoRA, LoRA+, 가능하면 Time-PEFT, 제한 Full FT다.

[판정] 방법 논문 진행에는 fresh final data의 독립 source 둘 이상에서 best strong baseline 대비 ≥1.0%F0 개선 또는 품질 손해 ≤0.25%F0와 wall-clock ≥10% 절감이 필요하다. 실패하면 method contribution 종료다. 이 역시 내부 운영 기준이다.

## Final 봉인과 금지 사항

[판정] final은 claim/method/hyperparameter/threshold freeze 뒤 처음 연다. unseen family ≥2, 새로운 chronological period, 가능하면 다른 backbone ≥1을 목표로 한다. final 뒤 threshold/dataset/seed/rank/method 선택을 변경하지 않는다. 실패는 그대로 보고한다. contamination UNKNOWN을 clean pretraining으로 바꾸지 않는다.

[판정] Study36 현재 Markov DGP GPU, Study31~33 threshold 재튜닝, 이유 없는 rank sweep, 같은 E/D 반복 튜닝, 좋은 dataset/seed만 보고, F0 제외, validation=future 취급, parameter 감소=wall-clock 감소 취급, seed/bootstrap/origin/실험번호를 독립 데이터로 세기, Point/Hurdle 축 혼합은 금지한다. 기존 sealed/preregistered artifact와 checkpoints는 수정·삭제·덮어쓰지 않는다.

## Phase 3 — 논문 story map (GPU 전 고정)

- [판정] 질문 1 “무슨 문제야?” → Introduction + Figure 1: C, G(V), G(D), U의 정의와 시간축. **빈 증거:** fresh unit에서 estimand 불일치 반복.
- [판정] 질문 2 “왜 중요해?” → Motivation + Table 1: F0 대비 실제 이득/손해와 비용. **빈 증거:** 외부 독립 원천에서 의사결정 영향.
- [판정] 질문 3 “왜 F0/HEAD/WIDE/LoRA/Full?” → Controls section + Table 2: 용량·head·업데이트·선택 기회·실측 비용. **빈 증거:** 새 기간 공정 대조, 제한 Full FT 여부의 사전 결정.
- [판정] 질문 4 “chronological split과 설정 근거는?” → Data/protocol section + Figure 2: 노출 장부, train/V/D, rolling dependence, final reserve. **빈 증거:** metadata로 확정한 fresh manifest와 untouched final.
- [판정] 질문 5 “결과가 현상을 지지해?” → Results + Figure 3(G1), Figure 4(V→D), Table 3(C vs U), 조건부 Figure 5(fresh head). **빈 증거:** Stage A/B/C 및 동일-prefix 독립 재현. 그림 하나는 질문 하나에 답하도록 하고 subplot을 채우기 위해 만들지 않는다.
- [판정] 질문 6 “한계와 다음은?” → Discussion + Table 4: source/기간/seed 의존, pretraining UNKNOWN, 선택편향, backbone 범위, method STOP. **빈 증거:** second backbone과 final validation. 수행하지 않은 실험은 not measured로 쓴다.

[판정] 빈 칸은 위처럼 명시한다. 우선 감사와 설계로 빈 칸을 해결하고, 남는 실증 항목만 게이트를 통과한 실험으로 채운다. 매끄러운 설명으로 미측정 항목을 덮지 않는다.

### 예상하지 못한 독자 질문

- [미검증] WIDE/JOINT 동수라도 head가 LoRA보다 최적화하기 어려운가? 유한 recipe 대조 밖의 필요성은 미입증이다.
- [미검증] V→D 차이는 계절·target 교체·결측·정규화 분모가 설명하는가? 동일 target 재현과 loss 분해가 필요하다.
- [미검증] 현재 contribution이 다음 update를 예측해야 한다는 전제 자체가 필요한가? 논문은 동일성을 가정하지 않고 실제 불일치와 평가상의 영향을 보여야 한다.
- [미검증] online adaptation 문헌과 무엇이 다른가? [novelty audit](37_novelty_boundary.md)의 직접 검증 범위 밖으로만 claim을 한정한다.
- [미검증] final 값이 실제로 봉인됐다는 것을 어떻게 재현하는가? metadata/접근 경계/선택 hash/첫 평가 시각으로 증명해야 한다.

## 기록의 지위

[확인] origin fetch 후 최신 PEFT head는 `33ede9bf9694c5199f7f8cfce6c8b34551b451f8`이고 기존 `peft-paper-closure-v1`과 `peftpaperclosurev1`는 같은 commit이다. 이번에는 현재 branch를 재사용한다. main의 README를 최신 연구 근거로 삼지 않았다.

[확인] 시작 시 미추적 `.pytest_tmp_objective_analysis/`, `tmp/`, `results/peft_paper_closure_v1/data_exposure_ledger.csv`가 있었다. 이전 장부의 모호한 target/기간은 검증 근거가 아니며 원본을 보존한 뒤 provenance 기반으로 감사한다. [저장소 감사](../../../../results/peft_paper_closure_v1/repository_audit_initial.json)에 실제 상태를 저장했다.

[판정] 모든 새 연구 사실은 [확인]/[추정]/[미검증]/[판정]으로 표시하고 관측과 해석을 구분한다. 종료 판정은 허용된 일곱 값 중 정확히 하나를 PAPER_READINESS에 기록한다. 중간 gate 실패 시 실행 예산을 채우지 않는다.
