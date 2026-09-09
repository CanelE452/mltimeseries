# 시간 블록 반복 결과와 현재 A 분기의 종료

2026-09-08. [고정 계획](13_peft_temporal_replication_plan_20260908.md)에 따라 두 원천의 다음 190일 구간에서 전체 선택·학습·보정 절차를 반복했다. **전체 진입 기준은 미충족이다. 현재 배포 조건과 예산에서 새로운 내부 PEFT의 필요성을 찾는 A 분기를 닫는다.** 이는 PEFT 전체가 효과 없다는 결론이 아니다.

## 무엇을 실제로 확인했나

28 fit(26 adaptation, 2 F0/cache), 선택된 모델의 미래 추론 14개, RAW 회귀 2개를 완료했다. S0 8개와 통합 CPU 29검사, 기존 분석 회귀 21검사가 통과했다. 독립 NumPy 검산은 32개 절차의 보정값 오차0, 점수 최대 오차2.78e-17, CI 최대 오차4.34e-16으로 일치했다. 마지막 추론은 Git 프로세스 급증으로 한 번 안전 중단됐고, 같은 가중치와 설정으로 그 추론만 재개했다. 실패를 포함한 실행 기록은 아래에 구분한다.

새 블록은 원 12번 전체 기간과 원시 시점이 겹치지 않는다. Bike 평가 기간은 2011-10-24~2012-01-16 미포함, Household는 2007-10-09~2008-01-01 미포함이다. 동일 원천의 다른 계절이며 독립 데이터셋 재현은 아니다. 각 원천은 64일 train, 14일 V_select, 14일 C_cal, 84일 평가; origin 수는 63/13/13/83이다. 매일 발행하는 48시간 예측끼리 24시간의 target이 겹친다.

원 Chronos-2 checkpoint부터 새로 적합했다. 이전 adapter를 계속 학습하거나 이전 calibration offset을 재사용하지 않았다. 원 12번의 소스·데이터·모델·결과 261개 보호 항목을 유지했다. 전체 HPO 절차를 반복했으며 성능을 본 뒤 LR나 기간을 추가하지 않았다.

## 주효과와 사전 판정

아래 효과는 `100 × (H_QCAL − LoRA_QCAL) / F0_QCAL`이다. H 대비 상대 개선율이 아니라 **F0 점수 단위의 백분율**이다. 7일 moving block 4,000회, 원천별 97.5% CI, 하한이 1%F0를 넘고 모든 seed가 양수이며 RAW 점수 veto가 없어야 한다는 기준을 사용했다.

```text
원천          12번 효과 / 97.5% CI          13번 효과 / 97.5% CI          13번 판정
Bike          +3.93% [ 2.05, 5.18]%       +7.42% [4.55, 11.26]%        실용 기준 통과
Household     +1.36% [-0.07, 3.24]%       +1.78% [0.53,  3.15]%        1% 기준 보류
```

Household의 13번 CI는 0을 제외한다. 따라서 “효과가 없다”가 아니라 “양수 효과는 관측했으나 사전에 요구한 최소 실용 효과를 확증하지 못했다”가 정확하다. 3일/14일 민감도 CI 하한도 각각 0.5207%/0.8699%로 1%를 넘지 않았다. Bike는 두 민감도 모두 기준을 통과했다. 모든 seed의 H 대 LoRA 차이는 양수이고 RAW veto는 두 원천 모두 없었다.

CI는 선택된 설정과 세 번의 fitted optimizer seed에 조건부인 시간 블록 불확실성이다. 전체 HPO·훈련·원천 추출의 불확실성을 포함하지 않는다. 12·13을 합쳐 새 유의성을 계산하지 않았으며, 앞선 Household의 보류 결론을 수정하지 않는다. 두 블록의 점수는 각각의 train 표준편차로 정규화됐으므로 절대 점수가 작아졌다는 사실을 기간 간 모델 개선으로 읽어서도 안 된다.

## Bike의 큰 효과를 그대로 적응 필요성으로 읽으면 안 된다

평가 proper score는 낮을수록 좋다. 각 target의 train 표준편차로 정규화한 21분위수 2-pinball을 target별 동등 가중했다. H와 LoRA는 세 seed 평균이며 F0/RAW는 한 결정적 절차다.

```text
원천          절차        SORT          QCAL
Bike          F0          0.167942      0.168719
              H           0.182751      0.181882
              LoRA        0.169685      0.169369
              RAW         0.449453      0.397193
Household     F0          0.478851      0.487131
              H           0.478753      0.486450
              LoRA        0.471734      0.477800
              RAW         0.691476      0.573547
```

Bike의 LoRA QCAL 점수는 F0보다 약 0.3852% 나쁘다. 반면 H는 F0보다 약 7.80% 나쁘다. **LoRA가 H보다 7.42%F0 좋다는 큰 차이는 원 모델을 능가한 개선이 아니라, 주로 H의 악화와 관련된다.** V_select에서 seed0의 F0/H/LoRA 점수는 0.216948/0.203400/0.198500으로 두 적응이 모두 좋아 보였다. 짧은 선택 구간의 개선이 그 뒤 전체 배포 구간의 개선을 보장하지 않았다는 관측이다. 이를 검증 잡음, 계절 변화 또는 새 PEFT 기법의 필요성으로 곧바로 단정하지 않는다.

Household에서는 LoRA가 F0보다 QCAL 약 1.9156% 좋았고 H는 거의 F0에 가까웠다. 표준 LoRA의 제한된 이득은 있지만, 범용 새 adapter가 필요하다는 주장을 뒷받침하기에는 이번 두 원천의 결과가 충분하지 않다.

Bike는 H_MLP LR1e-3과 LoRA LR1e-4를 선택했다. 앞선 블록의 H_FULL LR3e-5/LoRA LR1e-5와 다르다. Household는 앞선 블록과 동일한 H_FULL LR3e-5/LoRA LR1e-4다. LR 상단에 선택이 걸렸어도 추가 탐색으로 구제하지 않았다.

선택된 best step은 Bike H 120/120/40, LoRA 80/40/40; Household H 0/0/40, LoRA 40/40/40이다. Household seed0의 H 후보 6개는 모두 200 optimizer update와 nonzero gradient를 기록한 뒤 step0을 선택했다. 이를 학습 미실행이나 오류로 해석하지 않는다. H는 6개, LoRA는 3개 후보를 검색했으므로 후보군 크기에 따른 선택 분산도 남아 있다.

## 보정은 별도 실패를 보였다

Household의 LoRA는 SORT coverage 77.84%에서 QCAL 66.01%로 떨어졌고 score도 0.471734→0.477800으로 악화했다. F0 역시 coverage 76.54→65.20%, score 0.478851→0.487131로 악화했다. 따라서 이 현상을 LoRA만의 예측 능력 망각으로 설명할 수 없다. 고정된 짧은 C에서 얻은 offset을 이후 84일에 적용하는 절차의 전이 실패도 경쟁 설명이다.

Bike에서는 QCAL이 H/LoRA score를 소폭 개선했지만 F0 score는 악화했다. LoRA coverage는 75.68→73.71%로 nominal 80%에서 멀어졌다. Coverage와 proper score를 따로 해석해야 하며 단순 보정이 모든 실제 원천에서 해결책이라는 결론은 성립하지 않는다. 반대로 Gaussian의 [11번 보정 진단](11_peft_calibration_closure_results_20260908.md)이 통과한 사실도 그대로 유지한다.

RAW는 두 원천 모두 FM보다 나빴다. 이는 이번 full-past ridge 및 train 잔차 분위수 절차에 한정된다. 다른 고전 모형이나 계절성 기준모형까지 배제한 결과는 아니다. 표준 LoRA에는 native output projection도 포함되므로 attention의 독립 인과 효과를 확인한 실험 역시 아니다.

## 안전 중단·재개와 전체 비용

[재개 감사 기록](../../../results/peft_temporal_replication_v1/recovery_audit.json)에 실패 파일 5개와 완료된 기존 산출물 86개의 보존 해시를 남겼다. 16:16:38 Git 프로세스 56개가 한도 32개를 넘어 마지막 추론이 6.125초 만에 중단됐다. 당시 RAM 여유 15.489GiB, commit 여유 11.227GiB, GPU 1339MiB/44°C였다. 급증의 생성 부모는 조회 전에 종료돼 확인되지 않았다. 16:17:06과16:17:59 Git 0개를 확인한 뒤 같은 명령·소스·선택·checkpoint로 마지막 추론만 16.25초에 재개했다. 재학습·한도 완화·무관 프로세스 종료·드라이버/Defender 변경은 없었다.

성공 GPU 작업은 42개, 실패를 포함한 실제 시도는 43개다. 성공 guard 합계 936.080초(fit708.940/forecast227.140)에 실패 6.125초를 더하면 942.205초다. 첫 GPU guard 시작부터 마지막 종료까지는 대기·점검을 포함해 1,092.898초, 약 18분13초다. 원 시도 구간은 927.414초, 중단 종료~재개 guard 시작은149.233초였다. `completed.json`과 `costs.json`의 runner invocation 18.879초는 **재개한 두 번째 호출 시간만** 뜻한다. 전체 학습 시간으로 사용하지 않는다.

미선택 fit 14개의 324.158초도 비용에 포함했다. S0는 별도 113.400초, RAW CPU guard 4.140초/실제 적합 0.534초, CPU 분석 guard 12.125초였다. Guard 시간과 이를 포함하는 전체 경과시간을 더하면 중복 계산이다.

전체 GPU 시도 116개 자원 표본에서 RAM 여유 최소13.861GiB, commit9.334GiB, child RSS 최대1.674GiB, GPU2732MiB/58°C, Git최대56개였다. [기존 형식의 자원 요약](../../../results/peft_temporal_replication_v1/resource_summary.json)은 성공 작업만 포함하므로 그 파일의 Git최대2·중단0을 전체 실행의 안전 기록으로 인용하면 안 된다. 전체 시도는 재개 감사 기록을 사용한다. RAW의 RSS0은 시작 시점 표본만 있다는 뜻이며 실제 최대 메모리0이 아니다.

[Windows 조회](../../../results/peft_temporal_replication_v1/windows_event_audit.json)는16:20:44에 S0 시작 이후 Application1000/1002 및 System41/4101/153/2004를 확인했고 모두0건이었다. 이번 조회 범위에서 충돌 기록이 없다는 뜻이며 향후 무충돌을 보장하거나 Git 급증의 원인을 규명한 것은 아니다.

## 자기 평가와 다음 행동

현재 A 분기는 닫고 계획된 OUT_ONLY 후속으로 진행하지 않는다. Household가 실용 기준을 통과하지 못했고, Bike의 큰 H 대 LoRA 효과도 F0 초과 개선을 뜻하지 않았다. H 대비 우위만으로 내부 적응 필요성을 판단하는 기준에는 이처럼 불충분한 면이 있다. 이번 결과를 기준 완화에 사용하지 않고 이후 후보의 진입 검사에서는 F0를 반드시 포함한다.

다음에는 새 adapter를 먼저 만들지 않고 **선택 후 실제 미래 손실이 생기는지**를 B의 별도 최소 진단으로 확인한다. 직접 관측은 Bike의 validation 개선과 미래 악화다. 원 12·13의 저장된 후보 가중치를 재사용해 미선택 후보의 미래 예측을 추가하면, F0·고정 작은 LoRA·최근 validation 같은 단순 선택으로 문제가 해소되는지 확인할 수 있다. 이 데이터는 이미 일부 성능을 본 개발 자료이므로 새 방법 확증으로 사용하지 않는다. LR 내부 regret, 적응 여부의 regret, 단순 보정의 전이 실패를 구별한다. 모든 학습 step의 checkpoint가 남아 있지 않아 early-stopping 전체의 regret은 이번 재사용만으로 복원할 수 없다.

[후속 후보 검토](13_peft_next_candidate_review_20260908.md)에 기존 temporal model selection 연구와 C 후보의 선행 경계를 기록했다. 아직 구체적인 신규 ML 방법을 확보한 것은 아니며 [전체 주제 탐색](11_peft_topic_search_protocol_20260908.md)은 진행 중이다.

## 산출물

- [검증 및 원본 해시](../../../results/peft_temporal_replication_v1/verification.json), [32행 선택 결과](../../../results/peft_temporal_replication_v1/selected_results.csv), [30행 전체 fit/RAW 결과](../../../results/peft_temporal_replication_v1/all_results.csv), [효과·민감도](../../../results/peft_temporal_replication_v1/effects.json), [비용](../../../results/peft_temporal_replication_v1/costs.json)
- [12/13 별도 비교 JSON](../../../results/peft_temporal_replication_v1/temporal_comparison.json), [선택·학습 진단](../../../results/peft_temporal_replication_v1/diagnostics.json)
- [점수·coverage 그림](../../../results/peft_temporal_replication_v1/figures/01_scores_and_coverage.png), [효과·민감도 그림](../../../results/peft_temporal_replication_v1/figures/02_primary_effect_and_block_sensitivity.png), [전체 학습 궤적](../../../results/peft_temporal_replication_v1/figures/03_all_development_trajectories.png), [두 블록 별도 비교](../../../results/peft_temporal_replication_v1/figures/04_separate_temporal_replication.png). 각각 PDF도 저장했으며 PNG 네 개를 실제 열어 확인했다.
