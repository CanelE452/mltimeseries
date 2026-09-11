# 저장된 PEFT 후보의 선택 손실: 현재 B 분기 종료

2026-09-08. [고정 계획](14_peft_selection_regret_plan_20260908.md)에 따라 미선택 후보 28개의 추가 추론과 분석을 완료했다. **현재 B 진입 기준은 실패했다.** 검증 구간으로 고른 LoRA가 나빠지는 사례는 있었지만 두 원천에 걸친 실용적 손실이 반복되지 않았고, 고정 LR1e-5가 사전에 정한 단순 규칙 veto를 만족했다. 이 결과는 새 선택 알고리즘의 필요성을 뒷받침하지 않는다. PEFT 전체나 모든 선택 연구가 불필요하다는 결론은 아니다.

## 실제 비교와 주결과

[확인] Study12/13 × Bike/Household의 네 cell에서 seed12000의 저장 후보 10개씩을 평가했다. 원 F0/H/LoRA 예측 12개를 재사용하고 미선택 후보 28개를 추론했다. 추가 학습은 0회다. 후보별 C13/E83 origin, target2, quantile21, horizon48, stride24가 동일하다. 기존 선택 결과 일부를 이미 본 개발 자료이며 두 물리 원천의 두 기간이다. 네 독립 데이터셋이나 새 방법의 외부 확증이 아니다.

주효과는 `100 × (SORT_LORA_V − SORT_FIXED_LOW) / SORT_F0`다. 양수는 검증으로 선택한 LR의 손해다. 네 주비교 각각 98.75% 신뢰구간, 7일 moving block 4,000회, seed2026090814를 사용했다.

```text
조건                 선택 손실(%F0)       98.75% CI
12 Bike                    0.000           [ 0.000, 0.000]
12 Household              -0.388           [-0.991, 0.102]
13 Bike                   +3.079           [+0.414, 5.838]
13 Household              -0.268           [-1.351, 0.599]
```

Bike13은 CI가 0을 제외하지만 하한 +0.414%가 사전 실용 문턱 +1%를 넘지 못했다. 다른 원천에서 같은 손해가 반복되지 않았고, 반대로 고정 작은 LR의 손해가 1%를 넘는다는 CI 조건도 없었다. Bike12의 0은 두 정책이 같은 checkpoint를 선택했기 때문에 발생한 정확한 동일성이다. 서로 다른 학습 절차의 일반적 동등성 증거로 해석하지 않는다. 이 CI는 저장된 한 optimizer seed와 고정 선택 규칙에 조건부이며 훈련·HPO 선택의 불확실성을 모두 포함하지 않는다.

SORT 점수는 다음과 같다. 원12/13 보고서의 적응 모델 점수는 세 seed 평균인 경우가 있으므로 이번 seed12000 값과 혼동하지 않는다. 블록별 train 표준편차가 달라 서로 다른 블록의 절대 점수를 직접 개선율로 비교하지 않는다.

```text
조건                 F0         고정 LoRA     V 선택 LoRA   V 선택 head
12 Bike             0.410224     0.394322       0.394322      0.409553
12 Household        0.380649     0.371085       0.369608      0.377112
13 Bike             0.167942     0.166212       0.171382      0.179777
13 Household        0.478851     0.472700       0.471417      0.478851
```

네 cell 모두 `LORA_V = LORA_RECENT7 = ALL_V = ALL_RECENT7`이었다. 최근 V를 쓴 규칙이 별도 후보를 선택하지 않았다는 결과도 그대로 남겼다. 이 퇴화를 없애기 위해 다른 기간이나 선택 규칙을 추가하지 않았다.

## 단순 규칙과 사후 최적 후보를 구분

[확인] `FIXED_LOW`가 두 사전 정책 `LORA_V/FIXED_LOW` 중 더 나은 점수에서 벗어나는 양은 네 cell 순서로 **0.000 / 0.388 / 0.000 / 0.268%F0**다. 모두 1% 이내이므로 `simple_rule_veto=true`다. 이것은 고정된 두 정책에 대한 기술적 점추정 검사이며 통계적 동등성 확증이 아니다. FIXED_LOW도 각 fit 안에서는 원 validation-best checkpoint를 사용하므로 검증을 전혀 쓰지 않은 방법은 아니다.

**고정 작은 LR이 모든 후보 중 최적이라는 뜻은 아니다.** Bike12에서 사후 최적 LoRA LR1e-4는 FIXED_LOW보다 3.849%F0 좋았다. 따라서 선택 여지가 완전히 없어진 것은 아니다. 다만 그 사후 최적값은 배포 시 알 수 없고, 현재 B의 사전 조건인 두 원천 반복과 반대 방향의 실용적 비용을 통과하지 못했다. 이 oracle gap을 새 기준으로 바꿔 같은 결과를 양성으로 만들지 않는다. Head family의 선택 손해도 범용 PEFT 선택 실패로 확대하지 않는다.

QCAL은 보조 진단이다. 같은 네 정책 차이는 0.000 / −0.309 / +2.571 / −0.889%F0이며, Bike13 CI는 [−0.097, +5.156]%다. 이 보조 지표로 SORT 주판정이나 원 A의 QCAL 판정을 교체하지 않는다. 모든 후보의 C 적합 offset·기간별 점수·odd/even 및 3/14일 민감도는 전체 산출물에 보존했다. C 점수는 QCAL 적합 자료에 대한 재평가이므로 미래 성능으로 사용하지 않는다.

## 실행, 안전, 수치 검증

[확인] CPU 47검사와 4subtests가 통과했고, 마지막 그림 배치 수정 후 관련 2검사를 다시 통과했다. S0 4개는 원 선택 LoRA의 C/E 전체 SORT/native-unsorted 예측과 최대 차이 0이었다. 원12/13 모델·소스·데이터·결과·복구 로그·보고서·그림 등 557개 보호 항목의 hash를 유지했다. 계약 SHA256은 `4a0bab22a946ab0e6eb21d88edb72e61be15ae9c0e8da162b1dd3bae146d98da`다.

본 추론은 28/28 exit0, runner 457.341초(7분37초), guard 합 450.275초였다. S0는 별도 71.262초/guard64.921초다. CPU 분석 guard는 16.125초/exit0, 그림 생성은 2.60초/exit0으로 완료했다. 이번에는 실패·재시도·안전 중단이 없었다. 원12/13의 이미 지출한 학습비용은 이번 추가비용에 합치지 않았다.

본 추론 자원 56표본에서 RAM 여유 최소14.746GiB, commit여유10.424GiB, child tree RSS 최대0.674GiB, GPU 최대 관측1447MiB·43°C, Git 최대1개였다. 짧은 GPU 실행은 표본 사이 peak를 놓칠 수 있으므로 이 수치를 실제 절대 최대치나 향후 안정성 보장으로 해석하지 않는다. [Windows 조회](../../../../results/peft_selection_regret_v1/windows_events.json)는 17:04:32~17:21:46 KST의 Application1000/1002와 System41/4101/153/2004에서 각각 0건이었다. 기존 크래시의 인과가 해결됐다고 단정하지 않는다.

[확인] 별도 data-engineer가 14번 분석기를 import/copy하지 않고 원 prediction archive에서 NumPy로 80개 SORT/QCAL 점수와 네 주효과·CI를 재계산했다. 저장 분석과 최대 오차 0이었고 target/origin/quantile/count와 B gate도 일치했다. Root는 실제 PNG 세 개의 범례·축·수치·겹침을 확인했다.

## 자기 평가와 다음 방향

이번 진단은 선택 실패의 존재와 새 방법의 필요성을 구분하는 데 필요했다. 그러나 이러한 비교 결과를 모아 놓는 것만으로 사용자가 원하는 ML 방법론 논문이 완성되지 않는다. 현재 A의 범위 비교, B의 저장 후보 선택, C의 기존 synthetic 보정 현상은 각각 진입 근거가 약해 닫거나 보류했다. 다음에는 같은 LR·기간을 늘려 양성을 찾기보다, **표준 PEFT가 배포 입력이나 학습 목표에서 잃는 정보와 그 정보가 필요한 조건**을 먼저 특정해야 한다.

[다음 방법 선행 경계](../07_research_direction/14_next_method_literature_boundary_20260908.md)를 병행 검토했다. 공변량 정규화로 평균·규모 정보가 지워지는 현상은 코드상 단서가 있지만 기존 논문과 Chronos-2 공개 구현에 직접 선행이 있어 새 발견으로 채택하지 않았다. 정규화 손실을 raw pinball로 바꾸는 것도 기존 구현이 있어 단독 신규성은 약하다. 다음 후보는 실제 입력의 이용 가능 시점과 강한 단순 대조를 먼저 정하고, 그 대조를 넘을 수 있는 한 번의 반증 실험으로 좁힌다. 아직 다음 방법의 성능이나 신규성이 확보된 것은 아니며 전체 연구 목표는 미완료다.

## 산출물

- [40후보 × 2절차](../../../../results/peft_selection_regret_v1/candidate_results.csv), [7선택 규칙 × 4cell × 2절차](../../../../results/peft_selection_regret_v1/selected_results.csv)
- [효과·고정 gate](../../../../results/peft_selection_regret_v1/effects.json), [사후 oracle 손실](../../../../results/peft_selection_regret_v1/oracle_regret.json), [전체 진단](../../../../results/peft_selection_regret_v1/diagnostics.json)
- [비용 원장](../../../../results/peft_selection_regret_v1/costs.json), [자원 요약](../../../../results/peft_selection_regret_v1/resource_summary.json), [분석 검증](../../../../results/peft_selection_regret_v1/verification.json)
- [전체 후보의 V/미래 순위](../../../../results/peft_selection_regret_v1/figures/01_all_candidates_validation_and_future.png), [선택 손실과 비중복 진단](../../../../results/peft_selection_regret_v1/figures/02_selection_loss_and_nonoverlap.png), [사후 oracle 점추정](../../../../results/peft_selection_regret_v1/figures/03_hindsight_oracle_point_regret.png)

그림마다 PDF도 같은 폴더에 저장했다. [선택 계약](../../../../runs/peft_selection_regret_v1/selection_contract.json)과 [구현](../../../../experiments/peft_selection_regret_v1/README.md)에서 재현 경로를 확인할 수 있다.
