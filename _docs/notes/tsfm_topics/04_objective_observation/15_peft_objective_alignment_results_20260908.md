# 목적함수 정렬 진단 결과: 현재 변경의 실용 기준 미충족

2026-09-08. [사전 계획](15_peft_objective_alignment_plan_20260908.md)에 따라 두 개발 블록에서 같은 native LoRA의 손실만 바꾼 6개 학습과 6개 평가 추론을 완료했다. **현재 목적함수 정렬 진단은 닫는다.** Raw loss의 추가 효과는 Bike −0.218%F0, Household +0.122%F0로 두 97.5% CI가 모두 0을 포함하고 사전 +1% 기준에 미달했다. 새로운 PEFT 방법의 성능이나 필요성은 확보하지 못했으며 전체 연구 주제 탐색은 미완료다.

## 무엇을 비교했는가

[확인] Study12의 Bike/Household 자료와 amazon/chronos-2 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`를 재사용했다. 모든 학습은 OFF_LORA 97개 projection, rank8/alpha16, trainable1,206,912개, seed12000, LR3e-5, 200updates다. Train63/V13/C13/E83 origins, target2, L336/H48, 일별 origin, native21분위수를 유지했다. 동일 초기 가중치·sampler·모든 초기 gradient 측정이 실제로 일치했다.

- NATIVE: 기존 context-normalized/asinh pinball을 그대로 사용한다.
- NORM_ALIGNED: 정렬한 normalized 예측에 전체 train의 표적별 유효 셀 분모, Q평균, target macro를 적용한다.
- RAW_ALIGNED: 같은 정렬·분모·평균 규칙에서 raw 예측과 target의 pinball을 train-global target std로 나눈다.

Aligned 손실에는 train의 고정 8origin에서 측정한 초기 gradient norm 비율을 상수로 곱했다. 이는 초기 norm을 맞출 뿐 이후 AdamW·clipping을 같게 만들지 않는다. LoRA의 초기 B=0 때문에 이때 A gradient는 0이므로, 보정이 이후 양쪽 factor의 유효 업데이트를 모두 통제한다는 해석도 피한다. 모든 fit이 끝나고 validation 선택을 동결한 뒤 새 C/E 예측을 생성했다. C는 기존 QCAL 적합에만 사용했다.

Raw-loss fine-tuning은 이미 [Darts Chronos2Model](https://github.com/unit8co/darts/blob/master/darts/models/forecasting/chronos2_model.py)에 존재한다. 이번 실험은 관측한 구현 차이를 확인하는 진단이며 새 목적함수의 제안이 아니다. NATIVE↔NORM은 정렬·reduction을 함께 바꾸므로 reduction 단독 대조도 아니다.

## 주효과와 고정 판정

주효과는 `100 × (SORT_NORM_ALIGNED − SORT_RAW_ALIGNED) / SORT_F0`다. 양수는 raw 학습의 이득이다. 동일 평가 origin의 7일 moving block4,000회, seed2026090815, 원천별97.5% CI를 사용했다.

```text
원천           raw 추가 효과       97.5% CI                  하한 > +1%F0
Bike             −0.218%          [−0.680%, +0.142%]             아니오
Household        +0.122%          [−0.059%, +0.255%]             아니오

고정 판정: CLOSE_CURRENT_OBJECTIVE_ALIGNMENT_SCREEN
```

두 주 CI의 상한도 +1%F0보다 작다. 따라서 단순히 표본이 부족해 큰 실용 이득을 판별하지 못했다는 설명보다는, **저장된 이번 실행에 조건부인 날짜 변동 범위에서는 큰 개선 신호가 없었다**는 해석이 맞다. 다만 학습 seed/LR/예산의 불확실성까지 포함한 동등성 검정은 아니다. 목적 정렬이 어떤 조건에서도 무관하다는 뜻으로 확대하지 않는다.

[확인] SORT 점수는 다음과 같다. 낮을수록 좋으며 블록 간 절대 점수의 차이를 직접 성능 개선율로 비교하지 않는다.

```text
원천              F0          NATIVE       NORM_ALIGNED    RAW_ALIGNED
Bike             0.410224     0.395552       0.395569        0.396462
Household        0.380649     0.370996       0.371011        0.370547
```

모든 적응 arm은 이번 SORT 점추정에서 F0보다 좋았다. Native의 F0 대비 개선율은 Bike3.577%/Household2.536%, raw는3.355%/2.654%다. 이는 새 손실의 추가 효과와 다른 비교이며 여기서 별도의 유의성 주장을 하지 않는다. 현재 실패 판정은 “LoRA가 작동하지 않았다”가 아니라 “목적 정렬 변경이 사전 실용 이득을 만들지 못했다”다.

3/14일 block에서도 주효과 CI가 모두 0을 포함했다. 비중복 target window를 만든 even/odd 효과는 Bike−0.550%/+0.142%, Household+0.065%/+0.180%다. Bike의 parity 방향 차이는 그대로 남기며 좋은 날짜 묶음을 선택하지 않는다. 각 target·QCAL·parity는 기술적 보조이며 새로운 진입 기준으로 바꾸지 않았다.

QCAL에서 raw 추가 효과는 Bike−0.213%/CI[−0.636,+0.112]%, Household+0.053%/[−0.130,+0.178]%였다. 이번에도 QCAL의 proper score는 F0와 세 적응 arm 모두 SORT보다 나빴다. 명목80% coverage가 일부 개선됐다는 이유로 불확실성 문제가 해결됐다고 주장하지 않는다.

## 기전 진단과 자기 평가

[확인] NATIVE↔NORM의 초기 gradient cosine은 Bike0.999984/Household0.999991, NORM↔RAW는0.895358/0.988098이었다. Train std로 정규화한 raw-loss Jacobian `context_scale × cosh(z) / train_global_target_std`의 median/p95/max는 Bike1.056/2.686/12.355, Household1.255/2.859/11.377이었다. 학습 공간을 바꾸면서 gradient가 실제로 달라졌다는 것은 확인했지만, 그 차이를 성능 이득으로 연결하지 못했다.

```text
조건                     고정 multiplier       clip 발생 / 200        선택 step
Bike NATIVE                    1.000                  171                 40
Bike NORM_ALIGNED              7.971                  171                 40
Bike RAW_ALIGNED               5.540                  121                 40
Household NATIVE               1.000                  169                 80
Household NORM_ALIGNED        10.513                  168                 80
Household RAW_ALIGNED          7.527                  133                 80
```

이번에는 세 arm의 선택 step도 원천별로 같았다. 이는 동일 가중치라는 뜻은 아니지만, 서로 다른 validation checkpoint 시점이 주효과를 만든 설명은 해당하지 않는다. Clipping 감소 자체도 최종 성능 향상이나 더 나은 최적화를 뜻하지 않는다. NORM↔RAW는 context scale와 asinh Jacobian의 곱 전체를 바꾸므로 asinh만의 인과 효과로 해석할 수 없다.

자기 평가: 실제 코드의 손실 차이를 발견한 뒤 이것을 새 방법의 근거로 낙관하지 않고 원 trajectory 재현과 단순 raw-loss 대조를 먼저 실행한 것은 유효했다. 그러나 gradient 차이나 보기 좋은 미분 공식은 논문 기여의 증거가 아니었다. 같은 두 블록에서 LR·step·threshold를 다시 바꾸는 탐색을 이어갈 근거가 부족하다. 이번 분기는 사전 규칙대로 종료하며 이 변경의 큰 sweep이나 더 복잡한 gradient 조정을 추가하지 않는다.

다음 작업은 [이전 후보의 선행 경계](../07_research_direction/14_next_method_literature_boundary_20260908.md)에 남은 **한 시퀀스 안의 이종 관측 연산(point/interval mean/sum)** 문제의 연구 진입 가능성을 검토하는 것이다. 과거 interval-integrated Fourier 음성 실험을 확대하기에 앞서, 관측 연산 정보가 없을 때의 식별 불가능성과 그 정보를 받은 단순 state-space/출력 보정의 충분성을 CPU에서 분리해야 한다. 단순 대조가 해결하면 새 FM PEFT를 만들지 않는다. 이는 다음 주제를 확정했거나 새 성능을 얻었다는 선언이 아니며, 아직 해당 학습 실험은 실행하지 않았다.

## 적용 범위와 사전학습 노출

이번 자료는 우리가 이미 여러 가설에서 본 개발 블록이다. 97.5% CI는 이 두 주대비의 날짜 불확실성에 관한 것이며, 전체 반복 탐색의 다중 비교나 학습 변동까지 통제하지 않는다. 독립 원천·새 기간·다른 backbone의 확증 결과가 아니다. OFF_LORA에는 출력 projection도 포함하므로 이번 결과로 내부 encoder 적응의 필요성을 따로 주장하지 않는다.

[고정 revision 모델 카드](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/README.md)는 Chronos/GIFT-Eval pretraining 자료의 일부와 합성 자료 사용 및 GIFT test 제외를 설명한다. [Chronos-2 논문 §4.1·Appendix A·§5.1](https://arxiv.org/html/2510.15821v1)의 공개 전체 학습 목록에는 두 UCI 원천이 명시되지 않고 fev-bench의 미사용을 저자가 주장한다. Mexico City Bikes를 우리의 Washington D.C. Bike Sharing과, Electricity370계열을 단일 Household 원천과 혼동하지 않는다. 다만 정확한 원천·기간과 이 checkpoint의 실제 학습 manifest를 독립 대조하지 못했으므로 contamination 상태는 unknown이다. 모델 카드만 본 이전 기록보다 강한 논문의 저자 주장을 추가로 확인한 것이지, 엄격한 비중복을 증명한 것은 아니다.

## 실행과 검증

[확인] CPU53검사 통과, S0 6개 및 본6fit·6forecast가 모두 exit0이었다. 두 native의 S0와 본 fit은 원12의 초기 tensor, sampler, 전체 validation history, 선택 step, 복원 tensor와 저장 V 예측을 정확히 재현했다. S0 이후 분석기의 독립 metadata·gradient 보정 대조도 통과했다. 부모 산출물885개를 보호했으며 최종 분석은1,057개 입력 산출물 hash와6개 출력 hash를 기록했다. 계약 SHA256은 `e20970e05d8099ada23219980c41fd0e638af5564ea31f6ad0dcee8edef0ea37`다.

최종 계약 재검증과 종료 처리를 포함한 전체 S0 runner는102.505초/guard99.469초, 본 runner는458.546초(7분39초)/fit guard297.141초/forecast guard157.890초였다. `smoke_completed.json`의101.050초와 `completed.json`의457.126초는 그 뒤의 최종 검증을 포함하지 않은 중간 완료 기록이다. 전체 비용은 `invocations`의 finally 종료 시각을 사용한다. 새 optimizer update는 본1,200회+S0 30회다. CPU분석 guard36.218초/exit0, 그림 생성2.49초/exit0을 확인했다. 실패·재시도·안전 중단은0회다. [비용 원장](../../../../results/peft_objective_alignment_v1/costs.json)의 실행 중이던 자기 CPU분석 비용은 외부 완료 audit에서 별도로 추가한다. 이전 실험·cache·F0의 매몰 비용은 이번 추가비용에 합치지 않는다.

GPU 18시도의 자원60표본에서 여유 RAM 최소13.924GiB, commit여유8.074GiB, child tree RSS 최대1.803GiB, GPU 최대 관측2581MiB/57°C, Git 최대2개였다. [Windows 조회](../../../../results/peft_objective_alignment_v1/windows_events.json)는18:06:06~18:18:42 KST의 Application1000/1002 및 System41/4101/153/2004가 모두0건이었다. 표본 간 절대 peak나 향후 안정성을 보장하지 않으며 과거 프리즈 원인의 해결을 의미하지 않는다.

[확인] [독립 NumPy 검산](../../../../results/peft_objective_alignment_v1/independent_audit.json)은 이번 분석기를 import/copy하지 않고 예측 archive와 train std에서16개 SORT/QCAL 점수 및 두 주효과·CI를 다시 계산했다. 점수 최대 오차0, 효과/CI 최대 오차2.62e-16, origin·target·분위수 유효 count 차이0, 모든 bootstrap4,000회 유효였다. Root가 실제 PNG3개의 축·범례·수치·배치를 확인했다.

## 산출물

- [16개 절차 점수](../../../../results/peft_objective_alignment_v1/selected_results.csv), [효과와 고정 판정](../../../../results/peft_objective_alignment_v1/effects.json), [gradient·trajectory 진단](../../../../results/peft_objective_alignment_v1/diagnostics.json)
- [자원 요약](../../../../results/peft_objective_alignment_v1/resource_summary.json), [분석 검증](../../../../results/peft_objective_alignment_v1/verification.json), [실험 계약](../../../../runs/peft_objective_alignment_v1/study_contract.json), [재현 경로](../../../../experiments/peft_objective_alignment_v1/README.md)
- [점수와 주효과 그림](../../../../results/peft_objective_alignment_v1/figures/01_scores_and_primary_effect.png), [validation과 미래 점수](../../../../results/peft_objective_alignment_v1/figures/02_validation_and_future.png), [gradient 방향과 clipping](../../../../results/peft_objective_alignment_v1/figures/03_gradient_directions_and_clipping.png). 같은 폴더에 각 PDF를 저장했다.
