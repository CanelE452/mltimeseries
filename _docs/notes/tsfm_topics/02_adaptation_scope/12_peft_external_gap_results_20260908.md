# 실제 수요·전력 데이터의 PEFT 비교 결과

2026-09-08. [계획](12_peft_external_gap_plan_20260908.md) · [신규성 경계](12_peft_external_gap_literature_20260908.md) · [지속 탐색 계약](../07_research_direction/11_peft_topic_search_protocol_20260908.md).

## 판정

[확인] 자전거에서는 표준 LoRA가 선택된 출력층보다 실용적으로 개선됐다. 가정 전력에서는 평균 이득은 양수지만 조건부 신뢰구간이 넓어 결론 보류다. 두 원천 모두에서 사전 기준을 통과해야 한다는 전체 진입 조건은 충족하지 못했다. RAW ridge는 두 원천 모두 LoRA보다 나빴다. 이는 새 PEFT 방법의 완성이 아니라 실제 원천에서 후속 확인할 신호다.

주효과는 `(H_QCAL − LoRA_QCAL) / F0_QCAL`이며 아래 백분율은 **H 대비 상대개선율이 아니라 F0 점수 단위**다.

```text
원천          평균 효과      조건부 97.5% CI       사전 1%F0 판정
자전거        +3.9253%       [+2.0523, +5.1823]%   통과
가정 전력     +1.3553%       [−0.0682, +3.2396]%   결론 보류
```

각 원천에서 세 seed의 효과는 모두 양수다. 7일 moving block4000회가 주분석이고, 3/14일 민감도에서도 자전거는1%문턱을 넘었다. 가정 전력의14일 CI는[+0.1688,+3.1011]%로0을 제외하지만1%문턱은 넘지 못한다. 세 fittedseed와 개발 선택에 조건부인 CI이며 전체 훈련·HPO 불확실성을 포함하지 않는다. 한 원천만 유의하다는 사실을 두 원천의 효과 차이가 유의하다는 주장으로 바꾸지 않는다.

## 실제 실행과 정보 조건

[확인] 새 adaptation26개, F0/cache2개를 완료하고14개 모델 선택을 고정한 뒤14개 유보 추론을 수행했다. 그다음 RAW2개를 별도CPUguard에서 적합했다. S0는train-only8개이며 본실험과 별도다. LR·checkpoint 선택은V_select만, QCAL은별도C_cal만 사용했다. 분석은 모든 선택·RAW 적합이 끝난 다음 수행했다.

두 원천은14일 과거문맥→64일train→14일V→14일C→84일eval의 고정 기간이다. L336/H48, 자정 기준 일별origin63/13/13/83개이며 인접target창은24시간 겹친다. Eval83개를 독립표본83개로 해석하지 않는다. Bike는casual/registered와과거날씨3열, Household는active/reactive power와과거Voltage/Intensity를입력했다. 미래보조열은제공하지않고비target열의미래loss도마스킹했다. Train 통계와과거ffill을사용하고결측정답은채우지않았다. Household eval의target관측률은약96.79%여서관측가능cell에조건부인평가다.

## 점수·불확실성·선택 결과

확인한 그림: [점수·coverage](../../../../results/peft_external_gap_v1/figures_verified/01_scores_and_coverage.png), [주효과·block 민감도](../../../../results/peft_external_gap_v1/figures_verified/02_primary_effect_and_block_sensitivity.png), [전체 개발 곡선](../../../../results/peft_external_gap_v1/figures_verified/03_all_development_trajectories.png). 같은 폴더에 PDF도 보존한다. 초기 그림의 F0 상수값 로그축 범위가 붕괴해 별표가 다른 패널에 표시되던 문제를 고쳤으며, 원 그림은 `figures/`에 보존하고 최종 `figures_verified/`의 PNG 세 개를 실제 열어 확인했다.

점수는target별train표준편차로정규화한관측cell평균2-pinball을두target동일가중으로합친다. 낮을수록좋다. H/LoRA는세seed평균, F0/RAW는각단일결정적절차다.

```text
원천          방법          SORT 점수   QCAL 점수   coverage80 SORT→QCAL
자전거        F0            0.410224    0.411637    78.39% → 85.19%
              H_FULL        0.410306    0.411241    80.90% → 85.37%
              OFF_LORA      0.393567    0.395083    82.42% → 85.46%
              RAW           1.149738    0.922020    38.19% → 67.92%
가정 전력     F0            0.380649    0.381728    73.30% → 77.61%
              H_FULL        0.377604    0.380376    72.65% → 75.82%
              OFF_LORA      0.370266    0.375202    75.92% → 77.52%
              RAW           0.438720    0.442691    76.82% → 76.58%
```

QCAL은여기서모든선택FM의proper score를소폭악화시켰다. Bike의coverage는nominal80%에서오히려멀어졌다. HouseholdLoRA의coverage는가까워졌지만score는악화됐다. 보정이불확실성을해결했다고주장하지않는다. 결과를보고SORT로주지표를교체하지않고사전QCAL주분석과SORT보조결과를함께보존한다. SORT기준H대LoRA효과는Bike+4.0805%F0,Household+1.9279%F0다.

H는두원천모두H_FULL/LR3e-5가선택됐다. Bike의H beststep은40/0/40,LoRA LR1e-5의beststep은120/160/120이다. Household의H beststep은120/200/160,LoRA LR1e-4의beststep은모두40이다. HouseholdLoRA의높은LR·이른최적checkpoint와H의최종step선택을예산·최적화한계로함께기록하며추가LR/step으로사후구제하지않았다. H후보6개와LoRA3개는같은HPO비용이아니다.

RAW는두원천모두lambda10을골랐다. 가정전력의RAW검증점수0.418514는seed0LoRA0.420459보다낮았지만,유보QCAL점수는0.442691로LoRA평균0.375202보다나빴다. 이는선택구간과미래구간의순위가바뀐관측이며,짧은V의추정잡음이원인인지계절이동·회귀외삽이원인인지분리하지못했다. 이한종류의ridge열세를모든단순예측기의열세로일반화하지않는다.

## 비용·안전·검증

[확인] 본GPU실행42개는15:19:42~15:32:47KST,runner wall784.703초(13분4.7초)였다. Fit guard합626.806초,forecast156.688초이며,미선택fit14개의280.045초도포함한다. S0 wall85.781초,RAW자체적합합0.539초/CPUguard합4.156초는별도다. Invocation wall과그안의guard시간을중복합산하지않는다.

H_MLP589,301params,H_FULL3,653,280params,OFF_LORA1,206,912params다. 출력층은동결특징cache를재사용하여한fit의내부wall약8초,LoRA는약39초였다. 파라미터수감소와실제학습시간은같은지표가아니다. Cache생성·추론·후보탐색비용을포함한[전체비용](../../../../results/peft_external_gap_v1/costs.json)을본다.

[확인] 본GPU98자원표본에서최소RAM여유13.95GiB/commit8.65GiB,최대childRSS1.67GiB/Git2/GPU2619MiB·58°C였다. GPU42개와RAW2개의guard는모두exit0/안전중단0이다. 15:34:33Windows조회에서S0시작이후Application1000/1002및System41/4101/153/2004는0건이다. CPU RAW는짧아시작표본만있고RSS0기록은작업전체의실제peak0을뜻하지않는다. 표본간peak나미래안정성을보장하지않는다.

최초CPU분석은GPU센서가없는CPU기록의`gpus:null`처리오류로6.094초/exit1이었다. 로그를보존하고자원요약의null처리만수정했다. 실제CPU/GPU기록을섞은회귀를포함한분석검사21개통과후,검증용CPU분석은6.078초/exit0로완료했다. GPU재학습이나원결과변경은없다.

[검증JSON](../../../../results/peft_external_gap_v1/verification.json)은선택32행/전체fit30행,28fit·14forecast정확한키,원source/plan/data/checkpoint/cache/초기값/sampler보존,결측마스크·점수재계산,RAW세lambda재적합·정규방정식·예측복원,별도C_cal보정·pairedblock계산을확인했다. [원CSV](../../../../results/peft_external_gap_v1/selected_results.csv),[효과](../../../../results/peft_external_gap_v1/effects.json),[진단](../../../../results/peft_external_gap_v1/diagnostics.json),[자원](../../../../results/peft_external_gap_v1/resource_summary.json),[Windows조회](../../../../results/peft_external_gap_v1/windows_event_audit.json)를보존한다.

## 자기 평가와 다음 실행

이번에는합성Gaussian의“ridge로충분”결론을실제원천에옮기지않았고,Bike에서표준LoRA의추가효용을찾았다. 다만두원천전체진입기준은실패했으므로방법론연구가완성됐다고부를수없다. 또OFF_LORA가native출력projection도바꾸기때문에내부attention수정이필요한지원인분리는남는다.

다음은두원천모두에서[사전고정한다음190일블록](13_peft_temporal_replication_plan_20260908.md)으로선택·보정·평가절차를한번반복한다. 양성원천만남기거나현재eval을다시유보표본으로부르지않는다. 새블록에서두원천진입기준을통과하면원블록Household의보류를유지한채두블록을나란히해석하고,native OUT_ONLY재학습대조로출력저랭크제약이라는대안설명을검사한다. 반복에서기준을못넘으면이배포·예산조건에서범용내부PEFT필요성을확인하려는분기를닫고,관측순위반전의원인을별도검사할지판단한다. 같은결과를구하기위해기간·LR·보정법을연속변경하지않는다.

원천은각하나이며현재둘다첫반년중심,단일Chronos-2backbone,기존pretraining포함여부unknown이다. 미래기간반복도새데이터원천이나독립backbone의확증과같지않다. TRACE/MixFT/HiP-LoRA와구분되는새학습규칙과강한대조는아직없으며전체연구goal은진행중이다.
