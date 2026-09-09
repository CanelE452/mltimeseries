# 저장된 PEFT 후보의 실제 선택 손실: 최소 B 진단

2026-09-08. [13번 결과](13_peft_temporal_replication_results_20260908.md)에서 현재 A 분기를 닫은 뒤 작성했다. 아직 예측하지 않은 후보의 미래 성능을 열기 전에 이 계획을 고정한다. 이미 선택된 일부 후보의 성능은 12·13에서 보았으므로 이번은 개발 자료를 이용한 탐색적 진단이다. 새 방법의 확증으로 부르지 않는다.

## 관측과 질문

Bike13에서 validation은 H와 LoRA 모두 F0보다 좋았지만, 미래에는 H가 명확히 나빴고 LoRA도 평균적으로 F0를 넘지 못했다. 그러나 H보다 LoRA를 선택하는 기존 규칙은 큰 손실을 피했다. 이 관측만으로 B가 성립하는 것은 아니다. 먼저 고정 후보의 전체 성능을 완성해 “더 작은 고정 LoRA나 최근 validation으로 피할 수 있는 선택 손실인가”를 검사한다.

질문은 저장된 validation-best checkpoint 사이의 LR/적응 범위 선택이다. Step별 가중치가 보존되지 않았으므로 early stopping 전체의 미래 regret을 주장하지 않는다. 조건은 12/13 × Bike/Household의 4개 cell이다. 동일 두 물리 원천의 다른 시간 블록이며 독립 원천4개가 아니다.

## 후보와 정보 고정

각 cell은 seed12000의 F0 한 개, H_MLP LR{1e-4,3e-4,1e-3}, H_FULL LR{3e-5,1e-4,3e-4}, OFF_LORA LR{1e-5,3e-5,1e-4}의 10개 후보를 사용한다. 다른 seed는 전체 grid가 없어 주분석에 넣지 않는다. RAW는 기존 참고점으로만 보존하고 PEFT 후보 universe와 oracle에 넣지 않는다.

원 fit validation NPZ의 origin별 손실과 best_trainable.pt를 읽기 재사용한다. 원천·블록별 이미 선택된 F0/H/LoRA의 3개 미래 예측은 재사용하고, 나머지7개를 모두 추론한다. **총28개 추가 미래 추론, 추가 학습0회**다. Step0 후보도 생략하지 않아 F0 동등성 예외를 추가하지 않는다. 각 후보는 C13/E83 origin을 동일 native batch/정규화/결측 경로로 예측한다.

원 12/13 study contract, 분석 verification, fit result/checkpoint, prepared data, 선택 및 기존 forecast를 보호한다. 기존 선택 파일을 바꾸거나 원 forecast의 selected-only gate를 완화하지 않는다. 별도 namespace의 진단용 후보 계약에서 numerical `load_base/construct/load_trainable/predict` 경로를 재사용한다. 선택되지 않은 후보의 추론은 진단 목적이고 당시 배포에 선택된 모델인 것처럼 기록하지 않는다.

## 사전에 고정할 단순 선택 규칙

1. `F0`: 원 모델 유지.
2. `FIXED_LOW`: OFF_LORA LR1e-5. 각 fit 내부의 기존 V checkpoint 선택은 남으므로 validation을 전혀 사용하지 않은 방법은 아니다.
3. `LORA_V`: LoRA3개 중 전체 V의 SORT 점수가 가장 낮은 후보. 원 LR 선택과 일치해야 한다.
4. `LORA_RECENT7`: 동일 LoRA3개 중 마지막7개 V origin의 SORT 점수가 가장 낮은 후보.
5. `ALL_V`: F0와 적응9개 중 전체 V 최저 후보.
6. `ALL_RECENT7`: 동일10개 중 마지막7개 V origin 최저 후보.
7. `HEAD_V`: H6개 중 원 선택 규칙으로 고른 후보. 저렴한 head family 안의 선택 손실을 별도로 보여준다.

전체 V의 기존 H/LoRA 동률 규칙은 원 runner와 같게 유지한다. 새 ALL 규칙의 동률 우선순위는 F0, LoRA LR 오름차순, H_FULL LR 오름차순, H_MLP LR 오름차순이다. 먼저 V에서 모든 규칙의 선택 JSON을 저장하고 해시를 고정한 후 전체 미래 예측을 읽는다. C는 선택에 쓰지 않는다. QCAL은 각 후보의 C에서 원 절차대로 적합한다.

## 분석과 중단 기준

주비교는 각 cell의 `100 × (S_LORA_V − S_FIXED_LOW) / S_F0`이다. **SORT가 주지표**다. B는 SORT를 최적화하는 선택 규칙의 미래 손실을 묻기 때문이다. 이는 원 A의 QCAL 주지표나 그 음성 판정을 수정하지 않는다. QCAL 결과는 보정 상호작용의 보조 진단으로 모두 보존한다.

7일 moving block 4,000회, seed2026090814, 네 주비교 각각98.75% CI로 계산한다. 양수는 선택된 LR의 손해다. 3/14일 block 및 짝수·홀수 origin(48h target 비중복, 여전히 시간 의존 가능)을 민감도로 기록한다. 2일 stride의 부분집합에서는 block origin 수를 ceil(요청일수/2)로 정해 3/7/14일 요청에2/4/7 origin을 사용한다. 실제 target 시간 범위는4/8/14일이므로 요청값과 실제 범위를 함께 기록한다. 13개 V origin의 bootstrap을 큰 독립 표본처럼 해석하지 않는다.

Family 내부(H6/LoRA3) 및 전체10개에서 `선택 score − 해당 universe의 미래 최저 score`를 별도 진단한다. 이 최저는 사후 oracle이며 실행 가능한 선택자가 아니다. Oracle regret는 기술적 점추정으로만 보고하고, 미래 최저 후보를 고정한 뒤 일반적인 paired CI를 붙여 선택 편향을 숨기지 않는다. 각 후보의 V/C/평가전반41/평가후반42 origin 손실도 기록해 시간별 순위 변화의 단서를 본다. 이를 계절 변화의 인과 증명으로 부르지 않는다.

현재 B에서 추가 방법 탐색으로 진입하려면 LORA_V 대 FIXED_LOW 손해의 CI 하한이1%F0를 넘는 cell이 두 원천에 걸쳐 반복돼야 한다. 동시에 다른 cell에서는 같은 주비교 CI 상한이−1%보다 작아, FIXED_LOW를 항상 쓰는 방법에도 실용적 손해가 있어야 한다. 이는 고정 작은 LoRA 하나로 해결되는 현상을 새 selector의 필요성으로 포장하지 않기 위한 조건이다. 같은 양성 cell에서 LORA_RECENT7 대 FIXED_LOW 손해 점추정도1%F0를 넘고, LORA_V와 LORA_RECENT7 모두 odd/even 부분집합에서 손해 점추정이 양수여야 한다. 최근7일·부분집합 조건은 기술적 진입 조건이며 별도 유의성 주장으로 사용하지 않는다. 통과해도 이는 제한된 저장 후보에 조건부인 문제 후보이며, 예측 가능한 실패 기전과 기존 selector 대조 및 새 데이터 재현은 별도로 필요하다. 고정 작은 LoRA/F0/최근 validation 같은 단순 규칙으로 손실을 피할 수 있거나 차이가 실용 문턱보다 작으면 현재 B 분기를 닫고 더 많은 LR·기간·선택 규칙으로 양성을 찾지 않는다. Head만의 손실을 범용 PEFT 선택 실패로 확대하지 않는다.

[Han et al., ICML2024](https://proceedings.mlr.press/v235/han24b.html)의 적응형 rolling window·쌍별 tournament는 이미 존재한다. 최근7일 baseline을 그 방법의 재현이라고 부르지 않는다. 향후 새 selector를 제안할 때는 해당 원문의 가정·정보·비용을 맞춘 비교가 필요하다. 현재는 새 selector를 제안하지 않는다.

위 단순 해결의 운영 정의는 다음과 같다. F0/FIXED_LOW/LORA_RECENT7/ALL_V/ALL_RECENT7/HEAD_V 중 동일한 규칙 하나가 네 cell 모두에서 `[S_rule − min(S_LORA_V, S_FIXED_LOW)] / S_F0 ≤ 1%`를 만족하면 `simple_rule_veto=true`로 진입을 막는다. 이는 이미 고정한 두 LoRA 선택 정책 중 더 나은 결과를 기준으로 하는 기술적 점추정 검사이며, 전체 후보의 미래 oracle을 사용할 수 있다는 뜻이 아니다. Veto와 모든 개별 값을 출력하고 이 비교를 통계적 동등성 확증으로 부르지 않는다.

## 실행과 검증

`experiments/peft_selection_regret_v1/`, `runs/peft_selection_regret_v1/`, `results/peft_selection_regret_v1/`를 사용한다. CPU 검사 후 각 원천·블록의 이미 완료된 seed0 LoRA 예측을 그대로 재현하는 4개 S0 추론에서 원 예측과 수치 경로·checkpoint 복구를 확인한다. 그다음 고정28추론을 실행한다. S0에서 성능으로 후보를 선택하지 않는다.

Root만 단일 GPU shared guard를 사용한다. RAM5GiB/commit6GiB/RSS8GiB/Git32/GPU10500MiB·85°C 한도를 유지하며, Git 급증으로 중단되면 실패 로그를 보존하고 원인을 점검한 후 그 작업만 재개한다. 원 12/13 학습비용은 이미 지출한 비용이고 이번 추가 추론·진단 비용과 구분한다. Commit/push·시스템 설정 변경·무관 프로세스 종료는 없다.
