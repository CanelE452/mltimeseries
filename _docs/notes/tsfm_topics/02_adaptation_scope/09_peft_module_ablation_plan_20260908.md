# Q00 native LoRA 모듈 삭제 대조 계획

2026-09-08. 사용자가 “그렇게 진행해줘”로 [이전 결과](08_peft_shift_mechanism_results_20260908.md)의 출력층-only/attention-only 후속 대조를 승인했다. 목적은 ML 방법론 주제의 근거 확보 → 일반 LoRA 개선 경로 구분 → Q00 모듈 삭제 진단이다. 새 두 방법의 결과를 보기 전에 이 실행 계약을 저장한다. 이전 Q00 evaluation을 이미 분석했으므로 이번은 탐색적 후속이며 blind confirmatory test가 아니다.

## 선택과 대안

백본·생성법·데이터·loss·학습 순서를 유지하고 원 OFF_LORA 모듈 집합의 일부를 제거한 뒤 원 checkpoint에서 다시 학습한다. [Chronos-2](https://arxiv.org/html/2510.15821v1)의 native quantile head를 유지한다. 새 head를 추가하면 지난 실험의 혼동이 다시 생기므로 제외한다. 모듈 map과 LoRA rank/초기화는 검증된 로컬 소스를 기준으로 한다.

대안은 네 조건을 모두 확대하는 것과, 이미 학습한 BOTH에서 추론 때만 LoRA를 끄는 것이다. 전자는 비용에 비해 Q00의 가장 기본적인 귀속 문제를 먼저 해결하지 못한다. 후자는 흥미로운 coadaptation 검사지만 학습 중 다른 경로의 보상 가능성을 허용하지 않는 다른 질문이다. 이번에는 Q00 한 조건의 재학습 삭제 대조를 택한다. 이 결과를 학습 완료 모델의 내부 인과 기전이나 동일 예산의 모델 순위라고 부르지 않는다.

```text
이름        학습하는 원 OFF의 부분집합                  파라미터
F0          없음                                       0
BOTH        time/group qkvo96개 + output projection1개  1,206,912
OUT_ONLY    native output_patch_embedding.output_layer 27,264
ATTN_ONLY   time/group attention qkvo96개               1,179,648
```

모든 LoRA는 r8/alpha16, 새 head 없음, 그 밖의 pretrained parameter 동결이다. 남긴 module의 A 초기화는 동일 seed·module 이름을 사용하고 B는0이다. OUT과 ATTN map은 겹치지 않으며 합집합이 BOTH다. OUT_ONLY는 native head 전체를 갱신하는 것이 아니라 마지막 projection만 갱신한다.

## 데이터·실행 계약

기존 Q00 corpus0/1/2의 train64/validation128/evaluation512 독립 episode를 그대로 사용한다. Y/U/V 과거256 → Y 미래16, 원본 미래 보조변수 없음. Validation/evaluation은 corpus 간 공유하며 corpus별 optimizer seed7100/7101/7102를 유지한다. 기존 F0/BOTH 결과와 cache는 source/data/selection/prediction hash를 확인한 뒤 읽기 재사용한다. 기존 파일과 계획은 수정하지 않는다.

새 OUT_ONLY/ATTN_ONLY × LR{3e-5,1e-4} × corpus3 = **12 fit**. 200 updates, validation40간격 및step0, effective8/micro4 groups, 동일 sampler, AdamW/weight decay0/clip1, native normalized quantile loss, raw mean2-pinball selection을 유지한다. 같은4group 추론, BF16 cache-disabled, TF32 off, CPU2 threads/interop1/CUDA cap.67이다.

주비교는 **3e-5 고정**이다. 이는 이전 Q00 BOTH의 선택 LR이며 삭제 조건에서 같은 업데이트 설정을 사용한다. 부비교는 corpus0 validation에서 두 LR 중 하나를 고르고 corpus1/2에서도 그 LR을 고정한 절차다. 모든 corpus에서 두 LR을 실행하므로 선택되지 않은 추가 반복의 비용도 숨기지 않는다. 더 큰 LR/더 긴 학습으로 결과를 구제하는 후속 탐색은 이번 범위에서 하지 않는다.

S0는 OUT_ONLY/ATTN_ONLY/BOTH wrapper 각각5updates, train/val/eval8개로 검증한다. 원 모듈 map·동결·optimizer 대상·출력부 미추가·zero-update F0 일치·checkpoint 복구·미래/그룹 격리를 확인한다. Wrapper의 BOTH S0를 이전 동일 설정 BOTH S0와 비교해 재사용 경로의 수치 재현성을 확인한다. 같은 module seed의 A/B 초기화와 sampler hash도 대조한다. S0를 통과한 현재 소스·계획 계약에서만 본학습을 실행한다.

## 효과·해석·중단 기준

주효과 두 개는 다음처럼 정의하며, 양수는 해당 경로를 제거했을 때 점수가 나빠짐을 뜻한다.

```text
attention 삭제 비용 = (S_OUT_ONLY − S_BOTH) / S_F0
output 삭제 비용    = (S_ATTN_ONLY − S_BOTH) / S_F0
```

512개 evaluation episode를 paired bootstrap4000회 재표집한다. Corpus 평균 후 같은 표본 가중치로 분자·분모를 계산한다. 두 대비로 구성된 한 가족에97.5% CI를 사용하고 운영 문턱δ=.01을 유지한다. CI 하한>+.01 및 세 corpus 모두 양수면 반복적 실용 악화다. CI 전체가[−.01,+.01] 안에 있으면 **이 조건·학습 절차의 세 fitted repetition에 조건부인 평균 효과에서** 실용 동등 신호로 분류한다. CI가0을 포함한다는 이유만으로 동등이라고 하지 않는다. 반대 방향의 큰 개선, 그 밖의 미결정을 구분한다.

CI는 학습된 세 반복에 조건부이고 train/optimizer·validation 선택의 전체 불확실성을 포함하지 않는다. 선택 LR 대조는 별도 탐색적 가족으로 보고한다. 서로 다른 LR에서 학습한 최적화 절차 비교이며 모듈을 제외한 모든 경로가 같다고 해석하지 않는다. HPO 선택은 validation만 사용한다.

F0 대비 score/MSE, 80% coverage/width, 참 자기지연/U/V 성분에 대한 median 예측의 OLS 계수, oracle 평균 MSE를 기술한다. OLS는 출력의 기술적 분해이며 hidden 정보 부재나 공변량 무사용을 입증하지 않는다. 원 RAW/oracle/ALIGNED는 맥락 대조로 읽기 재사용하며 새 방법의 동등 정보·탐색 예산 대결로 해석하지 않는다.

OUT_ONLY가 BOTH와 비슷하면 이번 개선에 attention 갱신이 필수라는 주장을 약화한다. ATTN_ONLY가 비슷하면 native output projection 갱신이 필수라는 주장을 약화한다. 둘 다 나쁘면 이 절차에서 결합 경로가 유용할 수 있지만 낮은 모듈 용량·최적화·재학습 보상을 분리한 인과 증명은 아니다. 어느 결과에서도 자동으로 새 adapter를 제안하지 않는다. Q00 결과에 따라 더 유리한 generator를 찾아가는 연쇄 실험은 하지 않는다.

## 자원·산출물

사전 조회: available RAM17.83GiB/commit17.89GiB, Git0, GPU1123MiB/44°C, 이전 lock 없음. 모든 GPU trial은 기존 공유 guard 아래 직렬 실행한다. 경계는 available RAM5GiB/commit6GiB, child RSS8GiB, Git32, GPU10500MiB/85°C다. 기존 시스템 설정·드라이버·Defender와 무관한 프로세스는 변경하지 않는다. 오류면 소유 trial만 중단하고 원인 해결 전 다음 단계로 가지 않는다.

새 코드는 `experiments/peft_module_ablation_v1/`, 실행 artifact는 ignored `runs/peft_module_ablation_v1/`, 작은 결과는 `results/peft_module_ablation_v1/`에 둔다. 기존 source/plan/data/cache/result hash를 보존한다. 종료 후 점수 재계산·정확한12trial/선택 key·LR선택·source/data/초기화/sampler·checkpoint·자원·Windows 이벤트를 검증하고 결과 보고서를 저장한다. Commit/push는 하지 않는다.
