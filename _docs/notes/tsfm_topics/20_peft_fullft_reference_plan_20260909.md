# Study20: Chronos-2에서 실제 full fine-tuning과 PEFT의 차이 확인

2026-09-09. 사용자가 승인한 다음 단계다. 새 방법을 만들기 전에 F0, native head-only, 표준 LoRA, 실제 전체 모델 fine-tuning과 단순 예측기의 성능·비용을 비교한다. 이전 H_FULL은 출력 head 전체 학습이며 full-model FT가 아니었다. 이번 실험 자체를 새 ML 방법의 성공으로 부르지 않는다.

## 질문과 해석

주질문은 같은 정보와 학습 목적, 유한 예산에서 FULL_FT가 LoRA보다 안정적으로 유리한가이다. LoRA가 head-only보다 유리한지도 함께 본다. FULL_FT는 성능 상한이 아니며, 차이가 나도 표현력과 최적화·정규화 차이가 함께 포함된다. LoRA와 FULL_FT가 비슷하면 이 조건에서 새로운 적응 구조의 필요성이 약하다는 뜻이지 모든 PEFT 연구가 끝났다는 뜻은 아니다. 적응이 F0나 단순 기준선보다 나쁘면 이 조건에서 추가 적응의 필요성을 재검토한다.

관련 선행은 [SFF](https://arxiv.org/html/2606.08578v1)의 LP/FF 비교와 [TRACE](https://arxiv.org/html/2503.16991v3)의 적응 위치 선택이다. 비교 결과만으로 새 rank/gate를 명명하지 않는다. 긍정 신호가 있으면 이후 별도 계획에서 데이터량·최적화 예산과 실패 구간을 통제해 원인을 반증한다.

## 사용하지 않았던 다음 기간

Study12/13의 두 190일 블록 이후 세 번째 블록을 날짜만으로 선택한다. Study15는12번 자료 재사용이다. 원천과 시기는 여전히 과거 UCI 자료이고 Chronos-2 사전학습 포함 여부는 UNKNOWN이다. 로컬 실험에서 미평가였다는 뜻이며 사전학습 오염이 없다는 주장은 아니다.

Hourly L336/H48, origin stride24h. 처음14일은 context 확보에만 사용한다. 이후 train91일, embargo2일, V31일, embargo2일, C21일, E81일로 총242일이다. 각 split에서 target 48h 전체가 구간에 들어오는 origin만 사용하므로 정확히 train90/V30/C20/E80 origins다. C와E는 end-exclusive target 구간이 겹치지 않는다. E의 후속 origin에서는 그때까지 실제 관측된 과거값을 context로 쓰는 rolling-origin 평가이며 모델 업데이트는 없다.

```text
                         Bike                         Household
block start              2012-01-16 00:00             2008-01-01 00:00
first train origin       2012-01-30 00:00             2008-01-15 00:00
train target end          2012-04-30 00:00             2008-04-15 00:00
V interval               2012-05-02 -> 2012-06-02      2008-04-17 -> 2008-05-18
C interval               2012-06-04 -> 2012-06-25      2008-05-20 -> 2008-06-10
E interval               2012-06-25 -> 2012-09-14      2008-06-10 -> 2008-08-30
```

입력은 기존 Bike5/Household4채널과 표적2개, native21quantiles를 유지한다. 원 관측·결측을 보존하고, 입력 결측은 첫14일 precontext에서만 얻은 median fallback과 인과적 forward fill로 채운다. 미래 bfill은 없다. 이는 OOF 앞부분에 전체train 통계가 유입되지 않도록 기존 preprocessing에서 명시적으로 바뀐 부분이다. 모든 방법에 동일하게 적용한다. 평가용 target-global mean/std는 관측된 train target 구간만으로 계산한다. fit archive는 V 끝까지만 물리적으로 포함하고 C/E target은 별도 holdout archive에 저장한다.

## 고정한 학습과 선택

로컬 amazon/chronos-2 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`를 매 fit마다 다시 불러온다. native parameters119,477,664, HEAD_ONLY3,653,280, LORA1,206,912개를 실제 requires_grad와 optimizer 등록으로 확인한다. HEAD_ONLY는 output_patch_embedding 전체 residual block, LORA는 기존97 projection/rank8/alpha16, FULL_FT는 native 전체 파라미터다. 출력 head의 직접 forward를 공통으로 쓰며 head-only 전용 feature cache는 만들지 않는다.

Native pinball loss, 200updates, V every40, step0 선택 가능, strict `<` best checkpoint. AdamW weight_decay0/foreachFalse, clip1, effective batch8(4+4), FP32 weights/BF16 autocast, cacheFalse, TF32False, dropout0, CPUthreads2다. 학습·선택 목적을 새 손실로 바꾸거나 gradient norm 보정하지 않는다.

```text
seeds       20000, 20001, 20002
HEAD_ONLY   LR 3e-5, 1e-4, 3e-4
LORA        LR 1e-5, 3e-5, 1e-4
FULL_FT     LR 1e-6, 3e-6, 1e-5
```

FULL_FT grid는 로컬 Chronos2Pipeline.fit 기본1e-6을 포함한다. 두 원천×세arm×세LR×세seed=54fits. 각 fit의 best step을 V로 정한 후 원천/arm별 세seed 평균 V score가 가장 작은 LR 하나를 선택한다. 동점이면 나열된 낮은 LR을 선택한다. grid 경계가 최선이어도 이번 결과를 구하기 위해 확장하지 않으며 유한 grid 한계로 보고한다. 모든54fits와 단순 기준선 선택을 전역 동결한 다음에만 C/E forecast를 실행한다. E로 checkpoint/LR/seed/방법을 고르지 않는다.

S0는 각 원천×세arm의 train-only5updates, seed20000, arm별 첫 LR이다. train 첫8origin과 train index10:14의 pseudoV4만 쓴다. FULL_FT 첫 update에서 encoder/head 실제 변화, 정확한 optimizer scope, finite nonzero gradient를 검증한다. 모든 arm의 step0 F0 동일성, checkpoint CPU 재로딩 후 V예측 복원, 적용 가능한 frozen parameter 보존을 확인한다. FULL_FT의 frozen tensor 검사는 해당없음으로 남긴다.

## 단순 기준선과 분석

계절성 기준선은 1주 전 값에 train-only 계절성 residual quantiles를 더한다. Ridge는 모든 채널의 L336 context를 펼쳐 target/lead별 직접 예측하며 lambda0.01/1/100이다. Quantile offset은 시간순 OOF residual로 얻는다: 첫30origin으로 다음30, 첫60으로 마지막30을 예측하되 fold 시작 시점에 H48 전체 정답이 도착하지 않은 fit row는 제외한다. feature scaler/target center도 fold 과거만으로 구한다. 최종 모델은 train90으로 적합한다. 주단순기준선은 V에서 seasonal1+ridge3=4후보 중 선택한다. 각 neural arm의3LR와 단순4후보 비용을 모두 보고한다.

주지표는 SORT21quantile의 train-global target std 정규화 mean2pinball을 표적별 동일 가중 평균한 값이다. 모든방법의 주평가는 C를 쓰지 않는다. 보조 QCAL은 모든방법에 동일한 C residual quantile offset을 적용하고 다시 정렬하며, SORT/QCAL 중 E가 좋은 것을 선택하지 않는다. median MSE,80%coverage/width,crossing도 함께 보고한다.

세seed의 loss를 평균하며 예측을 평균한 ensemble로 바꾸지 않는다. 주대비는 원천별 FULL_FT vs LORA, LORA vs HEAD_ONLY의 `(left loss-right loss)/F0 loss`이고 양수는 right가 유리하다. 7일 moving-block bootstrap4000회, seed2026090920,95% CI와 개별seed 범위를 기록한다. 두 원천 네 대비의 탐색적 CI이며 다중비교 보정 확증이 아니다. 기간80origin과 선택된 모델에 조건부인 시간 불확실성이며 HPO/사전학습/전체 훈련 불확실성을 포괄하지 않는다. 두 원천 모두1% 같은 임의의 hardgate로 연구 가능성을 결정하지 않는다.

학습시간, optimizer seconds/step, V·load·checkpoint시간, CUDA max allocated/reserved, sampled RSS, trainable count와 실제 checkpoint 크기,54fits 전체 선택비용을 기록한다. trainable count만으로 실행 효율을 주장하지 않는다. FULL_FT도 최저 V CPU state 한 벌만 유지하고 마지막에 한 번 저장한다. 저장 파일은 weights-only이며 optimizer/RNG exact resume를 제공하지 않는다.

## 실행·보존

새 namespace `peft_fullft_reference_v1`만 사용한다. 계획/데이터준비/학습/선택 코드를 S0 전에 동결하고, 분석코드는 E예측 전 동결한다. 이전19번 보호계약과 완료 결과를 시작/끝에 확인한다. 실패 기록은 보존하고 오류 수정 시 기존 계약·실행을 덮어쓰지 않는다.

Root만 기존 shared guard를 통해 한 번에 GPU child 하나를 실행한다. RAM5GiB/commit6GiB/childRSS8GiB/Git32/GPU10500MiB·85°C 기준을 유지한다. CPU 준비·검사·분석도 guard로 실행한다. S0 실측으로 전체 예상시간을 갱신하며 리소스 이상이면 해당 child를 중단하고 원인을 조사한다. Windows 드라이버·Defender 등 시스템 설정과 무관 프로세스는 변경하지 않는다. 크래시가 절대 발생하지 않는다고 보장하지 않는다.
