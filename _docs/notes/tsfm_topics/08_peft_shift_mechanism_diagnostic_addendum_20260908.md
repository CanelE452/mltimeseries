# 통제 screen의 추가 추론 진단

2026-09-08. 원 [실행 계획](08_peft_shift_mechanism_plan_20260908.md)과 124개 본실행의 source/data 계약은 유지한다. 아래 항목은 본실행 중 검증 경로를 검토하면서 추가한 **탐색적 진단**이다. 원계획의 주대비나 학습 방법을 바꾸지 않는다. 본실행의 FM evaluation 점수는 이 문서를 저장하는 시점에 분석하지 않았다. 작은 RAW unit test에서는 evaluation의 oracle 평균 회수 오차를 이미 확인했으므로, 전체 개발 과정이 평가에 완전히 눈가림됐다고 표현하지 않는다.

## 1. 순열 대칭성과 수치 정밀도

Chronos-2의 채널 순열 대칭성은 수학적 함수 제약이며 BF16에서 bitwise 동일성을 보장하지 않는다. S0에서 U/V 순서를 바꾼 차이가 관측됐으므로, Q00 corpus0의 train 첫 4개 episode에 한해 원본 checkpoint의 FP32 및 BF16 출력 차이를 기록한다. Optimizer update는 0회다. FP32 normalized 최대 차이가 1e-5를 초과하면 대칭성 구현 해석을 다시 조사한다. 이 허용값은 예측 성능의 동등성 문턱이나 오차 하한이 아니다.

## 2. 알려진 과거의 지연 정렬을 미래 공변량 경로에 제공

묻는 것은 **같은 관측 정보의 표현을 바꾸면, 내부 학습 없이 회수 가능한 성능이 있는가**다. Lag alignment 자체는 새 방법으로 주장하지 않는다. [LIFT, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/file/b52b07a239a7afa155ca25cf17a55074-Paper-Conference.pdf)는 이미 지연 정렬과 frozen backbone의 예측 보정을 다룬다. [Chronos-2](https://arxiv.org/html/2510.15821v1)의 native known-future-covariate 경로를 이용한다.

입력 길이 L=256, 예측 길이 H=16, 후보 지연 ℓ∈{32,48,64}. Y 과거는 그대로 두고 각 driver Z∈{U,V}에 대해 다음을 만든다.

```text
정렬 과거: Z_aligned[t] = Z_original[t−ℓ],  ℓ≤t<L
           첫 ℓ개 위치는 결측으로 표시
정렬 미래: Z_aligned[L+h] = Z_original[L+h−ℓ],  0≤h<H
Y 미래:   전체 마스킹, target 값은 모델에 전달하지 않음
```

ℓ≥H이므로 사용한 원본 인덱스는 모두 L 미만이다. 원본 미래 U/V 관측을 추가하지 않는다. 원본 driver의 마지막 ℓ−H개 값은 사용되지 않는다. 시간 정렬, 과거 결측·길이, 정규화 통계, 미래 공변량 경로 활성화가 함께 바뀌는 **공동 개입**이다. 개선되더라도 group attention 하나의 인과적 실패라고 해석하지 않는다.

각 조건 Q의 corpus0 validation128에서 세 후보를 모두 평가하고 최소 점수의 ℓ를 선택한다. 동점은 후보 순서32→48→64를 따른다. 선택 JSON을 저장한 뒤 선택된 네 설정만 evaluation512에서 한 번 평가한다. Corpus 사이 validation/evaluation이 같고 모델도 동결돼 있으므로 이를 세 독립 학습 반복이라고 세지 않는다. 후보 지연 사전은 생성식을 아는 사람이 구성했고 참 지연48을 포함한다. Validation 선택으로 이 사전지식 이점이 없어지지는 않는다.

원본 F0 대비 Δ=(S_F0−S_aligned)/S_F0를 조건별로 보고한다. 같은 512개 episode를 paired bootstrap4000회 재표집하고 네 조건 가족의98.75% CI를 사용한다. 이 진단 가족을 원계획 모든 비교와 합친 전역 오류율 보장이라고 부르지 않는다. 원본 LoRA와의 수치 비교는 맥락 제공이며 학습·탐색 예산이 같은 새 방법 대결이 아니다.

BF16 cache_disabled, TF32 off, encoder4 groups/12 rows, CPU2 threads, 원본 checkpoint를 유지한다. 본실행이 완료되고 공유 guard lock이 해제된 후 순차 실행한다. 학습 update0, 설정 추가·후속 LR 탐색0. 입력 인덱스, 결측 위치, 원본 입력 불변과 선택/evaluation 순서를 검증한다.

## 3. 예측값의 생성 성분 회수

각 median 예측을 알려진 세 조건부 평균 성분 `.5Y[t−s]`, `sqrt(.39)cosθU[t−48]`, `sqrt(.39)sinθV[t−48]`와 절편에 선형 투영해 계수를 기술한다. 이 결과는 출력 예측에 어떤 성분이 나타나는지를 요약한다. Hidden representation에 정보가 없다는 증거나 time/group 모듈의 인과적 역할 식별은 아니다. Oracle 평균에 대한 예측 MSE도 함께 기록한다. 별도 유의성 검정을 늘리거나 이 값으로 학습·LR을 재선택하지 않는다.

## 해석의 경계

정렬로 성능이 회수되면, 그 개선에 내부 PEFT가 필수라는 주장은 약해진다. 반대로 개선이 없어도 표현 문제가 없다는 결론은 나오지 않는다. 후보 변환이나 원본 모델의 해당 입력 경로가 적절하지 않을 수 있다. 어느 결과도 새 방법의 신규성·실데이터 효용을 확정하지 않는다. 단순 회귀/기존 지연 정렬이 해결하는 합성에서 새 adapter를 억지로 설계하지 않는다.
