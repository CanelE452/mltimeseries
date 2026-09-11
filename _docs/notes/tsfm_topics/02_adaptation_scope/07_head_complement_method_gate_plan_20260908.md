# 출력부 보완 LoRA 후보: 방법론 진입 검사

작성: 2026-09-08. 사용자 요청: 기존 분석에서 방법론 가설 하나를 골라 선행연구, 최소 구현, 구분 가능한 실험을 진행한다. 본 문서는 아래 CPU 실험 실행 전에 저장한다. 문헌 검색에서 이미 확인한 중복 위험을 숨긴 사전등록은 아니며, 예측 성능을 확증하는 실험도 아니다.

## 가설과 최소 개입

[추정] 동결 특징의 선형 출력부가 설명할 수 있는 변화까지 내부 LoRA가 중복 학습하면 불필요한 적응이 생긴다. 내부 예측 변화에서 그 성분을 제거하면 일반화가 좋아질 수 있다.

Fit 특징에 bias를 추가한 행렬을 Z, 미적응 예측을 f0, LoRA 예측 변화를 dθ, 선형 출력부를 ZW라고 둔다. Hard projection 후보는 Cθ=Z†dθ, f=f0+ZW+dθ−ZCθ다. 새로운 표본에서는 **fit에서 구한 Cθ를 유지**하고 z_new Cθ를 뺀다. 평가 배치에서 C를 재추정하지 않는다.

## 실행 전 발견한 선행연구

- [SDDR](https://arxiv.org/abs/2002.05777): 구조화된 회귀와 DNN의 중복을 orthogonalization cell로 제거한다.
- [PHO, ICML 2023](https://proceedings.mlr.press/v202/rugamer23a.html): 같은 형태의 online orthogonalization, 함수 클래스, 작은 배치 붕괴와 예측 배치 의존성을 다룬다. Post-hoc 재분해는 예측을 보존한다.
- [Train Like a (Var)Pro](https://arxiv.org/abs/2007.13171): 선형 출력부를 부분 최적화해 제거하는 DNN 학습 선행이다.

이 때문에 현재 후보를 새 학습 원리라고 부르지 않는다. 단순히 TSFM에 적용했다는 차이로 대규모 학습에 들어가지 않는다.

## 고정한 검사와 중단 조건

1. CPU float64 최소 구현: fit support의 SVD로 투영한다. N×N 투영행렬과 명시적 Gram 역행렬을 만들지 않는다.
2. 합성 데이터에서 자유로운 joint linear head+nonlinear residual의 재매개화 및 PHO 예측 보존을 train/new-input에서 검사한다. 함수 클래스 동일성이 finite-step optimizer의 동일성이라는 주장은 하지 않는다.
3. MSE에서 최적 초기 linear head를 동결한 projected residual과, 매번 linear head를 정확히 refit하는 profiled baseline의 예측·손실·gradient를 비교한다. 이 등가성을 native pinball·regularized/early-stopped/nonlinear head까지 확대하지 않는다.
4. ETTm2의 기존 **train origin만** 사용해 실제 frozen Chronos 특징을 확인한다. 처음/중간/마지막 연속 4-origin 배치와 전체 fit support의 rank를 SVD로 측정한다. Bias 포함 769열, 4-origin 배치는 4×7×6=168행이다. 임의 출력 변화와 그 VJP cotangent의 잔존 norm은 구현·기하 진단이며 실제 LoRA gradient/예측 성능 측정이 아니다.
5. 검사 2~3의 수치 오차 허용치는 1e-9, hard projection 기하 잔차는 relative norm 1e-8이다. 작은 배치가 full row rank면 어떤 출력 변화도 그 배치의 hard complement에서 소멸한다.
6. 해당 연산·실패 모드가 직접 선행과 겹치고 별도 기여가 남지 않으면 **이 후보의 새 방법 주장을 중단**한다. 수학적으로 남은 차별점이 있을 때만 후속 GPU 학습으로 넘어간다. 논문·수치 검사는 긍정 결과를 만들기 위한 rescue가 아니다.

## 이후 학습으로 넘어가는 경우의 필수 대조

Head-only, ordinary LoRA, head-first→LoRA, 동일 joint head+LoRA의 projection on/off, update-norm을 맞춘 LoRA, same-fit profiled head, 동일 checkpoint의 PHO를 포함해야 한다. 추가 fit forward/backward와 projector 갱신 비용도 총비용에 포함한다. 겹침 감소만으로 성능 기여를 입증하지 않는다.

## 자원·범위 계약

이번 진입 검사는 GPU 모델 로드/학습/새 checkpoint 생성 없이 CPU 2 threads로 실행한다. 기존 guard의 RAM/commit/Git 감시와 Windows Job Object를 재사용하며 timeout은 180초다. 기존 guard의 study lock을 공유해 S1 GPU 실행과 동시에 시작하지 않는다. 기존 S1의 modeling.py/train.py/data.py hash와 예측을 보존한다. 검사는 ETT fit 특징만 사용하고 새 validation/evaluation 점수를 계산하지 않는다.

코드는 `experiments/peft_head_complement_v1/`, 실행 로그는 ignored `runs/peft_head_complement_v1/`, 작은 결과는 `results/peft_head_complement_v1/`에 둔다. 결과를 확인한 뒤 별도 판단 문서와 기존 history에 기록한다.
