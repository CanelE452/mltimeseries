# 출력부 보완 LoRA 후보의 선행연구·최소 구현 판정

2026-09-08. **판정: 현재 투영 연산을 핵심으로 하는 새 방법 주장은 중단한다.** 최소 구현과 진입 검사는 완료했다. GPU 학습·새 forecasting score는 생성하지 않았다. 기존 A S1의 결과와 판정은 유지한다.

## 무엇을 제안했고 왜 확인했나

사용자는 비교·분석에 머무르는 연구에서 ML 방법론으로 나아가기를 요청했다. 이에 “출력부가 설명할 수 있는 변화와 중복되는 LoRA 업데이트를 제거하면 일반화가 좋아질 수 있다”는 후보를 제안했다. 그 시점에는 신규성·효과가 확인되지 않은 가설이었다. 이번에는 이 제안을 선행연구와 수치 대조에 비추어 검증했다.

원래 [A S1](04_peft_adaptation_scope_s1_results_20260908.md)은 ETTm2/Jena의 출력 적응과 내부 적응을 비교한 pilot이다. ETT의 LoRA 추가 이득과 Jena의 추가 손해는 각각 block CI가 0을 포함했다. **이 결과는 ‘중복 업데이트 때문에 일반화가 나빠진다’는 원인을 입증하지 않는다.** 데이터·초기화·최적화·정규화로도 차이가 생길 수 있다.

## 가장 가까운 선행과의 대조

[확인: 원문] SDDR의 orthogonalization cell은 구조화된 선형 예측기가 설명하는 성분을 DNN에서 제거한다. 우리의 동결 특징을 그 선형 예측기의 입력으로 놓으면 핵심 연산이 겹친다. [SDDR, Lemma 2.2 / Eq.(1)](https://arxiv.org/pdf/2002.05777).

[확인: 원문] ICML 2023 PHO는 같은 학습 중 투영을 ONO라고 부르며, 함수 공간의 동일성·작은 배치의 소멸·예측 배치 의존성을 다룬다. 학습 후 출력 성분을 재분해하면서 전체 예측을 보존하는 PHO와 새 표본에 대한 확장도 제시한다. 따라서 “출력 공간에서 투영한다”, “fit에서 계수를 구해 새 표본에 적용한다”만으로 차별화할 수 없다. [PHO, §3–4 및 Appendix B.1](https://proceedings.mlr.press/v202/rugamer23a/rugamer23a.pdf).

[확인: 원문] 선형 출력부를 부분 최적화해 제거하는 접근도 DNN의 variable projection에 선행한다. *Train Like a (Var)Pro*는 비이차 목적함수까지 확장한다. 아래의 정확한 MSE 등가식을 pinball loss에 그대로 적용한다는 뜻은 아니다. [VarPro](https://arxiv.org/abs/2007.13171).

OPLoRA의 pretrained weight SVD 보존은 투영 대상이 다르지만, 이 차이가 위 직접 선행과의 중복을 없애지는 않는다. LP→LoRA는 출력부 초기 적응의 효과를 분리할 필수 대조다. [OPLoRA](https://ojs.aaai.org/index.php/AAAI/article/view/40703), [LP→FT/LoRA, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fcc22e5b7d5d2155d994da22d045f0a6-Abstract-Conference.html).

## 최소 구현에서 확인한 등가성

Fit 특징에 bias를 더한 행렬을 Z, 미적응 예측을 f0, 내부 적응의 예측 변화를 dθ라 둔다. P=ZZ†는 Z의 열공간에 대한 투영이다. 구현은 작은 SVD와 행렬 곱을 쓰며 N×N 행렬이나 Gram의 명시적 역행렬을 만들지 않는다. 합성 데이터에서 다음 관계를 독립적인 normal-equation baseline과 비교했다.

```text
후보                     f0 + ZW + (I−P)dθ
자유로운 head 재매개화    f0 + Z(W−Z†dθ) + dθ

MSE의 초기 최적 head     W0 = Z†(y−f0)
매번 refit하는 head      W*(θ) = Z†(y−f0−dθ)

따라서                   f0 + ZW0 + (I−P)dθ
                       = f0 + ZW*(θ) + dθ
```

새 입력에서도 fit에서 구한 계수를 유지하면 첫 등식이 성립한다. 합성 검사의 실제 최대 절대 오차는 다음과 같다.

```text
검사                                        최대 오차
자유로운 head 재매개화: fit / 새 입력        5.33e−15 / 3.55e−15
PHO의 예측 보존: fit / 새 입력              3.77e−15 / 3.55e−15
MSE profiled head와 예측 / 손실             3.55e−15 / 0
MSE profiled head와 θ gradient              1.67e−16
MSE 최적 head에서 cotangent 재투영 차이     6.07e−18
```

[확인] 8개 검사가 실행 전 허용치 1e−9를 통과했다. 이는 유한 표본에서 구현을 검산한 것이며 수학적 일반성은 위 등식의 가정에 의존한다. **함수 공간이 같다고 Adam의 유한 step 학습 경로나 정규화 효과까지 같지는 않다.** MSE의 최적 선형 head 등식도 early stopping, ridge penalty, H_MLP/H_FULL, native pinball에 자동으로 성립하지 않는다.

반대로, 중복 성분이 작아졌다는 측정만으로 성능 개선 원인을 주장할 수 없다. PHO처럼 예측을 그대로 유지하면서 분해만 바꿔도 그 수치는 달라질 수 있기 때문이다.

## 실제 Chronos 특징의 소멸 검사

기존 ETTm2 cache에서 train origin 379개만 선택했다. Validation/evaluation 특징과 target 값은 사용하지 않았다. Cache는 BF16 경로에서 생성해 float32로 저장된 특징이며, 이번 선형대수는 float64로 계산했다. 더 높은 원래 특징 정밀도를 복원한 것은 아니다.

선형 residual head는 768차원 특징과 bias의 총 769열을 가진다. 4-origin 배치는 7채널·6개 forecast patch이므로 168행이다. 처음/중간/마지막의 세 배치를 고정해 확인했다.

```text
support             Z 크기          rank    임의 출력 변화의 잔존 norm 비율
처음 4 origins      168 × 769        168     1.88e−15
중간 4 origins      168 × 769        168     2.08e−15
마지막 4 origins    168 × 769        168     1.95e−15
전체 fit            15918 × 769      769     0.974919
```

[확인] 세 작은 배치는 모두 full row rank다. 따라서 해당 배치에서 P=I가 되어 hard complement가 소멸한다. 임의 cotangent의 역전파 잔존 norm도 1.94e−15~2.03e−15였다. 이는 출력 변화에 대한 VJP 검사이며 실제 LoRA parameter gradient를 계산한 것은 아니다. 다만 정확히 이 작은 support의 hard projector를 적용하면 어떤 출력 변화에도 같은 소멸이 생긴다는 선형대수적 결론을 검산한다.

전체 fit의 rank는 769이고 complement 차원은 15,149이므로 같은 소멸은 생기지 않는다. 최대/최소 singular value는 935.217/0.326631, condition number는 약 2,863이다. 보고한 모든 rank는 singular-value 상대 cutoff 1e−12~1e−4에서 동일했다. 전체 support의 약 97.5% 잔존은 **임의 probe norm**이다. LoRA의 실제 업데이트가 97.5% 보존되거나 예측이 좋아진다는 측정이 아니다.

전체 fit support, ridge, truncated projection은 각각 소멸을 피할 수 있지만, 그 선택만으로 새 원리가 되지는 않는다. 특히 ridge는 hard orthogonal complement가 아니며 penalty·update norm·계산 예산을 맞춘 대조가 필요하다. 이 이유로 이 후보를 여러 변형으로 계속 확장하지 않았다.

## 실행과 재현성

[확인] 검증 실행은 2026-09-08 09:46:16~09:46:20 KST, guard 4.079초, CPU 계산 약 1.015초, exit0, safety stop 없음, stderr 0바이트로 완료했다. CPU thread 2개, interop 1개, GPU 모델 로드·TSFM forward·optimizer step·새 예측 점수 모두 0회다. 종료 시 여유 RAM은 16.86 GiB였다. 4초 실행은 정상 자원 로그의 10초 간격보다 짧아 상세 child RSS peak를 저장하지 않았으며, 이 수치를 전체 PC 안정성의 보장으로 해석하지 않는다.

첫 실행도 검사는 통과했지만 metric tensor의 scalar 변환에서 PyTorch 경고가 있었다. metric에만 detach를 명시하고 별도 경로에서 재검증했다. 첫 JSON/log는 보존했다. 최종 검증 JSON은 현재 audit/projection 코드 hash, 실행 전 plan hash, 입력·cache hash를 포함한다. S1의 modeling.py/train.py/data.py hash는 이전 계약과 일치한다.

- [실행 전 계획](07_head_complement_method_gate_plan_20260908.md)
- [최종 수치 JSON](../../../results/peft_head_complement_v1/entry_gate_verified.json)
- [최소 구현 및 재실행 설명](../../../experiments/peft_head_complement_v1/README.md)
- [guard 종료 상태](../../../runs/peft_head_complement_v1/entry_gate_verified/guard/status.json)

## 논문 주제로 나아갈 판단

현재 상태를 “새 PEFT 방법이 완성됐다”고 말할 수 없다. **A는 적응 위치를 비교한 pilot, 이번 후보는 방법론 진입 검사에서 중단**이다. 시계열 FM에서 이 연산의 예측 성능이 나쁘다고 증명한 것도 아니며, 별도의 효율적 최적화 기여가 원천적으로 불가능하다는 뜻도 아니다. 현재 제안에는 그 기여가 없다.

사용자가 원하는 방법론 논문으로 가려면 다음 문장을 채울 근거가 필요하다: “기존 PEFT가 **특정 변화 조건**에서 **어떤 예측 관계를 잘못 갱신**하며, 이를 **구체적인 제약/업데이트 규칙**으로 고치면 **강한 절차 대조**로 설명되지 않는 이득이 생긴다.” 지금은 가운데 실패 기전이 확인되지 않았다. A의 평균 성능 차이만으로 그 자리를 ‘중복 업데이트’라고 채웠던 비약을 이번에 제거했다.

후속 주제 선정에서는 이미 작성한 [A 실험 계획](04_peft_adaptation_scope_experiment_plan.md)의 통제 변화 검사를 작게 활용할 수 있다. 예를 들어 출력 scale/offset으로 충분한 변화와 시간·변수 관계를 바꾼 변화를 구분하되, 고정된 원자료 입력·동일 학습 예산에서 head-only, LoRA, head-first→LoRA가 반복적으로 실패하는 조건이 실제로 남는지 먼저 본다. 그 결과가 없으면 구조나 gate의 이름을 먼저 만들지 않는다. 기존 Time-PEFT/CoRA/TRACE와 비교해 변화 조건·목적함수·업데이트 원리 중 무엇이 다른지 명시할 수 있어야 새 방법 후보로 올린다. **이 후속 실험은 이번에는 실행하지 않았다.**

추가 확증에는 기존 ETT/Jena 평가를 반복해서 선택에 쓰지 않을 새로운 평가 구간/원천, cache와 직접 추론의 동일 배치 계약, 사전에 정한 실용적 개선 문턱이 필요하다. 이렇게 좁혀야 분석이 실패 원인을 특정하고 실제 방법을 검증하는 역할을 하며, 분석 자체를 논문 기여로 과장하지 않게 된다.
