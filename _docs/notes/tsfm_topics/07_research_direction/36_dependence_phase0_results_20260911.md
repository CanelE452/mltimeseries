# 36 실행 결과: 준비 단계에서 합성 대조의 한계 확인

2026-09-11. **현재 설계 STOP. CPU 준비 검증 수행, GPU 학습 0회.**

사용자가 1–4단계 진행을 승인하여 시작했다. [원래 계획](36_dependence_peft_paper_plan_20260911.md)의 ‘통제가 성립하지 않거나 원하는 문제를 검증하지 못하면 GPU에 진입하지 않는다’는 기준을 적용했다. 원래 계획 본문은 수정하지 않고 보존한다.

## 판단

제가 제안한 3상태 전/역 대조는 주변분포·PSD·최적 위험을 통제하지만, **시간 관계의 효과와 값의 부호 변환 효과를 분리하지 못한다.** 또한 이 데이터의 미래 예측에는 마지막 상태 외의 과거가 필요하지 않다. 따라서 이 설정에서 수십 회 학습을 해도 당초 의도한 시간 관계 PEFT의 필요성을 분명히 검증하기 어렵다.

이는 **실험 설계의 구성 타당도 문제**다. LoRA가 실패했다거나 사용자의 원래 순서 의존성 연구 주제가 틀렸다고 판단한 것이 아니다. PEFT 모델은 이번에 학습하지 않았다.

## 실제 확인한 것

관측값 `v=(-1,0,1)`에서 상태 순서를 뒤집는 행렬을 J라고 하면 다음이 성립한다.

```text
J P_forward J = P_reverse
J v = -v
```

즉 forward 경로의 모든 값에 -1을 곱한 과정은 reverse 과정과 분포가 같다. 독립 표본 두 개가 값별로 정확히 부호 반전된다는 뜻은 아니다. 생성 법칙이 그 변환으로 서로 대응한다는 뜻이다.

이 대조에서 어느 모델의 forward/reverse 성능이 다르면, 양수·음수 변환에 대한 모델의 반응 차이라는 경쟁 설명을 배제할 수 없다. 원래 계획에는 이 설명을 분리할 대조가 없었다.

더 직접적인 문제는 조건부 예측식이다.

```text
Forward: E[Y(t+1) | Y(t)=x] =  0.6 - 0.2x - 0.9x²
Reverse: E[Y(t+1) | Y(t)=x] = -0.6 - 0.2x + 0.9x²
```

현재 상태가 관측되는 1차 Markov 과정이므로, 전체 과거를 알고 있을 때도 이 조건부 평균이 최적 MSE 예측이다. 계수 3개의 이차식이 표현할 수 있다. 합성의 생성 파라미터를 알고 계산한 oracle의 표현 가능성과, 학습 데이터에서 계수를 추정한 실제 성능은 구분한다.

독립 수치 검산:

```text
J P J = P.T 오차                       0
관측 부호 반전 대응 오차                0
전/역 ACF 차이 (lag 1–128)             5.55e-17 이하
전/역 PSD 차이 (257개 주파수)          2.22e-16 이하
전/역 Bayes MSE 차이 (horizon 1–128)   2.22e-16 이하
두 과정의 1-step Bayes MSE             각각 0.46
이차식과 조건부 평균 오차               1.11e-16 이하
```

PSD/위험의 통제는 실제로 성립한다. 그러나 그것만으로 연구 질문에 적절한 데이터가 되는 것은 아니다. 이 점을 처음 설계할 때 더 먼저 점검했어야 했다.

![모집단 스펙트럼·조건부 평균·설계 판정](../../../../results/peft_dependence_screen_v1/independent_gate_audit/01_population_audit.png)

그림은 실제 수학 계산 결과다. 가운데 선은 관측 상태 세 개의 조건부 평균을 연결한 것이며, TSFM 학습 점수가 아니다.

추가로 조건·분할별 RNG stream을 독립으로 분리한 data gate를 실행했다. 처음 계산은 조건별로 같은 seed의 RNG를 공유해 `results/peft_dependence_screen_v1/data_gate_attempt01_shared_rng/`에 보존했고, 최종 계산은 `SeedSequence([seed, condition_index]).spawn(train,val,devtest)`로 재실행했다. 이는 성과 선택이 아니라 원래 계획의 "독립 경로" 조건에 맞춘 수정이다.

최종 data gate에서 전/역 두 seed의 train-only simple control은 아래 수준이었다.

```text
full train transition excess risk mean  0.0032% over oracle
full train transition excess risk max   0.0052% over oracle
origin-only transition excess mean      0.2368% over oracle
quadratic latest-value excess mean      0.2368% over oracle
```

full transition estimator는 seed·조건마다 train 64경로의 모든 인접 전이 `64 * 511 = 32704`개를 사용했다. origin-only transition과 quadratic readout은 계획된 train origin `64 * 16 = 1024`개만 사용했다. 이 비율들은 oracle risk를 분모로 한 excess risk라서, 뒤 단계의 2% 모델 개선 기준과 같은 분모가 아니다. 그래도 결론은 같다. 최신 상태 기반의 작은 train-only 대조가 oracle에 거의 붙기 때문에, 이 DGP로 내부 PEFT 필요성을 주장할 여지가 없다.

계획과 같은 분모로 다시 계산한 개선 여지는 아래와 같다. 예측기는 1,024개 train 예제로만 추정했고, 학습 후 모집단 위험을 적분할 때만 알려진 생성 법칙을 사용했다.

```text
조건 / seed        이차식 모집단 MSE    oracle까지 가능한 상대 개선 상한
Forward / 36000    0.46161050           0.34889%
Forward / 36001    0.46022192           0.04822%
Reverse / 36000    0.46080734           0.17520%
Reverse / 36001    0.46171693           0.37186%
Oracle             0.46000000
```

상한은 `100 × (1 - oracle_risk / fitted_control_risk)`이다. 이차식 대조 이후의 실제 모집단 개선 여지가 모두 2% 미만이다. 유한 D 표본에서는 우연히 oracle보다 낮은 MSE도 가능하므로, 모집단 상한과 표본 점수는 구분한다. 실제 LoRA의 성능을 측정한 결과는 아니다.

실데이터는 이번에 새로 수집·검증하지 않았다. 과거 covariate-trust-pilot의 강한 interval lag-1 상관 지원 부족은 역사적 주의사항이며, 새 방향의 실데이터 실패 판정이나 이번 중단 근거로 사용하지 않았다.

![data gate: simple controls](../../../../results/peft_dependence_screen_v1/data_gate/02_data_gate.png)

## 요청한 단계별 상태

```text
1. 문제/대안 원인 확인       CPU 검증 수행, 현재 합성 대조의 문제 발견
2. 기존 방법 진단 패키지     GPU 72fit 미진입: 준비 단계 통과 실패
3. 관계 adapter 개발        조건 미충족으로 미실행
4. 외부 데이터·백본 확인    조건 미충족으로 미실행
```

예산 72회는 반드시 채워야 하는 실행량이 아닌 상한이었다. 관련 없는 차이를 학습한 뒤 이를 새 방법의 근거로 해석하는 일을 피하기 위해 현재 설계는 종료한다. 수치 gate를 낮추거나 다른 DGP를 즉석에서 반복 탐색하지 않는다.

## 연구를 이어가려면 먼저 충족할 조건

1. **추가 과거의 가치:** 최근 상태가 같아도 더 이전의 관측 관계에 따라 미래의 조건부 예측이 달라지는 문제여야 한다. 현재 toy는 이 조건을 충족하지 않는다.
2. **다른 설명의 통제:** 단순한 부호·크기·상태 이름 변환의 차이로 목표 현상을 설명할 수 있는지 먼저 확인한다.
3. **실데이터 연결:** 실제 train 자료에서 그 관계를 안정적으로 측정할 수 있어야 한다. 낮은 lag-1 상관을 독립성 증거로 쓰지 않는다.
4. **단순 대조 이후의 여지:** raw history를 사용하는 작은 predictor가 충분하면 그 이상으로 FM을 적응시킬 실용적 이유가 필요하다. ‘과거가 중요하다’와 ‘내부 PEFT가 필요하다’는 별개의 주장이다.

이 조건은 다음 설계의 검토 기준이며 새로운 실행 승인을 요청하거나 새 학습을 자동 시작하는 문서가 아니다. 현재 PEFT 분야의 가능성을 부정하는 결론도 아니다.

## 근거와 재현

- [실행 코드](../../../../experiments/peft_dependence_screen_v1/independent_gate_audit.py)
- [모집단 계산 JSON](../../../../results/peft_dependence_screen_v1/independent_gate_audit/audit.json)
- [data gate 코드](../../../../experiments/peft_dependence_screen_v1/data_gate.py)
- [data gate 요약](../../../../results/peft_dependence_screen_v1/data_gate/summary.json)
- [data gate 보고서](../../../../results/peft_dependence_screen_v1/data_gate/data_gate_report.md)
- [실행 범위](../../../../experiments/peft_dependence_screen_v1/PURPOSE.md)
- [예측기별 원시 점수](../../../../results/peft_dependence_screen_v1/data_gate/predictor_metrics.csv)
- [최종 독립 검산](../../../../results/peft_dependence_screen_v1/verification.json)

`verify_summary.py` 실행 exit 0: 2,688개 경로 중복 없음, 108행 예측기 CSV, 핵심 4조합의 MSE/모집단 위험을 상태별 표본평균으로 독립 재계산, 원계획 해시와 이전59파일 보존 확인. 두 PNG를 실제 시각 확인했다. 계산 시간과 설계·코드 작성·검토를 포함한 전체 작업 시간은 구별한다.

`independent_gate_audit.py` 실행 exit 0, 명령 wall time 약 1.02초. `data_gate.py` 최종 실행 exit 0, 실제 계산 시간 0.165초. 기본 Codex Python에는 matplotlib가 없어 첫 data gate 실행은 import 단계에서 멈췄고, 기존 `.venv-peft`에서 재실행했다. 기존 Study35 source/input 59개 hash 불변 확인. 기존 실험 결과·OS 설정·다른 프로젝트 프로세스 변경 없음. commit/push 없음.
