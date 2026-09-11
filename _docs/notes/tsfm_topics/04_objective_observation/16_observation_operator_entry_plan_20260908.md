# 이종 관측 연산 후보의 CPU 진입 감사

2026-09-08. [15번 손실 변경 종료](15_peft_objective_alignment_results_20260908.md) 뒤의 다음 후보 진입 검사다. 사용자가 승인한 후보별 실험·자기 평가·다음 실행의 지속 탐색에 따른다. 최상위 목표는 필요한 새 FM PEFT 방법을 찾는 것이며, 이번 단계의 목적은 기존 선형 Gaussian 조건부 추론으로 해결되는 현상을 새 방법의 근거로 오인하지 않는 것이다. 학습 성능이나 신규성을 주장하는 실험이 아니다.

## 선택과 범위

원02는 한 history 전체에 같은 간격·연산을 적용했다. 관측마다 support가 다른 경우는 직접 시험하지 않았지만, 관측 연산을 입력받는 원리 자체에는 [MFSS §3](https://www.jstatsoft.org/article/download/v104i10/4404)와 [선형 관측 GP §3.3](https://jmlr.org/papers/volume14/hennig13a/hennig13a.pdf)의 선행이 있다. [FlowState](https://arxiv.org/html/2508.05287v3)의 sampling-rate 대응과 실제 측정 연산 likelihood를 구분한다.

두 대안은 (a) 즉시 metadata token/LoRA를 학습하는 것과 (b) 정확한 조건부 분포와 단순 복원부터 검사하는 것이다. (b)를 선택한다. (a)는 물리량·단위·구간의 metadata 계약과 강한 단순 대조가 부족한 상태에서 새 성능을 해석하기 어렵다. 역문제의 nullspace를 이용하는 관점은 집계로 지워진 정보와 단지 연산을 몰라서 생긴 오류를 분리한다. 관측 support가 겹칠 때 공유 sensor noise를 상태로 다루는 것이 핵심 구현 대조다. 이를 새 기법이라고 부르지 않는다.

잠재 과정은 정상 AR(1), `a=.8`, innovation variance `.36`, stationary variance1로 고정한다. 역사 길이24, 미래4, 원점23이다. 비자명하지만 직접 Gaussian 계산이 가능한 가장 작은 조건을 고른다. 공유 base sensor noise variance는 `.04`, 단위 정규화 후 개별 보고서 noise variance는 `.0025`다. 복수 관측이 같은 base sensor 값을 집계하면 noise covariance도 공유된다.

각 관측은 알려진 이산 support의 END_BASE/MEAN/SUM이다. SUM은 sample sum이며 물리적 연속 적분으로 해석하지 않는다. 아래13개 중 원점23까지 도착한12개만 사용한다. 원점 이전 support라도 늦게 도착하면 제외하며 미래 support 관측은 허용하지 않는다. 상세 일정과 noise variance를 첫 실행 전 소스 계약에 저장하고 결과에 그대로 출력한다.

```text
operator    inclusive support    arrival
END_BASE          0                  0
MEAN            0..2                 2
SUM             1..4                 5
END_BASE          4                  4
MEAN            3..6                 6
SUM             6..7                 9
END_BASE         10                 10
MEAN            8..12               12
SUM            11..15               16
END_BASE         17                 18
MEAN           16..20               20
SUM            19..23               23
MEAN           18..22               25    excluded at origin23
```

## 독립적인 두 정확 경로

직접 Gaussian 경로는 `K[i,j]=a**abs(i-j)`, 관측 행렬H, `R=.04*H@H.T + diag(report_variances)`를 사용해 과거/미래 조건부 평균·공분산을 계산한다. Noise의 단위도 mean/sum 변환과 함께 바꾼다.

두 번째 경로는24차원 잠재 과거와24차원 공유 base noise를 함께 둔48차원 상태에서 보고서를 하나씩 Kalman update하는 유한 창 Gaussian 필터다. 개별 보고서 noise만 update의 독립 R로 둔다. 마지막 잠재 상태의 posterior를 AR(1)로4step 예측한다. 이는 batch conditioning과 같은 가정을 사용하는 기존 정확 계산이며 새로운 효율적 online filter라고 주장하지 않는다.

고정 seed2026090816의 한 관측 벡터와 전체 선형 gain을 비교한다. Gain과 covariance가 일치하면 특정 관측 표본에서만 우연히 맞는 경우를 피할 수 있으므로 성능 순위를 위한 큰 Monte Carlo sweep은 하지 않는다. Hyperparameter 추정이나 작은 MLP 학습은 이번 정확성 감사에 필요하지 않아 실행하지 않는다. 알려진 동역학의 Bayes 해가 되는 toy를 새로운 PEFT의 필요성으로 사용하지 않는다.

## 검사와 중단 규칙

- 두 경로의 전체 미래 gain·평균·공분산 최대 오차가1e-10 이내이고 posterior covariance가 허용 반올림 오차 내 PSD인지 확인한다.
- 동일 support에서 mean↔sum의 H/y/noise 변환 후 posterior가 같아야 한다. 단위를 잘못 준 대조의 열세를 새 방법의 장점으로 쓰지 않는다.
- 겹치는 관측의 공유 sensor-noise covariance를 일부러 대각으로 바꾼 대조가 정확 posterior와 달라지는지 확인한다. 차이는 정보·noise 모형을 잘못 지정한 영향이지 PEFT 필요성의 증거가 아니다.
- 동일 숫자 y가 서로 다른 알려진 H에서 다른 posterior를 만들 수 있음을 작은 반례로 보인다. 별도2차원 AR(1) 과거에서 도착1/y=1/독립 noise variance .04를 같게 두고 END_BASE(x0)와 MEAN(x0,x1)의 x2 posterior를 비교한다. 동시에 H를 알면 기존 Gaussian conditioning 또는 연산별 선형 head가 해결한다는 것을 함께 기록한다.
- 잡음 없는 두 점 평균에서 `ker(H) ⊆ ker(L)`을 SVD로 검사한다. 평균은 식별되지만 개별 base 값은 식별되지 않는 반례와 두 점 관측이 모두 있는 대조를 함께 둔다. 미래의 과정 잡음을 이 결정론적 복원 조건과 혼동하지 않는다.
- 늦게 도착한 관측값을 큰 수로 바꾸어도 원점23 posterior가 같아야 한다. 허용되지 않은 미래 정보를 쓴 모델의 우위를 보고하지 않는다.

어느 정확성 검사가 실패하면 원인을 고치기 전 새로운 과학적 효과를 해석하지 않는다. 모두 통과하면 `EXISTING_LINEAR_CONDITIONING_SUFFICIENT`로 남긴다. 이는 알려진 선형 Gaussian 조건에 대한 진입 veto이며, 이종 관측 문제 전체나 FM의 nonlinear prior 이득을 부정하는 판정이 아니다. 이 toy에서 좋은 결과를 만드는 추가 학습은0회다.

실제 자료 검토는 별도로 수행한다. Base-resolution 값이 구간 평균이면 END_BASE도 그 구간 평균이며 순간값이 아니다. 한 물리량에 대해 support·단위·유효 개수·관측 도착 시각을 검증하지 못하면 실제 이종 관측 배포 실험에 진입하지 않는다. 알려진 base 배열에서 집계를 구성할 수는 있으나 통제된 변형으로만 표시한다.

### 값 파일을 열기 전 고정한 USCRN 의미 검증

공식 문서에서 같은 기온의 서로 다른 support를 확인했으므로, 모델 평가와 별도로 한 station/year의 두 공개 파일만 점검한다. 공식 파일명 예시에서 확인한 `AZ_Tucson_11_W`, 완결된2024년을 선택한다. 이 선택은 예측 성능이나 자료 품질을 본 뒤 정한 것이 아니다. 전체 station archive는 받지 않는다.

- [hourly 파일](https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2024/CRNH0203-2024-AZ_Tucson_11_W.txt)
- [subhourly 파일](https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/2024/CRNS0101-05-2024-AZ_Tucson_11_W.txt)

[Hourly README](https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/README.txt)는 `T_CALC`를 시간 끝5분 평균, `T_HR_AVG`를 시간 전체 평균으로 정의한다. [Subhourly README](https://www.ncei.noaa.gov/pub/data/uscrn/products/subhourly01/README.txt)의 대응 값 이름은 `AIR_TEMPERATURE`이며5분 구간 끝 timestamp를 쓴다. 숫자의 exact equality는 문서만으로 확정하지 못해 실제로 점검한다.

파일당32MiB/총64MiB 상한과 시간 제한을 두고 다운로드 URL·응답 metadata·SHA를 보존한다. UTC timestamp를 join하고 duplicate·격자·missing sentinel을 기록한다. `hourly T_CALC(t)`와 `subhourly AIR_TEMPERATURE(t)`의 차이 및 완전한12개 valid5분 구간의 평균과 `T_HR_AVG(t)`의 차이를 계산한다. 원 파일이 소수1자리로 반올림됐으므로 시간 평균 차이는 반올림 범위와 분리해 기술하고, 유효5분 구간이 부족한 hour를 몰래 채우지 않는다. 값이 다르면 다른 station/year로 바꾸지 않고 정의·QC·제품 version의 원인을 확인한다.

T_MAX/MIN은 원10초 센서의 극값 처리가 관련되어 공개5분 평균의 max/min과 같다고 주장하지 않는다. 두 연도별 파일은 수정된 최신 archive일 수 있으며 과거의 실제 공개 시각은 확정하지 못한다. 이번 결과는 관측 support의 의미 검증이고 forecast 성능·pretraining 비중복·배포 지연 검증이 아니다. 기록은 `data_external/uscrn_operator_entry_v1/`에 따로 보존하며 새 학습 업데이트는0회다.

## 실행·보존

새 namespace는 `experiments/peft_observation_operator_entry_v1/`, `runs/peft_observation_operator_entry_v1/`, `results/peft_observation_operator_entry_v1/`다. 기존15 계약·소스·결과는 바꾸지 않는다. 새 소스와 계획을 hash로 동결한 후 한 CPU child를 기존 shared guard의 CPU 모드로 실행한다. CPUthreads2, timeout120초, 원 RAM/commit/Git 한도를 유지한다. 예상 계산은 수초 이내다. 실패·부분 산출물은 덮어쓰지 않고 원 시도를 보존한다. CPU 결과와 독립 검산, 실제 데이터 metadata의 확인/미확인을 기록한 뒤 다음 방법 후보의 진입 여부를 판단한다.
