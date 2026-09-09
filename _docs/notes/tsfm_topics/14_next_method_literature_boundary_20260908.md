# B 진단 중 확인한 다음 방법의 선행연구 경계

2026-09-08. 14번 미선택 후보의 추가 미래 예측을 읽기 전 검토다. [14번 계획](14_peft_selection_regret_plan_20260908.md)의 선택 규칙·판정 기준을 변경하지 않는다. 아래 후보는 채택한 방법이 아니며 실행 성과도 없다.

**18:30 KST 상태 갱신:** 아래 17시대 “15번 미실행”은 당시 기록이다. 이후 [15번 목적함수 진단](15_peft_objective_alignment_results_20260908.md)의6fit·6forecast를 완료하고 사전 실용 기준 미충족으로 닫았다. 문서 끝에 실제 결과 이후의 OA 진입 검토를 추가한다. 원14/15의 고정 계획과 판정은 변경하지 않는다.

[CoRA 공개 v1](https://arxiv.org/html/2510.12681v1)의 §3은 동결 backbone의 covariate embedding을 정적 학습 벡터의 softmax로 합치고, 영 초기화한 shift/scale 모듈을 원 예측 head에 주입한다. 따라서 동결 특성, 공변량 선택, 초기 예측 보존의 조합 자체는 신규성이 아니다. 본문 식(6)의 가중치는 입력마다 달라지는 함수가 아니라 학습 파라미터다. 이를 조건별 관계 변화의 한계로 볼 수는 있지만, 실제 실패와 강한 기존 조건부 선택 대조 없이는 새 연구의 근거가 아니다. 그 가중치를 인과적 개입 효과로 해석해서도 안 된다. 본 검토는 공개 v1 원문이며 학회 채택 확인이 아니다.

[AdapTS 공개 v2](https://arxiv.org/html/2502.12920v2)는 시간순 피드백으로 경량 forecaster를 갱신하고 foundation forecast와 가중 결합한다. 따라서 온라인 잔차 모델·예측 결합이라는 넓은 아이디어도 이미 존재한다. 원문의 실험은 점예측 지표 중심이고, 우리 고정 prefix PEFT의 quantile 예측과 정보 도착·적응 비용이 다르다. 같은 미래 label을 허용하지 않은 offline baseline과 비교해 온라인 방법의 우수성을 주장하면 안 된다. 온라인 진단을 하더라도 각 48시간 label의 실제 관측 완료 시점을 지켜야 한다.

자기 평가: 지금까지 실패를 찾는 분석이 새 학습 규칙을 만드는 단계보다 길었다. 그러나 기존 기법의 간단한 조합에 새 이름을 붙이는 것도 해결책이 아니다. B 판정 후에는 실패의 존재, 관측 가능한 원인, 기존 방법으로 해결되지 않는 이유를 각각 적고, 한 번의 구체적 반증 실험으로 좁힐 수 있는 후보만 다음 실행으로 넘긴다. 현재 문헌 검토만으로 horizon별 adapter나 covariate gate를 새 방법이라고 채택하지 않는다.

## 정규화로 사라지는 공변량 정보: 구조적 단서와 반증

17:01 KST 추가. [설치된 InstanceNorm](C:/Users/User/anaconda3/envs/mlts/Lib/site-packages/chronos/chronos_bolt.py:95)과 [Chronos-2 토큰화](C:/Users/User/anaconda3/envs/mlts/Lib/site-packages/chronos/chronos2/model.py:408), [encoder 입력](C:/Users/User/anaconda3/envs/mlts/Lib/site-packages/chronos/chronos2/model.py:593)을 독립 코드 검토했다. 채널별 loc/scale은 정규화와 최종 해당 채널 역변환에 사용되며 encoder에는 값·시간·마스크가 들어간다. Future covariate가 없고 target context와 mask/group/time가 동일한 조건에서, 양의 affine 변환을 가한 다른 공변량도 정규화 후 같은 입력이 된다. 이상적인 산술에서 `Phi(a*x+b)=Phi(x), a>0`이며, 양의 분산을 가정한다. 이때 표준 LoRA의 파라미터를 바꾸어도 지워진 구분을 되살릴 수 없다. 실제 부동소수점 동일성 GPU 검사는 아직 하지 않았다.

그러나 이 단서는 새 발견의 확정이 아니다. [Non-stationary Transformers §3.2](https://arxiv.org/html/2205.14415v4)는 이미 정규화 후 서로 다른 시계열이 구별되지 않는 문제와 원자료·평균·표준편차를 통한 attention 복원을 다룬다. [SAN](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html)도 정규화 통계의 변화를 별도로 모델링한다. 통계를 FiLM이나 attention에 넣는다는 설명만으로 충분한 신규성이 생기지 않는다.

예를 들어 공변량 `U=r+b`, `b=-1/+1`, 동일한 target 과거, 미래 `Y=b+noise`를 구성하면 정규화 입력의 충돌을 보일 수 있다. 동시에 raw covariate 평균을 쓰는 회귀가 이 예제를 해결한다. 따라서 이런 toy의 개선만으로 내부 adapter가 필요하다고 주장할 수 없다. 후속 채택 조건은 실제 자료에서 사라진 통계가 추가 예측 정보를 갖고, 단순 통계 기반 출력 보정으로 그 이점을 충분히 회수하지 못한다는 증거다. 아직 그 조건을 검증하지 않았으며 15번 실험이나 새 방법명을 확정하지 않았다.

17:11 KST 선행 공개 구현 추가 확인. [fusiontimeseries 작성자 저장소의 Operating-parameter conditioning 절](https://github.com/lukaskurz/fusiontimeseries#operating-parameter-conditioning-without-training)은 Chronos-2의 공변량 행별 양의 affine 불변성, 상수 채널의 값 소실, 같은 정규화 행을 만든 서로 다른 값에서 bit-identical forecast를 이미 보고한다. 같은 문서에 static operating parameter를 사용하는 BilinearLoRA 적응도 나온다. 이는 peer-reviewed 방법의 성능을 독립 재현했다는 뜻은 아니지만, 해당 현상과 통계 조건부 LoRA를 처음 제시한다는 주장에는 직접적인 선행 경계다. 이 확인을 바탕으로 정규화 충돌 자체를 다음 새 방법으로 채택하지 않는다. 별도 실측·통계 보정 진단의 가능성과 신규성은 구분한다.

[Towards Principled Test-Time Adaptation 공개 v1](https://arxiv.org/html/2605.17250v1)도 확인했다. 부분적으로 드러난 label과 전체 horizon이 관측된 label의 사용 시점을 구분하고, 주파수 영역의 경량 입력·출력 보정을 제안한다. 따라서 label 지연을 지키는 온라인 PEFT나 frequency residual이라는 넓은 전환도 그 자체로 새 방법이 아니다. 현재 실험은 offline 고정 후보 비교이며 이 온라인 절차를 실행한 것이 아니다.

## 정규화 학습 손실과 평가 목표

17:27 KST 추가. [기존 native loss](../../../experiments/peft_adaptation_scope_v1/modeling.py:49)는 context 표준화·asinh 공간의 pinball을 사용한다. [12/13 실제 numerical loop](../../../experiments/peft_external_gap_v1/train.py:438)는 이 손실로 backward하지만 선택·평가는 SORT한 raw quantile을 train-global 표준편차로 나눈 proper score다. 표적별 유효 관측 분모와 Q reduction도 구분해야 한다.

동일 reduction·정렬 조건의 한 관측에서 `z=asinh((prediction-loc)/s)`라 하면 raw-loss 출력 gradient는 native-loss gradient의 `s*cosh(z)/global_std`배다. 작은 공유 adapter의 제한된 함수 공간에서는 하나의 LR 상수로 이 방향 차이를 항상 제거할 수 없다. 반면 full context에 조건부인 자유로운 quantile 함수에서는 단조 변환 전후 같은 최적 quantile이므로 native loss가 improper하다고 부르는 것은 틀리다. 실제 원천의 PEFT 실패가 이 차이 때문이라는 인과는 아직 미확인이다.

[Darts 공식 Chronos2Model](https://github.com/unit8co/darts/blob/master/darts/models/forecasting/chronos2_model.py)은 이미 역정규화한 출력으로 fine-tuning loss를 계산하며 원 Chronos의 normalized loss와 다르다는 TODO를 명시한다. 따라서 raw pinball 적응 자체 또는 그 Jacobian 재표현만을 새 방법으로 채택하지 않는다. 목표 정렬 대조가 후속 실험에 필요하면 가장 단순한 기존 baseline으로 넣을 수 있지만, 이를 이번에 학습·검증한 것으로 기록하지 않는다.

## 다음 후보를 거르는 기준에 대한 수정

입력 불확실성 통합과 관측 연산도 검토했다. [Taylor–Buizza](https://users.ox.ac.uk/~mast0315/EnsemblesForLoad.pdf)는 날씨 ensemble을 수요 모델에 각각 통과시키는 접근, [Höhlein 외](https://arxiv.org/html/2309.04452v2)는 DeepSets/SetTransformer 기반 확률적 ensemble 후처리, [ensemble distribution distillation](https://arxiv.org/html/2002.11531v2)은 분포 증류를 이미 다룬다. 정확한 forecast vintage와 적은 target label, 엄격한 추론 비용 조건에서 frozen TSFM 적응이 이 대조를 넘어서는지는 별도 미검증 질문이다. K개를 한 배치로 호출한 것을 1회 계산량이라고 부르지 않으며, 혼합 CDF의 역산과 분위수 평균을 구분한다.

관측 연산의 경우 [FlowState](https://arxiv.org/html/2508.05287v3)의 연속 basis와 sampling-rate 변경, [MFSS](https://www.jstatsoft.org/article/download/v104i10/4404)의 point/sum/average 측정 방정식, [temporal hierarchies](https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf)의 sample reconciliation이 강한 선행이다. 이들은 서로 같은 문제를 모두 해결하는 것은 아니다. 특히 주변 quantile만으로 합계의 분포를 결정할 수 없으므로 단순 분위수 합을 정상 baseline으로 사용하면 안 된다.

자기 평가: 구성요소가 기존에 있다는 이유만으로 모든 구체적 방법 조합을 자동 기각하는 것도 과도하다. 다음 판단은 **정확한 정보·표본·계산 제약에서 가장 가까운 기존 방법이 어디까지 해결하는지**에 둔다. 새로운 문제와 동작 원리, 강한 대조 대비 재현된 이득이 함께 있어야 한다. 이 문헌 검토는 아직 새 방법 채택이나 실험 성과가 아니다.

## Marginal을 유지하는 joint forecast 적응의 직접 선행

17:31 KST 추가. 합계·ramp·peak의 확률 예측은 개별 시점의 quantile만으로 결정되지 않는다. 그러나 이 사실이나 copula를 추가하는 원리도 새롭지 않다. [Baron 외, 2025 공개 v1](https://arxiv.org/html/2510.02224v1)은 Chronos-Bolt/TimesFM/TiRex marginal에서 Gaussian copula 경로를 생성한다. Appendix B는 raw history를 받는 작은 MLP/TCN/GRU가 AR(1) 계열 상관 파라미터를 예측하도록 variogram loss로 학습한다. 따라서 동결 FM marginal과 경량 학습 dependence 모듈의 조합까지 직접 선행이 있다.

[Wen–Torkkola, 2019 공개 v1](https://arxiv.org/html/1907.10697v1)은 quantile forecaster를 먼저 학습·동결하고 horizon context를 사용하는 conditional copula를 추가 학습한다. [Salinas 외, 2019](https://arxiv.org/abs/1910.03002)의 조건부 저랭크 Gaussian copula도 강한 대조다. Chronos2 hidden에 작은 저랭크 head를 붙이는 것만으로 새로운 학습 원리가 생겼다고 주장하지 않는다.

남는 미검증 질문은 소량 target label 조건에서 동결 FM 표현이 같은 크기의 raw-history copula보다 의존성 학습에 실제로 유용한가다. 이 조건의 성능을 확인한 것이 아니며, 확인되더라도 그 뒤의 구체적 방법 기여와 외부 재현이 필요하다. Cheap falsification은 같은 marginal/보간/tail/학습origin에서 historical-rank, shrinkage-Gaussian, raw-history 조건부 head, frozen-hidden 조건부 head를 비교하는 것이다. Hidden을 섞거나 raw-history head로 바꿔도 이득이 유지되면 FM 표현의 필요성 가설을 기각한다. 첫 지표는 미리 정한 누적량 분포 CRPS 하나로 좁히고 결과를 보고 ramp/peak로 주지표를 바꾸지 않는다.

PSD이고 대각이1인 `R`에 대해 `Z~N(0,R)`, `U=Phi(Z)`, `Y_i=Q0_i(U_i)`를 쓰면 설정한 주변 분포를 유지한다. 다만 21개 knot를 보존한 것과 전체 CDF 보존을 구분하고 모든 방법의 tail 규칙을 같게 둬야 한다. 주변 분포가 고정되면 합의 평균은 copula를 바꿔도 변하지 않는다. Marginal calibration 오류를 dependence 모듈이 해결했다는 해석도 하지 않는다. 아직 이 후속 CPU/GPU 실험은 실행하지 않았다.

## 후속 후보의 과거 실행 누락 정정

17:34 KST 추가. 위 새 후보 검토에서 초기02/03 주제 노트를 먼저 읽고, 다른 작업의 완료 결과를 뒤늦게 발견했다. [09-07 history](../../history/2026-09-07.md)와 [UCP 실제 STATUS](../../../results/uncertain_covariate_path_pilot_v1/STATUS.md), [판정 JSON](../../../results/uncertain_covariate_path_pilot_v1/verdict.json)을 확인했다. UCP는 ECMWF/ENS GB2019train/2020V/2021test와 BMRA8농장, K50을 사용해 이미21fit을 완료했다. `NO_INCREMENTAL_PATH_VALUE`: P 대 D−1.194%/95%CI[−2.252,−0.270], P 대 S−2.046%/[−3.726,−0.555]였다. 같은 원천의 경로 encoder를 다시 시작할 이유가 없다. 원 forecast base time·member 경로는 확보되어 있으며 실제 공개 지연 미검증이라는 한계가 있다. “예보 archive를 아직 확보하지 않았다”는 일반화는 잘못이었다.

UCP의 미래 날씨는 외부 adapter가 처리했고 backbone은 모든 방법에서 같은 target-history cache였다. MC는 작은 adapter를 member마다 호출한다. 따라서 약50배 차이를 K회 backbone 대비 효율로 해석하지 않으며, native conditional FM의 다중 시나리오 추론을 1회로 압축한 실험과도 구분한다. 하지만 현재 원천에서 평균 예보 이상의 추가 성능 근거가 부족했다는 음성 결과를 피할 수는 없다. P와 broken-path의 CI가0을 포함한다는 사실만으로 모든 경로 정보가 없다고 증명한 것도 아니다.

OA 본 결과는 현재 작업트리가 아닌 commit `4c6c805`에 보존돼 있다. 독립 agent가 그 commit의 본 보고서와 `audit_verdict.json`을 읽어 `INCONCLUSIVE` / 현재 interval-integrated Fourier O 확대 중단 / 넓은 문제 `OPEN_NOT_DIRECTLY_TESTED`를 확인했다. FM PEFT나 heterogeneous observation likelihood가 시험된 것은 아니다. 이번에는 원 OA/UCP 소스·결과를 변경하지 않고 [현재 프로젝트 인덱스](../../PROJECT_LOG.md)를 정정했다.

자기 평가: 최근 PEFT history만 확인한 채 다른 주제의 실제 완료 기록을 충분히 교차 조회하지 않은 것이 이번 후속 검토의 오류였다. 신규 후보마다 전체 결과 namespace와 원 source/평가 노출까지 먼저 확인한다. 이 정정 이후에도 새15번 실험은 채택·실행하지 않았으며, 직접 선행이 있는 copula head나 이미 음성인 path encoder에 이름을 붙여 새 방법이라고 보고하지 않는다.

외부 원천의 가용성도 별도 확인했다. Data-engineer가 [NOAA GEFS 공개 archive](https://registry.opendata.aws/noaa-gefs/)의 [2024-01-01 00UTC member1 +6h index](https://noaa-gefs-pds.s3.amazonaws.com/gefs.20240101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006.idx)에서 cycle/member/lead와 TMP/UGRD/VGRD 등의 필드를 확인했다. [S3 객체 metadata](https://noaa-gefs-pds.s3.amazonaws.com/?list-type=2&max-keys=1&prefix=gefs.20240101/00/atmos/pgrb2sp25/gep01.t00z.pgrb2s.0p25.f006.idx)의 LastModified는2024-01-01T03:49:47Z다. 이는 실제 예보 이용 가능 시각의 완전한 보증이 아니라 저장 객체의 공개 시각 proxy다. [BPA 공식 wind/load/solar 페이지](https://transmission.bpa.gov/business/operations/wind/)가 target 후보지만 공개지연·revision·UTC 정렬·공간 매핑은 미검증이다. 두 자료는 조사한 프로젝트 기록에서 미사용이지만, 새 원천이라는 사실이 기존 음성 path 표현의 신규성이나 필요성을 되살리지는 않는다. Bulk download나 추가 학습을 시작하지 않았다.

## 15번 종료 후: 이종 관측 연산의 진입 검토

18:30 KST 추가. Raw-loss 추가 효과는 Bike−0.218%F0/97.5%CI[−0.680,+0.142]%, Household+0.122%/[−0.059,+0.255]%였다. 초기 gradient 방향과 clipping은 바뀌었으나 사전 실용 기준을 통과하지 못했다. 현재 objective-alignment screen을 닫고 같은 두 블록에서 LR·step·가중치를 바꾸는 후속 탐색은 하지 않는다. 수치·검증·한계는 위15번 결과에 보존했다.

독립 reviewer는 원OA commit `4c6c805`의 `operators.py`를 직접 읽었다. `observe(history,r,op)`와 `token_features(r,op)`는 한 history 전체에 같은 r/op를 적용하고 op_id를 상수로 채운다. 따라서 관측마다 support·연산이 달라지는 시퀀스는 직접 실험하지 않았다. 다만 원자료 자체가 구간 평균이므로 `END_BIN`도 마지막 구간 평균이다. 이를 물리적 순간값 실험이라고 재해석하지 않는다. 이전 Fourier O 확대 중단 판정은 유지한다.

CPU 진입 질문은 “연산 metadata가 없어서 구별할 수 없는 것”과 “집계 때문에 원래 사라진 정보”를 구분하는 것이다. 잡음 없는 선형 관측 `y=Hx`, 평가량 `Lx`의 정확한 식별 조건은 `ker(H) ⊆ ker(L)`이다. 두 점의 평균만 관측하면 `(1,−1)` 방향은 보이지 않아 각 점은 식별되지 않지만 그 평균은 식별된다. SVD로 nullspace와 작은 특이값에 따른 불안정성을 확인할 수 있다. 미래 예측에는 별도의 동역학 가정이 필요하고 과정 잡음의 불확실성을 adapter 실패로 취급하면 안 된다.

동일 구간·단위의 mean/sum은 길이 또는 유효 개수로 변환된다. 이 단순 변환으로 차이가 사라지면 새 표현을 만들 이유가 없다. [MFSS §3·Table1](https://www.jstatsoft.org/article/download/v104i10/4404)은 point/sum/average를 누적 상태와 측정 방정식에 넣고 Kalman으로 처리한다. [FlowState §3](https://arxiv.org/html/2508.05287v3)의 연속 basis·sampling-rate 적응도 강한 선행이다. 연산을 입력받는다는 원리나 metadata token 자체는 신규성 주장이 될 수 없다.

다음의 작은 반증 순서는 아직 실행하지 않은 작업이다.

1. 알려진 선형 Gaussian 동역학에서 operator-aware Kalman의 평균·공분산을 직접 Gaussian conditioning과 대조한다. 겹치는 support와 실제 도착 시각을 반영한다. Kalman이 Bayes 해를 달성하는 toy는 PEFT 필요성의 증거로 사용하지 않는다.
2. 같은 관측·학습 표본으로 동역학을 추정한 작은 상태공간 모델, 상태 추정치·불확실성·metadata를 받는 ridge/작은 MLP가 정보 손실을 회수하면 그 조건의 내부 적응 가설을 닫는다.
3. 실제 FM 비교에 진입하려면 올바른 관측 복원＋동결 FM＋metadata 출력 head가 해결하지 못하는 반복적 실패가 먼저 필요하다. 현재 CPU 검토가 이 대조를 이미 통과했다고 주장하지 않는다.

현재 자료에서는 같은 물리량의 각 관측에 대해 support·단위·유효 개수·공개 시각을 검증한 실제 이종 관측 계약이 부족하다. 서로 다른 변수의 point/mean/sum을 섞으면 변수 의미까지 함께 바뀐다. 따라서 다음 행동은 metadata 확인과 작은 식별가능성·상태공간 반증이며 GPU 학습에 바로 진입하지 않는다. 이 문서는 다음 PEFT 주제를 확정하거나 새로운 성능을 확보한 결과가 아니다.

## 16번 진입 검사 완료 후의 경계

19:03 KST 추가. 위 ‘아직 실행하지 않은’ 항목 중 알려진 Gaussian 정확성·단위 변환·정보 소실 반례는 [16번 결과](16_observation_operator_entry_results_20260908.md)로 완료했다. 새학습0회이며 직접/순차 posterior와 별도 precision 검산이 일치했다. 실제USCRN 마지막5분 평균은8,087쌍 전부 일치했지만 완전한8,076시간의 재집계에는0.125°C 예외 한 건이 남는다. 알려진 조건의 해법을 새PEFT 증거로 쓰지 않으며, 실제 archive 전체가 단순한 동일 선형 관측이라는 주장도 하지 않는다.

집계 표적만으로 미세 미래를 예측하는 후보를 추가 검토했다. [Sax–Steiner의 temporal disaggregation §2](https://journal.r-project.org/articles/RJ-2013-028/)는 집계한 고빈도 설명변수로 GLS를 적합하고 미세 초기 패턴에 집계 잔차를 분배한다. FM 패턴＋선형 집계 보정은 이 선행을 넘어서는 설명이 필요하다. [Rangapuram23](https://proceedings.mlr.press/v206/rangapuram23a/rangapuram23a.pdf)은 미세 수준을 포함한 시간 계층 표적으로 학습하므로 정확한 coarse-only 감독 선행이라고 쓰지 않는다. 제한 검색에서 정확한 frozen TSFM·PEFT·집계 감독 구성을 확인하지 못한 상태는 UNKNOWN이며 신규성 증명이 아니다.

현실적인 정보 제약의 근거는 [PGE의 운영 연구, ACEEE1998 pp.4–8](https://www.aceee.org/files/proceedings/1998/data/papers/0625.PDF)에서 확인했다. 월별 검침 고객과 별도의 시간별 계량 표본을 구분하고, 다음 날 시간별 포트폴리오 부하 예측(DOF−1)과 청구량 도착 후 정산(DOF+35)을 분리한다. 이미 profile 이전·날씨 회귀를 사용했다. 현재 모든 고객이 이런 제약을 갖는다는 주장이나 개인별 미세 예측의 정확성 보장은 아니다.

따라서 target fine history를 입력으로 사용하면서 학습 라벨만 집계로 가리는 설정을 실제 coarse-only 배포라고 부르지 않는다. 같은 target의 fine 과거를 모두 차단한 계량 패널 접근 제한 실험은 가능하지만 통제 실험임을 명시해야 한다. Donor와target 고객을 분리하고, target 월별 값으로만 학습·선택하며, 미세 미래 값은 평가 전용으로 둔다. 예측하는 달의 실제 월합이나 아직 관측되지 않은 donor 부하·실제 미래 날씨를 입력하면 누수다.

후속 실험이 성립하려면 고정 profile, 집계 profile/공변량 회귀, frozen feature 정규화 head, LoRA가 같은 정보와 집계 감독을 써야 한다. 고빈도 평가를 보며 head/LoRA를 고르지 않는다. 현재는 자료·API의 가능성을 읽기 전용으로 확인 중이고 해당 실험은 아직 고정하거나 실행하지 않았다. 이 검토는 새 방법의 채택이 아니다.

로컬 가능성 조회에서 `data/electricity/electricity.csv`는26,304시간×321채널, 표시날짜2016-07~2019-07로 확인됐다. 그러나 fetch script가가리키는Time-Series-Library미러와 [원UCI370고객/2011–2014/15분kW](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) 사이의날짜·고객ID·집계변환이확인되지않았다. HQ01에서이미평가했고 [Chronos2 AppendixA Table6](https://arxiv.org/html/2510.15821v1)에Electricity370여러해상도가사전학습원천으로명시돼있다. 정확한로컬파일의중복범위는미확인이지만새로운미노출고객자료로주장할수없다. Household는한가정이라donor/target고객분리자료로부적합하다.

고정Chronos2 config는patch16/context8192/최대출력1024로744시간출력자체는가능하다. 다만기존12번runner는H48·3patch에고정돼있고native pinball은fine label을요구한다. 시간별주변분위수의합을월합의분위수라고사용하면안된다. 긴18개월월별이력을무조건hourly입력으로펼치면context상한을넘으며월별행과hourly donor행을같은index로붙이는것도시간정렬이아니다. 실제학습경로는새자료계약과별도S0가필요하다. 현재가변horizon이가능하다는코드조회만으로긴horizonGPU안전성이나집계감독의과학적타당성을확인한것은아니다.

새전력패널의 metadata-only 후보로Ausgrid Solar Home을조회했다. [현재공식연구자료페이지](https://www.ausgrid.com.au/about-us/about-ausgrid/research-data-sets)에서원half-hour패널을확인하지못했고, 과거공식 `Industry/Our-Research/Data-to-share/Solar-home-electricity-data` 및 두 `Solar-household-data.aspx` 주소는404였다. 비공식단서의300고객/2010–2013/kWh/GC·CL·GG 정의·연도간고객연결·이용조건은현재공식원문으로검증하지못했다. 값다운로드·실험은하지않았으며현재자료계약에는채택하지않는다. 이접근실패는연구가설반증도해당데이터의존재부정도아니다. 다른계량패널또는확인가능한원공식배포자료를찾는후속이남는다.
