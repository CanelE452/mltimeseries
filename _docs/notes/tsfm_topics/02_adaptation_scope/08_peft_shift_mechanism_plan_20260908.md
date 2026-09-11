# PEFT 실패 조건의 통제 screen

2026-09-08. 사용자의 “그렇게해줘”에 따라, 기존 PEFT가 반복적으로 실패하는 변화 조건을 확인하고 방법론 주제의 근거로 사용하는 작업을 진행한다. 기존 A S1와 중단한 출력부 투영 후보를 보존한다. 이 문서는 새 데이터 성능을 보기 전에 저장한다.

## 목적과 선택

목적 위계는 ML 방법론 논문 → 기존 방법의 구체적인 실패 원인과 해결 규칙 → 이번 통제 screen이다. 이번 결과의 소비처는 다음 방법 개발 여부를 판단하는 사용자다. 새 adapter를 이번 screen에 추가하지 않는다.

실데이터 확대는 적용성을 보여주지만 변화 유형을 식별하기 어렵다. 기존 A §8의 oracle을 아는 합성은 실패를 진단하기 좋지만 단순 회귀가 풀 수 있다. 이번에는 후자를 택하고 강한 회귀를 공개한다. Teacher의 특정 attention weight를 바꿔 데이터를 만드는 후보도 검토했으나 같은 모듈이 이기도록 답을 설계에 넣을 위험 때문에 제외한다. 흥미로운 추가 대조인 무정보 distractor는 조건 수를 늘리므로 이번 범위에서 제외한다.

백본은 이미 안전한 backward를 확인한 Chronos-2 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`다. Native time/group 구조와 공개 fine-tuning 경로를 사용한다. [Chronos-2](https://arxiv.org/abs/2510.15821). 출력부 먼저 학습하는 비교는 기존 LP→FT/LoRA의 순서 효과를 분리하기 위한 것이다. [NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/fcc22e5b7d5d2155d994da22d045f0a6-Abstract-Conference.html). 생성과정은 안정적인 autoregressive Gaussian 과정과 알려진 조건부 covariance를 이용한다. [VAR 문서](https://www.statsmodels.org/stable/vector_ar.html). 아래 수치·조합은 우리 설계이며 선행논문의 검증된 최적값이 아니다.

## 데이터·정보 계약

독립 U,V,ε ~ N(0,1)에 대해 Y[t]=0.5Y[t−s]+sqrt(.39){cosθ U[t−48]+sinθ V[t−48]}+0.6ε[t]. Burn-in 768 이후 L256 context와 H16 target을 가진 독립 episode를 생성한다. Y/U/V 과거만 입력하고 미래는 Y만 감독·평가한다.

```text
조건    s     θ        직접 바꾼 항
Q00     32    π/12     기준
Q10     64    π/12     자기지연
Q01     32    π/4      driver 결합 방향
Q11     64    π/4      두 항
```

분산은 각 조건에서 1, 전체 이력을 아는 oracle의 조건부 미래 분산은 .36이다. H≤min(s,48)이라 예측에 필요한 lag가 모두 관측돼 있다. θ 변화는 Y의 단변량 Gaussian 법칙을 유지한다. s 변화는 직접 자기지연을 바꾸지만 간접 cross-lag covariance도 달라진다. Time/group 모듈도 서로 영향을 주므로 결과를 독립적인 원인의 인과식별이라고 부르지 않는다.

실행 전 코드 감사에서 추가한 해석 계약: Chronos-2 group attention에는 채널 ID/위치 인코딩이 없고 Y/U/V가 projection을 공유한다. 따라서 Y를 유지한 U/V 순열에 Y 예측이 수학적으로 불변이며 현재 LoRA/head도 이 성질을 유지한다. θ=π/4 oracle은 대칭이지만 θ=π/12 oracle은 이름별 계수가 다르다. S0에서 swap 출력을 기록하고 RAW가 고정된 변수 순서를 이용한다는 함수 제약 차이를 명시한다. BF16 연산 순서의 작은 수치 차이와 수학적 비대칭성은 구분한다. **긴 과거의 Y/driver 관계에서 역할을 추론할 수 있으므로 이 대칭성만으로 실제 분포의 큰 오차 하한이나 실패 원인을 단정하지 않는다.** 역할 상호작용의 가능한 대안 설명으로 유지한다.

독립 train corpus 3개, 각각 64 episodes다. 별도 validation128/evaluation512 episodes는 모든 corpus에서 공유한다. 조건 간에는 동일 충격 stream을 사용하고 episode끼리는 독립이다. Corpus별 optimizer seed도 다르므로 세 반복은 **데이터·초기화가 함께 달라지는 반복**이며 둘의 분산을 분리하지 않는다. 새 evaluation은 이 생성 family의 개발 진단이지 실세계 일반화의 blind test가 아니다.

사전 QC는 split seed·ID 분리, CRN 재생, oracle lag 정렬, 정상 분산/ACF/잔차, target-free 입력을 확인한다. 작은 train64에서 경험 분산이 이론값과 정확히 같다고 요구하지 않는다. 긴 별도 QC 표본을 사용하며 그 label을 학습에 추가하지 않는다.

## 비교군과 학습 계약

```text
F0          미적응 native backbone
H_LIN       동결 forecast 특징의 선형 residual head
H_MLP       동결 특징의 768→533→336 residual head
OFF_LORA    기존 S1과 같은 time/group qkvo + output projection, rank8
JOINT       time/group qkvo rank4 + H_LIN, 처음부터 함께 학습
LP          동일 JOINT 구조, 처음80 updates head만, 이후120 updates 함께
TIME        time qkvo rank8 + H_LIN, 처음부터 함께 학습
GROUP       group qkvo rank8 + H_LIN, 처음부터 함께 학습
RAW         공통 후보 lag{32,48,64}의 Y/U/V 9features로 ridge
F0_RAW      같은 raw-lag 회귀로 F0 median의 residual 보정
ORACLE      알려진 생성계수와 σ를 쓰는 비학습 기준
```

TIME/GROUP/JOINT의 attention LoRA 예산은 동일하며 출력 head도 같다. OFF_LORA는 더 큰 공식 module-map 대조다. LP와 JOINT는 같은 구조·초기값·200개 sampler updates를 쓰며 phase별 optimizer를 분리해 head optimizer momentum을 유지한다. LP 단계 전환에서 LoRA optimizer만 시작하고 head 학습은 이어간다. Phase1의 validation best로 되돌리지 않고 update80 현재 상태에서 이어가므로 sampler 순서·update 예산을 숨기지 않는다. LP의 전체 best는 step0/40/80/120/160/200 중 선택한다.

최대200 updates, effective8 groups, micro4 groups, AdamW weight_decay0, grad norm clip1, native normalization 및 Y-only quantile objective를 모든 방법에 동일 적용한다. Validation은 step0 및40마다 같은 raw mean **2-pinball**로 선택한다. 모든 조건에서 Y 분산이1이므로 별도 target-std scaling은 하지 않는다. 미래 U/V row는 loss에서 제외한다.

Corpus0에서 H_LIN/H_MLP LR {3e−4,1e−3}; 내부 LoRA LR {3e−5,1e−4} 두 후보를 비교한다. JOINT/LP/TIME/GROUP의 head LR은1e−3 고정이다. 각 조건·방법의 LR을 validation으로 선택한 뒤 corpus1/2에서 고정한다. 7학습방법×4조건×(LR2+추가반복2)=112 fit이며 그중 내부backward80 fit다. F0/cache는12개 corpus×조건 자료에 공유된다. 새로운 LR rescue는 이번 screen에서 하지 않고 경계·step200 선택을 한계로 공개한다.

RAW는 모든 lead에서 context[t+h−lag]로 feature를 정렬한다. 같은 nH future Y label만 학습한다. 4fold episode OOF residual로 21개 quantile offset을 추정하고 ridge alpha{.001,.1,10}를 validation으로 고른다. 참 σ는 oracle에만 제공한다. F0_RAW는 F0 median에 같은 residual estimator를 더한다. RAW는 정답을 포함한 lag dictionary를 받은 강한 구조적 경쟁자라는 이점을 명시한다.

## 판정·불확실성·중단

모든 corpus·조건·방법에서 같은 evaluation episode ID를 paired bootstrap4000회 재표집한다. CI는 훈련된 세 모델과 생성 family에 조건부이며 corpus별 효과도 공개한다. H_LIN/H_MLP 중 corpus0 validation이 낮은 head를 H로 고정한다.

주대비 Δ=(S_H−S_OFF)/S_F0를 네 조건에서 계산한다. 이번 screen의 운영 문턱 δ=.01을 사용한다. 일반적인 학술 최소효과라고 주장하지 않는다. 네 조건 가족에 대해 Bonferroni98.75% CI를 보고한다. 세 corpus에서 방향이 일치하고 CI 상한<−δ이면 반복적 실용 악화 신호다. 상한<+δ인 추가 이득 부재와, CI가 넓어 결정할 수 없는 경우를 구별한다. 같은 방식으로 RAW 대비 격차를 별도 가족으로 공개한다.

위치 대조 D=S_TIME−S_GROUP에서 평균 조건 효과 Iθ=[D01−D00+D11−D10]/2, Is=[D10−D00+D11−D01]/2를 미리 고정한다. 두 효과 가족의97.5% CI를 사용하며 기준점 단순효과와 비가산성 J=D11−D10−D01+D00는 탐색으로 표시한다. 상대 효과가 달라도 한 위치가 모든 조건에서 우세하면 변화별 routing 필요성을 주장하지 않는다.

LP−JOINT는 절차 대조이며 좋아지더라도 기존 학습 순서 효과로 해석한다. RAW가 oracle에 가까우면 학습 가능한 실패 조건이라는 의미는 있지만 내부 PEFT 필요성 주장은 약해진다. 차이가 없거나 단순 절차로 해결되면 그 조건에서 새 adapter 개발을 중단한다. 차이가 남아도 합성 결과만으로 새 방법·인과 기전·실세계 효용을 선언하지 않는다. Generator를 바꿔 양성 결과를 찾는 연쇄 실험은 하지 않는다.

## 안전·구현 검증

기존 S1 소스·guard를 수정하지 않는다. 새 코드는 `experiments/peft_shift_mechanism_v1/`, 큰 산출물은 ignored `runs/peft_shift_mechanism_v1/`, 작은 결과는 `results/peft_shift_mechanism_v1/`에 둔다. GPU trial마다 guard를 사용하고 직렬 실행한다. 기존 study lock을 공유한다. 모델 deepcopy·DataLoader worker·torch.compile은 사용하지 않는다.

CPU2threads/interop1, CUDA allocator cap .67, BF16 cache_enabled=False, dropout0. Cache와 미적응/적응 직접 평가 모두 **micro4 groups**를 사용한다. Synthetic64/128/512는4로 나누어져 마지막 크기 차이가 없다. S0에서 같은4group 경로의 zero-update identity, target/future/group isolation, trainable map·동결 보존·LP head 포함 checkpoint 복구를 검사한다.

현재 여유 RAM 약17GiB, Git0, GPU사용약1.1GiB다. Guard는 여유 RAM5GiB/commit6GiB, childRSS8GiB, Git32, GPU10500MiB/85°C를 경계로 사용한다. 감시 간격 때문에 모든 OS 장애를 예방한다고 보장하지 않는다. 시스템 설정과 무관한 Trading Python 프로세스는 변경하지 않는다. S0의 실제시간으로 본학습 예상시간을 계산한 뒤 알린다. 안전중단이나 검증오류는 원인 해결 전 전체 학습으로 진행하지 않는다.
