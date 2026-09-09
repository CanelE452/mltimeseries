# 월합 감독 실험: 평균은 맞추지만 시간별 패턴은 손상됐다

2026-09-08. **현재 고정 head + attention LoRA 실험은 종료한다. 새 PEFT 방법의 성공은 확보하지 못했다.** 다만 월합 검증으로 선택한 head가 시간별 패턴을 크게 손상시키고, 같은 월합으로 학습한 LoRA가 이를 회복하지 못하는 현상을 확인했다. 큰 악화는 LoRA 이전 head 단계에 이미 존재했다.

이 결과는 [자료 진입 계획](17_coarse_supervision_data_entry_plan_20260908.md)과 [학습 계획](17_coarse_supervision_learning_plan_20260908.md)의 필요성 검사다. 표준 LoRA의 적용이며 신규 방법 실험으로 세지 않는다. 판정은 `CLOSE_CURRENT_COARSE_SUPERVISION_SCREEN`이다.

## 어떤 정보를 사용했나

BDG2 raw electricity/metadata revision `9b97ccbe90096aff42ed4fd6493bf7ae692d7118`에서 Eagle/Lamb의 2016 완전 자료를 가진 건물을 사전 규칙으로 골랐다. 각 site의 donor 8개는 2016 시간별 calendar template를 제공하고, target 8개는 월합만 학습과 입력 구성에 제공했다. Target fine history의 표준편차·자기상관·profile을 입력에 쓰지 않았다.

Train은 2016년 2–12월의 176창, 월합 V는 2017년 1–3월의 48창, fine E는 4–6월의 48창이다. V 월합은 48개 모두 완전했고 E는 34,944시간이 모두 유효했다. E의 다음 origin은 이미 끝난 직전 월합을 입력에 쓸 수 있지만 현재 예측 월합은 알지 못한다. 7–12월 값은 이번 처리·튜닝·평가에서 제외했다.

[데이터 독립 감사](../../../runs/peft_coarse_supervision_v1/data/prepared_data_audit.json)는 train/V 월합·연간 scale·donor template를 오차 0으로 재현했다. Proxy 차이는 scale 대비 최대 1.35e-7이었다. 4월 E 입력까지 과거 월합을 재구성했고, 5–6월 E 입력은 template 일관성과 소스의 인과 경로만 확인했다. 이후 별도 평가 감사가 E 원값과 정답 대응을 확인했다.

Chronos-2 checkpoint revision `29ec3766d36d6f73f0696f85560a422f50e8498c`, context 512, 월 길이 672–744시간, FP32 weights/BF16 autocast를 사용했다. F0는 원소별 raw 역변환 후 native 21출력의 평균이라는 고정 점예측이다. 이를 분포 평균이나 월합 분위수라고 해석하지 않는다.

## 선택과 실제 결과

두 ridge는 lambda 0.001/0.1/10만 비교했다. Head는 실제 저장되는 FP32 계수로 V 점수를 계산했다. Head lambda는 0.001, 수준 보정 lambda는 10이 선택됐다. 단순 policy는 V 정규화 월평균 MSE가 가장 낮은 COARSE_LIFT로 고정했다.

```text
방법                 V 정규화 월평균 MSE   E Eagle MSE       E Lamb MSE
PROFILE                  0.320214        0.057475          0.226156
F0                       0.323650        0.057493          0.232606
COARSE_LIFT              0.216308        0.068778          0.269865
FROZEN_HEAD              0.263508        1.596804          0.515387
ATTN_LORA                0.260896        1.596974          0.514724
```

오차는 target의 2016 연평균으로 나눈 후 제곱한다. 각 target 안 월을 균등 평균하고, site 안 target을 균등 평균한다. V는 월평균 오차이고 E는 시간별 오차이므로 두 열의 절대 크기를 직접 비교하면 안 된다.

LoRA는 선택된 head를 고정한 상태에서 1,179,648계수만 200 update 학습했다. V로 step 40을 선택했다. Step 0 대비 V 개선은 약 0.99%였지만 단순 수준 보정의 V 점수에 도달하지 못했다. Step 0 GPU V 점수와 독립 FP32 head 점수의 차이는 약 7.65e-9였다.

사전 주효과는 `(COARSE_LIFT MSE − LoRA MSE) / F0 MSE`다. 아래는 **비율 단위**이며 상대 개선율 백분율로 잘못 읽으면 안 된다.

```text
Eagle    −26.5804    95% CI [−42.9027, −18.3221]
Lamb      −1.0527    95% CI [ −1.2825,  +0.1409]
Pooled    −6.1119    95% CI [−11.0626,  −4.4362]
```

두 site의 최소 +1% 효과 조건과 pooled CI 양수 조건을 통과하지 못했다. Lamb 단독 CI는 0을 포함하므로 유의한 열세라고 단정하지 않는다. 패턴 오차는 LoRA가 head보다 두 site 모두 조금 낮췄지만, 이미 손상된 head를 출발점으로 한 작은 개선이었다. E에서 PROFILE/F0가 단순 V policy보다 좋았어도 선택 결과를 바꾸지 않았다.

[전체 수치](../../../results/peft_coarse_supervision_v1/metrics.json)와 [오차 분해 그림](../../../results/peft_coarse_supervision_v1/figures/coarse_supervision.png)을 함께 본다.

## 무엇을 발견했고 무엇은 아직 추정인가

[확인] Eagle의 FROZEN_HEAD는 월평균 오차 항을 F0의 0.003994에서 0.003829로 조금 줄였다. 반면 월평균을 제거한 패턴 오차는 0.053499에서 1.592975로 약 29.78배 커졌다. 전체 MSE는 약 27.77배였다. Lamb에서도 head 패턴 오차는 F0의 약 3.88배였다. **월합만으로는 이 fine 손상을 검증 단계에서 벌점으로 줄 수 없었다.**

[확인] 월합 head의 12,304개 명목 계수는 독립적인 fine 감독 12,304개를 뜻하지 않는다. Patch16과 월 길이의 나머지 0/8 때문에 각 8슬롯 안의 design 열이 같고, ridge 계수도 정확히 같았다. 축약 design의 실제 수치 rank는 176개 train 창에서 24였다. [독립 ridge 감사](../../../results/peft_coarse_supervision_v1/ridge_independent_audit.json)가 다른 SVD/primal 공식으로 lambda·policy·계수를 재현했다.

[확인] [보정 성분 감사](../../../results/peft_coarse_supervision_v1/ridge_contrast_independent_audit.json)는 정답 없이 head−F0 예측 변화만 분해했다. Eagle에서는 이미 train에서 보정의 패턴 에너지가 수준 에너지의 약 519배, V에서 약 315배였다. E에서도 약 817배였다. 따라서 fine 변화를 크게 추가하는 현상 자체는 E에서 처음 생긴 것이 아니다.

이를 단순히 “앞 8시간과 뒤 8시간이 서로 반대인 진동”으로 설명할 수도 없다. 두 계수 벡터의 cosine은 +0.9748이었다. E의 완전한 patch에서 보정 패턴 에너지 중 두 8슬롯의 대비 성분은 Eagle 약 3.58%, Lamb 약 6.46%였고, 대부분은 patch 사이 공통 성분의 변화였다. 마지막 8시간만 남는 부분 patch의 성분을 따로 식별했다고 주장하지 않는다.

[추정] 관측하지 못하는 fine 방향으로 큰 보정이 생기는 식별 문제는 이 실패를 설명하는 유력한 경로다. 하지만 해당 방향만 제거한 통제 실험을 아직 하지 않았으므로 유일한 인과 원인으로 확정하지 않는다. 현재 결과는 나쁜 고정 head 출발점의 문제이며, head 없이 학습한 LoRA나 모든 coarse supervision·출력 적응을 반증하지 않는다.

## 자기평가와 다음 판단

이번 설계는 월합에 head를 충분히 맞추면 내부 LoRA의 추가 필요성을 공정하게 볼 수 있다고 가정했다. 결과는 **월합 적합이 좋은 fine 출발점을 보장하지 않는다**는 약점을 드러냈다. 따라서 현재 LoRA 실패를 새 내부 방법이 필요하다는 근거로 바꾸면 안 된다.

다음은 현재 adapter를 더 학습하는 것이 아니라, 월평균 예측은 유지하면서 fine 패턴 변경만 제한하는 단순 대조의 충분성을 확인하는 것이다. 예를 들어 `p0 + mean(p − p0)`는 p와 같은 월평균을 가지면서 p0의 월내 패턴을 보존한다. 이는 기존 선형 출력 보정의 범주이며 그 자체를 새 PEFT로 명명하지 않는다. 이번 보고서에서 새 arm으로 평가하거나 사전 선택을 바꾸지는 않았다.

이 단순 대조가 충분하면 현재 실패를 위한 새 adapter 탐색도 닫아야 한다. 이후 donor fine supervision으로 유용한 패턴을 학습하는 방법을 검토하더라도, 같은 정보를 사용하는 강한 head와 기존 temporal disaggregation 방법을 먼저 넘어야 한다. [Chow–Lin 계열 설명](https://journal.r-project.org/articles/RJ-2013-028/)과 월별 고객·별도 fine 표본을 다룬 [PGE 사례](https://www.aceee.org/files/proceedings/1998/data/papers/0625.PDF)는 기존 접근의 경계다. 현재는 해결할 현상을 확보했으며 신규성·해결 방법·독립 재현은 미확보다.

## 검증과 실행 범위

[확인] CPU 68검사/17.24초 통과. S0 3 update에서 초기 head/LoRA/cache 예측 차이 0, 유효 LoRA B gradient 72개, 동결 파라미터·고정 head 불변, 체크포인트 예측 재현을 확인했다. 본학습 200 update와 평가 5arm×48창을 완료했다. 전체 선택은 fine E를 열기 전에 고정됐으며 모든 예측을 저장한 뒤 평가했다.

[평가 독립 감사](../../../results/peft_coarse_supervision_v1/independent_evaluation_audit.json)는 원 CSV의 E 정답 34,944개와 저장값의 차이 0, 960개 cell 지표 차이 최대 4.44e-16, macro 차이 2.22e-16, CI 차이 8.88e-16을 확인했다. 원천은 두 Office site이고 seed는 하나다. Target cluster CI는 날짜 일반화·학습 seed·여러 주제 탐색의 불확실성을 포함하지 않는다. 사전학습 비중복, 실제 청구 공개 지연과 DST 계량 정책도 완전히 확인하지 못했다.

[자원 기록](../../../results/peft_coarse_supervision_v1/resources.json): 다운로드·자료 처리·캐시·ridge·S0·본학습·추론·분석·그림의 9 guard 모두 exit0, 합 164.030초였다. 캐시 완료 뒤 전체 runner는 126.624초, 본학습 guard는 78.828초였다. 관측 21표본에서 여유 RAM 최소 14.044GiB, commit 10.012GiB, child RSS 최대 1.732GiB, GPU 최대 2330MiB/55°C, Git 0개였다. 주기 표본으로 정확한 순간 최대치는 아니다. 19:31:10~20:14:35 KST의 조회 대상 Windows 오류 이벤트는 0건이며 과거 프리즈 원인이 해결됐다는 뜻은 아니다.

이 결과로 전체 연구 목표를 완료 처리하지 않는다. 원 16번과 17번의 계약·결과를 보존하고, 실패한 조건을 같은 E에서 튜닝으로 구제하지 않는다.
