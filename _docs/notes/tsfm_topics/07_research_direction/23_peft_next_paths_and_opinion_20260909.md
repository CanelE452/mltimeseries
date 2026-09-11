# 23. 다음 경로 두 개와 내 의견 — 실행하지 않고 적은 계획

2026-09-09. [인벤토리 21](21_peft_method_pilot_local_inventory_20260909.md)과 [후보 스크린 22](22_peft_method_candidate_screen_20260909.md)의 후속이다. 사용자 요청: "문제 명세 다시 하는 것과 짧은 계열 자료 확보를 둘 다 적되, 실행하지 말고, 내 의견과 내가 보기에 좋은 방법을 함께 적어라." 이 문서는 그 답이다. **새 GPU 학습 0회, 새 실험 0회.** 아래 수치는 전부 저장된 결과를 읽은 것이고, 제안은 전부 미실행이다.

문서 안의 단정에는 근거 태그를 붙였다. `[확인]`은 파일·코드·저장 결과를 직접 읽은 것, `[추정]`은 그로부터 추론한 것, `[미검증]`은 실행 전에는 알 수 없는 것이다.

## 0. 이 문서가 답하려는 질문

지시문의 최상위 목적은 "새로운 ML 학습/추론 규칙 하나를 구체화하고 강한 대조와 비교하는 것"이다. 스크린 22의 결론은 네 방향 모두 `NO_METHOD_SPECIFICATION`이다. 그러면 남는 질문은 하나다.

> 20계열이 닫힌 뒤에도 새 방법이 나올 수 있는 자리가 이 저장소에 남아 있는가, 있다면 어디이고, 거기에 가는 가장 싼 길은 무엇인가.

이 문서는 그 자리를 **"20계열이 공유한 설정의 바깥"**에서 찾는다. 이유는 §1에 있다.

## 1. 지금 어디에 서 있나 — 배제된 것과 안 건드린 것을 나눠서

20계열이 실제로 배제한 것은 다음 조건 아래의 명제들이다.

```text
공통 설정 (20계열 전부)                                            [확인] 22 §1·§2b
  백본        amazon/chronos-2 한 revision (29ec3766…)
  단위        데이터셋 하나 = cell 하나, cell마다 별도 fit
  학습량      원천당 63~90 origins (Study17만 176창), 한 번도 변화시킨 적 없음
  자료        UCI hourly(Bike/Household) + ETT/Jena + BDG2 + 합성 Gaussian
  지표        train std 정규화 21분위 2-pinball (SORT), F0 대비 %
  적응 형태   head-only / rank8 attention LoRA / Full FT

이 설정에서 배제된 명제
  "LoRA보다 정확한 새 adapter 구조가 필요하다"         Study20: Full FT도 LoRA를 못 넘음
  "모듈 위치 선택으로 얻을 것이 있다"                    Study09: attention만으로 유지 (합성 한 조건, 원문 유보 있음)
  "LR/checkpoint 선택 규칙이 필요하다"                   Study14: 고정 LR 1e-5 단순 규칙 veto
  "적응을 거부해야 하는 대상이 있다"                     22 방향 A: F0 최선 cell 0/4, 오라클 상한 0.34%F0 < 1%
  "집계 감독에서 null-space 제어가 새 방법이다"          22 방향 B: Sax&Steiner Σ=I 와 동일
  "학습 없는 컨텍스트 정렬이 LoRA를 대체한다"            22 방향 D: 실데이터에 정렬할 지연 없음
  "raw-loss / 정규화 손실 정렬이 새 방법이다"            Study15
  "관측 연산·revision 조건에서 새 PEFT가 필요하다"       Study16 / 19 (CPU 선형 진단 수준)
```

그리고 20계열이 **한 번도 변화시키지 않은** 축이 넷 있다(22 §2b). 이 중 축 3(구간 보정)은 저장 결과로 대조해 보니 혼재라 열린 문제로 세지 않는다. 남는 셋이 이 문서의 재료다.

```text
축 1  백본        TiRex-2 / FlowState / chronos-2-synth 가 로컬 실물인데 미사용     [확인] 21 §4
축 2  학습량      "언제 적응이 손해로 뒤집히는가"를 한 번도 관측 안 함             [확인]
축 4  그룹 공유   여러 계열에 하나의 adapter 를 학습한 적 없음                     [확인]
```

이 셋이 왜 중요한가. **파운데이션 모델의 실제 배포 조건이 정확히 축 2와 4의 교집합이기 때문이다.** 계열이 수백~수만 개이고, 계열마다 이력이 짧아서 계열별 적응은 불가능하고, 적응 예산은 풀(pool) 전체가 공유한다. 이 저장소의 long-horizon closure는 바로 그 조건(10 태스크, 계열당 14~209 관측)에서 처음부터 학습한 specialist가 TSFM에 25~48% 졌다는 것을 보였고, §16에 "FM fine-tuning은 시험하지 않았다"고 명시했다 `[확인]`.

즉 20계열의 음성은 "한 계열을 60~90 origins로 적응시키는 설정"의 음성이다. 그 설정은 파운데이션 모델을 쓸 이유가 가장 약한 설정이기도 하다(이력이 충분하면 specialist도 된다). 새 방법이 나올 자리가 있다면, 그 반대편이다.

## 2. 경로 1 — 문제 명세부터 다시

### 2.1 왜 이 순서인가

Study20의 마지막 문장 `[확인]`:

> 다음에 재개할 구체적인 작업은 실제 적용 태스크 한 개의 제약과 실패 조건을 정하는 문제 명세다. 어떤 자료가 언제 들어오고, 정답이 언제 사용 가능하며, 한 번의 업데이트에 허용되는 시간·메모리와 반드시 유지해야 하는 성능이 무엇인지 관측 가능한 값으로 정한다.

이 순서가 맞는 이유는 20계열의 실패 패턴 자체에 있다. 매번 방법(또는 방법 후보)이 먼저 있었고 문제는 그 뒤에 붙었다. 그래서 강한 단순 대조가 나타나면 방법이 설 자리가 없어졌다. 문제를 먼저 고정하면 "이 조건에서 단순 대조가 어디까지 가는가"가 먼저 측정되고, 방법은 그 잔여에만 붙는다.

### 2.2 명세 양식

아래 필드가 전부 관측 가능한 값으로 채워져야 명세다. 채울 수 없는 칸은 `UNKNOWN`으로 두고 그 칸이 결정에 영향을 주는지 적는다.

```text
[태스크]      무엇을, 어느 해상도로, 얼마나 앞을 예측하나
[계열 구조]   계열 수 / 계열당 관측 수(분포) / 계열 간 이질성의 관측 가능한 지표
[입력 가용]   예측 시점 t 에 무엇이 보이나 (target 과거, 공변량, 다른 계열의 과거)
[정답 가용]   t 의 정답이 언제 관측되나 / 수정되나 (지연, revision)
[적응 시점]   언제 파라미터를 바꿀 수 있나 (배포 전 1회 / 주기적 / 매 origin)
[업데이트 예산] 1회 업데이트에 허용되는 wall-time·GPU 메모리·저장량, 풀 전체 기준인지 계열별인지
[유지 조건]   어떤 계열/구간에서도 F0 보다 나빠지면 안 되는가 (있으면 그 허용 폭)
[주지표]      하나. 계열 간 집계 방식과 가중치까지
[강한 대조]   F0 / 표준 절차 / 이 조건에 맞는 고전 방법 — 방법 설계 전에 이 셋의 점수를 먼저 확보
[실패 조건]   위 대조가 이 조건에서 무엇을 못 하는지, 관측된 수치로
[종료 조건]   실패가 관측되지 않으면 이 명세는 닫힌다 — 그 문턱을 지금 적는다
[오염]        백본 사전학습과 자료의 중복 상태
```

### 2.3 후보 태스크와 내가 고른 것

이 저장소에 이미 자료가 있고, 위 양식을 채울 수 있는 태스크 후보는 셋이다.

```text
후보                          자료 위치 [확인]                       왜 후보인가                           왜 안 고르나
(가) 풀 공유 짧은 계열 예측     runs/dscache fev 캐시 22개 태스크        축 2·4 교집합. specialist 실패 확인됨.    —
                              (hospital 767, m5_1M 30,490 계열 등)     FM fine-tune 미시험(§16)
(나) 집계 감독 미세 예측        data_external/bdg2_coarse_supervision_v1  실패 관측 가장 큼(29.8배)               22 방향 B: 잔여가 월평균 한 스칼라.
                                                                                                             그 스칼라 예측은 ridge 문제지 PEFT 문제 아님
(다) revision 있는 macro       data_external/alfred_revision_entry_v1    실제 정답 수정 존재                     Study19: 2계열, 선형에서 ZERO_GROWTH가 최선.
                                                                                                             FM 진입 근거 없음. 계열을 수백 개로 늘려야 열림
```

**(가)를 고른다.** 근거는 성능 기대가 아니라 §1의 논리다. 20계열이 공유한 설정의 바깥이면서, 자료·평가기·specialist 대조·오염 감사가 이미 로컬에 있어 GPU 0회로 명세를 채울 수 있는 유일한 후보다.

### 2.4 (가)를 양식에 채워 본 것

전부 저장된 파일에서 읽었고, 실행한 것은 없다.

```text
[태스크]      fev-bench Track U (target 과거만 입력). 태스크별 horizon 5~28 step, 일·주·월·년 해상도.
              long-horizon closure 의 10 태스크를 1차 후보로 둔다 (dev 4 + holdout 6).        [확인] STATUS §4·§5
[계열 구조]   hospital 767 / m5_1M 30,490 / restaurant 817 / solar_1D 137 / us_consumption_1Y 31 /
              world_co2 191 계열. 계열당 보이는 과거 14~209 관측. horizon/context 비 0.21~0.45.  [확인]
[입력 가용]   각 origin 에서 그 계열의 과거만. 공변량 없음(Track U). 같은 태스크의 다른 계열 과거는
              "풀"로서 학습에 쓸 수 있으나 추론 입력에는 넣지 않는다 (specialist 와 같은 계약).   [확인] A11
[정답 가용]   fev rolling window. window k 의 정답은 window k+1 의 입력이 된다. revision 없음.   [확인]
[적응 시점]   첫 cutoff 이전 이력으로 1회 학습, 이후 refit 없음 (specialist 와 동일 계약 A07~A10). [확인]
[업데이트 예산] 풀 전체 기준 1회. Study20 실측 LoRA 85초/200 updates 를 기준으로 태스크당 수 분.    [추정]
              m5_1M 은 30,490 계열이라 예산이 다르다 — 첫 파일럿에서 제외.
[유지 조건]   태스크 단위로 F0(= 3 TSFM 중 최선, F_FAMILY_ENVELOPE) 대비 악화 금지.               제안
[주지표]      fev native SQL, relative_to_naive. 계열 간 집계는 fev 자체 규칙.                    [확인] A12·A13
[강한 대조]   F0 (chronos-2 zero-shot) / 표준 shared LoRA (풀 전체 1 adapter) /
              specialist suite (이미 저장: S_BEST, S_ENSEMBLE) / SeasonalNaive / LinearAR.       [확인] 저장됨
[실패 조건]   specialist 는 F_FAMILY_ENVELOPE 에 dev −25% / holdout −48% (median RI).          [확인] Table D
              표준 shared LoRA 의 F0 대비 효과는 UNKNOWN — 이것이 첫 측정 대상.
[종료 조건]   표준 shared LoRA 가 F0 를 넘지 못하면(태스크 median RI ≤ 0) 이 명세에서
              "적응 자체가 무의미"이므로 방법 설계로 가지 않는다.
              넘으면, 그 이득의 계열별 분포를 보고 이질성이 있을 때만 방법 후보를 연다 (§4.3).
[오염]        chronos-2 는 fev-bench 에 대해 OVERLAP_RISK_UNKNOWN. tirex-2 도 같음.
              timesfm-3.0 만 CLEAN_BY_OFFICIAL_EXCLUSION 이나 로컬 weights 미확인.               [확인] contamination_matrix
```

오염 칸이 결정에 영향을 준다. 사전학습에 들어간 자료로 fine-tuning 효과를 재면 "이미 본 것을 다시 본다"가 섞인다. 첫 파일럿의 결과 해석은 그 한계를 안고 가야 하고, 확증은 오염이 배제된 자료가 필요하다. 이 한계를 지우는 유일한 방법은 timesfm-3.0(공식 배제)을 두 번째 백본으로 두는 것인데 로컬 weights 가 확인되지 않았다 — 실행 전에 확인할 항목이다.

### 2.5 이 경로의 산출물과 끝

산출물은 채워진 명세 하나와, 그 명세에서 강한 대조 셋의 점수다. 그중 F0·specialist·naive 는 이미 있고 **표준 shared LoRA 하나만 새로 측정**하면 된다. 그것이 §4.2 의 "4 fits" 다. 방법 설계는 그 뒤에만 연다.

## 3. 경로 2 — 짧은 계열 자료 확보

### 3.1 "짧다"의 정의

두 출처가 있다. 둘 다 자료 자체의 속성이지 결과에서 역산한 것이 아니다.

- 외부: Break-Even (arXiv 2607.04919, 정식 채택 미확인) — `n_train < 700` 이고 계절성이 무시할 수 없으면 fine-tuning 을 건너뛰라는 규칙. LoRA 가 짧은 계열에서 실제로 해롭다고 보고. `[확인: 초록 수준]`
- 내부: long-horizon closure 의 long 정의 `horizon_to_context_ratio > 0.1652` — 보이는 과거가 horizon 의 6배 미만. `[확인]`

이 저장소의 20계열은 이 정의로 전부 "길다" (Bike/Household L336 context, 63~90 origins × 48 step). 그래서 짧은 계열은 20계열이 한 번도 들어가지 않은 영역이다.

### 3.2 이미 로컬에 있는 것 — 새로 받을 필요가 거의 없다

`runs/dscache/autogluon___fev_datasets/` 에 22개 태스크가 캐시돼 있다 `[확인]`. 그중 짧은 계열에 해당하는 것:

```text
태스크                    해상도  계열 수   계열당 보이는 과거(중앙값)  horizon  H/ctx    비고
australian_tourism         Q         89          20                    8     0.400   dev, 가장 짧음
jena_weather_1D            D          1          58                   28     0.483   단일 계열 — 풀 불가
ecdc_ili                   W         25          80                   13     0.163   경계
LOOP_SEATTLE_1D            D        323          85                   28     0.329   dev
ETT_1D                     D          2         164                   28     0.171   2 계열 — 풀 약함
hermes                     W     10,000         209                   52     0.249   dev, 공변량 1
hospital                   M        767          ~36 (STATUS 표)       12     0.333   holdout
m5_1M                      M     30,490          ~48                  12     0.250   holdout, 무거움
restaurant                 D        817          ~72                  28     0.389   holdout
solar_1D                   D        137          ~85                  28     0.329   holdout
us_consumption_1Y          Y         31          ~14                   5     0.357   holdout, 극단
world_co2_emissions        Y        191          ~15                   5     0.333   holdout, 극단
```

첫 6행은 benchmark gap discovery 의 `task_metadata.csv` `[확인]`, 뒤 6행은 long-horizon closure `STATUS.md` §5 표 `[확인]`(계열당 과거는 H/ctx 에서 역산한 `[추정]`).

**결론: 자료 확보 단계는 사실상 이미 끝나 있다.** 외부에서 새로 받는 것은 오염 없는 확증 자료가 필요해질 때의 일이다.

### 3.3 외부 후보와 오염 검사

확증 단계가 오면 후보는 M4(monthly/quarterly/yearly, 10만 계열, 짧음), M3, Tourism 이다. 단 Chronos-2 Appendix A 가 사전학습 원천을 명시하고 M4 계열이 거기 들어 있을 가능성이 크다 `[추정 — 확인 필요]`. 자료를 받기 전에 benchmark study 가 한 것과 같은 `contamination_matrix` 절차를 먼저 돌린다. 오염 상태가 UNKNOWN 인 자료로 확증을 주장하지 않는다.

### 3.4 자료 확보 뒤의 첫 검사 — 0 fit 과 4 fits

**0 fit.** 방향 A 에서 한 것과 같은 오라클 상한을 짧은 계열에서 다시 계산한다. 다만 지금은 짧은 계열에 대한 LoRA 예측이 없으므로, 이 검사는 §4.2 의 4 fits 뒤에 붙는다.

**4 fits — break-even 곡선.** 판별자 검토가 제안한 설계이고 내가 보기에도 맞다 `[미검증]`. 한 태스크(hospital 권장: 767 계열, 가벼움)에서 계열당 사용 가능한 과거 길이를 4수준으로 자르고(예: 전체 / 1/2 / 1/4 / 1/8), 각 수준에서 표준 shared LoRA 를 고정 LR 1 seed 로 학습한다. 관측하는 것은 "적응 이득이 어느 길이에서 0을 지나는가"다. 이것은 8-fit 한도 안에 있고, Break-Even 논문의 축과 직결되며, 이 저장소가 한 번도 변화시키지 않은 유일한 변수다.

```text
실행 단위      hospital × 과거 길이 4수준 × shared LoRA(rank8, LR 고정, seed 1) = 4 fits
대조           F0 (저장됨), specialist S_BEST/S_ENSEMBLE (저장됨), SeasonalNaive/LinearAR (저장됨)
지표           fev native SQL, relative_to_naive, 태스크 median RI vs F0
예상 시간      Study20 LoRA 85초/fit 기준 4 fits 약 6분 + 예측·평가                              [추정]
종료 조건      4수준 전부에서 shared LoRA ≤ F0 이면 이 규모의 짧은 계열에서 적응 무의미 → 명세 닫음
              어느 수준에서 > F0 이면 그 지점이 break-even 이고 방법 후보(§4.3)의 입력이 됨
```

## 4. 내 의견

### 4.1 두 경로는 하나다

"문제 명세 다시"와 "짧은 계열 자료 확보"를 따로 놓으면 두 번 일한다. 명세를 쓸 태스크가 짧은 계열 풀 예측이고, 그 자료가 이미 로컬에 있으므로 **경로 2는 경로 1의 §2.4 [계열 구조]·[오염] 두 칸을 채우는 작업**이다. 합치면 이렇게 된다.

```text
1  명세 (§2.4) 를 확정한다 — GPU 0회. 오염 칸과 timesfm-3.0 weights 만 실행 전 확인.
2  표준 shared LoRA 를 hospital 에서 4수준 길이로 학습 — 4 fits. break-even 관측.
3  결과가 종료 조건에 걸리면 여기서 끝. NO_METHOD_SPECIFICATION 을 유지하되
   "짧은 계열 풀에서 표준 적응은 F0 를 넘지 못한다"는 관측이 추가된다.
4  넘으면 이득의 계열별 분포를 본다. 이질성이 크면 §4.3 의 후보를 연다. 작으면 방법 없이 끝.
```

### 4.2 실행 순서에 백본 검사를 앞에 둔다

위 4단계 전에 **3 fits 짜리 검사**가 하나 있다. 축 1이다. 20계열의 음성이 전부 chronos-2 한 revision 이므로, "LoRA 를 넘는 여유가 없다"가 방법론적 사실인지 이 백본의 성질인지 분리되지 않았다. Chronos-2 는 group attention 으로 추론 시점에 이미 상당한 in-context 적응을 하는 모델이라, LoRA 의 한계 이득이 작게 나오는 것이 백본 특성일 가능성이 남는다 `[추정]`.

가장 싼 확인: s13_bike 의 F0 대 고정 LR LoRA 하나만 TiRex-2 에서 반복한다. TiRex-2 는 xLSTM 계열이라 LoRA 주입 지점이 다르고 bridge 작업이 필요하다 `[확인: 아키텍처]`. 약 3 fits. 이 결과가 chronos-2 와 같은 방향이면 §1의 "배제된 명제" 표가 백본 독립으로 강해지고, 다르면 20계열 전체를 다시 읽어야 한다. 어느 쪽이든 먼저 알아야 하는 정보다.

### 4.3 방법 후보 — 조건부로, 순위를 매겨서

아래는 §4.1 의 4단계에 도달했을 때만 여는 후보다. 각각 관측된 문제 / 제안 규칙 / 강한 대조 / 선행 경계 / 반증 조건 / 예산 / 내가 보는 위험을 적는다. 셋 다 `[미검증]` 이고 신규성은 `[미확정]` 이다.

**후보 P1 — 이력 길이에 따른 적응 신뢰 (length-aware shrinkage toward F0)**

```text
관측된 문제    (예상) break-even 곡선이 있으면, 같은 풀 안에서도 이력이 짧은 계열은 적응이 해롭고
              긴 계열은 이롭다. 표준 shared LoRA 는 모든 계열에 같은 Δ 를 준다.
제안 규칙      출력 = F0 + g(n_i)·(LoRA − F0). g 는 계열 i 의 이력 길이 n_i 의 단조 함수,
              풀의 V 에서 한 개 파라미터로 적합. 파라미터 수 1~2.
강한 대조      g ≡ 1 (표준 LoRA), g ≡ 0 (F0), g ≡ c (길이 무관 고정 shrinkage, V 로 c 선택)
기전 삭제      n_i 를 무작위로 섞은 g — 길이 정보가 필요한지 검사
선행 경계      forecast combination / shrinkage 는 고전. 길이 조건부 결합 가중은 model averaging
              문헌에 있을 가능성 큼 [추정 — 검색 필요]. Break-Even 논문은 데이터셋 단위 이진 규칙이라
              계열 단위 연속 가중은 다름 [확인: 초록].
반증 조건      g ≡ c 가 g(n_i) 와 같으면 길이 정보는 무가치 → 닫음. 또는 곡선 자체가 없으면 시작 안 함.
예산           추가 fit 0 (저장된 LoRA·F0 예측의 사후 결합). 4 fits 위에 CPU 만.
위험           너무 단순해서 방법 논문이 되기 어렵다. 다만 "언제 적응을 믿을지"의 첫 정량 규칙이고
              반증이 즉시 가능하다. 논문보다는 §4.4 의 실증 연구 안에 한 절로 들어갈 크기.
```

**후보 P2 — 계열 기술자 조건부 공유 adapter (descriptor-conditioned shared LoRA)**

```text
관측된 문제    (예상) 풀 안의 이득 분포가 이질적이고 그 이질성이 관측 가능한 기술자(길이·계절 강도·
              규모·zero fraction)로 설명된다면, 하나의 flat adapter 는 평균만 맞춘다.
제안 규칙      LoRA 의 B 행렬(또는 scaling α)을 계열 기술자 벡터 d_i 의 작은 함수로 둔다:
              ΔW_i = B(d_i)·A, B(d_i) = B_0 + Σ_k d_ik B_k. 기술자는 학습 전에 계산되는 통계.
              파라미터 증가 = K × rank × d_out.
강한 대조      flat shared LoRA (K=0), 기술자 없이 rank 만 K배 늘린 LoRA (용량 통제),
              계열 기술자를 입력 채널로 넣은 F0 (정보 통제 — Chronos-2 는 공변량을 받는다)
기전 삭제      d_i 를 계열 간 무작위 치환
선행 경계      LLM 의 hypernetwork/MoE-LoRA, CoDA(도메인 participation), ThanoRA(태스크 rank) [확인: 초록].
              시계열: fusiontimeseries 의 BilinearLoRA 가 static operating parameter 조건부 LoRA 를
              이미 보고 [확인: 14번 노트]. 이것이 직접 선행이며, 차이는 "정적 운영 파라미터"가 아니라
              "이력에서 계산한 통계"라는 점뿐이다 — 약하다.
반증 조건      용량 통제 대조(rank K배)가 같으면 조건부는 무가치. 정보 통제 대조(기술자를 입력으로)가
              같으면 adapter 안에 넣을 이유 없음.
예산           4 arm × 2 seed = 8 fits (hospital). 약 12분 [추정].
위험           직접 선행(BilinearLoRA)이 있다. 신규성이 남으려면 "통계가 정규화로 지워지는 Chronos-2
              특성(14번 노트 InstanceNorm 불변성)을 되살리는 것"으로 문제를 좁혀야 하는데, 그건
              fusiontimeseries 가 이미 지목한 현상이다. 방법보다 실증 결과로 남을 가능성이 크다.
```

**후보 P3 — 풀 분할 적응 (adapter per cluster, 예산 배분)**

```text
관측된 문제    (예상) 풀이 이질적이면 하나의 adapter 가 아니라 몇 개가 필요하다. 몇 개, 어느 계열에.
제안 규칙      기술자 공간에서 계열을 K개로 나누고 클러스터마다 adapter. K 와 배분은 V 로.
              전체 파라미터·업데이트 예산은 flat LoRA 와 동일하게 고정.
강한 대조      flat LoRA (K=1), 무작위 분할 (K 같음, 기전 삭제), 계열별 adapter (K=n, 예산 초과 — 상한)
선행 경계      task clustering for multi-task learning 고전. 시계열 FM 에서 adapter clustering 은
              제한 검색에서 미확인 [22 방향 A 조사]. 신규성 증명 아님.
반증 조건      무작위 분할과 같으면 구조는 무가치. K=1 과 같으면 분할 자체가 무가치.
예산           K∈{1,2,4} × seed 1 + 무작위 분할 1 = 4~5 fits. 다만 K=4 는 fit 4개를 뜻하므로
              실제 GPU 호출은 1+2+4+4 = 11회. 8-fit 한도 초과 — 예산 변경 승인 필요.
위험           이것이 지시문의 "방향 A"를 그룹 축에서 되살린 형태다. 22 에서 방향 A 를 닫은 근거는
              "cell 단위 적응 거부"였고, "풀 내부 배분"은 시험되지 않았다. 그러나 이득 분포가
              이질적이라는 전제가 먼저 관측돼야 한다 — 4단계 결과 없이는 열지 않는다.
```

순위: **P1 > P3 > P2**. P1 은 예산 0 이고 반증이 즉시 가능하며 P3·P2 의 전제(이득의 이질성)를 가장 싸게 검사한다. P3 는 P1 이 "길이만으로 설명 안 됨"을 보였을 때만 의미가 있다. P2 는 직접 선행이 있어 마지막이다.

### 4.4 가장 가능성 높은 결말 — 미리 적어 둔다

내가 보기에 확률이 가장 높은 순서는 이렇다 `[추정]`.

1. 백본 검사(§4.2)는 chronos-2 와 같은 방향 — 20계열 음성이 백본 독립으로 강해진다.
2. break-even 곡선(§3.4)은 존재한다 — 극단적으로 짧은 계열(us_consumption 14 관측 수준)에서는 적응이 해롭고 hospital 규모(~36)에서는 이롭거나 중립. 표준 shared LoRA 가 hospital 에서 F0 를 소폭 넘는다.
3. 이득의 계열별 분포는 길이로 대부분 설명된다 — P1 의 g(n) 이 g≡c 를 이기지만 차이는 작다. P3·P2 는 열리지 않는다.

이 결말이면 새 방법은 없다. 대신 남는 것은 **"시계열 파운데이션 모델의 PEFT 는 어디서 이득이 있고 어디서 없는가"의 통제된 지도** — 백본 2개 × 학습량 4수준 × 단일/풀 × 20계열의 단순 대조 — 이고, 이 저장소의 결과 규율(사전 등록, 봉인 평가, 독립 재계산, 오염 감사)이 정확히 그런 논문이 요구하는 것이다. 이건 방법 논문이 아니라 실증 연구 논문이다. 최상위 목적("새 방법 기여")과 다르므로 이 결말을 받아들일지는 사용자의 결정이다. 다만 20계열의 결과를 보면 이 결말을 미리 배제하는 것이 더 큰 위험이다.

새 방법이 나오는 경우는 2번에서 이득 분포가 길이로 설명되지 않는 **강한 이질성**이 관측될 때뿐이다. 그때 P3 가 열리고, 그 문제는 "예산 배분"이라는 방향 A 의 원래 질문이 실제 조건에서 되살아난 것이다.

### 4.5 하지 말 것

- 같은 UCI hourly 두 원천에서 adapter 변형을 하나 더 만드는 것. 20계열이 그 설정을 닫았다.
- 방향 B·D 를 다른 이름으로 다시 여는 것. 22 에 닫힌 근거가 있다.
- 4 fits 결과가 나오기 전에 P1~P3 중 하나를 골라 구현을 시작하는 것. 전제(이득 이질성)가 먼저다.
- fev-bench 오염 상태가 UNKNOWN 인 채로 첫 파일럿 결과를 "확증"이라 부르는 것.
- 8-fit 한도를 조용히 넘기는 것. P3 는 11회라 승인이 필요하다.

## 5. 승인이 필요한 것 — 한 번에

실행 전에 다음을 한 번에 확인받는다. 하나라도 거부되면 해당 단계는 빠진다.

```text
A  §2.4 명세를 이번 인계의 문제 명세로 확정하는가 (태스크: fev-bench 짧은 계열 풀, hospital 1차)
B  백본 검사 3 fits (TiRex-2, s13_bike F0 대 고정 LR LoRA) 를 먼저 하는가
C  break-even 4 fits (hospital, 과거 길이 4수준, shared LoRA 고정 LR 1 seed) 를 하는가
D  종료 조건 — 4수준 전부 ≤ F0 이면 명세를 닫고 NO_METHOD_SPECIFICATION 유지 — 를 받아들이는가
E  §4.4 의 "실증 연구 논문" 결말을 허용 가능한 결말로 두는가, 아니면 새 방법이 없으면 전체를 접는가
```

E 가 이 문서에서 가장 중요한 질문이다. 답에 따라 B~D 의 의미가 달라진다.

## 6. 이 문서가 주장하지 않는 것

- 짧은 계열 풀에서 적응이 이득이라는 것. 그것이 첫 측정 대상이다.
- P1~P3 가 새 방법이라는 것. 셋 다 선행 경계가 가깝고, 전제가 관측되기 전에는 후보도 아니다.
- fev-bench 자료가 오염에서 자유롭다는 것. chronos-2·tirex-2 는 OVERLAP_RISK_UNKNOWN 이다.
- 20계열의 음성이 백본 독립이라는 것. §4.2 검사 전에는 모른다.
- 이 문서의 시간·비용 추정이 실측이라는 것. Study20 의 fit 시간에서 외삽한 `[추정]` 이다.

## 근거 파일

- 저장 결과: `results/tsfm_benchmark_gap_discovery_v1/{task_metadata.csv,contamination_matrix.csv}`, `results/tsfm_long_horizon_specialist_closure_v1/{STATUS.md,verdict.json}`, `results/peft_selection_regret_v1/selected_results.csv`, `results/peft_fullft_reference_v3/metrics.csv`
- 이번 스크린: `results/peft_method_pilot_screen_v1/{pattern_preserving_control.json,adaptation_abstain_control.json,covariate_lag_control.json,source_hashes.json}`
- fev 캐시: `runs/dscache/autogluon___fev_datasets/` (22 태스크)
- 재사용 코드: `experiments/peft_adaptation_scope_v1/modeling.py`, `experiments/peft_fullft_reference_v3/`, `experiments/peft_adaptation_scope_v1/guard.py`
