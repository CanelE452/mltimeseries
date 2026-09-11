# 다음 시간 블록에서 동일 PEFT 절차 재현

2026-09-08. [12번 결과](12_peft_external_gap_results_20260908.md)를 본 뒤 작성한 후속 계획이다. Bike는주효과+3.9253%F0/97.5%CI[2.0523,5.1823]%로통과했고Household는+1.3553%/CI[−.0682,3.2396]%로보류됐다. 양성원천만선별하거나보정주지표를바꾸지않고두원천의시간반복을한번수행한다. 이문서는새모델실행·새구간성능열람전에고정한다.

## 반증할 가설

가설은같은64일적응/14일선택/14일보정/84일배포절차의LoRA효용이다른계절에서도유지된다는것이다. 특정원천의효과차이가유의하다는가설은아니다. 이번에도표준방법의필요성screen이며새PEFT방법의완성을뜻하지않는다. 기존checkpoint를시간만늘려평가하지않고,동일nativeChronos-2원checkpoint부터새train기간으로재학습한다.

## 데이터와 분할

원천·열·hourlyaggregation·target마스킹·과거ffill·train-only통계·L336/H48/stride24는12번과같다. **새pre-context시작은각원천첫완전일부터정확히190일뒤**, 즉12번eval끝이다. 성능이나target크기·분포로다른날짜를찾지않는다.

```text
              pre-context      train 시작        V 시작          C 시작          E 시작          E 끝(미포함)
Bike          2011-07-10        2011-07-24        2011-09-26      2011-10-10      2011-10-24      2012-01-16
Household     2007-06-25        2007-07-09        2007-09-11      2007-09-25      2007-10-09      2008-01-01
```

둘다pre14/train64/V14/C14/E84일,origin63/13/13/83개다. 위시간표는달력연산과metadata로검증한다. 이번전체190일은12번전체190일과원시시점이겹치지않는다. Source원본파일을읽을수있었다는사실과모델평가노출을구분하며새기간의성과는아직보지않았다. 미래preparedfile은따로분리하고선택전fit프로세스에는train/V만제공한다. 어떤target의split관측률이70%미만이면그원천은QC실패로기록하고기간을옮겨구제하지않는다.

## 동일 선택 절차와 비용

12번과동일F0/H_MLP/H_FULL/OFF_LORA를사용한다. H_MLP LR{1e-4,3e-4,1e-3},H_FULL{3e-5,1e-4,3e-4},LoRA{1e-5,3e-5,1e-4}; seed12000에서새V의SORT점수로H6후보/LoRA3후보를각선택한다. 선택한family/LR를seed12001/12002에고정하고각seed의Vcheckpoint를선택한다. **이전블록에서선택한LR값을고정하는것이아니라전체선택규칙을반복**한다. Randomseed도동일하게두되새훈련데이터와새checkpoint를사용한다.

200updates/validation0·40·80·120·160·200,effective8/micro4,원nativequantileloss,AdamWwd0/clip1,r8alpha16,97projections,같은초기화·cache수치경로다. 최종28fit(26adaptation+2F0/cache)과14선택을고정한뒤14유보추론을수행한다. 더긴학습이나추가LR은없다.

RAW는같은full-past표준화ridge와SSE+lambda||beta||²,lambda{.1,10,1000},관측train잔차quantile을사용한다. 새V로lambda선택후새C/E예측을저장한다. 모든방법에새C의target×21quantile empiricaloffset을적합하는QCAL을공통적용하고SORT도보존한다. 이전Coffset이나이전trainstats를재사용하지않는다.

## 판정과 정지 규칙

주효과는새블록의`(H_QCAL−LoRA_QCAL)/F0_QCAL`이다. 7일movingblock4000회,각원천97.5%CI와하한>1%F0/각seed양수/RAW점추정veto없음이라는12번진입기준을그대로사용한다. Bootstrapseed는2026090813,3/14일민감도는보조다. 12·13결과를합쳐가족보정없이새유의성을만들거나더좋은블록만선택하지않는다. 이반복은같은원천의다른계절평가이며새원천·독립backbone확증이아니다.

새블록에서두원천기준을통과하면12번과나란히해석하며,원블록Household가보류였다는사실은유지한다. 그다음native OUT_ONLY재학습대조로저랭크출력제약만으로설명되는지확인한다. 새블록이기준을못넘으면이배포·예산에서범용내부PEFT필요성을찾으려는현재A분기를닫고,추가기간반복·LR증가로양성을찾지않는다. B의선택regret또는C의보존문제를검토하려면실제관측과별도반증실험에근거한새계획이필요하다.

## 실행·보존

새namespace `experiments/peft_temporal_replication_v1/`, `runs/peft_temporal_replication_v1/`, `results/peft_temporal_replication_v1/`를사용한다. 12번소스·계획·데이터·cache·가중치·선택·예측·결과를읽기전용참조로보존한다. 공통구현재사용wrapper는새소스와생산/소비계약을모두기록하고원12파일을수정하지않는다. CPU검증과새train-only S0 8개가통과한고정계약에서만본학습한다. Root만단일GPU sharedguard를사용하며RAM5GiB/commit6GiB/GPU10500MiB·85°C중단기준을유지한다. 시스템설정변경·새환경설치·commit/push없음.
