# 17. 집계 감독 아래 내부 적응 필요성 검사

2026-09-08. [자료 진입 계획](17_coarse_supervision_data_entry_plan_20260908.md)에 따른 실제 다운로드·2016 QC가PASS였다. Eagle34/Lamb16 eligible에서 사전 규칙으로 각8donor/8target을 골랐다. 이 파일은2017 전력 값의 학습·평가용 가공과 모델 실행 전에 고정한다. 주제 탐색의 필요성 검사이며 표준 LoRA의 양성을 새로운 방법 완성으로 주장하지 않는다. 지속 탐색 승인을 따른다.

## 목적과 정보 계약

질문은 월별 관측만 있는 대상의 미세 미래 예측에서, 잘 맞춘 동결 출력 head에 내부 attention LoRA를 더할 이유가 있는가다. Target fine history로 창을 옮겨 fine label을 얻는 우회는 차단한다. [Chow–Lin](https://journal.r-project.org/articles/RJ-2013-028/)과 월별 검침 고객의 profile 이전·회귀를 쓰는 [PGE 연구](https://www.aceee.org/files/proceedings/1998/data/papers/0625.PDF)를 강한 대조의 근거로 삼는다.

원 electricity/metadata의 pinned SHA·선정 ID는 `data_external/bdg2_coarse_supervision_v1/{contract,qc}.json`을 그대로 사용한다. 다른 건물로 바꾸지 않는다. Raw17,544개 naive-hour 격자를 기준으로 하며 두 사이트를 같은UTC동시측정으로 묶지 않는다. 실제 청구 공개 지연·DST 원계량 정책은 완전히 확인되지 않은 통제된 자료 접근 실험이다. 월별 합은 해당 달의 모든 원 hour가 finite·비음수일 때만 생성한다. 결측 부분을 채워 bill을 만들지 않는다. 사전학습 비중복은 UNKNOWN이다.

Target2016년12개월 월합·시간 수만 train supervisor와 정규화에 사용한다. Target의 시간별 표준편차·자기상관·profile은 추출하지 않는다. 기준 scale은 target의2016 연합계/8784다. 양수임은진입에서 확인했다. Donor의2016 fine 값은 calendar template 작성에 사용한다. 미래 donor 실제값·날씨는 사용하지 않는다.

고정 구간:

- Train origins:2016-02~12의월첫시각,16target으로176창. January2016을처음과거월로사용한다.
- V origins:2017-01~03,48창. 완전한 현재월합만 학습모델·ridge정규화 선택에 사용한다. Target의 fine V 오차는 계산하지 않는다.
- E origins:2017-04~06,48창. 모델/선택을동결한뒤모든예측을저장하고fine값을연다.
- July~December2017은이번screen의성능계산·튜닝에사용하지않는다. 양성시별도고정확인에남긴다.

V는사이트별24창중16개이상완전월label과각target최소1개가필요하다. 미충족이면자료부족으로실험을중단하고건물/기간을바꾸지않는다. E의fine점수는finite·비음수원시간만사용하되각target/month에시간80%이상이필요하다. 각사이트24창중16개이상과각target최소1창이필요하다. 미충족을방법실패라고부르지않는다.

## 동일한 과거 proxy

Site별 donor8개의각2016연평균으로fine값을나눈뒤,2016 `(month,day_of_week,hour)`셀의평균을합쳐calendar template를만든다. 모든셀은2016자료만사용한다. 모델input은원점직전512개naive hours다. 각과거calendar월에template를그달전체격자평균으로나눠월평균1의shape를얻고,그원점까지알려진그과거월의target월평균을곱한다. 과거월label이불완전하면더이전의마지막완전월평균을carry한다. 이는결측bill의추정치이며실제bill로기록하지않는다. 모든origin앞에는완전한2016년월이존재한다.

E의다음달origin은이미끝난직전E월의완전월합을입력에사용할수있다. 예를들어May1은April합을알지만April1은알지못한다. 모델가중치는E에서업데이트하지않는다. 미래현재월의합은E input파일에넣지않는다. 모든방법은같은proxy·허용된월별과거·donor template·calendar를사용한다. 각방법의표현구현은다르며추가fine target정보는없다.

PROFILE은직전의마지막완전월평균에미래달의평균1 donor template를곱한점예측이다. Oracle현재월합을사용하지않는다.

## 동결 모델과 단순 대조

기존fixed Chronos2 revision `29ec3766d36d6f73f0696f85560a422f50e8498c`,FP32weights/BF16autocast,context512,한target row/독립group을쓴다. H는672/696/720/744등실제naive월hours이며patch16 padding은손실·점수에서제외한다. 원소별raw역변환뒤21native출력을평균내는고정함수`native_grid_point`를F0점예측p0로정의한다. 이를분포평균또는월합의분위수라고부르지않는다. 불확실성proper score나coverage주장은하지않는다.

PROFILE/F0 외에두CPU대조와한LoRA를둔다.

- COARSE_LIFT: site별ridge로정규화된월평균을예측한다. 특징은 `[1,mean(context)/s,std(context)/s,mean(last72context)/s,(mean(last72)-mean(first72))/s,mean(p0)/s,H/744]`다. targetfine표준편차가아닌proxy의표준편차다. 예측은 `p0+s*(predicted_normalized_month_mean-mean(p0)/s)`다. 같은fine p0 pattern에시간마다일정한수준만더한다.
- FROZEN_HEAD: 동결hidden768에 `Linear(768,16)` raw잔차head를붙인다. `p=p0+proxy_native_scale*head(hidden)`이며12,304계수다. 월평균출력의선형feature를만들어닫힌형식ridge로적합한다. 이렇게head를먼저충분히맞춘다.
- ATTN_LORA_FIXED_HEAD: V로고른위head를그대로고정하고attention96projection에r8/alpha16/dropout0 LoRA만추가한다. Native출력head도고정하지만p0경로에서계속사용한다. 학습가능계수1,179,648개다. 초기LoRaB=0이므로시작점은선택된FROZEN_HEAD다. 같은target월별label만사용한다.

두ridge는각각lambda `[0.001,0.1,10]`을고정한다. `K=X X^T`, `alpha=lambda*trace(K)/n`, `w=X^T solve(K+alpha I,y)`를쓴다. 별도intercept를더하지않고bias특징도같이regularize한다. Head y는정규화된월평균label에서F0월평균을뺀잔차다. 각각의lambda는사이트/target/month균등macro V월평균MSE로선택한다. 동점은작은lambda순이다. Fine V는사용하지않는다. FROZEN_HEAD의raw시간feature평균과월평균design이일치하는지독립검산한다.

LoRA는seed2026090817,AdamW lr3e-5/betas(.9,.999)/eps1e-8/weight_decay0,200updates,micro1/gradient accumulation4,clip1.0을고정한다. Train176창을고정rng permutation으로순환한다. 손실은 `((mean(point[:H])-monthly_total/H)/s)^2`다. 추가anchor loss·fine label·LRsweep은없다. Step0/40/80/120/160/200을같은Vcoarse점수로선택하며동점은이른step이다. 그룹이한row면group-attention q/k gradient0일수있으며모든LoRA계수의비영gradient를요구하지않는다. 실제module별gradient/업데이트상태를보고한다.

단순policy는PROFILE/F0/선택COARSE_LIFT/선택FROZEN_HEAD중Vcoarse최저방법으로고정한다(동점은열거순). LoRA와모든ridge/선택을동결한뒤Eforecast를생성한다. E에서더유리한baseline을보고바꾸지않는다. 모든단순armE점수를함께보고해선택policy의한계도드러낸다.

## 평가와 진행 규칙

각target/month의원유효hour에대해 `(p-y)/s`의MSE를계산한다. 월별validsupport에서평균오차제곱과평균을제거한pattern MSE로분해해합이전체MSE인지검사한다. 이level항은결측이있으면validsupport평균이며실제월bill오차와동일시하지않는다. Site별target/month균등macro를주평가로삼고MAE와count도보고한다.

주효과는 `(simple_policy_MSE-LoRA_MSE)/F0_MSE`이며두사이트를따로보고한다. 사전screen진행조건은두사이트에서각각point효과1%이상,두사이트각각LoRA의pattern MSE가FROZEN_HEAD보다낮음,pooled16target cluster bootstrap4000회/seed2026090817의주효과95%하한>0이다. Site내target을복원추출하고각target의세달을함께보존한다. 월별독립성·날짜일반화·seed변동을포괄하는CI라고주장하지않는다. 이조건은탐색진입용이며반복주제검색의오류율을제어한확증이아니다.

LoRA가월평균만개선하고pattern을개선하지못하거나단순대조가충분하면현재일반LoRA coarse-supervision screen을닫는다. 이후같은E에서LR/anchor조건을바꿔구제하지않는다. 통과해도새방법완성은아니며,donor의fine supervision으로학습한adaptation prior같은구체방법과동일정보의stronghead/metahead대조를분리한후새확인이필요하다.

## 구현·안전 검증

Model・train/cache는training agent,데이터/proxy・ridge는data agent,analysis는safety agent,계약/직렬runner/통합검증은root가담당한다. 기존16소스・결과는수정하지않는다. 새소스・계획・실행환경・modelweights・데이터각split SHA를학습전에고정한다. Targetfine순열이같은월합을유지할때proxy/정규화가불변인지,현재미래월변경이해당origin입력에영향없는지검사한다.

S0는첫train H744창에서fitted head 출력대LoRA초기출력,3updates의finite loss/유효LoRaB gradient,동결원가중치와head불변,padding제외,checkpoint재로드출력재현을검사한다. 본cache·학습·forecast는root의sharedguard에서직렬실행한다. CPUthreads2/원RAM5GiB·commit6GiB·childRSS8GiB·Git32/GPU10500MiB·85°C한도를유지한다. Cache600초,S0 120초,본학습900초,Eforecast300초를상한으로두고초과시원인확인없이재실행하지않는다. 실패·partial·원선택기록을보존한다.
