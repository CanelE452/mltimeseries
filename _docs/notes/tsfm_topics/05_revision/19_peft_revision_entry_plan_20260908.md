# 19. R1 수정 정답: 자료 진입과 CPU 진단 계획

2026-09-08 22:44 KST. 사용자 승인: “그러면 한번 진행해볼래? 얼마나 걸릴것같아?”
18번 후보 R1의 첫 실행이다. 아래 규칙은 자료 값 다운로드·점수 계산 전에 고정한다. 기존 17번 소스·계약·결과는 수정하지 않는다.

## 목적과 범위

실제 정답 revision을 인과적 시간 규칙으로 재현할 수 있는지, 간단한 현재 자료 ridge와 정확한 충분통계 갱신이 어떤 효과·비용을 가지는지 확인한다. 새 PEFT의 성능이나 필요성을 입증하는 실험이 아니다. 이번 계약은 CPU까지다. GPU 단계가 타당하면 별도 계약을 작성한다.

## 자료 계약

- ALFRED 공식 웹 다운로드 양식으로 PAYEMS와 INDPRO 두 계열만 받는다. 관측일 1989-12-01~2024-12-01, 선택 vintage는 1990-01-01~2025-06-30 범위의 모든 공식 선택지 및 경계일이다. 원값 units=lin, file_type=1(real-time periods), file_format=csv.
- 공식 HTML 양식의 선택지와 응답 ZIP/README, 요청 필드, SHA256, 수신 시각을 보존한다. API key·로그인 우회는 하지 않는다. ZIP당 16MiB, 총 압축 해제 128MiB 이하, 요청 timeout 30초, 다운로드 guard 300초. 파일이 아닌 오류 응답은 보존하고 실패로 기록한다.
- 원본을 바꾸거나 다른 계열로 대체하지 않는다. 필드 이름이나 인코딩 차이를 처리하는 것은 허용하며 그 변경을 기록한다. 첫 진입이 실패하면 자료 접근 실패로 보고한다.
- 실시간 유효기간은 양 끝을 포함한다. 같은 event의 유효기간 중복·충돌, 날짜 오류, 비양수/비유한 level은 실패다. 과거 snapshot은 미래 vintage로 채우지 않는다.
- 공통 변수는 같은 vintage의 두 level로 계산한 g(t,v)=100*[log L(t,v)-log L(t-1,v)]이다. 서로 다른 최초 발표 level을 차분하지 않는다. INDPRO 기준연도 변경으로 원값의 차이는 revision 크기 주지표로 쓰지 않는다. 성장률 변환이 모든 정의 변경을 없애는 것은 아니다.
- 최초 성장률 label은 두 level이 함께 존재하는 가장 이른 실제 vintage의 값이다. 자료 범위 시작 이전 event의 최초 보유 record를 최초 발표로 주장하지 않는다. 모델 event는 1991-01 이후만 사용한다.
- event t의 성숙일 M(t)는 6개월 뒤 월말이다. 평가 정답은 그 날짜까지 이용 가능한 snapshot에서 계산하고 이후 revision으로 바꾸지 않는다.

## 예측·학습 정보 계약

- target event r의 origin은 r월 1일 00:00 America/New_York. 허용 정보는 vintage_date < origin_date. 일중 수신 시각을 재현했다는 주장은 하지 않는다.
- 예: 2월 event는 2월 1일에 예측하며 대개 최신 관측은 12월이다. 달력상 다음 달 forecast이자 최근 관측 기준 2-step이다. feature는 고정 좌표 r-2,...,r-7의 성장률 6개와 intercept다. 실제 latest_available_event 및 gap도 기록한다.
- 1991-01~2024-12 target의 과거 origin 당시 feature를 저장한다. 학습에는 해당 origin보다 이전 target 중 최초 label이 도착한 row만 넣는다. label revision은 min(origin 전날, M(t)) snapshot으로 제한한다.
- FIRST_FIXED_X와 REVISED_FIXED_X는 과거 feature를 당시 버전으로 고정한다. REVISED_CURRENT_X는 동일 lag 좌표의 level 버전만 origin 현재 허용 자료로 갱신한다. 새 event를 과거 row에 추가하지 않는다. 모든 arm의 현재 예측 feature는 같다.
- MATURE_ONLY는 M(t)가 지난 row만 쓴다. FIRST_BIAS는 FIRST_FIXED_X 예측에 이미 성숙한 과거 event의 mean(mature-first)를 더한다. AGE_CORRECTED는 현재 provisional label에 동일 경과 개월의 과거 provisional→mature 평균을 더한다. 보정 통계에 미래 성숙값을 쓰지 않는다.

## CPU 비교와 선택

- ZERO_GROWTH, LAST_KNOWN_GROWTH, FIRST_FIXED_X, FIRST_BIAS, REVISED_FIXED_X, REVISED_CURRENT_X, MATURE_ONLY, AGE_CORRECTED를 비교한다. 앞의 두 arm은 통계 baseline이며 frozen FM가 아니다.
- Ridge는 float64, intercept 비정규화, 나머지 λ 정규화다. feature는 성장률 원 단위로 유지한다. λ 후보 {0.01,0.1,1,10,100}, 과거 학습 window 후보 {120개월, expanding}; 최소 유효 학습 row 100개. 각 학습 arm은 같은 10개 후보를 쓴다.
- V: 2016-01~2018-12 (36 origin/계열). V 정답은 2019-06-30까지 성숙하며 2019-07-01 선택 동결. E: 2020-01~2024-12 (60 origin/계열). 2019년은 causal 학습에만 추가할 수 있다. COVID 기간을 제외하지 않는다.
- 선택: 계열·arm별 V MSE 최소. 동률은 작은 λ, expanding 우선. V 선택과 예측을 먼저 저장한 뒤 E 점수를 계산한다. E에서 재선택하지 않는다.
- Primary: 계열별 E MSE, 보조 MAE. 평균 비교는 1991-01~2014-12 성숙 성장률의 분산으로 각 계열 MSE를 정규화한 평균이다. 이 scale은 V 이전에 이용 가능하다. 두 macro 계열을 독립 domain 둘로 주장하지 않는다.
- 미래 gap 및 입력·label availability, first/M coverage를 전수 확인한다. V36/E60 중 결측이 있거나 과거 유효 row가 100 미만이면 해당 데이터 진입은 보류하고 선택적 행 삭제 점수는 만들지 않는다.
- Revision 빈도(|mature-first|>1e-10), 절대 크기/성장률 분산 대비 크기, release lag, 6개월 내 변화 횟수를 보고한다. event별 count는 forecast 표본 수와 구분한다.
- 정확한 ridge 갱신: λ=1, expanding, fixed X에서 row 최초 도착 시 A,b를 추가하고 revision 시 b+=x*(new-old)만 적용한다. 각 origin의 batch A,b, 계수, 예측과 비교한다. 최대 계수/예측 차이 1e-8을 넘으면 수치 검토 후 결과를 확정하지 않는다. 이것은 기존 항등식의 구현 검증이며 신규성 지표가 아니다.
- 불확실성은 고정된 12개월 circular time block, 2,000회 seed1908로 두 계열 동일 origin을 함께 재표집한다. 95% percentile 구간은 작은 5년 표본의 기술적 불확실성이며 보편적 유의성 주장이 아니다.

## 진행 판단과 자원

- 자료 진입 PASS는 누수·coverage 계약을 만족한다는 뜻이다. 수정 크기 0이면 현 자료의 revision 질문을 닫는다.
- 수정 정보를 이용한 강한 CPU arm이 FIRST_FIXED_X 대비 두 계열 정규화 평균 MSE 1% 이상 개선하는지 기록한다. 이 문턱은 다음 FM 진단의 우선순위를 정하는 실용적 screen이며 논문 성공 기준이 아니다. 미달이면 현 두 계열의 GPU 투자를 보류한다. 이상이어도 frozen FM/current-vintage head 및 표준 corrected replay를 먼저 검증해야 한다.
- exact update가 batch와 일치하면 알려진 선형 해법의 충분성만 확인한 것이다. 그것으로 비선형 PEFT 가설 전체를 반증하지 않는다. 역으로 stale arm 개선만으로 새 optimizer 설계를 시작하지 않는다.
- 기존 experiments/peft_adaptation_scope_v1/guard.py의 공유 lock·CPU-only guard를 재사용한다. GPU 학습 0, OS/드라이버/Defender 변경 0. 학습/실험 모듈에는 새 의존성을 설치하지 않는다.
- 예상: 자료 확보·QC·CPU 분석 총 20~40분. 공식 다운로드 문제 또는 시간 계약 결함은 즉시 보고하고 원인과 남은 일을 구분한다.

## 공식 출처

- https://alfred.stlouisfed.org/help/downloaddata
- https://alfred.stlouisfed.org/help
- https://alfred.stlouisfed.org/series/downloaddata?seid=PAYEMS
- https://alfred.stlouisfed.org/series/downloaddata?seid=INDPRO
- https://www.bls.gov/web/empsit/cesnaicsrev.htm

출처의 메타데이터 확인과 값 payload 검사는 구분한다. 작성 시점 payload는 아직 받지 않았다.
