# 같은 head LoRA 이득의 최적화 대조 — 2026-09-10

사용자 요청: 앞서 제시한 1→2→3 검증을 진행한다. 1은 최적화 혼입 대조, 2는 과거 특성으로 효용 예측, 3은 별도 미래 평가다. 이 문서는 1단계 결과를 보기 전에 고정한다. 원 실험은 불변 입력으로 보존한다.

## 1. 경쟁 설명과 비교

기존 FULL90/SPREAD30/RECENT30의 같은 residual MLP 대비 attention LoRA 추가 이득이 자료 구성 때문인지, 학습률·반복 노출·40회 간격 검사 때문인지 구분한다. Chronos-2, 원 fit/V/E, seed25000/25001, MLP589301개와 MLP+ALL rank8 1768949개, native pinball, batch8/micro4, AdamW/clip1은 유지한다. 동일 head 비교이며 총 trainable parameter 수를 맞춘 대조는 아니다.

2원천 ×3자료조건 ×2arm ×2seed ×2학습률(각 head/LoRA 모두1e-4 또는1e-5)=48 trajectories. 모든 trajectory는180updates까지 진행한다. 같은 seed의 표본추출 인덱스 스트림을 사용한다. subset30은 FULL90보다 동일 step당 기대 반복 노출이3배다. 학습률은 각 dataset/condition/arm/regime에서 두 seed의 최소 V 점수 평균으로 선택한다. 동률은 낮은 학습률을 우선한다. E를 열기 전에 모든 선택을 저장한다.

하나의 trajectory에서 두 가지 사전 고정 선택 규칙을 평가한다.

- UPDATE: 모든 조건의 후보 step=[0,5,10,20,40,60,100,140,180]. 같은 업데이트 상한과9회 선택 기회.
- EXPOSURE: FULL90 후보=[0,15,30,60,120,180], subset30=[0,5,10,20,40,60]. 각 후보의 기대 표본 노출8*step/N이 일치하고6회 선택 기회다. 실제 unique coverage/gradient 경로를 동일화하는 것은 아니다. subset60 이후 계산은 UPDATE를 위해서만 수행하고 EXPOSURE 선택에는 넣지 않는다.

추가 결과를 보고 grid를 늘리지 않는다. checkpoint0 포함, strict V 개선만 채택. 공유 head 초기값 동일, 시작 F0 동일, frozen weights 불변, checkpoint 복원 V 완전일치와 독립 E score 재계산을 확인한다. 두 종류 각24개 선택된 모델의 E 평가=48개. 적은 seed에서 학습률을 seed별로 선택하지 않는다.

## 2. 사전 진행 기준

주효과 G=100*(MLPscore−ALLscore)/F0score. Bike가 기존 신호가 큰 discovery 원천이다. Household는 보조 반복이며 실패를 숨기지 않는다. 아래는 검정 유의수준이 아닌 후속 비용을 정하는 실용적 gate다.

두 regime 모두에서 (a) Bike FULL90 G가 두 seed 모두 양수이고 평균≥1%F0, (b) Bike FULL90−SPREAD30 G 차이가 두 seed 모두 양수이고 평균≥1%F0이면 기존 자료 구성 관련 설명을 후속 검증할 가치가 남는다. RECENT30−SPREAD30과 Household는 방향·크기를 함께 보고하며 이 gate를 대체하지 않는다. 통과하지 못하면 현재 관측을 기전→새 adapter로 확장하지 않고 2·3의 종속 학습을 중단한다. 이것이 모든 PEFT 가능성을 부정하는 것은 아니다.

통과해도 원인 확정은 아니다. 시기·coverage·고유 label 수 혼입은 남는다. 2단계는 과거만으로 산출한 소수 특성으로 G를 예측해야 하며, 기존6조건×2seed를12개 독립 도메인으로 취급하지 않는다. 추가 rolling episode 수와 노출 감사를 확인한 뒤 별도 프로토콜을 먼저 고정한다. 3단계는 그 규칙을 고정한 후 시간적으로 분리된 미래 block에서 평가한다. 같은 원천 새 시기와 독립 원천 일반화는 별개다. 기존 E는 반복 노출된 개발 평가이고 최종 test가 아니다.

## 3. 안전·시간·재현

원 코드·데이터·checkpoint hash 확인. GPU 한 작업씩, admission commit13GiB/RAM5GiB 두 번 확인, 실행 guard commit6GiB/RAM5GiB/85C, 각 child900초. 문턱 완화·자동 재시도·원 결과 덮어쓰기 없음. 실패 결과 보존. controller는 torch-free, CPU 분석2threads. 두3step smoke 후 본학습. 예상1단계40~60분(진입 대기/보고서 제외). 같은 head+LoRA 성능 확인과 새 방법의 우월성을 구분한다. commit/push/시스템 설정 변경 없음.

평가 근거: [rolling forecasting origin](https://otexts.com/fpp3/tscv.html). 미래 target이 도착하기 전의 정보만 사용해야 한다. 최적화 대조만으로 분포 이동/표현 부족이라는 인과를 식별할 수는 없다.
