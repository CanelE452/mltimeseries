# 17. 집계 감독 PEFT를 위한 BDG2 자료 진입 계획

2026-09-08. 사용자가 승인한 후보 실험→실패 분석→다음 실행의 지속 탐색에 따른다. 16번은 알려진 Gaussian 해법의 충분성을 확인했으며 새 PEFT 방법을 확보하지 못했다. 이번에는 실제 다건물 패널에서 target의 미세 과거를 학습 입력에서 차단할 수 있는 자료를 확보한다. 본 문서는 전력 값 다운로드 전 고정하는 자료 계약이며, 방법 성능의 성공 기준이나 완료된 학습 결과가 아니다.

## 선택과 선행 경계

월별 검침 고객과 별도 시간별 계량 표본을 이용하는 문제는 [PGE 운영 연구](https://www.aceee.org/files/proceedings/1998/data/papers/0625.PDF)에 근거한다. 실제 청구량 공개 지연·계량 날짜까지 확보한 배포 재현은 아니며, 공개 시간별 관측으로 월별 접근 제한을 구성하는 통제 실험을 준비한다.

로컬 Electricity321은 날짜·고객 ID의 원본 대응이 미확인이고 원 Electricity370은 Chronos 사전학습 원천이다. Ausgrid는 현재 공식 과거 주소가404였다. 따라서 공식 코드·metadata·원 meter와 LFS SHA가 확인되는 [Building Data Genome2](https://github.com/buds-lab/building-data-genome-project-2)를 선택한다. 전체 Zenodo archive595MB 대신 raw electricity174MB와 metadata272KB만 받는다. [공식 Zenodo](https://zenodo.org/records/3887306)는 CC-BY-4.0을 명시한다. Repository의 소프트웨어 라이선스와 데이터 이용 조건을 혼동하지 않는다.

[GIFT Table14](https://arxiv.org/html/2410.10393v2)는 BDG2 Panther/Fox/Rat/Bear/Hog/Bull/Cockatoo를 pretraining 목록에 포함한다. 이 사이트들은 제외한다. 남은 metadata에서 `electricity=Yes`, `primaryspaceusage=Office`가 많은 순으로 Eagle40개, Lamb17개, Robin17개다. 동률은 이름순이므로 Eagle/Lamb을 값 조회 전에 선택한다. 각각 US/Eastern, Europe/London이다. [Chronos2 AppendixA](https://arxiv.org/html/2510.15821v1)의 Buildings900K와 BDG2를 같은 자료라고 취급하지 않는다. 정확한 가중치의 전체 학습 manifest를 확인한 것은 아니므로 선택 사이트의 비중복은 여전히 UNKNOWN이다.

## 고정 소스와 크기

공식 repository revision: `9b97ccbe90096aff42ed4fd6493bf7ae692d7118`.

- Metadata: `https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/9b97ccbe90096aff42ed4fd6493bf7ae692d7118/data/metadata/metadata.csv`
  - bytes272024, SHA256 `992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02`.
- Raw electricity: `https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/9b97ccbe90096aff42ed4fd6493bf7ae692d7118/data/meters/raw/electricity.csv`
  - bytes174239039, SHA256 `039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418`.

[공식 meter 설명](https://github.com/buds-lab/building-data-genome-project-2/wiki/Meters-data-features)에 따라 raw electricity의 단위는 kWh, timestamp는 local time이다. Cleaned는 outlier/long zero 등을 NaN으로 바꾼 자료라 전체 기간을 이용한 전처리 영향을 피하기 위해 이번에는 쓰지 않는다. DST 처리와 구간 시작/끝 의미는 완전히 확인되지 않았으므로 우선 원 local-naive hourly 격자에서 검사한다. 이를 실제 utility 청구 원장이나 확정된 UTC 에너지 적분이라고 주장하지 않는다.

## 값 조회 전 고정한 선택 규칙

고정 metadata에서 두 사이트의 모든 Office/electricity 후보 ID를 사전 기록한다. 전력 파일은 선택 후보 열만 chunk로 파싱하고, 2016년 행만 완전성 판단에 쓴다. 2017년 값은 이 단계에서 수치 통계·모델 선택·대체 후보 선택에 사용하지 않는다. 전체 원 파일의 bytes/SHA 계산은 수치 라벨 이용과 구분한다.

첫 진입 기준은2016년 local-naive 8,784개 시간 격자에서 finite·비음수 값이 전부 있고 연간 합이 양수인 건물이다. 이 엄격한 기준은 월 label을 일부 관측의 평균으로 대체하지 않기 위한 것이다. 건물별 결측·음수·0 개수와 자격 여부를 모두 기록한다. 전력0은 자동 결측으로 바꾸지 않지만 실제 비가동인지 계량 문제인지 검증했다고도 주장하지 않는다.

각 사이트에서 자격 ID를 이름순으로 나열한다. `n=min(8, floor(eligible_count/2))`로 정하고 처음 n개를 donor, 다음 n개를 target으로 고정한다. n이4보다 작으면 해당 두 사이트의 현재 완전월 진입을 중단하고 원인을 기록한다. 기준을 실행 후 낮추거나 2017년을 보고 사이트를 바꾸지 않는다. 남은 ID는 사용하지 않는다. 최대16donor/16target, 최소8donor/8target의 작은 필요성 검사에 해당한다.

전체 행의 timestamp 열은2016–2017 원 격자·중복·간격 확인에만 사용한다. 서로 다른 local timezone의 같은 naive timestamp를 동시 UTC 관측이라고 묶지 않는다. 학습 전처리에서 target의 fine 표준편차·미세 profile·세부 패턴을 추출하지 않는다. 이번 진입 산출물에는 선정 target의2016년 월합·유효 개수만 보존하며 donor의 세부 값은 후속 입력 설계에서 별도로 사용한다.

## 후속 모델 실험의 방향

자료 진입이 통과하면2016년 월별 train,2017년1–3월 월별 validation,4–6월 시간별 평가,7–12월 이후 확인 구간을 분리하는 계획을 별도로 동결한다. 이때 target의 fine 과거는 proxy context에도 넣지 않는다. 과거에 알려진 월별 수준과 donor의 과거 calendar profile로 같은 입력을 만들어 모든 FM 대조에 제공한다. 실제 미래 donor 소비·미래 날씨·예측하는 달의 실제 월합은 입력하지 않는다.

검토하는 대조는 PROFILE, 동결 FM 점예측, 월평균 회귀＋동결 fine 패턴의 일정 수준 보정, 동결 hidden의 ridge point head, head를 고정한 attention LoRA다. Raw21출력의 평균을 쓴다면 고정 점예측 함수라고 정의하고 조건부 평균이나 월합의 분위수로 부르지 않는다. 월평균을 제거한 fine 패턴 오차로 내부 적응이 단순 수준 보정을 넘는지도 검사한다. 일반 LoRA의 양성 결과만으로 새 방법이 완성된 것은 아니다.

기발한 대안은 donor의 fine label로 ‘집계 gradient에서 fine 업데이트를 예측’하는 meta-adaptation prior다. 그러나 집계 nullspace의 정보를 공짜로 복원할 수는 없고, 기존 meta-learning·역문제 prior와 구분해야 한다. 먼저 단순 대조가 해결하지 못하는지 확인하므로 이번 자료 단계에서 이 방법을 구현·학습하지 않는다. Hard nullspace anchoring만으로 새 방법이라고 주장하지 않는다.

## 실행·보존

`data_external/bdg2_coarse_supervision_v1/`에 prefetch contract, 두 raw 파일, 응답 metadata, train-only QC·선택 결과를 둔다. 코드 namespace는 `experiments/peft_coarse_supervision_v1/`, 실행은 `runs/peft_coarse_supervision_v1/data_entry_guard`다. 계약에는 본 계획·실행 소스 SHA와 위 정확한 두 LFS SHA/크기를 기록한다. 다운로드 전에 계약을 기록하고 일치하는 완료 파일만 재사용한다. Partial·실패 기록을 삭제하거나 다른 파일로 덮지 않는다.

Root가 기존 shared guard CPU모드로 순차 다운로드·QC 한 작업만 실행한다. request30초/전체300초/guard330초, 파일당200MiB/총201MiB 상한을 둔다. CPUthreads2, 기존RAM여유5GiB/commit6GiB/childRSS8GiB/Git32 기준을 유지한다. 예상은 수십 초~수분이다. 새GPU작업·패키지설치·시스템설정변경은 없다. 자료가 통과하면 독립검산을 거쳐 실제 학습 계획과 S0로 진행하고, 미통과하면 자료 제약을 연구 가설의 음성 결과로 바꿔 부르지 않는다.
