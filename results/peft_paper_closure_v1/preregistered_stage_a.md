# Stage A — 실행 전 계약 초안 (상류 게이트 대기, 미봉인)

[판정] 이 파일은 사용자가 승인한 연구 순서를 구체화한 **비실행 초안**이다. 데이터 source/target/기간을 metadata만으로 확정하고 evidence·novelty gate를 통과하기 전에는 preregistration 완료라고 부르지 않는다. 실제 상태와 미해결 항목은 동반 JSON 및 PAPER_READINESS를 따른다. 빈 값을 추측해서 채우거나 이 파일만으로 GPU를 시작하지 않는다.

## 목적과 고정된 비교

[판정] fresh source-period에서 internal adaptation의 WIDE 대비 추가 이득과 F0 대비 실제 가치, V에서 선택한 이득의 다음 D 전이를 같은 실험으로 측정한다. 후보 A가 primary다. 새 adapter, selector, rank sweep, Study31~33 threshold 재사용, Study36 DGP는 없다.

[확인] 재사용할 기존 코드 계약은 `experiments/peft_head_convergence_v1/{fit.py,run.py,PURPOSE.md}`, `experiments/peft_capacity_probe_v1/model.py`, `experiments/peft_decision_transfer_v1/panel.py`에서 읽었다. 기존 runner는 이전 study 경로·결과를 직접 참조하므로 새 manifest를 그대로 넣어 실행할 수 없다. 재사용은 불변 모델/metric 구성에 한하며 실행 전 새 출력 경로 및 data-access 경계를 별도로 검증해야 한다.

[판정] F0는 target adaptation 없음. HEAD는 frozen backbone+기존 residual MLP. JOINT는 동일 기본 head+기존 standard LoRA rank8, 기존12 block. WIDE는 backbone frozen+기존 WideHead. 기본 head/JOINT는 동일 초기 seed·sample stream. F0 identity는 step0 후보에 포함한다. head와 adapter의 함수공간이 같다고 주장하지 않는다.

[확인] 기존 4채널 L336/H48 Chronos-2 구성의 trainable parameter는 HEAD589,301 / WIDE1,768,949 / JOINT1,768,949다. WideHead는 input768→1601, hidden bias1,109개와 고정 zero492개, output1601→336을 사용해 수를 정확히 맞춘다. 새 데이터가 이 계약을 못 만족하면 이 수를 실측값인 것처럼 복사하지 않고 실행 전 재설계·재봉인한다.

## 데이터와 봉인 경계

[판정] Stage A 단위는 metadata와 노출 장부로 고른 두 개의 fresh source-period다. target 이름·timestamp 범위·sampling frequency·train/V/D 경계·origin stride·gap·최대 horizon을 JSON의 실제 값으로 확정해야 한다. 동일 정답 구간의 두 seed는 독립 데이터 2개가 아니다. 같은 원천의 인접 기간도 독립 source로 세지 않는다.

[판정] train/V는 개발, next D는 선택 후 한 번 평가하는 Stage A replication이다. D를 본 뒤에는 개발 노출 장부에 즉시 기록한다. 논문 final reserve는 이 D와 다르며 끝까지 열지 않는다. calibration 역할은 만들 필요가 없고, 기존 archive에 있어도 unused로 명시한다. 모든 target 선택은 target 값을 열기 전 확정하고 사후 coverage를 이유로 좋은 target으로 교체하지 않는다. 사전 QC 실패는 해당 cell의 data gate failure로 보고한다.

[판정] pretraining overlap UNKNOWN은 유지한다. timestamp/채널 metadata 확인은 target-value exposure와 구분하되 과거 실험에서 값이 fit/QC/EDA/forecast에 읽혔는지도 검사한다. 실행 전 실제 파일 hash와 final 접근 금지 경계를 고정한다.

## 예산·metric·선택

[판정] 기존 Study35 제한 recipe 재사용을 초안으로 고정한다. seed30000/30001, FULL90, 720step, HEAD/WIDE LR `[1e-5,3e-5,1e-4,3e-4]`, JOINT(head,LoRA) LR `[(1e-5,1e-5),(3e-5,3e-5),(1e-4,3e-5),(1e-4,1e-4)]`. AdamW wd0/foreachFalse, batch8/micro4/clip1/threads2. 파라미터 수 외에 global clipping의 최적화 결합도 한계로 보고한다.

[판정] 동일 checkpoint 기회 `[0,1,2,4,8,15,30,60,120,180,240,360,540,720]`와4 recipes를 각 family에 부여한다. V 최소 loss, 동률은 작은 step 다음 작은 recipe index. D 개봉 전에 모든 family/seed의 선택과 fit hash를 봉인한다. 최종 모델은 ensemble하지 않는다. selected step이 상한이어도 수렴을 주장하지 않는다.

[판정] 기존 native21-quantile pinball을 train-only target scale로 정규화하고 같은 target/valid mask/windows를 사용한다. origin/target/quantile 집계식을 실코드와 독립 NumPy 검산으로 일치시킨다. 지표는 각 기간의 F0로 나눈 %F0다. D 정보가 train scaling/imputation/selection으로 넘어가지 않도록 별도 archive와 접근 검사를 둔다.

[판정] 상한은 2units×2seeds×3families×4recipes=48 fits. 선택된12모델의 D forecast와 source-period unit별 F0 1회, 총2회(D truth 동일 seed 간 재사용 가능)가 기본 예측 task 상한14다. F0 V와 checkpoint V 예측은 fit 안에서 별도 카운트한다. smoke는 family별 train-only 한 번, 최대3 fits로 별도 기록한다. Full FT는 이번 Stage A에서 **실행하지 않음**으로 사전 고정하며 기존 reference를 historical context로만 쓴다. Stage A 뒤 Full FT가 필요해도 별도 계약 없이는 자동 추가하지 않는다.

[판정] 실행 수는 목표가 아닌 상한이다. actual GPU fit/forecast/model-call count, selected step, recipes, parameter count, per-job wall-clock와 end-to-end wall-clock, peak allocated/reserved GPU memory 및 장치 전체 sampled memory를 구분하여 기록한다. 파라미터 수로 시간 절감을 추론하지 않는다.

## Gate와 불확실성

[판정] G1: 한 fresh unit 이상에서 WIDE−JOINT가 두 seed 모두 양수, 평균≥0.5%F0이고 JOINT가 F0도 이겨야 한다. F0 조건은 두 seed 모두 양수로 보수적으로 구현한다. WIDE만 이기면 `INTERNAL_EFFECT_WITHOUT_ADAPTATION_VALUE`. G1 실패이면 새 adapter/selector와 Stage B는 STOP.

[판정] G2A: 두 독립 source-period에서 V gain≥0.5%F0, D gain≤0. source별 seed 평균을 주 unit로 보고 두 seed 각각의 부호도 숨기지 않는다. G2B: 평균 V→D gain 감소≥0.5%F0와 paired time-block bootstrap interval의0 제외. 두 분모는 각 기간 F0로 재계산하고 source는 동일 가중, source 내 seed는 함께 재표집한다. 두 기간 사이 origin 대응 규칙·block 길이·interval 수준은 timestamp 배열과 실제 gap 확인 전 미정이다. 따라서 G2B는 현재 실행 가능하지 않다. 결과를 본 뒤 값을 고르는 것은 금지한다.

[판정] uncertainty에는 seed별 raw loss, paired contrasts, unit별 효과를 우선 보고한다. bootstrap 반복은 독립 표본수가 아니며 overlapping horizon은 하나의 block으로 묶어 의존을 보존한다. 데이터별 최소 block 수를 만족하지 못하면 interval 미측정으로 남기며 source 일반화 CI로 바꾸지 않는다.

[판정] G3: 전부 JOINT 우세→stable LoRA value, 조건별 효용 차이→unstable utility, 전부 F0 우세→no PEFT need 및 method STOP. Stage B/C는 부모 claim gate의 진입 조건과 별도 preregistration을 필요로 한다.

## 중단과 보존

[판정] `EVIDENCE_MISMATCH`, 미확정 clean data, novelty 중복/필수 원문 검토 미완료, source/input hash 변경, 선택 봉인 위반, metric 불일치이면 GPU 시작/후속 실행을 중단한다. 실패 receipt를 보존하고 기존 artifact를 덮어쓰지 않는다. data 결과·D에 맞춘 LR/step/rank/threshold/period 변경은 금지한다.

[확인] 기존 자원 계약은 입장 available commit≥13GiB/RAM≥5GiB, runtime emergency commit≥6GiB/RAM≥5GiB다. [판정] 이를 유지하고 single GPU serial execution, 기존 timeout900초/job을 적용한다. OS·보안 설정 변경이나 다른 프로젝트 종료로 우회하지 않는다. 실행 전 실제 자원 및 소유 training process를 다시 확인한다.

[미검증] 현재는 실행 가능한 데이터 manifest, source/input freeze, CPU data-boundary 검증과 independent fit/forecast 감사가 완료되지 않았다. 따라서 raw metric table/uncertainty/실측 training memory를 만들지 않았다. 상태가 바뀌면 초안을 덮어쓰지 않고 versioned frozen manifest와 hash를 추가한다.
