# Fresh Stage A Candidate Manifest

[확인] 이 문서는 target loss, forecast performance, GPU 학습 없이 작성한 다음 Stage A 후보 manifest다. 목적은 Household post-P1과 BDG2 alternate building/period가 L336/H48 fresh Stage A 개발 단위로 바로 실행 가능한지 판정하는 것이다. 이 문서는 sealed final reserve가 아니며, Chronos/기타 foundation-model pretraining overlap은 두 후보 모두 UNKNOWN으로 둔다.

## 공통 split 계약

[판정] 이번 manifest는 calibration을 만들지 않고 `train / V / E1 / E2`를 직접 분리한다. 기존 `train90/V30/cal20/eval80` 계약과 다르므로, 실행 전 이 계약을 Stage A plan에 그대로 freeze해야 한다.

```text
context                 336 h
horizon                  48 h
origin stride            24 h
precontext               14 d
train                    91 d -> 90 origins
embargo train -> V        2 d
V                        31 d -> 30 origins
embargo V -> E1           2 d
E1                       41 d -> 40 origins
E2                       41 d -> 40 origins
E1 -> E2 gap              0 h, half-open target windows touch but do not overlap
```

[판정] READY 판정에는 source hash, target-blind period/series 고정, 기존 PEFT target-label exposure 비중복, split 가능성, split별 target finite fraction >=70%, 각 48h target origin window finite fraction >=70%, 선택 채널 nonconstant를 요구했다.

## 후보 판정

```text
candidate                         status    reason
Household post-P1 earliest block   BLOCKED   길이와 평균 target coverage는 충분하지만 train 안의 48h origin window 하나가 0 finite target fraction이다.
BDG2 Bull Office 2016A             READY     Bull office target-label exposure가 기존 PEFT 장부에 없고, train/V/E1/E2 모든 origin availability gate를 통과했다.
```

## Household post-P1

[확인] 원천은 `data/uci_household_power/household_power_consumption.txt`, SHA256 `4259c9d7ece5dbee9ab8d53682baac68d791c864f0f64a52b4043cb3b90894b7`이다. source timestamp range는 `2006-12-16T17:24:00`부터 `2010-11-26T21:02:00`까지다.

[확인] 기존 Household P1 exposure는 `2008-08-30~2009-04-29`로 장부에 기록되어 있다. 후보는 P1 end-exclusive 직후인 `2009-04-29T00:00:00`에서 시작해 `2009-12-07T00:00:00`에 끝나므로 local PEFT target-label timestamp overlap은 없다.

```text
precontext          2009-04-29T00:00:00 -> 2009-05-13T00:00:00
train               2009-05-13T00:00:00 -> 2009-08-12T00:00:00
embargo_train_val   2009-08-12T00:00:00 -> 2009-08-14T00:00:00
V                   2009-08-14T00:00:00 -> 2009-09-14T00:00:00
embargo_val_e1      2009-09-14T00:00:00 -> 2009-09-16T00:00:00
E1                  2009-09-16T00:00:00 -> 2009-10-27T00:00:00
E2                  2009-10-27T00:00:00 -> 2009-12-07T00:00:00
```

[확인] target은 `Global_active_power`, `Global_reactive_power`; context-only channel은 `Voltage`, `Global_intensity`다. Hourly aggregation은 기존 Household parser와 맞춰 channel별 finite minute가 45개 이상인 hour만 유지했다.

[확인] aggregate target finite fraction은 train `0.9741`, V `1.0000`, E1 `1.0000`, E2 `1.0000`이다. 그러나 train의 최소 48h origin-window finite fraction은 `0.0000`이다. Post-P1 daily-shift scan에서도 local source end까지 full 222-day block 중 per-origin 70% gate를 통과하는 시작점을 찾지 못했다.

[판정] 이 후보는 BLOCKED다. 해결하려면 성능을 보지 않은 상태에서 origin-exclusion/gap 정책 또는 더 짧은 split을 먼저 freeze한 뒤 manifest를 다시 만들어야 한다.

## BDG2 Bull Office 2016A

[확인] 원천은 `data_external/bdg2_coarse_supervision_v1/raw/electricity.csv`, SHA256 `039d909d8981e2d69eaeb366144e6ab7e84fa5e7e216aee42bddd95384a66418`이고, metadata는 `data_external/bdg2_coarse_supervision_v1/raw/metadata.csv`, SHA256 `992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02`이다.

[확인] 기존 BDG2 PEFT exposure는 Eagle/Lamb coarse supervision과 Eagle 2017 contribution/future diagnostics다. Bull office meters는 장부상 기존 PEFT target-label unit이 아니다. 다만 BDG2 raw source family 자체는 이미 로컬에서 inspect되었으므로 final holdout으로 부르지 않는다.

[판정] selection rule은 target-blind다. Eagle/Lamb를 제외하고, prior BDG2 office diagnostics와 가까운 Office usage를 우선했다. site_id 알파벳 순서에서 first four metadata-order office electricity meters를 검사했고, Bobcat은 availability 실패, Bull은 첫 READY site였다.

```text
site                Bull
usage               Office
targets             Bull_office_Lilla, Bull_office_Hilton
context-only        Bull_office_Myron, Bull_office_Nicolas
block               2016-01-01T00:00:00 -> 2016-08-10T00:00:00

precontext          2016-01-01T00:00:00 -> 2016-01-15T00:00:00
train               2016-01-15T00:00:00 -> 2016-04-15T00:00:00
embargo_train_val   2016-04-15T00:00:00 -> 2016-04-17T00:00:00
V                   2016-04-17T00:00:00 -> 2016-05-18T00:00:00
embargo_val_e1      2016-05-18T00:00:00 -> 2016-05-20T00:00:00
E1                  2016-05-20T00:00:00 -> 2016-06-30T00:00:00
E2                  2016-06-30T00:00:00 -> 2016-08-10T00:00:00
```

[확인] aggregate target finite fraction은 train에서 `0.9968 / 0.9949`, V에서 `1.0000 / 0.9972`, E1에서 `1.0000 / 1.0000`, E2에서 `0.9948 / 0.9948`이다. split별 최소 48h origin-window finite fraction은 train `0.9583`, V `0.9583`, E1 `1.0000`, E2 `0.9375`다. 네 선택 channel은 precontext/train/V/E1/E2에서 nonconstant다.

[판정] 이 후보는 READY다. 실행 전에는 이 manifest를 Stage A plan input-selection artifact로 freeze하고, E1/E2 loss를 열기 전에 모델·selection·metric 규칙을 먼저 봉인해야 한다.

## 전체 상태

[판정] 현재는 `PARTIAL_READY_ONE_OF_TWO`다. BDG2 Bull Office는 준비됐지만 Household post-P1은 gap/exclusion 정책 없이는 READY가 아니다. Fresh Stage A를 두 source-period로 시작하려면 Household rescue manifest를 다시 만들거나, 같은 수준의 target-blind 후보 하나를 추가로 준비해야 한다.
