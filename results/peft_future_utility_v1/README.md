# 동일 상태에서 이후 LoRA 업데이트의 가치 진단

2026-09-11 완료. [전체 보고서와 논문 요건](../../_docs/notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md) · [실행 전 고정 프로토콜](../../experiments/peft_future_utility_v1/PURPOSE.md).

**현재 LoRA가 유익하다는 신호로 미래 업데이트의 이득을 판단하면 틀릴 수 있다. 그러나 이를 해결하는 새 정책은 아직 검증하지 않았다.** 모델·Adam·RNG가 같은 상태에서 JOINT, HEAD_ONLY, MASKED_UPDATE, STOP을 비교했다. 2원천×3훈련 구성×2seed×2분기 시점, 총 24분기이며 독립 표본 24개가 아니다. 기존 V 앞14개 S는 신호·선택에, 뒤14개 D는 진단에 사용했다. 두 origin의 간격은 target 중첩을 막지만 문맥 의존성은 남는다. 둘 다 이미 관측한 V이고 새 E 성능이 아니다.

```text
동일 최종 시점 JOINT 우세       7/24
동일 최종 시점 HEAD_ONLY 우세  16/24
작은 차이 구간                  1/24
둘 다 STOP보다 나쁨             9/24
두 seed의 방향 분류 일치       11/12쌍
현재 기여 양수지만 HEAD 우세   16/24
S 선택 후 같은 prefix로 복귀   21/24
```

우세는 실행 전 고정한 ±0.25%F0 밖의 차이이며 통계적 유의성 판정이 아니다. F0는 초기 손실이고 %F0는 그 값으로 정규화한 손실 차이다. 최종 평균 JOINT 이득은 -4.350%F0, S 선택 후 -0.727%F0다. MASKED 대조 후 원시 부호는 24/24 동일하지만 clipping 방식에 따른 차이는 최대1.140%F0로 남는다.

![현재 기여와 미래 가치](01_current_vs_future.png)

Jena의12분기는 모두 head만 지속하는 것이 joint 지속보다 좋지만, 그중8개는 둘 다 중단보다 나쁘다. 전체 상관 -0.726만 보고 현재 기여 신호를 역으로 쓰면 원천 차이에 과적합할 수 있다. BDG2/Jena 안의 상관은 각각 -0.161/-0.413이다.

![선택·clipping·중단](02_selection_clipping_stop.png)

다음 연구는 먼저 튜닝한 조기 종료·고정 동결을 넘어설 여지가 있는지 새 개발 기간에서 확인해야 한다. 남는 이득이 있을 때만 짧은 미래 반응 신호를 만들고, 측정 비용까지 포함하여 새 기간·원천에서 검증한다. 현재 발견만으로 방법 논문의 기여가 확보되지는 않았다.

## 결과와 근거

- [24행 전체 비교](metrics.csv), [고정 분석 집계](summary.json), [12조건 전체 학습 점수·선택·상태 확인 기록](histories.json).
- [독립 감사](evidence/independent_audit.json): 115개 hash, 1,264회 원시 점수 검산(중복 참조 포함), 최대 차이1.67e-16. 학습 prefix 및 분기 시작 예측 일치, S-only 선택 확인.
- [고정 계획](evidence/plan.json), [본실험 완료](evidence/completed.json), [환경](evidence/environment.json), [CPU 검사](evidence/cpu_checks.json), [13개 guard 기록](evidence/guard_statuses.json).
- [완료·자원·사후 기술 감사](completion_review.json), [지정 Windows 이벤트 조회](evidence/windows_event_audit.json), [6개 원본 동일 복사 내역](evidence/copies.json).

본실험 702.140초, S0 18.219초. 총 guard 시간660.108초는 S0를 포함한 자식 작업 시간 합이고, admission 등을 포함한 controller 시간과 정의가 다르다. 이 비용은 진단 전체의 비용이며 새 정책의 학습 절감 수치가 아니다. MASKED 대조는 계산을 줄이는 방법이 아니다.

13개 GPU 작업 모두 exit0,71개 측정의 최소 여유 RAM10.924GiB/commit11.617GiB, 최대 GPU1859MiB/55°C. 지정 Windows 오류 이벤트0건. 완료 감사 시 활성 학습·controller0, guard lock 없음. 과거 프리즈가 영구적으로 해결됐다는 주장은 하지 않는다.

## 코드와 재현 범위

[fit.py](../../experiments/peft_future_utility_v1/fit.py) · [run.py](../../experiments/peft_future_utility_v1/run.py) · [analyse.py](../../experiments/peft_future_utility_v1/analyse.py) · [independent_audit.py](../../experiments/peft_future_utility_v1/independent_audit.py) · [completion_review.py](../../experiments/peft_future_utility_v1/completion_review.py).

실행은 아래 순서였다. 완료 경로에 다시 실행하면 덮어쓰기를 거부한다. 새 재현에는 기존 study31 입력, 모델 캐시, 고정 환경, 별도 run 경로 설계가 필요하다.

```powershell
.\.venv-peft\Scripts\python.exe -m experiments.peft_future_utility_v1.run --prepare
.\.venv-peft\Scripts\python.exe -m experiments.peft_future_utility_v1.run --smoke-only
.\.venv-peft\Scripts\python.exe -m experiments.peft_future_utility_v1.run
.\.venv-peft\Scripts\python.exe -m experiments.peft_future_utility_v1.analyse
.\.venv-peft\Scripts\python.exe -m experiments.peft_future_utility_v1.independent_audit
.\.venv-peft\Scripts\python.exe -m experiments.peft_future_utility_v1.completion_review
```

원시 prediction NPZ, 모델·Adam·RNG snapshot, 개별 실행 로그는 로컬 `runs/peft_future_utility_v1`에 유지한다. 이 결과 폴더에는 GitHub에서 읽기 쉬운 점수·학습 기록·PNG·작은 증거 파일을 묶었으며 대형 원시 파일 전체를 복제하지 않았다. 이번 턴 commit/push는 하지 않았다.
