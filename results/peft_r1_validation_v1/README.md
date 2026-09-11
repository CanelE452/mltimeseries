# R1 후속 검증 — 학습 원점 구성과 과거 잔차 보정

**완료.** [최종 한국어 결과·해석](../../_docs/notes/tsfm_topics/07_research_direction/27_r1_coverage_residual_validation_20260910.md). 새16 fits/16 모델 평가, 과거 잔차 보정 진단 완료. CPU assertion 실패1회는 별도 `residual_dtypefix.py`/`recover.py`로 계산 정밀도만 보정해 마쳤다. 원 소스·실패 로그·보정 선택·학습 결과를 유지했으며 재학습은 없었다.

이 묶음은 [고정 실행 범위](../../experiments/peft_r1_validation_v1/PURPOSE.md)의 결과다. 완료 여부는 `summary.json`의 `completed`와 로컬 `runs/peft_r1_validation_v1/completed.json`으로 확인한다. 기존 FULL90 결과 8 fits는 재사용하고, SPREAD30/RECENT30의 MLP/ALL 두 seed·두 원천에 대해 새 16 fits/16 평가를 실행한다. 기존 E는 이미 노출됐으며 독립 test가 아니다.

- `summary.json`: 독립 점수·입력·선택·checkpoint 감사, 조건별 효과, 실행 자원.
- `metrics.csv`: 전체 24개 모델 평가(기존 FULL90 8 + 새16)의 seed별 점수·checkpoint·시간.
- `origin_coverage.png`: 조건별 F0 효용과 같은 head 대비 LoRA 추가 이득.
- `validation_learning_curves.png`: 200 updates 동안 V 손실 변화. 실선/점선은 두 seed다.
- `training_coverage.png`, `coverage_audit.json`: 학습 target 시간의 중복·공백과 실제 관측 범위.
- `residual_correction.png`, `residual/selection.json`, `residual/evaluation.json`: 과거 OOF, V 선택, E 기술 평가. Ridge가 나쁘다는 결과는 모든 단순 보정의 불가능성을 뜻하지 않는다.
- `evidence/`: 작은 실행 계약·선택·완료·입력 감사 기록. 대형 checkpoint와 학습 캐시는 기존 정책대로 로컬 runs에 둔다.

실행 코드는 [run.py](../../experiments/peft_r1_validation_v1/run.py), [subset_fit.py](../../experiments/peft_r1_validation_v1/subset_fit.py), [F0 cache](../../experiments/peft_r1_validation_v1/cache_f0.py), [residual.py](../../experiments/peft_r1_validation_v1/residual.py)다. 원 같은-head 모델 코드는 변경하지 않았다. `run.py`는 기존 run 디렉토리가 있으면 거부한다. 동결된 residual helper를 직접 재실행하면 결과 파일을 덮어쓸 수 있으므로 기존 산출물 위에서 재실행하지 않는다.

전체 로컬 run이 완료된 뒤 다음 명령으로 학습 없이 독립 감사와 그림을 다시 생성할 수 있다.

```powershell
& .\.venv-peft\Scripts\python.exe -m experiments.peft_r1_validation_v1.audit_inputs
& .\.venv-peft\Scripts\python.exe -m experiments.peft_r1_validation_v1.analyse
```

GitHub의 작은 산출물만으로 대형 모델·원자료·예측을 모두 재생성할 수 있는 것은 아니다. 이 작업에서 commit/push는 하지 않았다.
