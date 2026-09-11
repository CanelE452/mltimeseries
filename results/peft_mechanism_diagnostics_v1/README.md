# R1–R3 mechanism diagnostics — 2026-09-10

완료: 새 R1 24 fits + R2 12 fits, 평가 24회, Hospital 저장 예측 CPU 진단. [최종 한국어 보고서](../../_docs/notes/tsfm_topics/07_research_direction/26_mechanism_diagnostics_20260910.md)에 수치의 분모, 해석 한계, 그래프와 다음 조건을 정리했다. 새 독립 test나 새 방법의 성공을 주장하지 않는다.

- `controlled/`: 이번 실제 통제 학습의 주결과. `summary.json`은 독립 점수·선택·checkpoint 감사, `metrics.csv`는 seed별 결과. `r1_matched_head.png`, `r2_placement.png`는 새 학습 그림이다. 작은 계약/선택/완료 기록도 복사 보관한다.
- `reference/`: 기존 Study20의 origin/seed/horizon/target 및 과거 context 특성 재분석. 새 같은-head 결과와 구분한다.
- `hospital_strength/`: 원 Hospital alpha 손실/미래 전이, V2만 사용한 월 제외 민감도, 안정성과 이미 노출된 E 손해의 사후 연결.
- `layer_scope/`: 과거 합성 Q00 모듈 대조의 CPU 재요약. 이번 R2 실데이터 위치 실험의 결과가 아니다.
- `followup_summary.json`, `saved_artifact_snapshot.md`: 중간 CPU 재분석 스냅샷. 새 학습 완료 상태는 `controlled/summary.json`과 최종 보고서를 따른다.

코드 위치: [실행 범위](../../experiments/peft_mechanism_diagnostics_v1/PURPOSE.md), [runner](../../experiments/peft_mechanism_diagnostics_v1/run_controlled.py), [fit/forecast](../../experiments/peft_mechanism_diagnostics_v1/controlled_fit.py), [독립 감사와 그림](../../experiments/peft_mechanism_diagnostics_v1/summarize_controlled.py).

저장된 로컬 run이 있을 때 다음 명령은 재학습 없이 결과를 다시 계산한다. `.venv-peft`는 이번 Windows 환경의 Python이다. 실행 당시 버전과 원 입력 hash는 `controlled/original_input_audit.json`에 있다.

```powershell
& .\.venv-peft\Scripts\python.exe -m experiments.peft_mechanism_diagnostics_v1.summarize_controlled
& .\.venv-peft\Scripts\python.exe -m experiments.peft_mechanism_diagnostics_v1.diagnose_reference
& .\.venv-peft\Scripts\python.exe -m experiments.peft_mechanism_diagnostics_v1.reference_components
& .\.venv-peft\Scripts\python.exe -m experiments.peft_mechanism_diagnostics_v1.diagnose_followup
& .\.venv-peft\Scripts\python.exe -m experiments.peft_mechanism_diagnostics_v1.hospital_month_sensitivity
& .\.venv-peft\Scripts\python.exe -m experiments.peft_mechanism_diagnostics_v1.stability_transfer
```

대형 checkpoint와 예측은 기존 Git 제외 경로 `runs/peft_mechanism_diagnostics_v1/`에 보존한다. GitHub의 작은 결과만으로 학습/원 예측 재계산에 필요한 모든 자료가 제공되는 것은 아니다. 이번 변경은 아직 commit/push하지 않았다.
