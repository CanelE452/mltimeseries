# Study34 — Initial LoRA headroom

2026-09-11 완료. [한국어 분석·그래프·다음 검증](../../_docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md).

G1 통과: 강한 non-JOINT 대비 두 seed에서 +0.25%F0 넘는 이득이4조합 중3조합. G2 미통과: oracle의 고정 JOINT 대비 추가 이득 +0.0408/+0.0756%F0. 평균 JOINT 이득은 F0 +1.418/HEAD +2.037/WIDE +2.428/보정 +1.455%F0. BMRA SPREAD30은 F0보다 나쁘다. 두 기존 원천의 다른 과거 기간을 사용한 개발 진단이며 새 방법 확증이 아니다. 일부 head가 학습 상한을 선택하여 수렴 문제는 남아 있다.

![초기 LoRA 이득](01_g1_initial_lora_gap.png)

## 산출물

- [summary.json](summary.json): 고정 G1/G2, 평균, 조건부 bootstrap 구간.
- [metrics.csv](metrics.csv): 8조건 × 5방법 = 40행 D 성능.
- [trajectory.csv](trajectory.csv): 96개 LR 후보의 V 선택과 비용.
- [per_example_losses.npz](per_example_losses.npz): origin·target·quantile 손실 분자, 관측 수, scale, F0, row key. 원 예측·checkpoint가 아니라 작은 점수 재현 자료다.
- [selection_sealed.json](selection_sealed.json): D 평가 전 선택 봉인.
- [completion_review.json](completion_review.json): 123 guard, 자원, 출력 차이, 학습 상한, target별 차이.
- [그림2](02_family_heatmap.png), [그림3](03_validation_cost.png), [그림4](04_selected_trajectories.png), [그림5](05_quality_differences.png).
- [근거 복사 목록과 SHA256](evidence/copies.json), [전체 V 기록](evidence/validation_histories.json), [guard 기록](evidence/guard_statuses.json), [독립 감사](evidence/independent_audit.json), [Windows 조회](evidence/windows_event_audit.json).

## 재현 범위

저장소 루트의 기존 환경에서 가벼운 독립 테스트:

```powershell
.venv-peft/Scripts/python.exe experiments/peft_initial_headroom_v1/independent_audit.py --self-test
```

전체 감사는 `independent_audit.py --run`이며 기존 출력 덮어쓰기를 거부한다. 새 감사 출력 위치로 분리하려면 `--run-dir`/`--results-dir`의 원본 구조 계약을 먼저 확인해야 한다. 완료한 run/prepare/completion_review를 원 경로에 재실행하지 않는다.

전체 재학습·원 예측 감사에는 Git 제외된 runs, 준비 archive·원자료, Chronos-2 checkpoint, [환경](evidence/environment.json), 이전 실험 공통 코드가 필요하다. 이 폴더만으로 모델 학습 전체가 재현된다고 주장하지 않는다. [실행 계약](../../experiments/peft_initial_headroom_v1/PURPOSE.md), [데이터 계약](../../experiments/peft_initial_headroom_v1/PURPOSE_DATA.md), [소스46·입력5 해시](evidence/plan.json)를 보존했다.

첫 감사의 [실패 기록](evidence/independent_audit_failure.json)은 초 단위 완료 시각과 소수 초 guard 시각의 정밀도 차이다. 마지막 경계와 실제 파일mtime 대조 후 통과했고 동결 코드·결과는 불변이다. 학습은 종료했으며 이번 작업에서 commit/push하지 않았다.
