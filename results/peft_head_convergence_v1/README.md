# Study35 — head 예산 대조와 적응의 기간 간 손해

2026-09-11 완료. [분석 보고서·다음 방향](../../_docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md) · [사전 계획](../../experiments/peft_head_convergence_v1/PURPOSE.md).

**G1 미통과.** L720 JOINT는 동일 총용량 WIDE보다 평균8.804%F0 좋지만, 학습하지 않은 F0보다4.305% 나쁘다. 네 조건 모두 같은 부호이며 S180에서도 F0 대비 손해였다. head가 더 학습해서 LoRA를 따라잡은 결과가 아니다. 선택된12개 학습 모델은 모두 train/V 개선과 D 악화를 함께 보였다. 조건부 StageB는 미실행했으며 G2/G3는 미측정이다.

![F0를 포함한 선택 모델 비교](03_family_scores.png)

## 결과 파일

- [metrics.csv](metrics.csv): 두 원천×두 seed×두 예산×5방법,40행. D_score는 유효 target의 2×pinball 손실을 train 표준편차로 나누고 채널·분위에 대해 평균한 값이다. D_over_F0는1보다 작아야 F0보다 좋다.
- [summary_A.json](summary_A.json): 사전 G1, 각 조건의 결과, 개발 평가임을 표시.
- [per_example_losses.npz](per_example_losses.npz): pinball_numerator `[40,20,2,21]`, valid_target_count `[40,20,2]`, scale `[40,2]`, F0 및 행 키. 손실 재계산과 대응 재표집 근거.
- [fit_histories.json](fit_histories.json):48개 학습 경로 전체의 train/V, 선택 step/LR, 시간·의존성 기록.
- [intervals.json](intervals.json): 원천 기간 고정,2000회 paired circular block bootstrap의 조건부90% 구간. 새 원천 일반화 구간이 아니다.
- [descriptive_decomposition.json](descriptive_decomposition.json): 사후 train/V→D 부호, LoRA 손해의 채널·origin 기여. 모델 선택이나 판정을 바꾸지 않았다.
- [completion_review.json](completion_review.json): 완료 상태, 자원 표본, 이전 실험 보호 해시, 예산 차이와 비용 범위.
- [completed.json](completed.json): 결과 생성 완료 표시. 실제 주 실행의 완료 시각·시간은 [evidence/completed.json](evidence/completed.json).

## 그래프

1. [01_budget_gap](01_budget_gap.png): head 대비 LoRA 차이. 이것만으로 F0보다 좋다고 해석하면 안 된다.
2. [02_validation_trajectories](02_validation_trajectories.png): 선택된 L720 LR의 전체 검증 곡선과 반환 checkpoint.
3. [03_family_scores](03_family_scores.png): F0를 포함한 전체 성능.
4. [04_train_trajectories](04_train_trajectories.png): 같은 경로의 학습 손실.
5. [05_budget_effect](05_budget_effect.png): 같은 기간에서180→720 예산의 개발 손실 변화.
6. [06_validation_development](06_validation_development.png): V 개선과 D 손해의 부호 불일치.
7. [07_target_origin_diagnostics](07_target_origin_diagnostics.png): 전체 손실 차이에 대한 채널·origin 기여. origin별 자체 상대 손실률과는 다르다.

## 재현 근거와 실행 범위

48 fit +24 forecast +3 smoke =75 guard 전부 exit0. 본실행4546.778초(75.78분), 최초 smoke·CPU 준비 제외. StageB 실행0. 가용 RAM 표본 최저14.275GiB, commit13.430GiB, GPU 표본 최고1828MiB/58°C. 종료시 해당 학습0/guard lock없음.

[고정 plan](evidence/plan.json) SHA256 `33959c5c95e8a2cc91e7d278150bf008cfeb5d210025f7fc52dcda18ded19406`. Source52/input7 및 Study34/33 보호 해시 일치. [선택 봉인](evidence/selection_A.json), [독립 감사](evidence/independent_audit.json), [75 guard](evidence/guard_statuses.json), [환경](evidence/environment.json), [복사 해시](evidence/copies.json), [Windows 이벤트 조회](evidence/windows_event_audit.json).

데이터는 Jena2018/BMRA2017의242일 블록이다. train90/V30/cal20/D80 origin 중 D는 매4번째20개, cal은 미사용. 관측량 기준으로 BMRA target을 E_BRYBW-1/E_BURBO로 바꿨으므로 Study34와 같은 target 재현이 아니다. Jena2020은 기존 실험의 원시값 노출을 발견해 GPU 이전 제외했다. [데이터 요약](evidence/prepared_summary.json), [노출 검토](evidence/prepared_data_audit.json), [개별 창 QC](evidence/window_qc.json), [공식 데이터 다운로드](evidence/download_receipt.json), [사전 검토 기록](evidence/preflight_review.json).

실제 학습·평가 NPZ와 checkpoint는 `runs/peft_head_convergence_v1`에 보존되어 있다. 아래 검산은 이 로컬 실행 파일들이 필요하다. 이미 만들어진 결과에 학습/분석 명령을 다시 실행하면 exclusive-create 보호로 중단된다.

```powershell
.venv-peft/Scripts/python.exe -m unittest experiments.peft_head_convergence_v1.test_contract
.venv-peft/Scripts/python.exe experiments/peft_head_convergence_v1/independent_audit.py --self-test
```

full 독립 감사·완료 설명·사후 분해는 이미 실행 완료했다. 각각 `independent_audit.py`, `completion_review.py`, `completion_diagnostics.py`이며 frozen 학습 경로에 포함되지 않는다. 이번 정리는 로컬 파일 기준이고 commit/push하지 않았다. 새 최종test도 열지 않았다.

[최종 산출물 검사](artifact_review.json)는 문서 링크·7개 PNG·배열 점수·해시를 확인한다. 설명 그림06/07은 최초 생성 후 라벨만 개선했으므로 descriptive_decomposition.json의 PNG bytes는 최초 렌더링 기록이고 최종 파일 해시·크기는 이 검사에 남긴다. 수치 JSON은 보존했다.
