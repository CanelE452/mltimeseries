# 시계열 FM 연구 노트

**완료: [35번 head 학습량·성분 개입 검증](07_research_direction/35_head_convergence_20260911.md).** 180→720step 대조 완료. LoRA는 동일 용량 head보다 평균8.804%F0 좋지만 F0보다4.305% 나빠 G1 미통과. head가 따라잡은 결과는 아니며,12개 선택 모델 모두 train/V 개선과 D 악화. 조건부 head재학습·warm-start 미실행.75 GPU 실행·독립 검산 완료,학습 종료.

**완료: [34번 초기 LoRA 이득 진입 검사](07_research_direction/34_initial_headroom_20260911.md).** 초기 LoRA 이득 G1은 4조합 중3조합 통과, controller 여지 G2는 +0.0408/+0.0756%F0로 미통과. 평균 동일 용량 head 대비 +2.428%F0, F0 대비 +1.418%F0. 일부 head의 학습 상한 선택으로 수렴·표현 원인은 미확정. 123 GPU 실행·독립 검산·5개 그림 완료, 학습 종료.

**완료: [33번 시기 전이·신호 제거·실제 비용 검증](07_research_direction/33_decision_transfer_20260911.md).** 177 GPU 실행·독립 검산 완료, 세 기준 미통과. 후보는12/12 STOP과 같은 모델을 반환했고 조기 종료보다20.1~33.7% 느렸다. 동일 용량 head 대비 평균 LoRA 이득+.944%F0는 Jena/BMRA에서 부호가 다르다. 4개 그래프·전체 결과·실패 원인과 다음 연구 순서 정리; 이번 학습 종료.

**완료: [32번 미래 LoRA 업데이트 가치 진단](07_research_direction/32_future_utility_20260911.md).** 12조건·24분기와 독립 검산 완료. 현재 기여가 양수여도 미래 업데이트가 불리한 사례16개, checkpoint 선택 후21/24 같은 결과. 새 방법의 우월성은 미확보; 결과·그래프·논문 요건 정리.

**완료: [31번 단일 LoRA 적응량 검증](07_research_direction/31_contribution_freeze_20260911.md).** 112GPU실행정상,독립검산완료. 기여기반동결두seed 비용기준실패;고정동결약20%시간절감·평균성능우세. 그래프·전체기록·반례·다음검증방향 정리.

**최신: [30번 동일 파라미터 head·짧은 적응 반응](07_research_direction/30_capacity_probe_20260910.md) 완료.** 같은학습파라미터수에서도 P1 Bike FULL90 LoRA 추가이득 +4.779%F0. 짧은반응은효용크기단서를주지만현재이진선택은항상LoRA보다정확하지않고시간도늘었다. 새60학습/24평가/1smoke 종료,새방법기여는미확보. 아래는 이전 이력이다.

**[28번 최적화 대조](07_research_direction/28_optimization_control_20260910.md)·[29번 새 시기 검증](07_research_direction/29_overlap_transfer_20260910.md) 완료.** 새 시기 LoRA 추가 이득은12개 대응 비교 모두 양수였지만, 정적인 중복 비율 선택 규칙은 시간 약12% 절감에도 정확도 기준을 실패했다. 과거 구성 규칙보다 현재 적응 반응의 저비용 측정이 후속 가설이며 아직 새 방법의 성공은 아니다. 현재 학습은 종료됐고 아래는 이전 이력이다.

**최신: R1 학습 구간 구성·잔차 보정 검증 완료.** [27번 결과](07_research_direction/27_r1_coverage_residual_validation_20260910.md): 같은30개 원점에서도 분산/최근 선택에 따라 Bike LoRA 추가 이득이 −1.019/+1.267%F0로 달라졌다. 최적화·최근성·자료 구성을 아직 분리하지 못했고 새 독립 test는 없다. 새16 fits/16 평가 종료, 진행 중인 학습 없음.

**현재 상태: 2026-09-10 R1–R3 통제 실험·분석 완료.** [최신 결과와 그래프](07_research_direction/26_mechanism_diagnostics_20260910.md): 새 36 fits/24 평가. 같은 head 대비 내부 LoRA 추가 이득은 Bike 4.171%F0, Household .418%F0. 층 선택의 동일 예산 대비 우위는 미확보이고 Hospital 개인화 실패 원인은 부분적으로만 좁혔다. 진행 중인 학습은 없다. [앞선 Hospital 본결과](08_hospital_shared_strength/24_hospital_results_20260910.md)도 보존한다.

먼저 [지금까지의 과정·결과·판단](research_review_20260910/README.md)을 읽는다. 실제 결과 그래프 5개, 실험 관계도, 실패 원인과 미검증 항목이 포함돼 있다. 다음 행동은 [개선 방향과 재개 조건](research_review_20260910/next_steps.md)에 정리했다.

## 주제별 폴더

- [01 초기 후보](01_initial_topics/README.md): tokenization, observation-aware resolution, uncertain covariates.
- [02 A: 적응 위치와 메커니즘](02_adaptation_scope/README.md): A S1, 출력 보완, 통제 합성, 모듈 삭제, 지연 추정, 외부·시간 반복.
- [03 B/C: 선택과 보정](03_selection_calibration/README.md): 선택 안정성, 능력 보존, calibration, selection regret.
- [04 목적함수와 관측](04_objective_observation/README.md): raw loss, 관측 연산, 월합 감독.
- [05 정답 수정](05_revision/README.md): ALFRED 자료 확보와 revision 진단.
- [06 Full FT 기준선](06_fullft_reference/README.md): 실제 전체 학습 비교, 메모리 복구, 최종 결과.
- [07 연구 방향](07_research_direction/README.md): 탐색 규칙, 선행 경계, 후보 스크린과 의견.
- [08 Hospital](08_hospital_shared_strength/README.md): 공유 LoRA의 계열별 적용 강도, [중단 상태](08_hospital_shared_strength/24_hospital_pause_status_20260910.md).
- [Benchmark gap discovery](benchmark_gap_discovery_v1/README.md): 기존 별도 폴더 유지. 장기 예측 후속은 [최종 판정](../../../results/tsfm_long_horizon_specialist_closure_v1/STATUS.md).

## 기록을 읽는 방법

번호는 연구 이력을 보존한 식별자이며, 독립적인 실패 실험 개수가 아니다. 계획·자료 검사·후속 분석도 번호를 갖는다. 옛 문서의 ‘다음 실행’, ‘승인 대기’는 작성 당시 기록이며 현재 실행 상태는 위 요약이 우선한다. 23번의 길이 절단·백본 비교 구상은 최신 24번 Hospital 계약에서 제외됐다.

기존 50개 문서를 위 8개 폴더로 옮겼다. [이동 목록과 원본 hash](research_review_20260910/evidence/relocation_manifest.json)에 이전 경로와 새 경로가 있다. 실험 코드·설정·원 결과·run의 경로는 유지했다. 과거 hash 계약이 문서 자체를 포함했다면 이동 전 문서는 이전 commit과 이 manifest로 추적한다. 현재 문서를 과거 봉인 문서와 byte-identical하다고 주장하지 않는다.

그림과 결과 링크는 저장소 상대경로다. `runs/`와 `third_party/`는 기존 Git 제외 정책 때문에 로컬에만 있다. Hospital의 작은 검증·중단 근거는 [evidence](research_review_20260910/evidence/)로 별도 추출했다. 이번 정리는 로컬 변경이며 commit/push 여부는 별도로 확인해야 한다.
