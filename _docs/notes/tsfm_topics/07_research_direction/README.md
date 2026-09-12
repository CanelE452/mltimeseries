# 연구 후보 스크린과 다음 방향

- [38. Native-path LoRA-only 진단](38_lora_only_diagnostic_20260912.md) — [확인] 16fit/8D/4zero-update gate 완료. 모든8cell/view에서 LoRA-only와 JOINT가 F0보다 나빠 CASE C. LoRA-only는 JOINT보다7/8에서 좋지만 BMRA seed30001 S180은 반대. [판정] 추가 head만으로 손해 설명 불가; fresh 검증 우선. BDG2 Bull READY, Household post-P1 BLOCKED. Fresh 학습0.

- [37. PEFT 논문 claim·gate와 story map](37_peft_paper_claim_gate_20260911.md) — [판정] 분석 논문 A를 조건부 primary로 고정. [확인] 기존 핵심 수치936개 재검산 통과, 당시 새 GPU 실행0. [novelty 경계](37_novelty_boundary.md)는 2026-09-12 Time-PEFT 원문을 반영했으며, 좁은 신규성과 독립 실증은 미확정이다. [readiness·노출 감사·Stage A 초안](../../../../results/peft_paper_closure_v1/PAPER_READINESS.md)에서 실행 자격과 미해결 증거를 구분한다.

- [36. 순서 의존성 PEFT 준비 검증 결과](36_dependence_phase0_results_20260911.md) — 현재 설계 STOP/GPU 0회. 전/역 대조가 값 부호 반전과 동등하고, 마지막 값의 이차식으로 Bayes 예측을 표현함을 확인. 시간 관계 PEFT 필요성 식별에 부족해 72fit·adapter·외부 검증 미진입. [원래 계획 보존](36_dependence_peft_paper_plan_20260911.md).

- [35. head 예산 대조와 기간 간 적응 손해](35_head_convergence_20260911.md) — 180→720step 대조 완료. LoRA는 동일 용량 head보다 평균8.804%F0 좋지만 F0보다4.305% 나빠 G1 미통과. head가 따라잡은 결과는 아니며,12개 선택 모델 모두 train/V 개선과 D 악화. 조건부 head재학습·warm-start 미실행.75 GPU 실행·독립 검산 완료,학습 종료.

- [34. 초기 LoRA 적응: 강한 head 이후의 선택 여지](34_initial_headroom_20260911.md) — 완료. 초기 LoRA 이득 G1은 4조합 중3조합 통과, controller 여지 G2는 +0.0408/+0.0756%F0로 미통과. 평균 동일 용량 head 대비 +2.428%F0, F0 대비 +1.418%F0. 일부 head의 학습 상한 선택으로 수렴·표현 원인은 미확정. 123 GPU 실행·독립 검산·5개 그림 완료, 학습 종료.

[전체 정리](../research_review_20260910/README.md) · [전체 목록](../README.md)

- [33. 짧은 적응 반응의 시기 전이·신호 제거·실제 비용](33_decision_transfer_20260911.md) — 완료. 세 기준 미통과,177 GPU 실행·독립 감사·4개 그림.12/12 같은 prefix 출력,조기 종료보다20.1~33.7% 느림. 평균 초기 LoRA 이득은 남지만 원천별 부호 차이;판단 시점·목표의 한계와 다음 진입 순서 정리.
- [32. 미래 LoRA 업데이트 가치와 논문 요건](32_future_utility_20260911.md) — 완료. 12조건·24분기·독립 검산. 현재 기여와 미래 학습 가치의 불일치, STOP·선택·clipping 대조와 논문 요건 정리. 새 정책 우월성은 미확보.
- [31. 단일 LoRA 학습의 on/off 반응과 동결](31_contribution_freeze_20260911.md) — 완료. 48fit+50E+S0+시간재실행13,112guard정상. 후보시간7.42/3.77%절감으로기준실패;고정동결19.92/21.46%절감. 전체검산·그래프·현재기여와미래가치불일치반례·후속진단.

- [30. 동일 학습 파라미터 수 head 대조와 짧은 적응 반응](30_capacity_probe_20260910.md) — 완료. P1 Bike FULL90 LoRA 추가이득4.779%F0,같은수대조기준통과. 초기반응6cell rank0.543;현재선택은품질기준통과/비용증가로효율기준실패. 85guard정상종료·3그래프·독립감사.

- [29. 과거 자료 구성으로 LoRA 사용 여부 결정 → 새 시기 검증](29_overlap_transfer_20260910.md) — 완료. 24학습/26예측, LoRA 추가 이득12/12양수. OVERLAP은 비용 약12% 절감에도 정확도 기준 실패. SPREAD/RECENT 순위 반전과 V/E 적응 반응을 분석.
- [28. 최적화 대조 → 과거 효용 예측 → 미래 평가](28_optimization_control_20260910.md) — 완료. 48학습/48평가, LR·초반 선택·기대 노출 대조 후 Bike FULL90/SPREAD30 추가 이득 차이가 남아 후속 기준 통과. 기존 SPREAD30 평균 손해 해석은 수정.

- [27. R1 후속 검증: 학습 구간 구성과 과거 잔차 보정](27_r1_coverage_residual_validation_20260910.md) — 최신. 새16 fits/16 평가 완료. Bike 추가 이득 FULL90 +4.171 → SPREAD30 −1.019 / RECENT30 +1.267%F0. 구간 구성·최적화가 혼입돼 원인은 미확정. CPU 정밀도 검사 복구와 독립 감사 포함.
- [26. R1–R3 실제 통제 학습과 진단 결과](26_mechanism_diagnostics_20260910.md) — 앞선36 fits/24 평가 완료. 같은 head 뒤에도 Bike 내부 적응 이득은 남지만 층 선택 우위는 미확보. Hospital 월 제외 안정성과 미래 효용을 구분.
- [25. 관찰된 PEFT 이득에서 원인·방법으로: 순차 진단 후보](25_phenomenon_to_mechanism_roadmap_20260910.md) — 실행 전 로드맵. 실제 실행 범위와 미실행 원인 개입은 26번을 따른다.

- [PEFT 주제 반복 탐색: 운영 계약과 자기 평가](11_peft_topic_search_protocol_20260908.md)
- [B 진단 중 확인한 다음 방법의 선행연구 경계](14_next_method_literature_boundary_20260908.md)
- [18. PEFT 문제 중심 후보와 다음 진입 판단](18_peft_problem_first_candidates_20260908.md)
- [21. 방법 파일럿 인계: 로컬 상태 확인 (Part A1)](21_peft_method_pilot_local_inventory_20260909.md)
- [22. 방법 후보 스크린: 네 방향의 선행 경계와 실측 가지치기 (Part A2–A3)](22_peft_method_candidate_screen_20260909.md)
- [23. 다음 경로 두 개와 내 의견 — 실행하지 않고 적은 계획](23_peft_next_paths_and_opinion_20260909.md)
