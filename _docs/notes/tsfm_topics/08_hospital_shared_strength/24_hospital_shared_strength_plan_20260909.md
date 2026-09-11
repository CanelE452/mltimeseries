# 24. Hospital 공유 LoRA · 계열별 적용 강도 파일럿 — 착수 기록

> 최신 상태: 이후 사용자 재개 요청으로 실험을 완료했다. [최종 결과](24_hospital_results_20260910.md)를 우선하며 아래는 당시 기록이다.

> 2026-09-10 상태: 사용자 요청으로 중지 유지. 1/4 fit 완료, 두 번째는 자원 기준으로 중단, E 평가 없음. [최신 중단 기록](24_hospital_pause_status_20260910.md)이 아래 착수 시점 표현보다 우선한다.

2026-09-09. 외부 kit `hospital_shared_adaptation_kit`(ZIP sha256 `ef07bd6d…5cb8b22`, `third_party/hospital_shared_adaptation_kit/`, PROVENANCE.json)의 계약을 그대로 실행한다. 계약 본문은 kit의 `docs/EXPERIMENT_CONTRACT.md`와 `config/pilot.json`이며 여기서는 [소비처]/[문장]과 착수 상태만 적는다. 같은 내용을 두 곳에 쓰지 않는다.

## 1. 제안

- 가설: 단일 공유 LoRA의 예측을 모든 계열에 같은 강도(`GLOBAL` alpha 1개)로 섞는 것보다, 계열별 V2 손실로 고른 강도(`INDIVIDUAL` alpha 767개)가 E1·E2에서도 유리하다. [미검증]
- 방법: Chronos-2 `29ec3766…` + rank8 shared LoRA, LR{1e-5,3e-5}×seed{24000,24001}=4 fits, V1로 LR/checkpoint, V2로 alpha, E=2005/2006. `q(a)=q0+a(qL−q0)`, a∈{0,…,1}. 교환 대조 SHUFFLED×20(seed 2026090924).
- 판정 지표: 주대비 `100·(L_GLOBAL−L_INDIVIDUAL)/L_F0`, 기전 대조 `100·(mean_perm L_SHUFFLED−L_INDIVIDUAL)/L_F0`. 분모는 F0 손실 평균. 부호 허용오차 1e-8 %F0. 실용 문턱은 사전에 두지 않는다(kit 계약 §6).
- 예상 실패 모드: INDIVIDUAL이 V2 12개 정답에 과적합해 GLOBAL보다 나쁨 / F0=LoRA 퇴화로 alpha 무의미 / 200 updates에서 step0 선택.
- 중단 기준: S0 실패(동일 계약 1회 재시도 후) → IMPLEMENTATION_BLOCKED. 어떤 결과든 데이터·규칙·seed·grid를 바꾸지 않는다. 자동 재시도는 job당 1회.

## 2. 착수 상태

- 자료: fev 캐시 Hospital 767×84 전체(arrow sha256 `5fabbcef…8fde`), NaN 0·음수 0. kit에 Arrow 리더가 없어 `experiments/hospital_shared_strength_v1/export_canonical.py`로 canonical NPZ(values/series_ids/start_months)만 export한다 — kit CLI §2(2)가 허용한 유일한 추가.
- 환경: `.venv-peft` (torch 2.11.0+cu128, chronos 2.3.1, peft 0.20.0, transformers 5.16.1). verify_bundle PASS 37, pytest 52 passed.
- 결과 절은 실행 후 이어 쓴다.
