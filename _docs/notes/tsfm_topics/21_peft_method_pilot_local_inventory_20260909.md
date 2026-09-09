# 21. 방법 파일럿 인계: 로컬 상태 확인 (Part A1)

2026-09-09. 외부에서 받은 지시문 `PEFT_METHOD_PILOT_CLI.txt` 및 계획 초안 `PEFT_METHOD_PILOT_PLAN.md`의 Part A1 수행 기록이다. **새 GPU 학습 0회, 기존 파일 수정 0건, commit/push 0건.** 이 문서는 현황 확인이며 새 방법·새 결과가 아니다.

## 0. 인계 전제와 실제의 불일치 — 먼저 보고

지시문 §0은 함께 제공된 `peft_method_pilot_kit` 폴더를 읽으라고 지시하며, 그 안의 `peft_kit/audit.py`, `peft_kit/metrics.py`, `peft_kit/selection.py`, `peft_kit/projection.py`, `tests/`(37 CPU 테스트), `docs/01~03`, `verification/pytest.txt`, `config/plan.draft.json`, `evidence/study17_site_excerpt.json`을 재사용하라고 한다.

**[확인] 그 폴더는 이 머신에 없다.** 이번 세션에 업로드된 파일은 지시문 텍스트와 계획 초안 두 개뿐이다.

```text
확인한 경로                                          결과
C:/Users/User/.claude/uploads/<이번 세션>/            지시문 .txt + 계획 .md 두 개만
E:/CODING/proj/mltimeseries/ 전체 (maxdepth 4)        peft_kit / plan.draft.json / 02_LITERATURE_BOUNDARY.md 없음
C:/Users/User/{Downloads,Desktop,Documents}           peft_method_pilot 이름의 폴더·zip 없음
C:/Users/User/.claude/uploads/<다른 세션 6개>          해당 kit 없음
```

지시문 §0 자신이 "필요하면 기존 저장소의 검증된 평가기를 주분석으로 유지하고 이 패키지를 독립 교차검산용으로 써라"라고 쓴다. 즉 kit은 보조이고 주 자산은 이 저장소다. 따라서 kit 부재는 Part A 전체의 `BLOCKED`가 아니라 **교차검산 채널 하나의 상실**이다. 아래 §3에 그 영향을 적는다.

kit이 실제로 존재한다면 경로를 알려주면 해당 부분만 다시 채운다. 없는 파일의 API·키·내부 구조를 추정해서 쓰지 않는다.

## 1. 실제 HEAD와 작업트리

```text
repository        E:/CODING/proj/mltimeseries   (worktree 1개, 추가 worktree 없음)
branch            tsfm-long-horizon-specialist-closure-v1
HEAD              2b837a1869d5aad76c45b3e3355086bfb9bc827a
지시문 §A1 기준    2b837a1869d5aad76c45b3e3355086bfb9bc827a      -> 일치
작업트리 변경      추적 파일 수정 0건
미추적            .pytest_tmp_objective_analysis/   (이번 작업과 무관, 건드리지 않음)
```

읽기 전용 확인만 했다. `reset --hard`, `clean`, 강제 checkout, stash, 기존 결과 덮어쓰기, 자동 commit/push는 실행하지 않았다.

원격에 main으로 병합되지 않은 브랜치 6개가 있다: `hq-token-pilot-v1-audit-closure`, `oa-resolution-pilot-v1`, `oa-resolution-pilot-v1-audit-closure-v1`, `tsfm-benchmark-gap-discovery-v1`, `tsfm-long-horizon-specialist-closure-v1`, `uncertain-covariate-path-pilot-v1`. 현재 HEAD는 그중 마지막 계열의 tip이며 PEFT study 계열까지 포함한다. 병합은 사용자 지시 없이 하지 않는다.

## 2. 지시문 §A1이 지정한 파일의 로컬 실물

전부 존재한다. MISSING 없음.

```text
_docs/PROJECT_LOG.md                                                   OK   170 lines
_docs/notes/tsfm_topics/04_peft_adaptation_scope_s1_results_20260908.md OK   180
_docs/notes/tsfm_topics/10_peft_trainlag_results_20260908.md            OK   108
_docs/notes/tsfm_topics/14_peft_selection_regret_results_20260908.md    OK    64
_docs/notes/tsfm_topics/17_coarse_supervision_results_20260908.md       OK    74
_docs/notes/tsfm_topics/20_peft_fullft_reference_plan_20260909.md       OK    60
_docs/notes/tsfm_topics/20_peft_fullft_reference_results_20260909.md    OK   124
experiments/peft_adaptation_scope_v1/modeling.py                        OK   194
experiments/peft_fullft_reference_v3/                                   OK   10 modules + tests/
```

지시문이 목록에 넣지 않았지만 현재 브랜치의 실제 최신 작업 두 건이 더 있다. 방법 후보를 고르기 전에 반드시 읽어야 하므로 여기 적는다.

```text
results/tsfm_benchmark_gap_discovery_v1/            NO_STRONG_GAP_FOUND      (fev-bench 18 tasks, 3 TSFM 계열)
results/tsfm_long_horizon_specialist_closure_v1/    SHARED_TASK_DIFFICULTY_CLOSE
```

## 3. 재사용할 코드 — 다시 만들지 않는다

`experiments/peft_adaptation_scope_v1/modeling.py`가 이 저장소 PEFT 계열의 공통 계약이고, `peft_fullft_reference_v3`가 그것을 감싼 최신 실행기다. 아래는 실제로 읽고 확인한 내용이다.

| 자산 | 실체 | 재사용 방식 |
|---|---|---|
| `modeling.AdaptationModel(base, method, channels)` | method 7종: `F0`/`AFF`/`H_LIN`/`H_MLP`/`H_FULL`/`OFF_LORA`/`FULL` | import 해서 arm 구성 |
| `modeling.native_pinball` | 정규화·asinh 공간의 mean-horizon / sum-quantile / mean-series pinball | 학습 손실 |
| `modeling.forecast_scores` | train std 정규화 scaled 2-pinball, qmean MSE, median MAE, 80% coverage/width, crossing | 평가 |
| `modeling.encode / from_context / from_cache` | `from_cache`만 residual probe(`H_LIN`/`H_MLP`)를 더한다. `from_context`는 native 경로라 probe를 무시한다 | 지시문 B1 경고와 일치하는 실제 함정 |
| `fullft_reference_v3/model.py` | arm별 trainable 수 하드 검증(`F0` 0 / `H_FULL` 3,653,280 / `LORA` 1,206,912 / `FULL_FT` 119,477,664), LoRA B=0 초기화 검사, 97-projection map 검사, 파라미터 digest, mapped best-state | 그대로 import |
| `fullft_reference_v3/{data,train,forecast,analyse,baselines,contract,plot,run_study}.py` | 데이터 계약·학습 루프·예측·분석·단순 기준선·계약 동결·그림·드라이버 | 새 namespace에서 compose |
| `peft_adaptation_scope_v1/guard.py` | 공유 resource guard (RAM/commit/child RSS/Git/GPU MiB·°C) | 새 안전 체계를 만들지 않고 이것을 쓴다 |
| `experiments/peft_fullft_reference_v3/tests/` | `test_{analyse,baselines,data,model,run_study}.py` | 로컬 연결 후 1회 실행 |

**[확인] LoRA 계약의 실제 값:** rank 8 / alpha 16 / dropout 0 / bias none, 대상 97개 Linear = 12블록 × 2레이어 × `self_attention.{q,k,v,o}` 96개 + `output_patch_embedding.output_layer` 1개.

kit 부재의 실제 영향: 지시문이 예정한 "기존 평가기를 주분석으로 두고 kit 평가기로 교차검산" 중 **두 번째 채널이 없다.** 대신 이 저장소가 study마다 실행해 온 별도 stdlib/NumPy 독립 재계산(Study20에서 44개 평가 행 직접 재계산, 오차 0)을 교차검산으로 쓴다. 서로 다른 bootstrap·집계·missing 계약을 같은 결과처럼 섞지 않는다는 지시문 §0 요구는 이 방식에서도 유지된다.

## 4. 사용 가능한 모델·자료·환경

```text
GPU            NVIDIA RTX 4070, 12,282 MiB total (확인 시점 1,427 MiB 사용, 44°C)
환경           conda mlts (chronos_forecasting 2.3.1), .venv-peft, .venv-tsfm, .venv-tsfm-specialist
모델 weights   amazon/chronos-2, autogluon/chronos-2-synth, NX-AI/TiRex-2, ibm-research/flowstate  (HF 캐시 실물)
               고정 revision 29ec3766d36d6f73f0696f85560a422f50e8498c 로 전 study 사용
데이터         data/{electricity,ETT-small,jena_mpi_roof,uci_household_power,weather}
               data_external/{uci_bike_sharing,uci_air_quality,bdg2_coarse_supervision_v1,
                              alfred_revision_entry_v1,uscrn_operator_entry_v1,ucp_path_pilot_v1,
                              tsfm_benchmark_gap_discovery_v1,tsfm_long_horizon_specialist_closure_v1}
fev 캐시       runs/dscache/autogluon___fev_datasets  (511 MB)
```

- `timesfm-3.0`은 HF 캐시 목록에서 확인되지 않았다. benchmark 계열 결과에는 그 점수가 있으므로 **이번 세션에서 재실행 가능 여부는 UNKNOWN**으로 둔다. 필요해지면 그때 확인한다.
- Chronos-2 사전학습과 위 자료의 중복은 전 study에서 `UNKNOWN`으로 유지돼 왔다. 그대로 유지한다.
- fev-bench contamination: `chronos-2` / `tirex-2`는 `OVERLAP_RISK_UNKNOWN`, `timesfm-3.0`만 `CLEAN_BY_OFFICIAL_EXCLUSION`.

## 5. 보존해야 할 해석 — 지시문 §A1 목록의 로컬 근거

지시문이 열거한 일곱 항목을 로컬 문서에서 실제 확인했다. 아래는 그 근거다.

| 보존할 해석 | 로컬 근거 |
|---|---|
| Study20의 LoRA 효용은 기준선 결과다 | LoRA의 F0 대비 감소 Bike 4.935% / Household 0.915%. 표준 rank8 LoRA의 적용이며 새 방법 아님 |
| Full FT는 상한이 아니다 | Full의 LoRA 대비 이득 Bike −0.113% [−0.949,+1.043] / Household −0.433% [−0.950,−0.060]. 6개 선택 LR 중 5개가 grid 경계 |
| Study10 Gaussian family는 반복하지 않는다 | 추정 지연 선형 회귀가 oracle에 근접(RAW−oracle gap 0.128% F0). 이 조건의 adapter 탐색 중단 |
| Study14 LR 선택과 모듈 선택은 다른 질문 | Bike 선택 손실 +3.079%지만 고정 LR 1e-5 단순 규칙의 veto가 성립 |
| Study17의 손상은 head 단계부터 | Eagle MSE F0 0.05749 → head 1.59680 → LoRA 1.59697. LoRA 이전에 이미 악화 |
| 이미 본 기간·원천은 개발 자료 | Bike/Household 블록 3개 소진(12/13 두 블록 + 20 세 번째). ETT/Jena는 04 S1에서 노출 |
| 사전학습 중복은 UNKNOWN 유지 | 전 study 공통 계약 |

## 6. 없어서 막힌 것 / 재학습이 필요한 이유

**없어서 막힌 것**

- `peft_method_pilot_kit` 전체 (§0). 교차검산 채널 하나가 없다. 대체 수단은 §3에 적었다.
- `timesfm-3.0` 로컬 weights UNKNOWN. 세 TSFM 계열을 다시 돌리는 실험을 지금 계획에 넣지 않는 이유 중 하나다.
- Study02(OA) 본 결과는 현재 작업트리가 아니라 commit `4c6c805`에 있다. 필요하면 그 commit에서 읽는다.

**재학습이 필요한 이유**

기존 결과를 그대로 재사용할 수 없는 경우는 셋뿐이다. 그 외에는 저장된 점수를 쓴다.

1. 새 arm(제안 방법·기전 삭제 대조)은 저장된 예측이 존재하지 않는다.
2. 비교가 성립하려면 모든 arm이 **같은 origin·같은 target·같은 missing mask·같은 scale·같은 quantile grid**를 써야 한다. 과거 study는 블록·split이 서로 달라 그대로 섞으면 paired 비교가 깨진다.
3. F0와 표준 LoRA는 과거 값이 있어도, 새 split에서 재계산하지 않으면 위 1·2를 만족하지 않는다. 재사용할 경우 재사용 범위와 과거 HPO 비용을 명시한다.

## 7. 이 문서가 하지 않은 것

- 저장소 전체 감사 보고서를 쓰지 않았다. 지시문 §A1이 금지한다.
- 과거 결과를 재실행하거나 수정하지 않았다. 새 GPU 학습 0회.
- 방법 후보를 아직 고르지 않았다. A2~A5는 별도 문서에서 다룬다.
