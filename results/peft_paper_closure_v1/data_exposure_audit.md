# PEFT data exposure audit

[확인] 감사 범위는 기존 PEFT 연구에서 이미 열린 데이터 단위, 역할, 기간, 결과 노출 여부, Stage A 및 sealed final 후보 가능성이다.

[확인] 새 후보 타깃 값은 열지 않았다. 이 문서는 기존 준비 매니페스트, 결과 요약, 연구 노트, 그리고 타깃 값을 포함하지 않는 로컬 타임스탬프/파일 메타데이터 판단만 사용한다.

[확인] 원장의 `results_viewed=yes`는 해당 결과/분석 산출물이 연구 과정의 노출 단위였다는 artifact-level 판단이다. 이는 사람별 실제 열람 로그를 증명한다는 뜻이 아니다.

## 원본 원장 보존

[확인] 기존 untracked 원장은 교체 전에 `results/peft_paper_closure_v1/data_exposure_ledger.original_20260911_181304.csv`로 보존했다.

[확인] 보존본 SHA256: `b4b0edca0df8b74ca088e3e27fced2568270158767ec45aa7fc1b0f0a8103bd9`.

[판정] 보존본 해시 검증: PASS.

## 24-row snapshot preservation

[확인] 누락 지적 전 24-row audit 산출물도 versioned snapshot으로 보존했다.

[확인] Snapshot files: `data_exposure_ledger.v24_snapshot_20260911_183300.csv`, `data_exposure_audit.v24_snapshot_20260911_183300.md`, `holdout_availability.v24_snapshot_20260911_183300.json`, `holdout_availability.v24_snapshot_20260911_183300.md`.

## 핵심 수정

[확인] 기존 원장의 Study31 `bmra+jena from overlap experiments` 행은 틀렸다. Study31 준비 요약의 실제 단위는 `bdg2` 2017-05-04~2018-01-01과 `jena` 2023-05-04~2024-01-01이다.

[확인] 기존 결과가 열린 모든 Bike, Household, BDG2, BMRA, Jena, Hospital 단위는 development exposure로 처리한다. 이들은 sealed final 후보가 아니다.

[판정] 이 감사의 데이터 쪽 결론은 `NO_CLEAN_HOLDOUT_AVAILABLE`이 아니다. 확인된 clean fresh unit은 0개지만, 이는 후보 부재가 아니라 clean manifest와 target-blind certification 부재다.

## Stage A readiness

[판정] Stage A 데이터 조건은 `CLEAN_FRESH_UNITS_NOT_YET_CERTIFIED`다.

[확인] 현재 인증된 clean fresh unit 수는 0개이고, 사용자 요청의 최소 조건은 2개다.

[확인] 상위 novelty audit에서 paper novelty gate가 UNKNOWN/미통과로 전달되어 Stage A 전체는 이미 blocked다. 따라서 새 타깃 다운로드나 학습으로 데이터 결론을 억지로 확정하지 않았다.

[추정] Household의 2009-04-29 이후 로컬 타임스탬프 구간은 후보가 될 수 있으나, target coverage, prior exposure, pretraining overlap, target-blind manifest가 없으므로 clean unit으로 세지 않는다.

[미검증] BDG2 alternate meters/periods와 BMRA alternate periods/unit pairs는 후보일 수 있으나, 현 감사에서는 clean manifest가 없다.

## Sealed final readiness

[판정] sealed final은 `SEALED_FINAL_NOT_AVAILABLE_LOCALLY_YET`다.

[확인] final holdout은 claim, method, hparams, threshold, dataset rule이 고정된 뒤 열어야 한다. 기존 PEFT 산출물은 모두 development 과정에서 validation/evaluation/D 결과가 열렸다.

[판정] final용으로는 최소 2개 독립 fresh source/period unit을 새로 봉인해야 한다. 같은 결과를 본 뒤 seed·rank·threshold·dataset을 바꾸면 final이 아니라 development가 된다.

## 후보 가용성 해석

- [추정] Bike: 로컬 원천은 2011-01-01~2012-12-31이고 P1 eval이 2012-12-04까지 열려 있어, 같은 242일 계약의 post-P1 fresh 기간은 로컬상 거의 없다. 이는 로컬 타임스탬프 판단이며 전세계 데이터 부재가 아니다.
- [추정] Household: 로컬 원천은 2006-12-16~2010-11-26이고 P1 eval 이후 시간이 남는다. 그러나 타깃 커버리지와 clean prior-exposure 검증이 없으므로 `candidate metadata only`다.
- [확인] Jena: PEFT/S1에서 2018, 2019, 2021, 2022, 2023, 2024가 열렸고, 2020은 이전 weather exposure 때문에 Study35에서 제외됐다.
- [확인] BMRA: PEFT에서 2017, 2018, 2022, 2023 단위가 열렸고, prepared summaries는 2019~2021 BMRA가 비-PEFT UCP panel로 처리됐다고 기록한다. Stage A 사용 가능성은 별도 clean manifest 없이는 UNKNOWN이다.
- [미검증] BDG2: Study31의 Eagle 2017 단위는 열렸지만 다른 building/period 조합은 이번 감사에서 target-blind로 인증하지 않았다.
- [확인] Hospital: 로컬 canonical artifact는 2000-01~2006-12이고 E1/E2가 열린 development pilot이다. 다른 병원 데이터셋 존재 여부는 조사하지 않았다.

## 산출물

[확인] 구조화 원장: `results/peft_paper_closure_v1/data_exposure_ledger.csv`.

[확인] Holdout availability JSON: `results/peft_paper_closure_v1/holdout_availability.json`.

[확인] Holdout availability Markdown: `results/peft_paper_closure_v1/holdout_availability.md`.

[확인] PEFT directory coverage inventory CSV: `results/peft_paper_closure_v1/data_exposure_dir_inventory.csv`.

[확인] PEFT directory coverage inventory JSON: `results/peft_paper_closure_v1/data_exposure_dir_inventory.json`.

## 원장 행 수

[확인] 현재 원장 행 수: 27.

## 원장 상세 요약

- [확인] 01. ETTm2 / `adaptation_scope_s1_ettm2`
  - [확인] Target/series: all 7 channels: HUFL,HULL,MUFL,MULL,LUFL,LULL,OT
  - [확인] Range/roles: 2018-03-02~2018-06-25; input-only 4d + fit 64d + validation 16d + development evaluation 32d / fit|validation|evaluation
  - [확인] Studies/results: Study04 S1; Study07 reuses ETTm2 train cache / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/07_head_complement_method_gate_results_20260908.md; results/peft_adaptation_scope_v1/summary_and_costs.json
  - [미검증] No clean alternate ETTm2 period was certified in this audit.
- [확인] 02. Jena / `adaptation_scope_s1_jena2024`
  - [확인] Target/series: 21 Jena weather channels scored jointly
  - [확인] Range/roles: 2024-09-07~2024-12-31; input-only 4d + fit 64d + validation 16d + development evaluation 32d / fit|validation|evaluation
  - [확인] Studies/results: Study04 S1 / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md; results/peft_adaptation_scope_v1/summary_and_costs.json
  - [미검증] 2024 Jena cannot be treated as final-fresh after S1 exposure.
- [확인] 03. synthetic Gaussian lag/covariate / `peft_shift_module_trainlag_synthetic`
  - [확인] Target/series: Y target with U/V covariates; synthetic episodes
  - [확인] Range/roles: episode data, not calendar time; L=256/H=16; train/validation/evaluation episodes / train|validation|evaluation|diagnostic
  - [확인] Studies/results: Study08; Study09; Study10; Study11 calibration closure / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NOT_APPLICABLE
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/08_peft_shift_mechanism_results_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/09_peft_module_ablation_results_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/10_peft_trainlag_results_20260908.md; _docs/notes/tsfm_topics/03_selection_calibration/11_peft_calibration_closure_results_20260908.md; results/peft_calibration_closure_v1/summary.json
  - [판정] Synthetic episodes do not satisfy the requested real fresh source/period requirement.
- [확인] 04. USCRN / `observation_operator_entry_uscrn_az_tucson_2024`
  - [확인] Target/series: AZ_Tucson_11_W hourly T_CALC/T_HR_AVG and subhourly AIR_TEMPERATURE
  - [확인] Range/roles: hourly 2024-01-01 01:00~2025-01-01 00:00 UTC; subhourly 2024-01-01 00:05~2025-01-01 00:00 UTC / data_qc|operator_semantics_check|no forecast score
  - [확인] Studies/results: peft_observation_operator_entry_v1 data_qc; Study16 observation-operator entry / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NOT_APPLICABLE
  - [확인] Provenance: experiments/peft_observation_operator_entry_v1/README.md; data_external/uscrn_operator_entry_v1/contract.json; data_external/uscrn_operator_entry_v1/qc.json; results/peft_observation_operator_entry_v1/final_audit.json
  - [판정] This is exposed metadata/data-QC evidence and cannot be sealed final.
- [확인] 05. Bike Sharing / `external_gap_first_block`
  - [확인] Target/series: casual|registered
  - [확인] Range/roles: 2011-01-01~2011-07-10; precontext 2011-01-01~2011-01-15; train 2011-01-15~2011-03-20; V 2011-03-20~2011-04-03; C 2011-04-03~2011-04-17; eval 2011-04-17~2011-07-10 / train|V_select|C_cal|eval
  - [확인] Studies/results: Study12 external gap; Study15 objective alignment reuse; Study14 selection regret reuse / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_external_gap_v1/prepared/manifest.json; _docs/notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_plan_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_results_20260908.md; _docs/notes/tsfm_topics/04_objective_observation/15_peft_objective_alignment_results_20260908.md; _docs/notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md; results/peft_external_gap_v1/diagnostics.json
  - [확인] Exact retained timestamp boundaries were traced from runs/peft_external_gap_v1/prepared/manifest.json.
- [확인] 06. Household Power / `external_gap_first_block`
  - [확인] Target/series: Global_active_power|Global_reactive_power
  - [확인] Range/roles: 2006-12-17~2007-06-25; precontext 2006-12-17~2006-12-31; train 2006-12-31~2007-03-05; V 2007-03-05~2007-03-19; C 2007-03-19~2007-04-02; eval 2007-04-02~2007-06-25 / train|V_select|C_cal|eval
  - [확인] Studies/results: Study12 external gap; Study15 objective alignment reuse; Study14 selection regret reuse / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_external_gap_v1/prepared/manifest.json; _docs/notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_plan_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_results_20260908.md; _docs/notes/tsfm_topics/04_objective_observation/15_peft_objective_alignment_results_20260908.md; _docs/notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md; results/peft_external_gap_v1/diagnostics.json
  - [확인] Exact retained timestamp boundaries were traced from runs/peft_external_gap_v1/prepared/manifest.json.
- [확인] 07. Bike Sharing / `temporal_replication_block`
  - [확인] Target/series: casual|registered
  - [확인] Range/roles: 2011-07-10~2012-01-16; train 2011-07-24~2011-09-26; V 2011-09-26~2011-10-10; C 2011-10-10~2011-10-24; eval 2011-10-24~2012-01-16 / train|V|C|eval
  - [확인] Studies/results: Study13 temporal replication; Study14 selection regret reuse / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_plan_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_results_20260908.md; _docs/notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md; results/peft_temporal_replication_v1/selected_results.csv
  - [확인] This block predates the Study20 and P1 Bike windows.
- [확인] 08. Household Power / `temporal_replication_block`
  - [확인] Target/series: Global_active_power|Global_reactive_power
  - [확인] Range/roles: 2007-06-25~2008-01-01; train 2007-07-09~2007-09-11; V 2007-09-11~2007-09-25; C 2007-09-25~2007-10-09; eval 2007-10-09~2008-01-01 / train|V|C|eval
  - [확인] Studies/results: Study13 temporal replication; Study14 selection regret reuse / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_plan_20260908.md; _docs/notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_results_20260908.md; _docs/notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md; results/peft_temporal_replication_v1/selected_results.csv
  - [확인] This block predates the Study20 and P1 Household windows.
- [확인] 09. Bike Sharing / `fullft_reference_P0`
  - [확인] Target/series: casual|registered
  - [확인] Range/roles: 2012-01-16~2012-09-14; train 2012-01-30~2012-04-30; V 2012-05-02~2012-06-02; C 2012-06-04~2012-06-25; eval 2012-06-25~2012-09-14 / train|val|cal|eval
  - [확인] Studies/results: Study20 fullft reference; Study26 mechanism diagnostics; Study30 P0 capacity probe / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_fullft_reference_v3/prepared/manifest.json; results/peft_fullft_reference_v3/summary.json; _docs/notes/tsfm_topics/06_fullft_reference/20_peft_fullft_reference_results_20260909.md
  - [확인] Existing development exposure excludes this unit from any final holdout.
- [확인] 10. Household Power / `fullft_reference_P0`
  - [확인] Target/series: Global_active_power|Global_reactive_power
  - [확인] Range/roles: 2008-01-01~2008-08-30; train 2008-01-15~2008-04-15; V 2008-04-17~2008-05-18; C 2008-05-20~2008-06-10; eval 2008-06-10~2008-08-30 / train|val|cal|eval
  - [확인] Studies/results: Study20 fullft reference; Study26 mechanism diagnostics; Study30 P0 capacity probe / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_fullft_reference_v3/prepared/manifest.json; results/peft_fullft_reference_v3/summary.json; _docs/notes/tsfm_topics/06_fullft_reference/20_peft_fullft_reference_results_20260909.md
  - [확인] Existing development exposure excludes this unit from any final holdout.
- [확인] 11. Bike Sharing / `overlap_transfer_P1`
  - [확인] Target/series: casual|registered
  - [확인] Range/roles: 2012-04-06~2012-12-04; train 2012-04-20~2012-07-20; V 2012-07-22~2012-08-22; C 2012-08-24~2012-09-14; eval 2012-09-14~2012-12-04 / train|val|cal|eval
  - [확인] Studies/results: Study29 overlap transfer; Study30 capacity probe / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_overlap_transfer_v1/prepared/manifest.json; results/peft_overlap_transfer_v1/evidence/prepared_manifest.json; _docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md
  - [추정] Local Bike timestamp span leaves too little post-P1 calendar for another comparable 242-day block, but this is not a global no-data claim.
- [확인] 12. Household Power / `overlap_transfer_P1`
  - [확인] Target/series: Global_active_power|Global_reactive_power
  - [확인] Range/roles: 2008-08-30~2009-04-29; train 2008-09-13~2008-12-13; V 2008-12-15~2009-01-15; C 2009-01-17~2009-02-07; eval 2009-02-07~2009-04-29 / train|val|cal|eval
  - [확인] Studies/results: Study29 overlap transfer; Study30 capacity probe / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_overlap_transfer_v1/prepared/manifest.json; results/peft_overlap_transfer_v1/evidence/prepared_manifest.json; _docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md
  - [추정] Later local Household timestamps may allow candidate periods, but no clean manifest exists yet.
- [확인] 13. BDG2 / `coarse_supervision_bdg2_eagle_lamb_2016_2017`
  - [확인] Target/series: Eagle office targets 8 ids; Lamb office targets 8 ids; donor context meters 8+8
  - [확인] Range/roles: raw 2016-01-01~2017-12-31; train months 2016-02~2016-12; validation 2017-01~2017-03; evaluation 2017-04~2017-06 / train|validation|evaluation_truth
  - [확인] Studies/results: peft_coarse_supervision_v1; Study17 coarse supervision / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/peft_coarse_supervision_v1/data/metadata.json; runs/peft_coarse_supervision_v1/data/data_build_contract.json; results/peft_coarse_supervision_v1/independent_evaluation_audit.json; _docs/notes/tsfm_topics/04_objective_observation/17_coarse_supervision_results_20260908.md
  - [판정] This closes the previous ledger gap for peft_coarse_supervision_v1 and excludes these BDG2 site/months from any sealed final.
- [확인] 14. BDG2 / `contribution_freeze_bdg2_eagle2017`
  - [확인] Target/series: Eagle_office_Elias|Eagle_office_Elvis
  - [확인] Range/roles: 2017-05-04~2018-01-01; train 2017-05-18~2017-08-17; V 2017-08-19~2017-09-19; C 2017-09-21~2017-10-12; D/eval 2017-10-12~2018-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study31 contribution freeze; Study32 future utility / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_contribution_freeze_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/31_contribution_freeze_20260911.md; _docs/notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md
  - [미검증] Other BDG2 buildings or periods may be candidates, but this audit did not certify a clean target-blind manifest.
- [확인] 15. Jena / `contribution_freeze_jena2023`
  - [확인] Target/series: p (mbar)|T (degC)
  - [확인] Range/roles: 2023-05-04~2024-01-01; train 2023-05-18~2023-08-17; V 2023-08-19~2023-09-19; C 2023-09-21~2023-10-12; D/eval 2023-10-12~2024-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study31 contribution freeze; Study32 future utility / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_contribution_freeze_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/31_contribution_freeze_20260911.md; _docs/notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md
  - [확인] Same-family Jena 2024 had already been exposed in S1, so this is development evidence only.
- [확인] 16. Jena / `decision_transfer_dev_jena2021`
  - [확인] Target/series: p (mbar)|T (degC)
  - [확인] Range/roles: 2021-05-04~2022-01-01; train 2021-05-18~2021-08-17; V 2021-08-19~2021-09-19; C 2021-09-21~2021-10-12; D/eval 2021-10-12~2022-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study33 decision transfer dev / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_decision_transfer_v1/evidence/prepared_summary.json; experiments/peft_decision_transfer_v1/PURPOSE_DATA.md; _docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md
  - [확인] Study33 labels this as new target-label period under local history, not a new raw source family.
- [확인] 17. Jena / `decision_transfer_test_jena2022`
  - [확인] Target/series: p (mbar)|T (degC)
  - [확인] Range/roles: 2022-05-04~2023-01-01; train 2022-05-18~2022-08-17; V 2022-08-19~2022-09-19; C 2022-09-21~2022-10-12; D/eval 2022-10-12~2023-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study33 decision transfer test / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_decision_transfer_v1/evidence/prepared_summary.json; experiments/peft_decision_transfer_v1/PURPOSE_DATA.md; _docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md
  - [확인] Results were opened as Study33 transfer evidence, so this is not sealed final.
- [확인] 18. BMRA / `decision_transfer_dev_bmra2022`
  - [확인] Target/series: E_BNWKW-1|E_BRYBW-1
  - [확인] Range/roles: 2022-01-04~2022-09-03; train 2022-01-18~2022-04-19; V 2022-04-21~2022-05-22; C 2022-05-24~2022-06-14; D/eval 2022-06-14~2022-09-03 / train|val|cal|D/eval
  - [확인] Studies/results: Study33 decision transfer dev / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_decision_transfer_v1/evidence/prepared_summary.json; experiments/peft_decision_transfer_v1/PURPOSE_DATA.md; _docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md
  - [확인] Same BMRA raw family had non-PEFT UCP processing for 2019~2021; FM pretraining overlap remains UNKNOWN.
- [확인] 19. BMRA / `decision_transfer_test_bmra2023`
  - [확인] Target/series: E_BNWKW-1|E_BRYBW-1
  - [확인] Range/roles: 2023-01-04~2023-09-03; train 2023-01-18~2023-04-19; V 2023-04-21~2023-05-22; C 2023-05-24~2023-06-14; D/eval 2023-06-14~2023-09-03 / train|val|cal|D/eval
  - [확인] Studies/results: Study33 decision transfer test / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_decision_transfer_v1/evidence/prepared_summary.json; experiments/peft_decision_transfer_v1/PURPOSE_DATA.md; _docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md
  - [확인] Results were opened as Study33 transfer evidence, so this is not sealed final.
- [확인] 20. Jena / `initial_headroom_jena2019`
  - [확인] Target/series: p (mbar)|T (degC)
  - [확인] Range/roles: 2019-05-04~2020-01-01; train 2019-05-18~2019-08-17; V 2019-08-19~2019-09-19; C 2019-09-21~2019-10-12; D/eval 2019-10-12~2020-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study34 initial headroom / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_initial_headroom_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md
  - [확인] Same raw source family; FM pretraining overlap UNKNOWN in prepared summary.
- [확인] 21. BMRA / `initial_headroom_bmra2018`
  - [확인] Target/series: E_BNWKW-1|E_BRYBW-1
  - [확인] Range/roles: 2018-01-04~2018-09-03; train 2018-01-18~2018-04-19; V 2018-04-21~2018-05-22; C 2018-05-24~2018-06-14; D/eval 2018-06-14~2018-09-03 / train|val|cal|D/eval
  - [확인] Studies/results: Study34 initial headroom / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_initial_headroom_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md
  - [확인] Same raw source family; FM pretraining overlap UNKNOWN in prepared summary.
- [확인] 22. Jena / `head_convergence_jena2018`
  - [확인] Target/series: p (mbar)|T (degC)
  - [확인] Range/roles: 2018-05-04~2019-01-01; train 2018-05-18~2018-08-17; V 2018-08-19~2018-09-19; C 2018-09-21~2018-10-12; D/eval 2018-10-12~2019-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study35 head convergence / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_head_convergence_v1/evidence/prepared_summary.json; experiments/peft_head_convergence_v1/PURPOSE.md; _docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md
  - [확인] This was selected after Jena2020 was rejected for prior raw-value exposure.
- [확인] 23. BMRA / `head_convergence_bmra2017`
  - [확인] Target/series: E_BRYBW-1|E_BURBO
  - [확인] Range/roles: 2017-05-04~2018-01-01; train 2017-05-18~2017-08-17; V 2017-08-19~2017-09-19; C 2017-09-21~2017-10-12; D/eval 2017-10-12~2018-01-01 / train|val|cal|D/eval
  - [확인] Studies/results: Study35 head convergence / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_head_convergence_v1/evidence/prepared_summary.json; experiments/peft_head_convergence_v1/PURPOSE.md; _docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md
  - [확인] PURPOSE notes target pair/season changed after availability-only QC, before model outcomes.
- [확인] 24. Jena / `excluded_jena2020`
  - [확인] Target/series: p (mbar)|T (degC) family
  - [확인] Range/roles: 2020 raw readings, exact PEFT block not used in Study35 / prior non-Study35 weather fit/validation/test exposure
  - [확인] Studies/results: excluded from Study35 candidate selection / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_head_convergence_v1/evidence/prepared_summary.json; experiments/peft_head_convergence_v1/PURPOSE.md; results/hq_token_pilot_v1/data_manifest.json; _docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md
  - [미검증] This audit did not re-open the old weather target arrays to prove every timestamp overlap.
- [확인] 25. ALFRED revision data / `revision_entry_payems_indpro_2020_2024`
  - [확인] Target/series: PAYEMS|INDPRO monthly revision targets
  - [확인] Range/roles: event months 1989-12-01~2024-12-01; selection cutoff 2019-07-01; evaluation origins/target events 2020-01-01~2024-12-01 for each series / validation_selection|evaluation_cpu_predictions
  - [확인] Studies/results: peft_revision_entry_v1; Study19 revision entry / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: results/peft_revision_entry_v1/data_qc.json; results/peft_revision_entry_v1/origin_audit.json; runs/peft_revision_entry_v1/validation_selection.json; results/peft_revision_entry_v1/cpu_results.json; _docs/notes/tsfm_topics/05_revision/19_peft_revision_entry_results_20260908.md
  - [판정] This fills the previous ALFRED gap; no GPU PEFT final seal exists for these revision targets.
- [확인] 26. Hospital shared-strength / `hospital_monthly_2000_2006`
  - [확인] Target/series: 767 monthly hospital series; ids_sha256 recorded
  - [확인] Range/roles: 2000-01~2006-12; train 2000~2002; V1 2003; V2 2004; E1 2005; E2 2006 / train|V1|V2|E1|E2
  - [확인] Studies/results: Study24 hospital shared strength / results_viewed=yes
  - [판정] Stage A candidate=NO; final candidate=NO
  - [확인] Provenance: runs/hospital_shared_strength_v1_prepared/manifest.json; results/hospital_shared_strength_v1/summary.json; _docs/notes/tsfm_topics/08_hospital_shared_strength/24_hospital_results_20260910.md
  - [확인] E1/E2 results were opened in the pilot; local canonical has no later months.
- [확인] 27. synthetic Markov DGP / `dependence_screen_v1`
  - [확인] Target/series: synthetic generated conditions
  - [확인] Range/roles: length 512; context 128; train 64; val 32; devtest 128 / data_gate_only
  - [확인] Studies/results: Study36 dependence screen / results_viewed=no forecast PEFT result
  - [판정] Stage A candidate=NO; final candidate=NOT_APPLICABLE
  - [확인] Provenance: results/peft_dependence_screen_v1/data_gate/summary.json; _docs/history/2026-09-11.md
  - [판정] Synthetic design gate is not a real fresh holdout.

## PEFT directory coverage inventory

[확인] 로컬 `experiments/peft_*` 및 `results/peft_*` 디렉터리 inventory 행 수: 55.

[판정] 아래 inventory는 디렉터리별로 원장 row 매핑 또는 명시적 UNKNOWN을 남기는 coverage 장부다. 후보 가용성 인증이 아니라 기존 노출 추적용이다.

- [확인] `experiments/peft_adaptation_scope_v1` -> adaptation_scope_s1_ettm2; adaptation_scope_s1_jena2024
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: ETTm2 2018-03-02~2018-06-25; Jena 2024-09-07~2024-12-31
  - [확인] Provenance: results/peft_adaptation_scope_v1/summary_and_costs.json; _docs/notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md
  - [확인] S1 opened validation/evaluation target results for both units.
- [확인] `experiments/peft_calibration_closure_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: Study11 CPU calibration reuses Study10 synthetic predictions; no new calendar source
  - [확인] Provenance: results/peft_calibration_closure_v1/summary.json; _docs/notes/tsfm_topics/03_selection_calibration/11_peft_calibration_closure_results_20260908.md
  - [확인] Calibration closure does not certify a fresh real holdout.
- [확인] `experiments/peft_capacity_probe_v1` -> fullft_reference_P0 Bike/Household; overlap_transfer_P1 Bike/Household
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Capacity probe uses P0 and P1 Bike/Household contracts and opened probe metrics
  - [확인] Provenance: results/peft_capacity_probe_v1/evidence/plan.json; results/peft_capacity_probe_v1/evidence/P0_FULL90_contract.json; results/peft_capacity_probe_v1/evidence/P1_FULL90_contract.json; results/peft_capacity_probe_v1/summary.json
  - [확인] Study30 is covered by P0/P1 ledger rows.
- [확인] `experiments/peft_coarse_supervision_v1` -> coarse_supervision_bdg2_eagle_lamb_2016_2017
  - [확인] Coverage: [확인] mapped_to_ledger_row
  - [확인] Source/range: BDG2 raw 2016-01-01~2017-12-31; train 2016-02~2016-12; validation 2017-01~2017-03; evaluation 2017-04~2017-06
  - [확인] Provenance: runs/peft_coarse_supervision_v1/data/metadata.json; runs/peft_coarse_supervision_v1/data/data_build_contract.json; results/peft_coarse_supervision_v1/independent_evaluation_audit.json; _docs/notes/tsfm_topics/04_objective_observation/17_coarse_supervision_results_20260908.md
  - [확인] Missing BDG2 coarse-supervision ledger gap filled.
- [확인] `experiments/peft_contribution_freeze_v1` -> contribution_freeze_bdg2_eagle2017; contribution_freeze_jena2023
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: BDG2 2017-05-04~2018-01-01; Jena 2023-05-04~2024-01-01
  - [확인] Provenance: results/peft_contribution_freeze_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/31_contribution_freeze_20260911.md
  - [확인] Corrects previous BMRA mislabel for Study31.
- [확인] `experiments/peft_decision_transfer_v1` -> decision_transfer_dev_jena2021; decision_transfer_test_jena2022; decision_transfer_dev_bmra2022; decision_transfer_test_bmra2023
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Jena 2021/2022 and BMRA 2022/2023 development/test transfer windows
  - [확인] Provenance: results/peft_decision_transfer_v1/evidence/prepared_summary.json; experiments/peft_decision_transfer_v1/PURPOSE_DATA.md; _docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md
  - [확인] Study33 opened transfer evidence; final status is NO.
- [확인] `experiments/peft_dependence_screen_v1` -> dependence_screen_v1 synthetic Markov DGP
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic length 512; context 128; train 64; val 32; devtest 128
  - [확인] Provenance: results/peft_dependence_screen_v1/data_gate/summary.json; _docs/history/2026-09-11.md
  - [판정] Data gate stopped before GPU Phase 1; not a real fresh holdout.
- [확인] `experiments/peft_external_gap_v1` -> external_gap_first_block Bike; external_gap_first_block Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows_with_manifest_boundaries
  - [확인] Source/range: Bike 2011-01-01~2011-07-10; Household 2006-12-17~2007-06-25, with train/V/C/eval split boundaries from manifest
  - [확인] Provenance: runs/peft_external_gap_v1/prepared/manifest.json; results/peft_external_gap_v1/diagnostics.json; _docs/notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_results_20260908.md
  - [확인] Parent spot-check gap fixed: exact retained timestamp boundaries exist in prepared manifest.
- [미검증] `experiments/peft_fullft_reference_v1` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [미검증] version_specific_outputs_UNKNOWN_superseded_experiment_version_mapped_via_v3_manifest
  - [확인] Source/range: UNKNOWN for version-specific v1 outputs because no local results/peft_fullft_reference_v1 directory exists; P0 exposure represented by v3 manifest: Bike 2012-01-16~2012-09-14, Household 2008-01-01~2008-08-30
  - [확인] Provenance: runs/peft_fullft_reference_v3/prepared/manifest.json; results/peft_fullft_reference_v3/summary.json
  - [미검증] Experiment directory is mapped by study lineage, but this audit did not replay v1 separately and does not certify version-specific v1 outputs.
- [미검증] `experiments/peft_fullft_reference_v2` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [미검증] version_specific_outputs_UNKNOWN_superseded_experiment_version_mapped_via_v3_manifest
  - [확인] Source/range: UNKNOWN for version-specific v2 outputs because no local results/peft_fullft_reference_v2 directory exists; P0 exposure represented by v3 manifest: Bike 2012-01-16~2012-09-14, Household 2008-01-01~2008-08-30
  - [확인] Provenance: runs/peft_fullft_reference_v3/prepared/manifest.json; results/peft_fullft_reference_v3/summary.json
  - [미검증] Experiment directory is mapped by study lineage, but this audit did not replay v2 separately and does not certify version-specific v2 outputs.
- [확인] `experiments/peft_fullft_reference_v3` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Bike 2012-01-16~2012-09-14; Household 2008-01-01~2008-08-30
  - [확인] Provenance: runs/peft_fullft_reference_v3/prepared/manifest.json; results/peft_fullft_reference_v3/summary.json; _docs/notes/tsfm_topics/06_fullft_reference/20_peft_fullft_reference_results_20260909.md
  - [확인] Full fine-tuning reference results are development-exposed P0 units.
- [확인] `experiments/peft_future_utility_v1` -> contribution_freeze_bdg2_eagle2017; contribution_freeze_jena2023
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Reuses Study31 BDG2/Jena development units
  - [확인] Provenance: _docs/notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md; results/peft_future_utility_v1/summary.json
  - [확인] Future-utility artifacts do not introduce sealed final data.
- [확인] `experiments/peft_head_complement_v1` -> adaptation_scope_s1_ettm2; synthetic geometry checks
  - [확인] Coverage: [확인] mapped_real_features_no_new_targets
  - [확인] Source/range: ETTm2 train-origin feature cache from S1; synthetic CPU checks; validation/evaluation target values marked unused in entry gate
  - [확인] Provenance: results/peft_head_complement_v1/entry_gate.json; results/peft_head_complement_v1/entry_gate_verified.json; _docs/notes/tsfm_topics/02_adaptation_scope/07_head_complement_method_gate_results_20260908.md
  - [확인] Directory does not add a clean holdout; ETTm2 source is already development-exposed by S1.
- [확인] `experiments/peft_head_convergence_v1` -> head_convergence_jena2018; head_convergence_bmra2017; excluded_jena2020
  - [확인] Coverage: [확인] mapped_to_ledger_rows_and_exclusion_row
  - [확인] Source/range: Jena 2018-05-04~2019-01-01; BMRA 2017-05-04~2018-01-01; Jena2020 excluded due prior raw exposure
  - [확인] Provenance: results/peft_head_convergence_v1/evidence/prepared_summary.json; experiments/peft_head_convergence_v1/PURPOSE.md; _docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md
  - [확인] Includes excluded Jena2020 so it is not silently treated as fresh.
- [확인] `experiments/peft_initial_headroom_v1` -> initial_headroom_jena2019; initial_headroom_bmra2018
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Jena 2019-05-04~2020-01-01; BMRA 2018-01-04~2018-09-03
  - [확인] Provenance: results/peft_initial_headroom_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md
  - [확인] Study34 opened development evidence.
- [확인] `experiments/peft_mechanism_diagnostics_v1` -> fullft_reference_P0 Bike/Household; hospital_monthly_2000_2006
  - [확인] Coverage: [확인] mapped_to_development_rows
  - [확인] Source/range: P0 Bike/Household via reference summaries; hospital strength eval years 2005/2006
  - [확인] Provenance: results/peft_mechanism_diagnostics_v1/reference/summary.json; results/peft_mechanism_diagnostics_v1/hospital_strength/summary.json
  - [확인] Diagnostics are downstream analyses of already-opened development outputs.
- [확인] `experiments/peft_method_pilot_screen_v1` -> external_gap_first_block Bike/Household; temporal_replication_block Bike/Household; coarse_supervision_bdg2_eagle_lamb_2016_2017
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Uses Study12/13 selected_results, Study17 BDG2 evaluation truth/predictions, and Bike raw covariate correlation screen
  - [확인] Provenance: results/peft_method_pilot_screen_v1/source_hashes.json; results/peft_method_pilot_screen_v1/adaptation_abstain_control.json; results/peft_method_pilot_screen_v1/covariate_lag_control.json; results/peft_method_pilot_screen_v1/pattern_preserving_control.json
  - [확인] Method pilot screen reuses already-opened development artifacts; [미검증] exact covariate-lag raw Bike analysis window is not separately frozen in the artifact, so it is conservatively mapped to the already exposed Bike raw/source family rather than certified as a clean unit.
- [확인] `experiments/peft_module_ablation_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic episodes, no calendar source
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/09_peft_module_ablation_results_20260908.md
  - [판정] Synthetic data is not a real fresh source/period unit.
- [확인] `experiments/peft_objective_alignment_v1` -> external_gap_first_block Bike; external_gap_first_block Household
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Study15 reuses Study12 Bike/Household train63/V13/C13/E83 origins at L336/H48
  - [확인] Provenance: _docs/notes/tsfm_topics/04_objective_observation/15_peft_objective_alignment_results_20260908.md
  - [확인] Objective alignment is not a new clean holdout source.
- [확인] `experiments/peft_observation_operator_entry_v1` -> observation_operator_entry_uscrn_az_tucson_2024; synthetic operator semantics checks
  - [확인] Coverage: [확인] mapped_to_qc_entry_row
  - [확인] Source/range: USCRN AZ_Tucson_11_W hourly 2024-01-01 01:00~2025-01-01 00:00 UTC; subhourly 2024-01-01 00:05~2025-01-01 00:00 UTC
  - [확인] Provenance: experiments/peft_observation_operator_entry_v1/README.md; data_external/uscrn_operator_entry_v1/contract.json; data_external/uscrn_operator_entry_v1/qc.json; results/peft_observation_operator_entry_v1/final_audit.json
  - [확인] Data were downloaded/QC'd for observation-operator semantics and no forecast score was treated as sealed final.
- [확인] `experiments/peft_optimization_control_v1` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: P0 Bike/Household from fullft reference v3 contract; FULL90/SPREAD30/RECENT30 training-row controls
  - [확인] Provenance: results/peft_optimization_control_v1/evidence/contract.json; results/peft_optimization_control_v1/summary.json; results/peft_optimization_control_v1/exposure_unit_audit.json
  - [확인] Optimization-control artifacts are development exposure over P0 units.
- [확인] `experiments/peft_overlap_transfer_v1` -> overlap_transfer_P1 Bike; overlap_transfer_P1 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Bike 2012-04-06~2012-12-04; Household 2008-08-30~2009-04-29
  - [확인] Provenance: runs/peft_overlap_transfer_v1/prepared/manifest.json; results/peft_overlap_transfer_v1/evidence/prepared_manifest.json; _docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md
  - [확인] P1 overlap-transfer results are development exposure.
- [확인] `experiments/peft_r1_validation_v1` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: P0 Bike/Household from fullft reference v3 contract
  - [확인] Provenance: results/peft_r1_validation_v1/evidence/contract.json; results/peft_r1_validation_v1/summary.json
  - [확인] R1 validation opened metrics/residual artifacts on P0 and cannot serve as final.
- [확인] `experiments/peft_revision_entry_v1` -> revision_entry_payems_indpro_2020_2024
  - [확인] Coverage: [확인] mapped_to_ledger_row
  - [확인] Source/range: ALFRED PAYEMS/INDPRO event months 1989-12-01~2024-12-01; evaluation origins/target events 2020-01-01~2024-12-01
  - [확인] Provenance: results/peft_revision_entry_v1/data_qc.json; results/peft_revision_entry_v1/origin_audit.json; runs/peft_revision_entry_v1/validation_selection.json; results/peft_revision_entry_v1/cpu_results.json
  - [확인] Missing ALFRED revision-entry ledger gap filled.
- [확인] `experiments/peft_selection_regret_v1` -> external_gap_first_block Bike/Household; temporal_replication_block Bike/Household
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Study14 uses Study12 and Study13 four cells; no separate fresh unit
  - [확인] Provenance: results/peft_selection_regret_v1/selected_results.csv; _docs/notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md
  - [확인] Selection/calibration artifacts reuse already-opened Bike/Household cells.
- [확인] `experiments/peft_shift_mechanism_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic episodes, no calendar source
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/08_peft_shift_mechanism_results_20260908.md
  - [판정] Synthetic data is not a real fresh source/period unit.
- [확인] `experiments/peft_temporal_replication_v1` -> temporal_replication_block Bike; temporal_replication_block Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows_with_manifest_boundaries
  - [확인] Source/range: Bike 2011-07-10~2012-01-16; Household 2007-06-25~2008-01-01
  - [확인] Provenance: runs/peft_temporal_replication_v1/prepared/manifest.json; results/peft_temporal_replication_v1/selected_results.csv; _docs/notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_results_20260908.md
  - [확인] Study13 opened selected results for both units.
- [확인] `experiments/peft_trainlag_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic episodes, no calendar source
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/10_peft_trainlag_results_20260908.md
  - [판정] Synthetic data is not a real fresh source/period unit.
- [확인] `results/peft_adaptation_scope_v1` -> adaptation_scope_s1_ettm2; adaptation_scope_s1_jena2024
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: ETTm2 2018-03-02~2018-06-25; Jena 2024-09-07~2024-12-31
  - [확인] Provenance: results/peft_adaptation_scope_v1/summary_and_costs.json; _docs/notes/tsfm_topics/02_adaptation_scope/04_peft_adaptation_scope_s1_results_20260908.md
  - [확인] S1 opened validation/evaluation target results for both units.
- [확인] `results/peft_calibration_closure_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: Study11 CPU calibration reuses Study10 synthetic predictions; no new calendar source
  - [확인] Provenance: results/peft_calibration_closure_v1/summary.json; _docs/notes/tsfm_topics/03_selection_calibration/11_peft_calibration_closure_results_20260908.md
  - [확인] Calibration closure does not certify a fresh real holdout.
- [확인] `results/peft_capacity_probe_v1` -> fullft_reference_P0 Bike/Household; overlap_transfer_P1 Bike/Household
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Capacity probe uses P0 and P1 Bike/Household contracts and opened probe metrics
  - [확인] Provenance: results/peft_capacity_probe_v1/evidence/plan.json; results/peft_capacity_probe_v1/evidence/P0_FULL90_contract.json; results/peft_capacity_probe_v1/evidence/P1_FULL90_contract.json; results/peft_capacity_probe_v1/summary.json
  - [확인] Study30 is covered by P0/P1 ledger rows.
- [확인] `results/peft_coarse_supervision_v1` -> coarse_supervision_bdg2_eagle_lamb_2016_2017
  - [확인] Coverage: [확인] mapped_to_ledger_row
  - [확인] Source/range: BDG2 raw 2016-01-01~2017-12-31; train 2016-02~2016-12; validation 2017-01~2017-03; evaluation 2017-04~2017-06
  - [확인] Provenance: runs/peft_coarse_supervision_v1/data/metadata.json; runs/peft_coarse_supervision_v1/data/data_build_contract.json; results/peft_coarse_supervision_v1/independent_evaluation_audit.json; _docs/notes/tsfm_topics/04_objective_observation/17_coarse_supervision_results_20260908.md
  - [확인] Missing BDG2 coarse-supervision ledger gap filled.
- [확인] `results/peft_contribution_freeze_v1` -> contribution_freeze_bdg2_eagle2017; contribution_freeze_jena2023
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: BDG2 2017-05-04~2018-01-01; Jena 2023-05-04~2024-01-01
  - [확인] Provenance: results/peft_contribution_freeze_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/31_contribution_freeze_20260911.md
  - [확인] Corrects previous BMRA mislabel for Study31.
- [확인] `results/peft_decision_transfer_v1` -> decision_transfer_dev_jena2021; decision_transfer_test_jena2022; decision_transfer_dev_bmra2022; decision_transfer_test_bmra2023
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Jena 2021/2022 and BMRA 2022/2023 development/test transfer windows
  - [확인] Provenance: results/peft_decision_transfer_v1/evidence/prepared_summary.json; experiments/peft_decision_transfer_v1/PURPOSE_DATA.md; _docs/notes/tsfm_topics/07_research_direction/33_decision_transfer_20260911.md
  - [확인] Study33 opened transfer evidence; final status is NO.
- [확인] `results/peft_dependence_screen_v1` -> dependence_screen_v1 synthetic Markov DGP
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic length 512; context 128; train 64; val 32; devtest 128
  - [확인] Provenance: results/peft_dependence_screen_v1/data_gate/summary.json; _docs/history/2026-09-11.md
  - [판정] Data gate stopped before GPU Phase 1; not a real fresh holdout.
- [확인] `results/peft_external_gap_v1` -> external_gap_first_block Bike; external_gap_first_block Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows_with_manifest_boundaries
  - [확인] Source/range: Bike 2011-01-01~2011-07-10; Household 2006-12-17~2007-06-25, with train/V/C/eval split boundaries from manifest
  - [확인] Provenance: runs/peft_external_gap_v1/prepared/manifest.json; results/peft_external_gap_v1/diagnostics.json; _docs/notes/tsfm_topics/02_adaptation_scope/12_peft_external_gap_results_20260908.md
  - [확인] Parent spot-check gap fixed: exact retained timestamp boundaries exist in prepared manifest.
- [확인] `results/peft_fullft_reference_v3` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Bike 2012-01-16~2012-09-14; Household 2008-01-01~2008-08-30
  - [확인] Provenance: runs/peft_fullft_reference_v3/prepared/manifest.json; results/peft_fullft_reference_v3/summary.json; _docs/notes/tsfm_topics/06_fullft_reference/20_peft_fullft_reference_results_20260909.md
  - [확인] Full fine-tuning reference results are development-exposed P0 units.
- [확인] `results/peft_future_utility_v1` -> contribution_freeze_bdg2_eagle2017; contribution_freeze_jena2023
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Reuses Study31 BDG2/Jena development units
  - [확인] Provenance: _docs/notes/tsfm_topics/07_research_direction/32_future_utility_20260911.md; results/peft_future_utility_v1/summary.json
  - [확인] Future-utility artifacts do not introduce sealed final data.
- [확인] `results/peft_head_complement_v1` -> adaptation_scope_s1_ettm2; synthetic geometry checks
  - [확인] Coverage: [확인] mapped_real_features_no_new_targets
  - [확인] Source/range: ETTm2 train-origin feature cache from S1; synthetic CPU checks; validation/evaluation target values marked unused in entry gate
  - [확인] Provenance: results/peft_head_complement_v1/entry_gate.json; results/peft_head_complement_v1/entry_gate_verified.json; _docs/notes/tsfm_topics/02_adaptation_scope/07_head_complement_method_gate_results_20260908.md
  - [확인] Directory does not add a clean holdout; ETTm2 source is already development-exposed by S1.
- [확인] `results/peft_head_convergence_v1` -> head_convergence_jena2018; head_convergence_bmra2017; excluded_jena2020
  - [확인] Coverage: [확인] mapped_to_ledger_rows_and_exclusion_row
  - [확인] Source/range: Jena 2018-05-04~2019-01-01; BMRA 2017-05-04~2018-01-01; Jena2020 excluded due prior raw exposure
  - [확인] Provenance: results/peft_head_convergence_v1/evidence/prepared_summary.json; experiments/peft_head_convergence_v1/PURPOSE.md; _docs/notes/tsfm_topics/07_research_direction/35_head_convergence_20260911.md
  - [확인] Includes excluded Jena2020 so it is not silently treated as fresh.
- [확인] `results/peft_initial_headroom_v1` -> initial_headroom_jena2019; initial_headroom_bmra2018
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Jena 2019-05-04~2020-01-01; BMRA 2018-01-04~2018-09-03
  - [확인] Provenance: results/peft_initial_headroom_v1/evidence/prepared_summary.json; _docs/notes/tsfm_topics/07_research_direction/34_initial_headroom_20260911.md
  - [확인] Study34 opened development evidence.
- [확인] `results/peft_mechanism_diagnostics_v1` -> fullft_reference_P0 Bike/Household; hospital_monthly_2000_2006
  - [확인] Coverage: [확인] mapped_to_development_rows
  - [확인] Source/range: P0 Bike/Household via reference summaries; hospital strength eval years 2005/2006
  - [확인] Provenance: results/peft_mechanism_diagnostics_v1/reference/summary.json; results/peft_mechanism_diagnostics_v1/hospital_strength/summary.json
  - [확인] Diagnostics are downstream analyses of already-opened development outputs.
- [확인] `results/peft_method_pilot_screen_v1` -> external_gap_first_block Bike/Household; temporal_replication_block Bike/Household; coarse_supervision_bdg2_eagle_lamb_2016_2017
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Uses Study12/13 selected_results, Study17 BDG2 evaluation truth/predictions, and Bike raw covariate correlation screen
  - [확인] Provenance: results/peft_method_pilot_screen_v1/source_hashes.json; results/peft_method_pilot_screen_v1/adaptation_abstain_control.json; results/peft_method_pilot_screen_v1/covariate_lag_control.json; results/peft_method_pilot_screen_v1/pattern_preserving_control.json
  - [확인] Method pilot screen reuses already-opened development artifacts; [미검증] exact covariate-lag raw Bike analysis window is not separately frozen in the artifact, so it is conservatively mapped to the already exposed Bike raw/source family rather than certified as a clean unit.
- [확인] `results/peft_module_ablation_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic episodes, no calendar source
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/09_peft_module_ablation_results_20260908.md
  - [판정] Synthetic data is not a real fresh source/period unit.
- [확인] `results/peft_objective_alignment_v1` -> external_gap_first_block Bike; external_gap_first_block Household
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Study15 reuses Study12 Bike/Household train63/V13/C13/E83 origins at L336/H48
  - [확인] Provenance: _docs/notes/tsfm_topics/04_objective_observation/15_peft_objective_alignment_results_20260908.md
  - [확인] Objective alignment is not a new clean holdout source.
- [확인] `results/peft_observation_operator_entry_v1` -> observation_operator_entry_uscrn_az_tucson_2024; synthetic operator semantics checks
  - [확인] Coverage: [확인] mapped_to_qc_entry_row
  - [확인] Source/range: USCRN AZ_Tucson_11_W hourly 2024-01-01 01:00~2025-01-01 00:00 UTC; subhourly 2024-01-01 00:05~2025-01-01 00:00 UTC
  - [확인] Provenance: experiments/peft_observation_operator_entry_v1/README.md; data_external/uscrn_operator_entry_v1/contract.json; data_external/uscrn_operator_entry_v1/qc.json; results/peft_observation_operator_entry_v1/final_audit.json
  - [확인] Data were downloaded/QC'd for observation-operator semantics and no forecast score was treated as sealed final.
- [확인] `results/peft_optimization_control_v1` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: P0 Bike/Household from fullft reference v3 contract; FULL90/SPREAD30/RECENT30 training-row controls
  - [확인] Provenance: results/peft_optimization_control_v1/evidence/contract.json; results/peft_optimization_control_v1/summary.json; results/peft_optimization_control_v1/exposure_unit_audit.json
  - [확인] Optimization-control artifacts are development exposure over P0 units.
- [확인] `results/peft_overlap_transfer_v1` -> overlap_transfer_P1 Bike; overlap_transfer_P1 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: Bike 2012-04-06~2012-12-04; Household 2008-08-30~2009-04-29
  - [확인] Provenance: runs/peft_overlap_transfer_v1/prepared/manifest.json; results/peft_overlap_transfer_v1/evidence/prepared_manifest.json; _docs/notes/tsfm_topics/07_research_direction/29_overlap_transfer_20260910.md
  - [확인] P1 overlap-transfer results are development exposure.
- [확인] `results/peft_paper_closure_v1` -> audit package only
  - [확인] Coverage: [확인] audit_output_not_data_source
  - [확인] Source/range: No PEFT target exposure introduced by this audit script; outputs summarize existing metadata/artifacts only
  - [확인] Provenance: results/peft_paper_closure_v1/build_data_exposure_audit.py; results/peft_paper_closure_v1/data_exposure_ledger.csv; results/peft_paper_closure_v1/holdout_availability.json
  - [확인] Owned closure artifacts only.
- [확인] `results/peft_r1_validation_v1` -> fullft_reference_P0 Bike; fullft_reference_P0 Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows
  - [확인] Source/range: P0 Bike/Household from fullft reference v3 contract
  - [확인] Provenance: results/peft_r1_validation_v1/evidence/contract.json; results/peft_r1_validation_v1/summary.json
  - [확인] R1 validation opened metrics/residual artifacts on P0 and cannot serve as final.
- [확인] `results/peft_revision_entry_v1` -> revision_entry_payems_indpro_2020_2024
  - [확인] Coverage: [확인] mapped_to_ledger_row
  - [확인] Source/range: ALFRED PAYEMS/INDPRO event months 1989-12-01~2024-12-01; evaluation origins/target events 2020-01-01~2024-12-01
  - [확인] Provenance: results/peft_revision_entry_v1/data_qc.json; results/peft_revision_entry_v1/origin_audit.json; runs/peft_revision_entry_v1/validation_selection.json; results/peft_revision_entry_v1/cpu_results.json
  - [확인] Missing ALFRED revision-entry ledger gap filled.
- [확인] `results/peft_selection_regret_v1` -> external_gap_first_block Bike/Household; temporal_replication_block Bike/Household
  - [확인] Coverage: [확인] mapped_to_existing_development_rows
  - [확인] Source/range: Study14 uses Study12 and Study13 four cells; no separate fresh unit
  - [확인] Provenance: results/peft_selection_regret_v1/selected_results.csv; _docs/notes/tsfm_topics/03_selection_calibration/14_peft_selection_regret_results_20260908.md
  - [확인] Selection/calibration artifacts reuse already-opened Bike/Household cells.
- [확인] `results/peft_shift_mechanism_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic episodes, no calendar source
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/08_peft_shift_mechanism_results_20260908.md
  - [판정] Synthetic data is not a real fresh source/period unit.
- [확인] `results/peft_temporal_replication_v1` -> temporal_replication_block Bike; temporal_replication_block Household
  - [확인] Coverage: [확인] mapped_to_ledger_rows_with_manifest_boundaries
  - [확인] Source/range: Bike 2011-07-10~2012-01-16; Household 2007-06-25~2008-01-01
  - [확인] Provenance: runs/peft_temporal_replication_v1/prepared/manifest.json; results/peft_temporal_replication_v1/selected_results.csv; _docs/notes/tsfm_topics/02_adaptation_scope/13_peft_temporal_replication_results_20260908.md
  - [확인] Study13 opened selected results for both units.
- [확인] `results/peft_trainlag_v1` -> peft_shift_module_trainlag_synthetic
  - [확인] Coverage: [확인] mapped_to_synthetic_ledger_row
  - [확인] Source/range: synthetic episodes, no calendar source
  - [확인] Provenance: _docs/notes/tsfm_topics/02_adaptation_scope/10_peft_trainlag_results_20260908.md
  - [판정] Synthetic data is not a real fresh source/period unit.

## 감사 한계

[미검증] 이 감사는 official Time-PEFT full-text 신규성 판정이나 새 외부 데이터 다운로드를 수행하지 않는다.

[미검증] FM pretraining overlap은 각 prepared summary가 UNKNOWN으로 남긴 경우 UNKNOWN으로 유지했다.

[판정] 후보가 local metadata로 보인다는 사실과 Stage A/final clean unit으로 인증됐다는 사실을 분리해야 한다.
