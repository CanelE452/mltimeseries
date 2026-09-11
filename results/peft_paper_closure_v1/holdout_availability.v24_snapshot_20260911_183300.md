# Holdout availability

[판정] Stage A: `CLEAN_FRESH_UNITS_NOT_YET_CERTIFIED`.

[확인] Certified clean fresh units for Stage A: 0 / required >=2.

[확인] Parent novelty audit reports Stage A is already blocked by method novelty gate UNKNOWN, so this audit does not justify downloading target values or running training to force a data-only conclusion.

[판정] Sealed final: `SEALED_FINAL_NOT_AVAILABLE_LOCALLY_YET`.

[확인] Certified sealed final units: 0 / required >=2.

[판정] Do not state `NO_CLEAN_HOLDOUT_AVAILABLE`. The accurate conclusion is that no certified clean manifests exist yet in local artifacts, while external/global availability remains unverified.

## Candidate inventory

- [추정] metadata-time candidate Household later chronological period after 2009-04-29: [추정] Local source timestamp span extends to 2010-11-26, so a later block may fit; no target coverage or prior-exposure clean manifest was produced.
- [추정] weak local candidate Bike later chronological period after 2012-12-04: [추정] Local Bike source ends 2012-12-31, leaving too little post-P1 calendar for a comparable 242-day block; this is local/timestamp reasoning only.
- [미검증] possible but uncertified BDG2 alternate buildings or periods: [미검증] Current audit did not build a target-blind manifest for other meters/periods and did not audit GIFT/Chronos pretraining overlap.
- [미검증] possible but uncertified BMRA unexposed PEFT periods or alternate unit pairs: [확인] PEFT exposed 2017/2018/2022/2023 BMRA windows; prepared summaries also record non-PEFT UCP processing for 2019~2021. Clean Stage A use would need a separate target-blind manifest and exposure decision.
- [판정] not certified from current local set Jena unused local years: [확인] Jena 2018/2019/2021/2022/2023/2024 were exposed in PEFT development or S1; Jena2020 was excluded due prior weather exposure. Other external Jena-derived sources were not audited here.
- [판정] no local future period in canonical artifact Hospital future months: [확인] Local canonical manifest spans 2000-01~2006-12 and E1/E2 were opened; this does not rule out other hospital datasets outside the local artifact.
- [미검증] not searched/downloaded New external source families: [미검증] This audit intentionally avoided new target downloads; global data availability remains unknown.

## Minimum safe next step

[판정] If the project resumes after novelty gate resolution, first write a target-blind candidate manifest for at least two units. The manifest should freeze source family, period, targets/series IDs, split boundaries, QC thresholds, exclusion rules, and whether the unit is Stage A development or sealed final before any target values are opened.

[판정] Stage A and final must be separated. Stage A can use fresh development units after preregistration, but final must remain sealed until claim/method/hparams/threshold/dataset rules are frozen.
