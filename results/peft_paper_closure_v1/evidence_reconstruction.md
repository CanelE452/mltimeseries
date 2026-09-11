# Existing PEFT evidence reconstruction

[판정] PASS_CORE_RECONSTRUCTION. [확인] 936 numeric checks; 486 hashed input files; inputs unchanged=True.

[확인] CPU NumPy reconstruction only: GPU fit 0 / forecast 0. Runtime 3.911 seconds. GPU memory not measured (no GPU used).

[확인] Float comparison uses absolute 1e-10, rtol=0. Printed prose uses half of its final displayed decimal separately; DOCUMENT_ROUNDING_MATCH is not numerical drift.

[확인] SORT mean 2-pinball: mean over targets/quantiles of valid origin+horizon loss, divided by each target train scale.

[확인] Shared target windows across seeds, conditions and overlapping rolling origins are not independent datasets. All reconstructed periods are historical development evidence, not a fresh final test.

[미검증] UNKNOWN; no new contamination evidence is inferred.

## Main document comparisons

- [확인] Study 20 bike LoRA incremental mean: computed 3.95344665667; document 3.953; delta +0.000447; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 20 household LoRA incremental mean: computed 0.12097935451; document 0.121; delta -2.06e-05; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 26 bike matched MLP incremental mean: computed 4.17070588748; document 4.171; delta -0.000294; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 26 household matched MLP incremental mean: computed 0.417965655511; document 0.418; delta -3.43e-05; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 30 P1 Bike FULL90 WIDE vs JOINT: computed 4.77892621673; document 4.779; delta -7.38e-05; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 31 ES2/27000/time: computed 7.53036487399; document 7.53; delta +0.000365; DOCUMENT_ROUNDING_MATCH (% fit wall-clock).
- [확인] Study 31 ES2/27001/time: computed 5.03718940886; document 5.04; delta -0.00281; DOCUMENT_ROUNDING_MATCH (% fit wall-clock).
- [확인] Study 31 FIXED_FREEZE/27000/loss: computed -0.492232703209; document -0.492; delta -0.000233; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 31 FIXED_FREEZE/27000/time: computed 19.9172514602; document 19.92; delta -0.00275; DOCUMENT_ROUNDING_MATCH (% fit wall-clock).
- [확인] Study 31 FIXED_FREEZE/27001/loss: computed -0.0318116462784; document -0.032; delta +0.000188; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 31 FIXED_FREEZE/27001/time: computed 21.4607347574; document 21.46; delta +0.000735; DOCUMENT_ROUNDING_MATCH (% fit wall-clock).
- [확인] Study 31 CONTRIB_FREEZE/27000/loss: computed 0.24011280475; document 0.24; delta +0.000113; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 31 CONTRIB_FREEZE/27000/time: computed 7.42197419683; document 7.42; delta +0.00197; DOCUMENT_ROUNDING_MATCH (% fit wall-clock).
- [확인] Study 31 CONTRIB_FREEZE/27001/time: computed 3.77412921595; document 3.77; delta +0.00413; DOCUMENT_ROUNDING_MATCH (% fit wall-clock).
- [확인] Study 32 mean U final: computed -4.3503002324; document -4.35; delta -0.0003; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 32 mean U selected: computed -0.727274993749; document -0.727; delta -0.000275; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 33 PROBE mean gain vs HEAD: computed 0.744033175583; document 0.744; delta +3.32e-05; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 33 PROBE mean gain vs WIDE: computed 0.944032280945; document 0.944; delta +3.23e-05; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 34 JOINT gain vs HEAD: computed 2.03722153385; document 2.037; delta +0.000222; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 34 JOINT gain vs WIDE: computed 2.42771700945; document 2.428; delta -0.000283; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 34 JOINT gain vs F0: computed 1.41826448883; document 1.418; delta +0.000264; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 35 JOINT gain vs HEAD: computed 12.1346322306; document 12.135; delta -0.000368; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 35 JOINT gain vs WIDE: computed 8.80432348134; document 8.804; delta +0.000323; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 35 JOINT gain vs F0: computed -4.30480924505; document -4.305; delta +0.000191; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study Hospital LoRA vs F0: computed 1.18257830155; document 1.183; delta -0.000422; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study Hospital individual vs global: computed -0.411889897154; document -0.41189; delta +1.03e-07; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study Hospital individual vs shuffled: computed -0.0553658781691; document -0.055366; delta +1.22e-07; DOCUMENT_ROUNDING_MATCH (%F0).
- [확인] Study 36 Bayes one-step MSE: computed 0.46; document 0.46; delta -5.55e-17; FLOAT_MATCH (MSE).
- [확인] Study 36 sign_conjugacy_max_error: computed 0; document 0; delta +0; FLOAT_MATCH (absolute population error).
- [확인] Study 36 observation_sign_max_error: computed 0; document 0; delta +0; FLOAT_MATCH (absolute population error).
- [확인] Study 36 acf_max_difference_h1_128: computed 5.55111512313e-17; document 0; delta +5.55e-17; FLOAT_MATCH (absolute population error).
- [확인] Study 36 risk_max_difference_h1_128: computed 2.22044604925e-16; document 0; delta +2.22e-16; FLOAT_MATCH (absolute population error).
- [확인] Study 36 psd_max_difference_257_freqs: computed 1.22124532709e-15; document 0; delta +1.22e-15; FLOAT_MATCH (absolute population error).
- [확인] Study 36 forward/coefficient/0: computed 0.6; document 0.6; delta +0; FLOAT_MATCH (polynomial coefficient).
- [확인] Study 36 forward/coefficient/1: computed -0.2; document -0.2; delta +0; FLOAT_MATCH (polynomial coefficient).
- [확인] Study 36 forward/coefficient/2: computed -0.9; document -0.9; delta +0; FLOAT_MATCH (polynomial coefficient).
- [확인] Study 36 reverse/coefficient/0: computed -0.6; document -0.6; delta +0; FLOAT_MATCH (polynomial coefficient).
- [확인] Study 36 reverse/coefficient/1: computed -0.2; document -0.2; delta +0; FLOAT_MATCH (polynomial coefficient).
- [확인] Study 36 reverse/coefficient/2: computed 0.9; document 0.9; delta +0; FLOAT_MATCH (polynomial coefficient).

## Interpretation and coverage

[판정] Study20/26/30 support procedure-specific internal adaptation benefit under the recorded controls. They do not prove absent representation information or optimality over all possible heads.

[확인] Study32 current contribution is positive in 21 of 24 dependent forks; 16 of those positive-contribution forks have negative future update utility. [판정] Current contribution and update utility are distinct estimands.

[확인] Study35 L720 JOINT F0-relative loss increase: 4.304809245%F0; it beats WIDE while losing to F0 in all four cells.

[판정] First-order observed-state Markov property makes latest state sufficient at every horizon; sign relabeling confounds direction. Current DGP remains STOP.

[확인] Main primary scores and contrasts reconstructed; no retraining, model-selection opportunity audit, interval recomputation, full secondary calibration audit or causal mechanism test in this script.

[미검증] Study32 S_off uses saved raw run JSON; off predictions absent.
[미검증] Hospital individual/shuffled selection and blended predictions are not reimplemented; elementary CSV contrast is independent, F0/LoRA raw scoring verified.
[미검증] Study34 correction and G2 are not independently reimplemented here.
[미검증] Study36 finite-sample data-gate metrics are not reimplemented; population audit is independent.

## Reproduction and provenance

[확인] Run `.venv-peft/Scripts/python.exe results/peft_paper_closure_v1/reconstruct_evidence.py`. Script imports only stdlib/NumPy and writes its two closure reports.

[확인] Per-check expected/computed/delta/tolerance and SHA-256 for every accessed raw NPZ, CSV, run JSON and document are in `evidence_reconstruction.json`. Existing scripts were read for contracts but not executed.
