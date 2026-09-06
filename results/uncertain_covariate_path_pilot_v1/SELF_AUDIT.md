# UCP-PATH-PILOT-v1 — SELF_AUDIT

Instruction section 34. Nothing below is asserted from memory; each row was
recomputed from the files this run wrote, or from git, at report time.

| check | result | evidence |
|---|---|---|
| artefacts_present | pass | `{"missing": []}` |
| integrity_tests | pass | `{}` |
| unit_tests | pass | `{"summary": "10 passed in 1.42s"}` |
| A18_seed_aggregation_matches_point_estimate | pass | `{"max_abs_diff": 5.960464477539063e-08, "per_arm": {"D": 5.551115123125783e-17, "H": 5.960464477539063e-08, "M": 9.934107481068821e-09, "MC": 9.934107481068821e-09, "P": 2.9802322387695312e-08, "P_BROKEN": 2.9802322387695312e-08, "S": 0.0}}` |
| A17_bootstrap_same_estimand_as_metrics | pass | `{"max_abs_diff": 0.0}` |
| A16_ri_sign_convention | pass | `{"definition": "RI(A over B) = 100 * (L_B - L_A) / L_B, positive means A is better"}` |
| A12_d_p_parameter_equality | pass | `{"D": 65921, "P": 65921}` |
| A07_whole_path_permutation_invariance | pass | `{"max_abs_diff": 4.76837158203125e-07, "tolerance": 1e-06}` |
| A08_A10_broken_path_marginals | pass | `{"per_lead_multiset_max_abs_diff": 0.0, "linkage_actually_broken_max_abs_diff": 6.594143867492676}` |
| A15_mc_uses_empirical_union | pass | `{"particles": 1600, "expected": 1600, "note": "member-conditional particles are concatenated; no quantile is ever averaged"}` |
| test_not_used_for_tuning | pass | `{"note": "checkpoints are chosen on the 2020 validation macro CRPS; the 2021 test split is scored once, after every fit is finished"}` |
| no_raw_or_cache_in_git | pass | `{"offending": []}` |
| oa_artefacts_untouched | pass | `{"note": "origin/main carries no OA artefacts; this branch was cut from origin/main, so there is nothing here to modify", "tracked_oa_files": []}` |
| origin_main_unchanged | pass | `{"at_start": "36b01d2b84df16f03447a8e214723fc61ae147fd", "now": "36b01d2b84df16f03447a8e214723fc61ae147fd"}` |
| working_tree | pass | `{"entries": ["M results/uncertain_covariate_path_pilot_v1/preprocessing_manifest.json", "?? _docs/history/.pending.md", "?? results/hq_token_pilot_v1_dryrun/", "?? results/oa_resolution_pilot_v1_dryrun/", "?? results/uncertain_covariate_path_pilot_v1/bootst...` |

All checks passed: **True**

Not covered by these checks, and stated rather than hidden: real BMRA
publication latency was not verified, so metered history is assumed available at
the origin; and `per_example_losses.npz` is gitignored with the rest of the
binary artefacts, so reproducing the bootstrap from a clean checkout means
re-running `evaluate` first.
