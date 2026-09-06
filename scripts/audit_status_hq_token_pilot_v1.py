"""Build the audit STATUS.md from the audit artifacts alone.

Reads only files inside results/hq_token_pilot_v1/audit_closure_v1/ and imports nothing
from the experiment package, so the report is regenerable from what it cites and every
number in it can be traced back to a stored file.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "hq_token_pilot_v1", "audit_closure_v1")
DATASETS = ["ETTm2", "weather", "electricity"]


def _f(v, d=3):
    return "n/a" if v is None else f"{v:+.{d}f}"


def build(out_dir=OUT) -> str:
    def j(n):
        with open(os.path.join(out_dir, n), encoding="utf-8") as f:
            return json.load(f)

    def rows(n):
        with open(os.path.join(out_dir, n), encoding="utf-8") as f:
            return list(csv.DictReader(f))

    ver, ari, cap = j("audit_verdict.json"), j("arithmetic_revalidation.json"), j("capacity_control_audit.json")
    seed, gap, scope = j("seed_uncertainty_audit.json"), j("compression_gap_audit.json"), j("scope_boundary.json")
    eff, ck, fr = j("efficiency_interpretation.json"), j("checkpoint_selection_audit.json"), j("foundation_reference_scope.json")
    imm, expo = j("original_artifact_immutability.json"), j("sampling_exposure_audit.json")
    boot = j("bootstrap_revalidation.json") if os.path.exists(
        os.path.join(out_dir, "bootstrap_revalidation.json")) else None
    curves, traj = rows("training_curve_summary.csv"), rows("validation_C_vs_I_trajectory.csv")
    qm, seeds_tbl = rows("query_mechanism_summary.csv"), rows("paired_seed_effects.csv")
    cvi = ari["recomputed"]["C_vs_I"]
    obs = ver["observations_behind_the_recommendation"]

    L = ["HQ-TOKEN-PILOT-v1 AUDIT RESULT", ""]
    L += [f"ORIGINAL SCIENTIFIC DECISION: {ver['scientific_decision_original']}", ""]
    L += [f"CURRENT IMPLEMENTATION RECOMMENDATION: {ver['audit_recommendation_current_implementation']}", ""]
    L += [f"BROADER TOPIC: {ver['broader_topic_status']}", ""]
    L += ["These are three separate statements. The first is the pre-registered decision and was "
          "not revised. The second is an engineering recommendation about this exact "
          "implementation. The third says the wider question was never directly tested.", ""]

    L += ["## 1. What was audited", "",
          f"Commit `{ver['audited_commit']}`. Every number below was recomputed from the stored "
          "artifacts by a script that imports nothing from the experiment package, so a "
          "disagreement would be a real disagreement and not a shared bug.", "",
          "- the six contrasts, recomputed from `metrics.csv`",
          "- the paired bootstrap, re-derived from the raw keyed error arrays",
          "- validation curves and checkpoint selection for the six paired I and C fits",
          "- the training-schedule hashes across arms",
          "- the synthetic capacity control, the seed study, the compression gap, the query "
          "diagnostics, the efficiency table and the reference scope", ""]

    L += ["## 2. What was NOT rerun", "",
          "No model was fitted. The 27 core fits, the 24 seed-noise fits, the synthetic capacity "
          "training and the two foundation references were all left as they were. No learning "
          "rate, budget, seed, token budget or threshold was changed, and the pre-registered "
          f"decision was not revised. Existing result files were read-only: "
          f"{imm['files_checked']} were hashed before and after this audit, result "
          f"**{imm['status']}**.", ""]

    L += ["## 3. Arithmetic reproduction", "",
          f"Status **{ari['status']}**. Largest absolute disagreement over every cell, macro and "
          f"per-seed value: {ari['max_abs_difference_pct_points']:.2e} percentage points against "
          f"a tolerance of {ari['absolute_tolerance']:.0e}.", "",
          "| contrast | macro RI | seeds used |", "| --- | --- | --- |"]
    for name, c in ari["recomputed"].items():
        L.append(f"| {name.replace('_', ' ')} | {_f(c['macro'])}% | {len(c['seeds_used'])} |")
    L += ["", "| dataset | C vs I at H=96 | at H=336 |", "| --- | --- | --- |"]
    for d in DATASETS:
        L.append(f"| {d} | {_f(cvi['cells'][f'{d}|96'])}% | {_f(cvi['cells'][f'{d}|336'])}% |")
    L.append("")
    if boot:
        r, o = boot["recomputed"], boot["original"]
        L += [f"Bootstrap re-derived independently under the same contract: **{boot['status']}**. "
              f"Recomputed macro {_f(r['macro_mean'])}% with interval [{_f(r['lower95'])}, "
              f"{_f(r['upper95'])}], against the stored {_f(o['macro_mean'])}% "
              f"[{_f(o['lower95'])}, {_f(o['upper95'])}]. Effective time blocks: "
              + ", ".join(f"{k} {v}" for k, v in boot["effective_blocks"].items()) + ".", ""]
    L += ["Per-seed macro, computed within each seed:", ""]
    for r in seeds_tbl:
        if r["dataset"] == "MACRO":
            L.append(f"- seed {r['model_seed']}: {float(r['RI_pct']):+.3f}%")
    L += ["", "Two seeds are not a sample from which a standard error or an interval is derived "
          "here, and none is reported.", ""]

    L += ["## 4. Training curves", "",
          "| dataset | seed | arm | best update | position | last interval | final / best |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in curves:
        L.append(f"| {r['dataset']} | ...{r['model_seed'][-2:]} | {r['arm']} | {r['best_update']} "
                 f"| {r['position_label']} | {r['last_interval_label']} | "
                 f"{float(r['final_over_best_ratio']):.3f} |")
    L += ["", "Three different optimization states, described rather than diagnosed:", "",
          "- **ETTm2** picked its best checkpoint at the very first validation point, update 600 "
          "of 3000, after which validation loss rose by 13 to 17 percent to the end of the "
          "budget. Both arms, both seeds. The reported ETTm2 numbers therefore come from models "
          "trained for 600 updates, and the rest of the budget made them worse.",
          "- **weather** is mixed: one seed was still improving at the last checkpoint, the other "
          "peaked in the middle and then worsened.",
          "- **electricity** was still improving at the final checkpoint in both seeds, so the "
          "budget was the binding constraint there.", "",
          "These labels describe the curves. They are not a diagnosis of overfitting or of "
          "non-convergence, and no checkpoint was re-selected.", ""]
    L += ["Sampling exposure, read next to those labels:", "",
          "| dataset | eligible channel-origin pairs | windows drawn | exposure ratio | curve position |",
          "| --- | --- | --- | --- | --- |"]
    for d, v in expo["per_dataset"].items():
        L.append(f"| {d} | {v['eligible_channel_origin_pairs']:,} | {v['windows_drawn']:,} | "
                 f"{v['sampling_exposure_ratio']:.3f} | {', '.join(v['position_labels'])} |")
    L += ["", expo["note"], "", expo["reading_rule"], ""]

    L += ["## 5. Checkpoint selection", "",
          f"{ck['pairs_with_identical_best_update']} of {ck['pairs']} paired fits selected the "
          f"same update for C and for I. " + ck["reading"], "", ck["no_reselection"], "",
          "The two arms also track each other throughout training and not only at the selected "
          f"point: across the {len(traj)} shared validation checkpoints the median absolute "
          f"difference in validation MSE is "
          f"{float(np.median([abs(float(r['delta_val_pct'])) for r in traj])):.3f} percent and the "
          f"largest is {max(abs(float(r['delta_val_pct'])) for r in traj):.3f} percent. The "
          f"{sum(1 for r in traj if abs(float(r['delta_val_pct'])) > 0.5)} checkpoints above 0.5 "
          "percent are all on ETTm2 and change sign between them, so the closeness at the "
          "selected checkpoint is not manufactured by the selection.", ""]

    L += ["## 6. Query mechanism", "",
          "| dataset | H | C vs I | C vs R | query swap | mean weight shift | mean centre shift |",
          "| --- | --- | --- | --- | --- | --- | --- |"]
    for r in qm:
        L.append(f"| {r['dataset']} | {r['horizon']} | {_f(float(r['C_vs_I_RI_pct']))}% | "
                 f"{_f(float(r['C_vs_R_RI_pct']))}% | {_f(float(r['query_swap_delta_pct']))}% | "
                 f"{float(r['mean_abs_weight_diff_96_vs_336']):.5f} | "
                 f"{float(r['mean_center_shift_samples']):.3f} samples |")
    L += ["", "This is the most direct mechanism evidence in the closure. Under this "
          "implementation the true horizon query bought no measurable accuracy over an "
          "input-only query, a random horizon query performed the same, and changing only the "
          "pooling query on fixed inputs barely moved the weights or the prediction.", ""]

    L += ["## 7. Synthetic capacity control, reinterpreted", "",
          "| arm | MSE H=96 | MSE H=336 | joint equal-weight mean |", "| --- | --- | --- | --- |"]
    for arm in ("I", "C"):
        p = cap["per_horizon_mse"][arm]
        L.append(f"| {arm} | {p['96']:.5f} | {p['336']:.5f} | "
                 f"{cap['joint_equal_weight_mean_mse'][arm]:.5f} |")
    L += ["",
          f"- QUERY_PATH_RESPONSIVE = {cap['QUERY_PATH_RESPONSIVE']} (weight separation "
          f"{cap['C_weight_separation_between_horizons']:.3f})",
          f"- JOINT_HORIZON_TASK_SOLVED = {cap['JOINT_HORIZON_TASK_SOLVED']} (C is "
          f"{cap['joint_equal_weight_mean_mse']['C_over_I_ratio']:.1f} times worse than I on the "
          f"joint objective)", "",
          cap["what_this_supports"], "", cap["what_this_does_not_support"], "",
          cap["design_caveat"], ""]

    L += ["## 8. Seed uncertainty, reinterpreted", "",
          "| dataset | mean validation MSE | seed sd | sd as percent of mean | seeds |",
          "| --- | --- | --- | --- | --- |"]
    for d, v in seed["I_absolute_seed_variability"].items():
        L.append(f"| {d} | {v['mean_validation_mse']:.5f} | {v['sd']:.5f} | "
                 f"{v['sd_pct_of_mean']:.2f}% | {v['n_seeds']} |")
    L += ["", seed["what_can_be_said"], "", seed["what_cannot_be_said"], "",
          f"`paired_CI_effect_seed_variability_estimated` = "
          f"{seed['paired_CI_effect_seed_variability_estimated']}, because {seed['reason']}. The "
          f"core pair has {seed['core_pair_seed_count']} seeds.", ""]

    L += ["## 9. Compression gap, reinterpreted", "",
          f"OBSERVED_DENSE_VS_UNIFORM_COMPRESSION_GAP = "
          f"{_f(gap['OBSERVED_DENSE_VS_UNIFORM_COMPRESSION_GAP']['macro_pct'])}%, at model seed "
          f"{gap['OBSERVED_DENSE_VS_UNIFORM_COMPRESSION_GAP']['seeds_used'][0]} only.", "",
          gap["why_not"], "",
          f"`DENSE_IS_NOT_EMPIRICAL_UPPER_BOUND` = {gap['DENSE_IS_NOT_EMPIRICAL_UPPER_BOUND']}: "
          f"{len(gap['cells_where_C_beat_DENSE'])} of six cells already have the compressed C arm "
          f"ahead of the uncompressed arm"
          + ("." if not gap["cells_where_C_beat_DENSE"] else " ("
             + ", ".join(f"{c['dataset']} H={c['horizon']}" for c in gap["cells_where_C_beat_DENSE"])
             + ")."), ""]

    L += ["## 10. Efficiency", "",
          "| config | tokens | end-to-end b64 (ms) | vs dense | peak (MiB) |",
          "| --- | --- | --- | --- | --- |"]
    for k, v in eff["per_config"].items():
        L.append(f"| {k} | {v['tokens_encoded']} | {v['end_to_end_median_ms_b64']:.3f} | "
                 f"{v['end_to_end_vs_DENSE_pct']:+.1f}% | {v['peak_allocated_mib_b64']:.1f} |")
    if "microprofile" in eff:
        mp = eff["microprofile"]
        L += ["", "Per-stage micro-profile, inference only, diagnostic:", "",
              "| stage | U (32) | C (32) | DENSE (64) |", "| --- | --- | --- | --- |"]
        for s in ("embed", "pool", "encoder", "unmerge_readout_head", "total_forward"):
            r = [mp["per_stage_median_ms"][k][s] for k in ("U_B32", "C_B32", "DENSE_B64")]
            L.append(f"| {s.replace('_', ' ')} | {r[0]:.3f} | {r[1]:.3f} | {r[2]:.3f} |")
        L += ["", f"Halving the token count saves "
              f"{mp['encoder_saving_from_halving_tokens_ms']:.3f} ms in the encoder, while the "
              f"content scorer alone adds {mp['scorer_extra_cost_ms']:.3f} ms and the unmerge "
              f"path adds a further {mp['extra_decode_cost_of_unmerge_ms']:.3f} ms.", ""]
    if "measurement_disagreement" in eff:
        md = eff["measurement_disagreement"]
        L += [f"A disagreement worth stating rather than smoothing over: {md['issue']}. The "
              f"stored table puts uniform pooling {md['efficiency_csv_U_vs_DENSE_end_to_end_pct']:+.1f} "
              f"percent against dense, while the longer micro-profile puts it at "
              f"{md['microprofile_U_total_ms']:.3f} ms against "
              f"{md['microprofile_DENSE_total_ms']:.3f} ms. " + md["resolution"], ""]
    L += [eff["scope"], ""]

    L += ["## 11. Foundation reference scope", "",
          f"- Chronos-2: {fr['chronos2']['status']}, {fr['chronos2']['condition']}.",
          f"- TiRex-2: {fr['tirex2']['status']}, {fr['tirex2']['condition']}.", "",
          "These are native reference rows. No performance claim in either direction is drawn "
          "from them and neither model was rerun in this audit.", ""]

    L += ["## 12. Strong-baseline gap", "", fr["strong_dynamic_baseline"]["statement"], "",
          "That is a statement about this search, not about whether such implementations exist, "
          "and it is not a claim of superiority over recent dynamic tokenization methods.", ""]

    L += ["## 13. Claims that remain valid", ""]
    lo = _f(boot["recomputed"]["lower95"]) if boot else "n/a"
    hi = _f(boot["recomputed"]["upper95"]) if boot else "n/a"
    for s in [
        "C and I were evaluated on exactly the same frozen keys, from the same initial state and "
        "the same training sample schedule within a seed; the schedule hashes match across arms "
        "for every dataset and seed.",
        f"The primary C-versus-I effect under this pilot is {_f(cvi['macro'])} percent, with a "
        f"re-derived interval of [{lo}, {hi}] percent.",
        f"A random horizon query performed essentially the same as the true one: C versus R is "
        f"{_f(obs['C_vs_R_macro_pct'])} percent.",
        f"Swapping only the pooling query on fixed inputs changed test MSE by at most "
        f"{obs['max_abs_query_swap_effect_pct']:.4f} percent, and the mean pooling weight moved "
        f"by at most {obs['max_mean_horizon_weight_shift']:.5f} between the two horizons.",
        f"Learned pooling did outperform uniform pooling in this pilot: C versus U is "
        f"{_f(ari['recomputed']['C_vs_U']['macro'])} percent.",
        f"The phase-shifted evaluation grid gives {_f(obs['phase_shifted_macro_pct'])} percent, so "
        "the grid phase does not reverse the null.",
        "On this small architecture the 32-token C model was not faster end to end than the "
        "64-token dense model, and the per-stage profile attributes that to the scorer.",
    ]:
        L.append(f"- {s}")
    L.append("")

    L += ["## 14. Claims that are withdrawn or weakened", "",
          "| earlier claim | corrected statement |", "| --- | --- |",
          "| CAPACITY_CONFIRMED rules out architecture and optimization failure | Only "
          "query-path responsiveness was shown. On that same synthetic task the "
          "horizon-conditioned arm was worse than the input-only arm on the joint multi-horizon "
          "objective, so architecture and optimization limits are not excluded. |",
          "| The 5.05 percent seed spread puts a 1 percent effect outside the resolution of the "
          "experiment | 5.05 percent describes the absolute variability of arm I across eight "
          "seeds. The seed variability of the paired C minus I difference was never measured and "
          "could be far smaller. |",
          "| DENSE versus U bounds what any pooling rule could gain | It is one observed "
          "comparison between two trained models, not an upper bound. Two of six cells already "
          "have the compressed C arm ahead of the uncompressed arm. |",
          "| The optimizer found no reason to use the horizon | Under this architecture and "
          "training budget, horizon-conditioned pooling produced no measurable advantage and the "
          "learned weights were nearly horizon-invariant. Why is not established. |",
          "| This topic should be stopped | Stop scaling this fixed-neighbourhood implementation. "
          "The wider question of forecast-query-conditioned budget allocation remains open and "
          "was not directly tested. |", ""]

    L += ["## 15. Current implementation recommendation", "",
          f"**{ver['audit_recommendation_current_implementation']}** — "
          f"{ver['meaning']['STOP_SCALING_CURRENT_FIXED_NEIGHBORHOOD_C']}", "",
          "The observations behind it, listed rather than combined into a new threshold:", ""]
    for k, v in obs.items():
        L.append(f"- `{k}`: {v}")
    L += ["", ver["no_new_gate_created"], ""]

    L += ["## 16. What remains untested", "", "This pilot never had the freedom to:", ""]
    for s in scope["what_C_could_not_do"]:
        L.append(f"- {s}")
    L += ["", "Also untested: query-conditioned tokenizer adaptation inside a pretrained "
          "foundation model, and any regime where the token saving outweighs the pooling "
          "overhead, which at 1.5M parameters it does not.", "",
          scope["scope_of_any_conclusion"], "",
          "The horizon embedding module is shared between the pooling query and the decoder, so C "
          "and I differ by slightly more than the presence of a query. That does not flatter the "
          "current null, but it would need separating before any future positive result from this "
          "design were believed.", ""]

    L += ["## 17. Exact next research choice", "",
          "Two options, and no new training starts until one is chosen.", "",
          "**A.** Close topic 1 and move to the second topic note.", "",
          "**B.** Keep topic 1, but not by adjusting the r=2 fixed-neighbourhood mixture. Design "
          "a separate pre-registered v2 around horizon-dependent variable token allocation with "
          "movable boundaries, which is the part of the original idea this pilot never had the "
          "freedom to express.", ""]

    return "\n".join(L) + "\n"


def main():
    body = build()
    with open(os.path.join(OUT, "STATUS.md"), "wb") as f:
        f.write(body.encode())
    print(f"wrote {os.path.join(OUT, 'STATUS.md')} ({len(body)} bytes)")


if __name__ == "__main__":
    main()
