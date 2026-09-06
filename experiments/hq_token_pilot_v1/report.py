"""STATUS.md writer.

Contract source: 01_forecast_query_tokenization_CLI.txt section 21. Kept apart from
run.py only so the orchestration stays readable; it performs no analysis of its own and
reads nothing the analysis stage did not already write.
"""

from __future__ import annotations

import json
import os


def _pct(v, digits=2):
    return "n/a" if v is None else f"{v:+.{digits}f}%"


def _num(v, digits=5):
    return "n/a" if v is None else f"{v:.{digits}f}"


def write_status(results_dir, spec, rows, analysis, diagnostics, efficiency, state):
    v = analysis["verdict"]
    planned = len(spec["planned_fits"])
    done = sum(1 for r in rows if r["status"] == "COMPLETE")
    execution_status = "COMPLETE" if done == planned else ("PARTIAL" if done else "BLOCKED")
    prim = analysis["primary"]
    boot = analysis["bootstrap"]
    L = []

    L.append("# HQ-TOKEN-PILOT-v1 — STATUS")
    L.append("")
    L.append(
        f"**execution_status = {execution_status}**, "
        f"**scientific_decision = {v['decision']}**, "
        f"evidence_level = DEVELOPMENT_SCREEN."
    )
    L.append("")
    L.append(
        f"{done} of {planned} pre-registered fits completed. "
        f"Total wall time this invocation: {state['wall_seconds']/60:.1f} min."
    )
    L.append("")

    L.append("## 1. What this run does and does not claim")
    L.append("")
    L.append(
        "This is a development screen of horizon-conditioned local pooling at a fixed token "
        "budget. It is not a foundation-model result, not a benchmark submission, and not a "
        "statement that the topic as a whole works or fails. The model is a small pilot that "
        "borrows patching and channel independence from PatchTST; it is not official PatchTST "
        "and has no instance normalization, so its absolute numbers are not comparable to "
        "published tables."
    )
    L.append("")

    L.append("## 2. Primary contrast — C versus I")
    L.append("")
    L.append("| dataset | H=96 | H=336 | dataset mean |")
    L.append("| --- | --- | --- | --- |")
    for d in ("ETTm2", "weather", "electricity"):
        L.append(
            f"| {d} | {_pct(prim['cells'].get(f'{d}|96'))} | "
            f"{_pct(prim['cells'].get(f'{d}|336'))} | "
            f"{_pct(prim['per_dataset_horizon_mean'].get(d))} |"
        )
    L.append(f"| **macro (6 cells, equal weight)** | | | **{_pct(prim['macro'])}** |")
    L.append("")
    L.append(
        f"Paired circular moving block bootstrap, {boot.get('draws')} draws, "
        f"block = {boot.get('block_origins')} origins: "
        f"macro mean {_pct(boot.get('macro_mean'))}, "
        f"95% interval [{_pct(boot.get('lower95'))}, {_pct(boot.get('upper95'))}]. "
        f"Model seeds are not resampled, so this is a time-sample interval conditional on the "
        f"two observed seeds."
    )
    if boot.get("flags"):
        L.append("")
        L.append("Bootstrap flags: " + ", ".join(boot["flags"]) + ".")
    if boot.get("effective_blocks"):
        L.append("")
        L.append(
            "Effective time blocks per dataset: "
            + ", ".join(f"{k} {n}" for k, n in boot["effective_blocks"].items())
            + "."
        )
    L.append("")
    L.append("Per-seed macro RI (sign condition of section 14):")
    L.append("")
    for s, val in prim["per_seed_macro"].items():
        L.append(f"- seed {s}: {_pct(val)}")
    L.append("")

    L.append("## 3. Pre-registered decision conditions")
    L.append("")
    L.append("| condition | met |")
    L.append("| --- | --- |")
    for k, ok in v["conditions"].items():
        L.append(f"| {k.replace('_', ' ')} | {'yes' if ok else 'no'} |")
    L.append("")
    if v["notes"]:
        L.append("Reading notes:")
        L.append("")
        for n in v["notes"]:
            L.append(f"- {n}")
        L.append("")

    L.append("## 4. Scope and explanation contrasts")
    L.append("")
    L.append("| contrast | macro RI | role |")
    L.append("| --- | --- | --- |")
    sec = analysis["secondary"]
    L.append(f"| C vs I | {_pct(prim['macro'])} | primary |")
    L.append(
        f"| C vs H_STATIC | {_pct(sec['C_vs_H_STATIC']['macro'])} | "
        f"separates content-conditioned pooling from a fixed horizon prior |"
    )
    L.append(
        f"| C vs R | {_pct(sec['C_vs_R']['macro'])} | "
        f"seed0 only; random query control |"
    )
    L.append(f"| C vs U | {_pct(sec['C_vs_U']['macro'])} | seed0 only; uniform pooling |")
    L.append(f"| C vs DENSE | {_pct(sec['C_vs_DENSE']['macro'])} | seed0 only; no compression |")
    hr = analysis["precondition_compression_headroom_DENSE_vs_U"]
    L.append(
        f"| DENSE vs U | {_pct(hr['macro'])} | precondition: how much accuracy the "
        f"64 to 32 compression costs at all |"
    )
    L.append("")
    L.append(
        "The DENSE versus U row bounds what any pooling rule could win at this budget. If it is "
        "smaller than the 1.0 percent gate, the gate was unreachable by construction and a null "
        "must not be read as evidence against the hypothesis."
    )
    L.append("")
    shift = analysis["robustness_phase_shifted_grid"]
    L.append(
        f"Robustness: on a phase-shifted evaluation grid the C versus I macro is "
        f"{_pct(shift['macro'])} against {_pct(prim['macro'])} on the pre-registered grid. "
        f"The pre-registered grid uses stride 96, which is exactly 24 h on ETTm2 and exactly "
        f"4 days on electricity, so all of its origins share one time of day."
    )
    L.append("")

    L.append("## 5. Can this architecture express the mechanism at all")
    L.append("")
    cap = analysis["mechanism_capacity_check"]

    def _w(d, h):
        return d.get(h, d.get(str(h)))

    L.append(f"Verdict: **{cap['verdict']}**.")
    L.append("")
    L.append(
        "The task gives every group one a-patch and one b-patch at independent levels; the "
        "short future is the mean of the a-levels and the long future the mean of the "
        "b-levels. Because the a-levels are mutually independent, no fixed allocation of the "
        "32 groups serves both futures."
    )
    L.append("")
    L.append("| arm | weight on the a-patch at H=96 | at H=336 | MSE H=96 | MSE H=336 |")
    L.append("| --- | --- | --- | --- | --- |")
    for arm in ("I", "C"):
        wd = cap.get(f"{arm}_mean_weight_on_a_patch", {})
        L.append(
            f"| {arm} | {_w(wd, 96):.3f} | {_w(wd, 336):.3f} | "
            f"{_w(cap[arm], 96):.4f} | {_w(cap[arm], 336):.4f} |"
        )
    L.append("")
    L.append(
        f"C moves its pooling weight by {cap['C_weight_separation_between_horizons']:.3f} "
        f"between the two horizons and improves the H=96 cell by "
        f"{_pct(cap['relative_improvement_pct'].get(96))}. "
        f"{cap['reading_note']}"
    )
    L.append("")
    L.append(
        "This is synthetic and is not a benchmark result. Its only job is to keep a null on real "
        "data from being ambiguous between a wrong hypothesis and an architecture that cannot "
        "learn the function. Two rejected earlier versions of this check are themselves "
        "informative and are recorded in the run notes: with a single shared level on the "
        "a-patches, the input-only pooler solved the task by giving different groups different "
        "roles, so horizon conditioning bought nothing. Spatial allocation across 32 groups is "
        "a real alternative to horizon conditioning, and H_STATIC is the arm that probes it."
    )
    L.append("")

    noise = analysis.get("seed_noise_floor")
    if noise:
        L.append("## 6. Resolution available to the 1.0 percent gate")
        L.append("")
        L.append("| dataset | mean validation MSE | seed sd | sd as percent of mean |")
        L.append("| --- | --- | --- | --- |")
        for d, s in noise["per_dataset"].items():
            L.append(f"| {d} | {_num(s['mean'])} | {_num(s['sd'])} | {s['sd_pct_of_mean']:.2f}% |")
        L.append("")
        L.append(
            f"Arm I trained at eight extra seeds, validation split only, before the core. "
            f"Worst per-seed spread {noise['worst_seed_sd_pct']:.2f} percent against a "
            f"{spec['gates']['macro_ri_C_vs_I_min']} percent gate. This is a measurement of "
            f"resolution, not an input to the verdict; the core uses exactly the two "
            f"pre-registered seeds."
        )
        L.append("")

    L.append("## 7. Accuracy table")
    L.append("")
    L.append("| dataset | H | B | seed | arm | MSE | MAE |")
    L.append("| --- | --- | --- | --- | --- | --- | --- |")
    path = os.path.join(results_dir, "metrics.csv")
    if os.path.exists(path):
        import csv

        with open(path) as f:
            for r in sorted(
                csv.DictReader(f),
                key=lambda r: (r["dataset"], int(r["horizon"]), r["arm"], r["model_seed"]),
            ):
                L.append(
                    f"| {r['dataset']} | {r['horizon']} | {r['B']} | {r['model_seed']} | "
                    f"{r['arm']} | {float(r['MSE']):.5f} | {float(r['MAE']):.5f} |"
                )
    L.append("")
    L.append("Seasonal-naive anchor on the same grid, same standardized space:")
    L.append("")
    L.append("| dataset | H=96 MSE | H=336 MSE | period (samples) |")
    L.append("| --- | --- | --- | --- |")
    for d, a in analysis["seasonal_naive_anchor"].items():
        L.append(
            f"| {d} | {_num(a['96']['mse'])} | {_num(a['336']['mse'])} | {a['96']['period']} |"
        )
    L.append("")

    L.append("## 8. Efficiency")
    L.append("")
    L.append(
        "| config | tokens | params | tokenizer params | model call b64 (ms) | "
        "end-to-end b64 (ms) | p90 (ms) | peak alloc (MiB) |"
    )
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for k, e in efficiency.items():
        peak = e["batch64"]["peak_allocated_bytes"]
        peak_txt = f"{peak/2**20:.1f}" if peak else "n/a"
        b64 = e["batch64"]
        L.append(
            f"| {k} | {e['tokens_encoded']} | {e['parameters']['total_parameters']:,} | "
            f"{e['parameters']['tokenizer_parameters']:,} | "
            f"{b64['model_call_median_ms']:.3f} | {b64['end_to_end_median_ms']:.3f} | "
            f"{b64['model_call_p90_ms']:.3f} | {peak_txt} |"
        )
    L.append("")
    L.append(
        "Fewer attention FLOPs is not the same claim as a faster forecast. The end-to-end column "
        "includes the host-to-device copy; the model-call column does not. Token counts from "
        "BPE-style or foundation tokenizers are not comparable to these numbers, because a token "
        "does not mean the same thing."
    )
    L.append("")

    if diagnostics:
        L.append("## 9. Query contribution diagnostics (C only)")
        L.append("")
        L.append("| dataset | H | MSE with true query | MSE with swapped query | change |")
        L.append("| --- | --- | --- | --- | --- |")
        for d, dg in diagnostics.items():
            for h, q in dg["query_swap"].items():
                L.append(
                    f"| {d} | {h} | {_num(q['mse_true_query'])} | "
                    f"{_num(q['mse_swapped_query'])} | {_pct(q['delta_pct'])} |"
                )
        L.append("")
        L.append("| dataset | mean absolute weight difference (96 vs 336) | mean token centre shift (samples) |")
        L.append("| --- | --- | --- |")
        for d, dg in diagnostics.items():
            wp = dg.get("weight_profile")
            if wp:
                L.append(
                    f"| {d} | {wp['mean_abs_weight_diff']:.4f} | "
                    f"{wp['mean_abs_center_shift_samples']:.2f} |"
                )
        L.append("")
        L.append(
            "A swapped query hurting is not by itself evidence for the method: the model never "
            "saw that mismatch during training. The separately trained I, H_STATIC and R arms "
            "carry the argument. Weight pictures are explanatory and cannot stand in for an "
            "accuracy effect."
        )
        L.append("")

    L.append("## 10. Fits, blocks and evaluation scope")
    L.append("")
    L.append(f"- completed fits / planned fits: {done} / {planned}")
    bad = [r for r in rows if r["status"] != "COMPLETE"]
    if bad:
        L.append("- fits that did not complete:")
        for r in bad:
            L.append(
                f"  - {r['arm']} {r['dataset_id']} seed {r['model_seed']}: "
                f"{r['status']} ({r['stop_note']})"
            )
    else:
        L.append("- no blocked or failed fits")
    collapsed = [r for r in rows if r.get("collapsed_to_constant")]
    if collapsed:
        L.append(
            "- BASELINE_UNRESOLVED candidates (near-constant forecast): "
            + ", ".join(f"{r['arm']}/{r['dataset_id']}/{r['model_seed']}" for r in collapsed)
        )
    L.append(
        "- evaluation scope: sparse origin grid at stride 96, both horizons, all channels. "
        "This is a development screen and is not the official stride-1 benchmark."
    )
    ef = spec.get("train_epoch_fraction", {})
    if ef:
        L.append(
            "- training budget as a fraction of one pass over eligible windows: "
            + ", ".join(f"{k} {v:.3f}" for k, v in ef.items())
            + ". Every arm gets the same budget, so the contrast is fair, but what is measured "
            "is early-optimization quality rather than converged quality."
        )
    res = state.get("resource")
    if res:
        L.append(
            f"- peak process-tree RSS {res['peak_process_tree_rss_bytes']/2**30:.2f} GiB, "
            f"peak GPU allocated {res['peak_gpu_allocated_bytes']/2**30:.2f} GiB, "
            f"minimum available RAM {res['min_available_ram_bytes']/2**30:.2f} GiB. "
            f"This guard bounds this job only and is not a guarantee about the machine."
        )
    L.append("")

    L.append("## 11. Stated limitations")
    L.append("")
    for k, text in spec["known_limitations"].items():
        L.append(f"- **{k.replace('_', ' ')}** — {text}")
    L.append("")

    L.append("## 12. Artifacts")
    L.append("")
    for name in sorted(os.listdir(results_dir)):
        if name != "STATUS.md":
            L.append(f"- `results/hq_token_pilot_v1/{name}`")
    L.append("")

    with open(os.path.join(results_dir, "STATUS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    append_reference_sections(results_dir)
    with open(os.path.join(results_dir, "verdict.json"), "w") as f:
        json.dump(
            {
                "execution_status": execution_status,
                "evidence_level": "DEVELOPMENT_SCREEN",
                "scientific_decision": v["decision"],
                "conditions": v["conditions"],
                "notes": v["notes"],
                "macro_RI_C_vs_I": prim["macro"],
                "bootstrap": {k: boot.get(k) for k in ("macro_mean", "lower95", "upper95", "flags")},
                "fits_completed": done,
                "fits_planned": planned,
            },
            f,
            indent=2,
        )


APPENDIX_MARKER = "<!-- appendix: references and figures -->"


def append_reference_sections(results_dir):
    """Add the section 16/17 and figure sections, which are produced after the report.

    Idempotent: anything below the marker is replaced, so re-running the driver and then
    re-running this leaves one copy.
    """
    path = os.path.join(results_dir, "STATUS.md")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        body = f.read().split(APPENDIX_MARKER)[0].rstrip()

    L = ["", APPENDIX_MARKER, ""]
    ref_path = os.path.join(results_dir, "optional_references.json")
    if os.path.exists(ref_path):
        with open(ref_path) as f:
            ref = json.load(f)
        L.append("## 13. Existing dynamic tokenization baselines (section 16)")
        L.append("")
        L.append("| method | official code found | status | how it was checked |")
        L.append("| --- | --- | --- | --- |")
        for r in ref["dynamic_tokenization_baselines"]:
            L.append(
                f"| {r['method']} | {'yes' if r['official_code_found'] else 'no'} | "
                f"{r['status']} | {r['how_checked']} |"
            )
        L.append("")
        L.append(ref["missing_strong_dynamic_baseline_note"])
        L.append("")

        L.append("## 14. Native foundation reference (section 17)")
        L.append("")
        grid = ref["reference_subgrid"]
        L.append(
            "Declared subgrid: "
            + ", ".join(
                f"{k} {v['origins']} origins x {v['channels']} channels"
                + (" (channel subsample)" if v["channel_subsample"] else "")
                for k, v in grid.items()
            )
            + "."
        )
        L.append("")
        L.append(ref["reference_subgrid_note"])
        L.append("")
        core = ref.get("core_arms_on_subgrid", {})
        L.append("| dataset | H | C (32 tok) | I (32 tok) | DENSE (64 tok) | Chronos-2 | TiRex-2 |")
        L.append("| --- | --- | --- | --- | --- | --- | --- |")
        for d in ("ETTm2", "weather", "electricity"):
            for h in ("96", "336"):
                def cell(arm):
                    v = core.get(d, {}).get(arm, {}).get(h, {}).get("mse")
                    return _num(v, 5)

                def rcell(model):
                    v = ref.get(model, {}).get("per_dataset", {}).get(d, {}).get(h, {}).get("mse")
                    return _num(v, 5)

                L.append(
                    f"| {d} | {h} | {cell('C')} | {cell('I')} | {cell('DENSE')} | "
                    f"{rcell('chronos2')} | {rcell('tirex2')} |"
                )
        L.append("")
        ch = ref["chronos2"]
        ti = ref["tirex2"]
        L.append(f"- Chronos-2 ({ch.get('weight')}): status {ch['status']}, point forecast is the {ch.get('point_forecast_type')}.")
        L.append(
            f"- TiRex-2 ({ti.get('weight')}): status {ti['status']}, device {ti.get('device')}. "
            f"{ti.get('device_note', '')}"
        )
        if ti.get("blocked_horizons"):
            for h, why in ti["blocked_horizons"].items():
                L.append(f"  - H={h}: {why}. Rolling the model forward to cover it was not attempted.")
        L.append("")
        L.append("Interpretation limits carried from the spec:")
        L.append("")
        for lim in ref["interpretation_limits"]:
            L.append(f"- {lim}")
        L.append("")

    fig_dir = os.path.join(results_dir, "figures")
    if os.path.isdir(fig_dir):
        L.append("## 15. Figures")
        L.append("")
        cap_path = os.path.join(fig_dir, "captions.md")
        caps = {}
        if os.path.exists(cap_path):
            with open(cap_path, encoding="utf-8") as f:
                current = None
                for line in f:
                    if line.strip().startswith("#") and ".png" in line:
                        current = line.strip().lstrip("#").strip()
                        caps[current] = []
                    elif current and line.strip():
                        caps[current].append(line.strip())
        for name in sorted(n for n in os.listdir(fig_dir) if n.endswith(".png")):
            L.append(f"### {name}")
            L.append("")
            L.append(f"![{name}](figures/{name})")
            key = next((k for k in caps if name in k), None)
            if key:
                L.append("")
                L.append(" ".join(caps[key]))
            L.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write(body + "\n" + "\n".join(L) + "\n")
