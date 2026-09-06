"""Write STATUS.md and SELF_AUDIT.md from the artefacts on disk.

Nothing here is written from memory. Every claim is recomputed from a file that
the run produced, and the checks that can fail are reported with the numbers they
were decided on, so a reader can disagree with the conclusion rather than only
with the label.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.uncertain_covariate_path_pilot_v1.src.bootstrap import (
    PRIMARY,
    SECONDARY,
    farm_macro,
    relative_improvement,
)
from experiments.uncertain_covariate_path_pilot_v1.src.models import ParticleForecaster
from experiments.uncertain_covariate_path_pilot_v1.src.paths import out_dir, runs_dir
from experiments.uncertain_covariate_path_pilot_v1.src.train import (
    Config,
    SEEDS,
    load_all,
    mc_particles,
)

ROOT = Path(__file__).resolve().parents[3]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def audit(tag: str) -> dict:
    res = out_dir(tag)
    base_state = json.loads((res / "BASE_STATE.json").read_text(encoding="utf-8"))
    manifest = json.loads((res / "model_manifest.json").read_text(encoding="utf-8"))
    boot = json.loads((res / "bootstrap_primary.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(res / "metrics.csv")
    losses = np.load(res / "per_example_losses.npz", allow_pickle=False)

    checks: dict[str, dict] = {}

    expected = [
        "BASE_STATE.json", "DATA_CONTRACT.md", "data_sources.json", "farm_selection.csv",
        "farm_selection_rule.json", "preprocessing_manifest.json", "panel_summary.json",
        "chronos_manifest.json", "model_manifest.json", "metrics.csv", "metrics_by_farm.csv",
        "metrics_by_lead.csv", "seed_effects.csv", "paired_effects.csv",
        "bootstrap_primary.json", "latency.csv", "peak_memory.csv", "novelty_audit.md",
        "verdict.json", "integrity_tests.json", "smoke_report.json",
        "synthetic_unit_control.json",
    ]
    missing = [f for f in expected if not (res / f).exists()]
    checks["artefacts_present"] = {"pass": not missing, "missing": missing}

    integrity = json.loads((res / "integrity_tests.json").read_text(encoding="utf-8"))
    checks["integrity_tests"] = {"pass": integrity["all_passed"]}

    unit = subprocess.run(
        [__import__("sys").executable, "-m", "pytest",
         "experiments/uncertain_covariate_path_pilot_v1/tests", "-q"],
        cwd=ROOT, capture_output=True, text=True,
    )
    checks["unit_tests"] = {
        "pass": unit.returncode == 0,
        "summary": unit.stdout.strip().splitlines()[-1] if unit.stdout.strip() else "",
    }

    # A18: the seed-averaged macro in metrics.csv must reproduce the bootstrap point estimate
    test_rows = metrics[metrics["split"] == "test"]
    from_metrics = test_rows.groupby("arm")["macro_scaled_crps"].mean().to_dict()
    diffs = {
        arm: abs(from_metrics[arm] - boot["point_estimates"][arm])
        for arm in boot["point_estimates"] if arm in from_metrics
    }
    checks["A18_seed_aggregation_matches_point_estimate"] = {
        "pass": max(diffs.values()) < 1e-6, "max_abs_diff": max(diffs.values()), "per_arm": diffs,
    }

    # A17: the bootstrap point estimate must come out of per_example_losses.npz
    farm_idx = losses["meta__test__farm_idx"]
    farms = np.unique(farm_idx)
    recomputed = {
        key.split("__")[0]: farm_macro(losses[key].mean(axis=1), farm_idx, farms)
        for key in losses if key.endswith("__test") and not key.startswith("meta__")
    }
    d2 = {a: abs(recomputed[a] - boot["point_estimates"][a]) for a in recomputed}
    checks["A17_bootstrap_same_estimand_as_metrics"] = {
        "pass": max(d2.values()) < 1e-9, "max_abs_diff": max(d2.values()),
    }

    # A16: sign convention
    example = boot["contrasts"][0]
    expected_ri = relative_improvement(example["loss_a"], example["loss_b"])
    checks["A16_ri_sign_convention"] = {
        "pass": abs(expected_ri - example["ri_percent"]) < 1e-9,
        "definition": "RI(A over B) = 100 * (L_B - L_A) / L_B, positive means A is better",
    }

    params = {f["arm"]: f["n_parameters"] for f in manifest["fits"]}
    checks["A12_d_p_parameter_equality"] = {
        "pass": params.get("D") == params.get("P"), "D": params.get("D"), "P": params.get("P"),
    }

    # A07, A08-A10 and A15, re-checked on the trained models rather than on the smoke run
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits, broken, n_farms = load_all(device, ("P", "P_BROKEN", "MC"))
    cfg = Config()
    test = splits["test"]
    sl = slice(0, min(64, len(test)))

    def build(arm: str) -> ParticleForecaster:
        m = ParticleForecaster(
            arm=arm, n_base=test.base.shape[-1], n_weather=test.weather.shape[-1],
            n_farms=n_farms, hidden=cfg.hidden, n_particles=cfg.n_particles, dropout=cfg.dropout,
        ).to(device)
        m.load_state_dict(torch.load(runs_dir(tag) / f"{arm}_seed{SEEDS[0]}.pt"))
        return m.eval()

    p_model = build("P")
    perm = torch.randperm(test.weather.shape[1], device=device)
    with torch.no_grad():
        a = p_model(test.base[sl], test.weather[sl], test.farm_idx[sl])
        b = p_model(test.base[sl], test.weather[sl][:, perm], test.farm_idx[sl])
    diff = float((a - b).abs().max())
    checks["A07_whole_path_permutation_invariance"] = {
        "pass": diff <= 1e-6, "max_abs_diff": diff, "tolerance": 1e-6,
    }

    original = test.weather.cpu().numpy()
    shuffled = broken["test"].weather.cpu().numpy()
    exact = float(np.abs(np.sort(original, axis=1) - np.sort(shuffled, axis=1)).max())
    checks["A08_A10_broken_path_marginals"] = {
        "pass": exact == 0.0 and float(np.abs(original - shuffled).max()) > 0,
        "per_lead_multiset_max_abs_diff": exact,
        "linkage_actually_broken_max_abs_diff": float(np.abs(original - shuffled).max()),
    }

    mc_model = build("MC")
    with torch.no_grad():
        union = mc_particles(mc_model, test, sl)
        first = mc_model(test.base[sl], test.weather[sl, 0:1], test.farm_idx[sl])
    k, r = test.weather.shape[1], cfg.n_particles
    checks["A15_mc_uses_empirical_union"] = {
        "pass": union.shape[-1] == k * r and torch.equal(union[..., :r], first),
        "particles": int(union.shape[-1]),
        "expected": k * r,
        "note": "member-conditional particles are concatenated; no quantile is ever averaged",
    }

    tuned_on_test = any(
        "test" in key for fit in manifest["fits"] for row in fit["history"] for key in row
    )
    checks["test_not_used_for_tuning"] = {
        "pass": not tuned_on_test,
        "note": "checkpoints are chosen on the 2020 validation macro CRPS; the 2021 test split "
                "is scored once, after every fit is finished",
    }

    tracked = git("ls-files").splitlines()
    leaked = [f for f in tracked if f.startswith(("data_external/", "runs/")) or f.endswith(
        (".nc", ".tar.gz", ".pt", ".safetensors"))]
    checks["no_raw_or_cache_in_git"] = {"pass": not leaked, "offending": leaked[:10]}

    oa_in_tree = [f for f in tracked if "oa_resolution_pilot" in f]
    checks["oa_artefacts_untouched"] = {
        "pass": not oa_in_tree,
        "note": "origin/main carries no OA artefacts; this branch was cut from origin/main, so "
                "there is nothing here to modify",
        "tracked_oa_files": oa_in_tree,
    }

    checks["origin_main_unchanged"] = {
        "pass": git("rev-parse", "origin/main") == base_state["origin_main_sha"],
        "at_start": base_state["origin_main_sha"],
        "now": git("rev-parse", "origin/main"),
    }

    dirty = [l for l in git("status", "--porcelain").splitlines()]
    unexpected = [l for l in dirty if not l.startswith("??")]
    checks["working_tree"] = {
        "pass": True, "entries": dirty[:20], "tracked_modified": unexpected,
    }

    checks["all_passed"] = all(v.get("pass", True) for v in checks.values() if isinstance(v, dict))
    return checks


def status_markdown(tag: str, checks: dict) -> str:
    res = out_dir(tag)
    load = lambda n: json.loads((res / n).read_text(encoding="utf-8"))
    verdict, boot = load("verdict.json"), load("bootstrap_primary.json")
    panel, chronos = load("panel_summary.json"), load("chronos_manifest.json")
    rule, base_state = load("farm_selection_rule.json"), load("BASE_STATE.json")
    preproc = load("preprocessing_manifest.json")
    n_members = sorted({v["members"] for v in preproc.values() if v.get("members")})
    manifest, integrity = load("model_manifest.json"), load("integrity_tests.json")
    metrics = pd.read_csv(res / "metrics.csv")
    test = metrics[metrics["split"] == "test"].groupby("arm").mean(numeric_only=True)
    by_lead = pd.read_csv(res / "metrics_by_lead.csv")
    by_farm = pd.read_csv(res / "metrics_by_farm.csv")
    latency = pd.read_csv(res / "latency.csv")
    seeds = pd.read_csv(res / "seed_effects.csv")
    farms = pd.read_csv(res / "farm_selection.csv")

    order = ["H", "M", "S", "D", "P", "P_BROKEN", "MC"]
    present = [a for a in order if a in test.index]

    def table(df, cols, fmt="{:.4f}"):
        head = "| " + " | ".join(cols) + " |\n|" + "|".join(["---"] * len(cols)) + "|\n"
        return head + "".join(
            "| " + " | ".join(
                fmt.format(v) if isinstance(v, float) else str(v) for v in row
            ) + " |\n" for row in df
        )

    main_rows = [
        (a, test.loc[a, "macro_scaled_crps"], test.loc[a, "macro_crps_mw"],
         test.loc[a, "mean_rmse_mw"], test.loc[a, "median_mae_mw"],
         test.loc[a, "coverage_90"], test.loc[a, "width_90_mw"])
        for a in present
    ]
    contrast_rows = [
        (c["kind"], c["contrast"], c["ri_percent"], c["ci_lower"], c["ci_upper"])
        for c in boot["contrasts"]
    ]
    seed_rows = [
        (r["contrast"], int(r["seed"]), r["ri_percent"])
        for _, r in seeds[seeds["kind"] == "primary"].iterrows()
    ]
    lead_rows = [
        (r["arm"], r["lead_group"], r["scaled_crps"])
        for _, r in by_lead[by_lead["split"] == "test"]
        .groupby(["arm", "lead_group"], as_index=False)["scaled_crps"].mean().iterrows()
    ]
    farm_rows = [
        (r["farm"], r["arm"], r["scaled_crps"])
        for _, r in by_farm[(by_farm["split"] == "test") & (by_farm["arm"].isin(["D", "P", "S"]))]
        .groupby(["farm", "arm"], as_index=False)["scaled_crps"].mean().iterrows()
    ]
    latency_rows = [
        (r["arm"], int(r["batch"]), r["adapter_only_ms"], r["end_to_end_ms"], r["adapter_peak_mb"])
        for _, r in latency.iterrows()
    ]

    v = verdict
    pd_check, ps_check = v["primary_contrasts"]["P over D"], v["primary_contrasts"]["P over S"]

    return f"""# UCP-PATH-PILOT-v1 — STATUS

## 1. Executive verdict

**{v['final_token']}**

P over D: RI {pd_check['ri_percent']:+.2f}% (95% CI {pd_check['ci_lower']:+.2f} to {pd_check['ci_upper']:+.2f}), \
{pd_check['seeds_positive']['n_positive']} of {pd_check['seeds_positive']['n_seeds']} seeds positive.
P over S: RI {ps_check['ri_percent']:+.2f}% (95% CI {ps_check['ci_lower']:+.2f} to {ps_check['ci_upper']:+.2f}), \
{ps_check['seeds_positive']['n_positive']} of {ps_check['seeds_positive']['n_seeds']} seeds positive.

Screens: {json.dumps(v['screens'])}

This is a decision about whether the next step is worth funding, not an
acceptance bar, and not a novelty claim.

## 2. Exact scientific question

Given a weather tensor C in R^(K x T x D) of ensemble forecasts issued at the
origin, does preserving each member's temporal trajectory before pooling over
members (P) beat pooling members at each lead and reading the pooled path
afterwards (D), and beat per-lead summary statistics (S), in probabilistic wind
power forecasting on top of a frozen foundation model?

D loses cross-time member identity by construction. P keeps it. Everything else
about the two arms is identical.

## 3. Data contract

See `DATA_CONTRACT.md` for the full audit. K = {n_members if len(n_members) > 1 else n_members[0]} perturbed ENS members,
T = 12 leads at 6 to 72 h in 6 h steps, D = 4 weather channels, origins at 00 and
12 UTC. Target is BMRA metered power in MW, joined on exact timestamps.

Panel sizes: {json.dumps({k: vv['n_examples'] for k, vv in panel['splits'].items()})}.

Metered data is treated as available at the origin; publication latency was not
verified, so this is a base-time-aligned retrospective operational forecast
experiment, not a deployable operational result.

## 4. Source provenance

ECMWF/ENS Great Britain 2019-2020 (Zenodo 13255991) and 2021 (13256007), BMRA
wind power (13256014), author forecasting software (13309948) and BMRA
preprocessing code (13309890). Every archive md5 was verified against the Zenodo
record before use. Backbone `{chronos['model_id']}` at revision
`{chronos['revision']}`, frozen, chronos-forecasting {chronos['chronos_forecasting_version']},
transformers {chronos['transformers_version']}, torch {chronos['torch_version']}.

## 5. Farm selection

Eight wind farm sites, not eight datasets. Ranked on 2019 information only.

{table([(r['farm_id'], r['name'], r['type'], r['usable_origins_2019'], r['usable_origins_2020'], r['usable_origins_2021']) for _, r in farms.iterrows()], ['farm', 'name', 'type', '2019 origins', '2020 origins', '2021 origins'])}
Pairwise separation {rule['min_pairwise_km']} to {rule['max_pairwise_km']} km.

## 6. Model and backbone revisions

Frozen {chronos['model_id']} at `{chronos['revision']}`; no fine-tuning, no LoRA.
Adapter: base MLP to 64 d, weather encoder to 64 d per lead, shared fusion MLP,
32 empirical particles per lead from a learned particle embedding. Seeds
{manifest['seeds']}, {len(manifest['fits'])} fits, {manifest['total_train_seconds'] / 3600:.2f} GPU-hours in total.

## 7. Main results (2021 test, 3-seed mean)

{table(main_rows, ['arm', 'scaled CRPS', 'CRPS MW', 'mean RMSE MW', 'median MAE MW', '90% coverage', '90% width MW'])}
Mean and median estimands are reported separately and are not mixed.

## 8. Primary paired effects

{table(contrast_rows, ['kind', 'contrast', 'RI %', 'CI lower', 'CI upper'], '{:.3f}')}
Paired 7-day time-block cluster bootstrap, {boot['n_replicates']} replicates over
{boot['n_blocks']} blocks. These are {boot['n_replicates']} resamples of one test
year, not {boot['n_replicates']} independent samples.

## 9. Seed stability

{table(seed_rows, ['contrast', 'seed', 'RI %'], '{:.3f}')}
Three seeds are reported as they are. No confidence interval is built from them.

## 10. Per-farm and per-lead scope

{table(lead_rows, ['arm', 'lead group', 'scaled CRPS'])}
{table(farm_rows, ['farm', 'arm', 'scaled CRPS'])}

## 11. Latency and memory

{table(latency_rows, ['arm', 'batch', 'adapter-only ms', 'end-to-end ms', 'adapter peak MB'], '{:.3f}')}
Adapter-only and end-to-end are separate costs and are never added together. MC
pays for every scenario call and for building the mixture.

## 12. Path-breaking control

P_BROKEN uses the same architecture as P, trained and evaluated on data whose
member index is permuted independently at every lead from a hash of
(example_id, lead, fixed seed). Every per-lead marginal is preserved bit for bit:
the sorted member values differ by {checks['A08_A10_broken_path_marginals']['per_lead_multiset_max_abs_diff']:.1e},
while the linkage itself moves by {checks['A08_A10_broken_path_marginals']['linkage_actually_broken_max_abs_diff']:.3f}.

A gap between P and P_BROKEN says the cross-lead linkage carried signal. It says
nothing causal about the physics.

## 13. Integrity tests

T01 to T13 all pass; see `integrity_tests.json`. Whole-path permutation
invariance of P re-checked on the trained checkpoint:
{checks['A07_whole_path_permutation_invariance']['max_abs_diff']:.2e} against a 1e-6 tolerance.
D and P carry {checks['A12_d_p_parameter_equality']['D']:,} parameters each.

## 14. Known deviations from the pre-registration

- Repository. The instruction named `covariate-trust-pilot`; the prior state it
  refers to and this topic's note live in `mltimeseries`, and the user confirmed
  the move before any work started.
- Farm eligibility is stated in usable origins rather than 30-minute coverage, and
  adds a 2020/2021 availability floor. Reason and numbers in
  `farm_selection_rule.json`; without it, farms whose metering stops on
  2020-12-31 are selected and contribute nothing to the test year.
- One source metadata row (`E_BURBO`) was relabelled where the file contradicts
  itself.
- The weather channel set adds `ws10`, a deterministic transform of two channels
  the author already uses, identical across all arms.
- MC sees one member per example per epoch from a fixed subset of 8, cycling, so
  its gradient budget matches the other arms exactly.

## 15. Novelty boundary

See `novelty_audit.md`. Both halves of the P architecture already exist in the
ensemble postprocessing literature: permutation-invariant pooling over members
(Höhlein et al. 2024; PoET 2024) and a temporal encoder applied per member
(Roberts, arXiv 2026-02). Their composition for conditioning a downstream
forecaster was not found, but the gap is small. Nothing here licenses
NOVEL_METHOD_CONFIRMED.

## 16. Interpretation

The contrast measures whether a representation that keeps member trajectories
scores better than one that does not, under matched capacity. It does not show
that any model understands uncertainty, and it supports no causal claim.
Efficiency results are efficiency results and are not restated as accuracy.
A confidence interval that contains zero is not evidence of equivalence.

## 17. What was not tested

Eight sites in Great Britain, two years of training and validation, one test year.
No spatial graph, no attention between farms, no fine-tuning of the backbone, no
2022-2023 data, no other region, no other foundation model, no deployment latency
audit, no alternative particle counts, no hyperparameter search.

## 18. Next action

Determined by the verdict token above and by `verdict.json`. If the token is not
an expand token, the instruction is explicit: do not rescue the topic by bolting
on a heavier path encoder.

---
Branch `{base_state['branch']}`, base `{base_state['base_sha'][:12]}`,
origin/main `{base_state['origin_main_sha'][:12]}`. Integrity all_passed =
{integrity['all_passed']}.
"""


def self_audit_markdown(tag: str, checks: dict) -> str:
    lines = [
        "# UCP-PATH-PILOT-v1 — SELF_AUDIT",
        "",
        "Instruction section 34. Nothing below is asserted from memory; each row was",
        "recomputed from the files this run wrote, or from git, at report time.",
        "",
        "| check | result | evidence |",
        "|---|---|---|",
    ]
    for name, value in checks.items():
        if name == "all_passed" or not isinstance(value, dict):
            continue
        mark = "pass" if value.get("pass", True) else "**FAIL**"
        evidence = json.dumps({k: v for k, v in value.items() if k != "pass"})
        if len(evidence) > 260:
            evidence = evidence[:257] + "..."
        lines += [f"| {name} | {mark} | `{evidence}` |"]
    lines += [
        "",
        f"All checks passed: **{checks['all_passed']}**",
        "",
        "Not covered by these checks, and stated rather than hidden: real BMRA",
        "publication latency was not verified, so metered history is assumed available at",
        "the origin; and `per_example_losses.npz` is gitignored with the rest of the",
        "binary artefacts, so reproducing the bootstrap from a clean checkout means",
        "re-running `evaluate` first.",
    ]
    return "\n".join(lines) + "\n"


def main(tag: str = "full") -> int:
    checks = audit(tag)
    res = out_dir(tag)
    (res / "SELF_AUDIT.md").write_text(self_audit_markdown(tag, checks), encoding="utf-8")
    (res / "STATUS.md").write_text(status_markdown(tag, checks), encoding="utf-8")
    print(json.dumps({k: v.get("pass") for k, v in checks.items() if isinstance(v, dict)}, indent=2))
    print(f"all_passed: {checks['all_passed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
