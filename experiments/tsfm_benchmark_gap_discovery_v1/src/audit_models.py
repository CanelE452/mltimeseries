"""Audit the candidate models against their official sources.

Every field here is either read live from the Hugging Face API at run time or
transcribed from the official model card / repository README, with the source URL
recorded alongside it. Nothing is carried over from conversation.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import certifi
import pandas as pd
import requests

from . import paths

# Transcribed from the official model cards on 2026-09-07. `card_url` is the
# exact page each claim came from.
MODELS = [
    {
        "model_id": "chronos-2",
        "role": "primary",
        "hf_repo": "amazon/chronos-2",
        "card_url": "https://huggingface.co/amazon/chronos-2",
        "official_repo": "https://github.com/amazon-science/chronos-forecasting",
        "paper": "Chronos-2: From Univariate to Universal Forecasting (2025, arXiv:2510.15821)",
        "publication_status": "arxiv_technical_report",
        "parameter_count_m": 120,
        "architecture_family": "transformer_encoder",
        "architecture_note": "T5-style encoder with a group attention mechanism",
        "univariate": True,
        "multivariate_target": True,
        "past_covariates": True,
        "future_known_covariates": True,
        "missing_value_support": "documented_via_nan_handling",
        "max_context_length": 8192,
        "max_prediction_length": 1024,
        "probabilistic_output": "quantiles",
        "public_weights": True,
        "public_inference_code": True,
        "public_benchmark_code": True,
        "finetuning_code": False,
        "benchmark_specific_checkpoint": None,
        "pretraining_corpus": (
            "Subset of autogluon/chronos_datasets (excluding the test portion of datasets that "
            "overlap with GIFT-Eval); subset of Salesforce/GiftEvalPretrain; synthetic univariate "
            "and multivariate data"
        ),
        "claimed_fev_bench_exclusion": False,
        "license": "apache-2.0",
        "inference_package": "chronos-forecasting",
    },
    {
        "model_id": "tirex-2",
        "role": "primary",
        "hf_repo": "NX-AI/TiRex-2",
        "card_url": "https://huggingface.co/NX-AI/TiRex-2",
        "official_repo": "https://github.com/NX-AI/tirex-2",
        "paper": "TiRex-2: Generalizing TiRex to Multivariate Data and Streaming (2026, arXiv:2607.01204)",
        "publication_status": "arxiv_technical_report",
        "parameter_count_m": 82.5,
        "architecture_family": "recurrent_xlstm",
        "architecture_note": "12 alternating mLSTM/sLSTM xLSTM blocks; 38.4M univariate + 44.1M multivariate",
        "checkpoint_note": (
            "Flagship checkpoint. The advertised fev-bench decontaminated repository "
            "NX-AI/TiRex-2-fevbench publishes a byte-identical model.ckpt (LFS oid "
            "5596fb4bd1ecc4fbf93c5d7d3c9c68bd4e8492b3), so it is not a distinct model; see "
            "MODEL_SUBSTITUTION.md"
        ),
        "univariate": True,
        "multivariate_target": True,
        "past_covariates": True,
        "future_known_covariates": True,
        "missing_value_support": "not_documented",
        "max_context_length": None,
        "max_prediction_length": None,
        "probabilistic_output": "quantiles",
        "public_weights": True,
        "public_inference_code": True,
        "public_benchmark_code": True,
        "finetuning_code": False,
        "benchmark_specific_checkpoint": "advertised_for_fev_bench_but_duplicate_of_flagship",
        "pretraining_corpus": (
            "autogluon/chronos_datasets and Salesforce/lotsa_data. The card advertises a "
            "fev-bench decontaminated variant, but the published artifact is identical to this "
            "flagship checkpoint, so no fev-bench exclusion is realised in the weights actually "
            "available"
        ),
        "claimed_fev_bench_exclusion": False,
        "license": "apache-2.0",
        "inference_package": "tirex-2",
    },
    {
        "model_id": "timesfm-3.0",
        "role": "primary",
        "hf_repo": "google/timesfm-3.0-pytorch",
        "card_url": "https://huggingface.co/google/timesfm-3.0-pytorch",
        "official_repo": "https://github.com/google-research/timesfm",
        "paper": "TimesFM 3.0 model card / TimesFM line (arXiv:2310.10688 cited by the card)",
        "publication_status": "company_release",
        "parameter_count_m": 330,
        "architecture_family": "transformer_patch_variate",
        "architecture_note": "Stacked Mixing Transformer with variate attention and CPM iterative RevIN, 20 layers",
        "univariate": True,
        "multivariate_target": True,
        "past_covariates": True,
        "future_known_covariates": True,
        "missing_value_support": "not_documented",
        "max_context_length": None,
        "max_prediction_length": None,
        "probabilistic_output": "quantiles",
        "public_weights": True,
        "public_inference_code": True,
        "public_benchmark_code": True,
        "finetuning_code": False,
        "benchmark_specific_checkpoint": None,
        "pretraining_corpus": (
            "GiftEvalPretrain excluding the datasets that overlap with fev-bench; Wikipedia "
            "Pageviews (cutoff Nov 2023); Google Trends top queries (cutoff EoY 2022); synthetic "
            "and augmented data"
        ),
        "claimed_fev_bench_exclusion": True,
        "license": "timesfm-non-commercial-license-v1.0",
        "inference_package": "timesfm",
    },
    {
        "model_id": "chronos-2-synth",
        "role": "diagnostic_contamination_control",
        "hf_repo": "autogluon/chronos-2-synth",
        "card_url": "https://huggingface.co/autogluon/chronos-2-synth",
        "official_repo": "https://github.com/amazon-science/chronos-forecasting",
        "paper": "Chronos-2: From Univariate to Universal Forecasting (2025, arXiv:2510.15821)",
        "publication_status": "arxiv_technical_report",
        "parameter_count_m": 120,
        "architecture_family": "transformer_encoder",
        "architecture_note": "Chronos-2 architecture trained on synthetic data only",
        "univariate": True,
        "multivariate_target": True,
        "past_covariates": True,
        "future_known_covariates": True,
        "missing_value_support": "documented_via_nan_handling",
        "max_context_length": 8192,
        "max_prediction_length": 1024,
        "probabilistic_output": "quantiles",
        "public_weights": True,
        "public_inference_code": True,
        "public_benchmark_code": True,
        "finetuning_code": False,
        "benchmark_specific_checkpoint": None,
        "pretraining_corpus": "Synthetic univariate and multivariate data only",
        "claimed_fev_bench_exclusion": True,
        "license": "apache-2.0",
        "inference_package": "chronos-forecasting",
    },
]

# Section 8 contamination labels, for the fev-bench benchmark specifically.
CONTAMINATION = {
    "chronos-2": (
        "OVERLAP_RISK_UNKNOWN",
        "The card documents exclusions against GIFT-Eval only. Its corpus (chronos_datasets, "
        "GiftEvalPretrain) shares upstream sources with autogluon/fev_datasets and no fev-bench "
        "exclusion is claimed, so overlap with fev-bench evaluation windows cannot be ruled out.",
    ),
    "tirex-2": (
        "OVERLAP_RISK_UNKNOWN",
        "The card advertises NX-AI/TiRex-2-fevbench as excluding all fev-bench eval datasets, but "
        "that repository publishes a byte-identical checkpoint to this one (same LFS oid "
        "5596fb4bd1ecc4fbf93c5d7d3c9c68bd4e8492b3), while the GIFT-Eval variants are genuinely "
        "distinct. The claimed fev-bench exclusion is therefore not realised in any published "
        "artifact, and overlap with fev-bench evaluation data cannot be ruled out.",
    ),
    "timesfm-3.0": (
        "CLEAN_BY_OFFICIAL_EXCLUSION",
        "Card states pretraining used GiftEvalPretrain excluding the datasets that overlap with "
        "fev-bench. Wikipedia/Google Trends portions are unrelated to fev-bench sources.",
    ),
    "chronos-2-synth": (
        "SYNTH_ONLY_DIAGNOSTIC",
        "Trained on synthetic data only, so no real-data overlap with fev-bench is possible. "
        "Used as a contamination anchor, not as a leaderboard entry.",
    ),
}


def _hf_revision(repo: str) -> dict:
    session = requests.Session()
    session.verify = certifi.where()
    response = session.get(f"https://huggingface.co/api/models/{repo}", timeout=60)
    if response.status_code != 200:
        return {"resolved_revision": None, "hf_api_status": response.status_code}
    payload = response.json()
    return {
        "resolved_revision": payload.get("sha"),
        "hf_api_status": 200,
        "hf_gated": payload.get("gated"),
        "hf_last_modified": payload.get("lastModified"),
    }


def build() -> pd.DataFrame:
    rows = []
    for model in MODELS:
        row = dict(model)
        row.update(_hf_revision(model["hf_repo"]))
        status, note = CONTAMINATION[model["model_id"]]
        row["contamination_status_fev_bench"] = status
        row["contamination_note"] = note
        row["status"] = "READY" if row.get("resolved_revision") else "BLOCKED_WEIGHTS"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    audit = build()
    audit.to_csv(paths.RESULTS / "model_audit.csv", index=False)

    contamination = audit[
        [
            "model_id",
            "role",
            "hf_repo",
            "resolved_revision",
            "benchmark_specific_checkpoint",
            "claimed_fev_bench_exclusion",
            "pretraining_corpus",
            "contamination_status_fev_bench",
            "contamination_note",
        ]
    ].assign(benchmark="fev-bench")
    contamination.to_csv(paths.RESULTS / "contamination_matrix.csv", index=False)

    primary = audit[audit.role == "primary"]
    families = sorted(primary.architecture_family.unique())
    coverage = {
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_primary_models": int(len(primary)),
        "n_primary_ready": int((primary.status == "READY").sum()),
        "architecture_families": families,
        "n_architecture_families": len(families),
        "coverage_sufficient": bool((primary.status == "READY").sum() >= 3 and len(families) >= 2),
    }
    (paths.RESULTS / "model_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    print(
        audit[
            ["model_id", "role", "hf_repo", "architecture_family", "status", "resolved_revision"]
        ].to_string(index=False)
    )
    print()
    print(json.dumps(coverage, indent=2))
    if not coverage["coverage_sufficient"]:
        raise SystemExit("HARD STOP: MODEL_COVERAGE_INSUFFICIENT")


if __name__ == "__main__":
    main()
