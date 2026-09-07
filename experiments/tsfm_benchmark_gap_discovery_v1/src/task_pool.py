"""Build the fev-bench task pool from benchmark metadata only.

No model is loaded and no score is read here. The pool carries the descriptors
that task selection is allowed to see (Section 7 of the study contract).
"""

from __future__ import annotations

import difflib
import hashlib
import html
import json
import re

import certifi
import pandas as pd
import requests
import yaml

from . import paths

PAPER_HTML_URL = "https://arxiv.org/html/2509.26468v2"
PAPER_HTML_LOCAL = paths.DATA_EXTERNAL / "fev_bench_paper_v2.html"
TASKS_YAML = paths.DATA_EXTERNAL / "fev_bench_tasks.yaml"
TASKS_YAML_URL = (
    "https://raw.githubusercontent.com/autogluon/fev/"
    "eadb28ed3a3f8fc2db8dd4d3d6850894efcbc4d1/benchmarks/fev_bench/tasks.yaml"
)
# Hash of the newline-normalised text, so the contract holds on Windows and Linux.
TASKS_YAML_SHA256 = "c7160f61a5e1ded66a3954ef1c514d55d13be18534b34fca817356312a6520a9"

# Appendix tables A.1-A.6 of the fev-bench paper, in order; together they list
# all 100 tasks with the official domain and frequency labels.
PAPER_TABLE_INDICES = [13, 14, 15, 16, 17, 18]

FREQ_SUFFIX = {
    "5T": "5t",
    "10T": "10t",
    "15T": "15t",
    "30T": "30t",
    "T": "t",
    "H": "1h",
    "D": "1d",
    "W": "1w",
    "M": "1m",
    "Q": "1q",
    "Y": "1y",
}

# Coarse frequency buckets used by descriptor D2, fixed before any model runs.
FREQ_BUCKET = {
    "T": "sub_hourly",
    "5T": "sub_hourly",
    "10T": "sub_hourly",
    "15T": "sub_hourly",
    "30T": "sub_hourly",
    "H": "hourly",
    "D": "daily_or_coarser",
    "W": "daily_or_coarser",
    "M": "daily_or_coarser",
    "Q": "daily_or_coarser",
    "Y": "daily_or_coarser",
}

LF = chr(10)
CRLF = chr(13) + chr(10)


def _session() -> requests.Session:
    session = requests.Session()
    session.verify = certifi.where()
    session.headers["User-Agent"] = "tsfm-benchmark-gap-discovery-v1"
    return session


def canonical_yaml_hash() -> str:
    text = TASKS_YAML.read_text(encoding="utf-8").replace(CRLF, LF)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fetch_sources() -> None:
    session = _session()
    if not TASKS_YAML.exists():
        TASKS_YAML.write_text(session.get(TASKS_YAML_URL, timeout=120).text, encoding="utf-8")
    digest = canonical_yaml_hash()
    if digest != TASKS_YAML_SHA256:
        raise SystemExit(
            f"HARD STOP: fev-bench tasks.yaml hash mismatch ({digest} != {TASKS_YAML_SHA256})"
        )
    if not PAPER_HTML_LOCAL.exists():
        PAPER_HTML_LOCAL.write_text(session.get(PAPER_HTML_URL, timeout=120).text, encoding="utf-8")


def _cells(row_html: str) -> list[str]:
    return [
        html.unescape(re.sub(r"<[^>]+>", "", cell)).replace("\xa0", " ").strip()
        for cell in re.findall(r"<t[hd].*?</t[hd]>", row_html, flags=re.S)
    ]


def _paper_table() -> pd.DataFrame:
    text = PAPER_HTML_LOCAL.read_text(encoding="utf-8")
    tables = re.findall(r"<table.*?</table>", text, flags=re.S)
    rows = []
    for idx in PAPER_TABLE_INDICES:
        trs = re.findall(r"<tr.*?</tr>", tables[idx], flags=re.S)
        header = _cells(trs[0])
        if header[:2] != ["Task", "Domain"]:
            raise SystemExit(f"HARD STOP: unexpected paper table at index {idx}: {header}")
        for tr in trs[1:]:
            cells = _cells(tr)
            if len(cells) >= 11:
                rows.append(cells[:11])
    columns = [
        "paper_task",
        "domain",
        "freq",
        "horizon",
        "num_windows",
        "median_length",
        "n_series",
        "n_targets",
        "n_past_cov",
        "n_known_cov",
        "n_static_cov",
    ]
    table = pd.DataFrame(rows, columns=columns)
    for column in columns[3:]:
        table[column] = table[column].str.replace(",", "", regex=False).astype(int)
    if len(table) != 100:
        raise SystemExit(f"HARD STOP: paper table yielded {len(table)} rows, expected 100")
    return table


def _normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _match_paper_to_yaml(paper: pd.DataFrame, yaml_tasks: pd.DataFrame) -> pd.DataFrame:
    """Deterministic bijection between paper rows and tasks.yaml entries.

    Candidates are restricted to entries with the same horizon and window count,
    then scored by name similarity with a bonus for a matching frequency suffix.
    Sub-tasks that share a dataset config (FRED CEE/Macro, UK COVID new/cumulative)
    are disambiguated by the explicit task_name the yaml gives them.
    """
    paper = paper.copy()
    used: set[int] = set()
    assigned: list[int] = []
    scores: list[float] = []
    for _, row in paper.iterrows():
        paper_name = _normalise(row.paper_task)
        pool = [
            (j, r)
            for j, r in yaml_tasks.iterrows()
            if j not in used and r.horizon == row.horizon and r.num_windows == row.num_windows
        ]
        if not pool:
            pool = [
                (j, r)
                for j, r in yaml_tasks.iterrows()
                if j not in used and r.horizon == row.horizon
            ]
        best_index, best_score = None, -1e9
        for j, candidate in pool:
            config_name = _normalise(str(candidate.dataset_config))
            score = difflib.SequenceMatcher(None, paper_name, config_name).ratio()
            suffix = FREQ_SUFFIX.get(row.freq, "")
            if suffix and config_name.endswith(suffix):
                score += 0.35
            if paper_name and (paper_name in config_name or config_name.startswith(paper_name[:6])):
                score += 0.30
            task_name = candidate.get("task_name")
            if isinstance(task_name, str) and "/" in task_name:
                sub_task = _normalise(task_name.split("/")[-1])
                score += 0.60 if sub_task and sub_task in paper_name else -0.60
            if score > best_score:
                best_index, best_score = j, score
        assigned.append(best_index)
        scores.append(best_score)
        used.add(best_index)
    paper["yaml_index"] = assigned
    paper["match_score"] = scores
    if paper.yaml_index.nunique() != len(yaml_tasks):
        raise SystemExit("HARD STOP: paper<->yaml task matching is not a bijection")
    return paper


def build() -> pd.DataFrame:
    fetch_sources()
    raw = yaml.safe_load(TASKS_YAML.read_text(encoding="utf-8"))["tasks"]
    yaml_tasks = pd.DataFrame(raw)
    paper = _match_paper_to_yaml(_paper_table(), yaml_tasks)

    records = []
    for _, row in paper.iterrows():
        task = raw[int(row.yaml_index)]
        target = task.get("target")
        target_columns = target if isinstance(target, list) else ([target] if target else ["target"])
        known = task.get("known_dynamic_columns") or []
        past = task.get("past_dynamic_columns") or []
        static = task.get("static_columns") or []
        task_name = task.get("task_name") or task["dataset_config"]
        records.append(
            {
                "task_uid": f"fevbench::{task['dataset_config']}::{task_name}",
                "benchmark": "fev-bench",
                "yaml_index": int(row.yaml_index),
                "dataset_path": task["dataset_path"],
                "dataset_config": task["dataset_config"],
                "task_name": task.get("task_name"),
                "paper_task": row.paper_task,
                "domain": row.domain,
                "freq": row.freq,
                "freq_bucket": FREQ_BUCKET[row.freq],
                "horizon": int(task["horizon"]),
                "num_windows": int(task["num_windows"]),
                "seasonality": int(task["seasonality"]),
                "eval_metric": task["eval_metric"],
                "quantile_levels": json.dumps(task["quantile_levels"]),
                "median_length": int(row.median_length),
                "n_series": int(row.n_series),
                "n_targets": len(target_columns),
                "target_columns": json.dumps(target_columns),
                "n_known_cov": len(known),
                "n_past_cov": len(past),
                "n_static_cov": len(static),
                "known_dynamic_columns": json.dumps(known),
                "past_dynamic_columns": json.dumps(past),
                "static_columns": json.dumps(static),
                "min_context_length": task.get("min_context_length"),
                "max_context_length": task.get("max_context_length"),
                "is_multivariate": len(target_columns) > 1,
                "has_known_cov": len(known) > 0,
                "has_past_cov": len(past) > 0,
                "horizon_to_median_length": round(int(task["horizon"]) / int(row.median_length), 6),
                "n_forecasts_track_u": int(row.n_series)
                * len(target_columns)
                * int(task["num_windows"]),
            }
        )
    return pd.DataFrame(records).sort_values("task_uid").reset_index(drop=True)


def main() -> None:
    pool = build()
    out = paths.RESULTS / "task_pool.csv"
    pool.to_csv(out, index=False)
    print(f"wrote {out}  ({len(pool)} tasks)")
    print(pool.groupby("domain").size().to_string())
    print(pool.groupby("freq_bucket").size().to_string())
    print(f"multivariate tasks: {int(pool.is_multivariate.sum())}")
    print(f"tasks with known future covariates: {int(pool.has_known_cov.sum())}")
    print(f"tasks with past covariates: {int(pool.has_past_cov.sum())}")
    print(f"total TRACK U forecasts in pool: {int(pool.n_forecasts_track_u.sum()):,}")


if __name__ == "__main__":
    main()
