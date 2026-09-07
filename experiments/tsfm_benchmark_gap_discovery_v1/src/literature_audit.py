"""Novelty audit for candidates that cleared the headroom screen (Section 22).

The searching itself is done against primary sources and written into
`literature_findings.json`; this module checks that every candidate that needs an
audit has one, and renders it. A candidate with no entry is reported as UNCLEAR
rather than quietly treated as open space.
"""

from __future__ import annotations

import json

import pandas as pd

from . import paths

FINDINGS = paths.RESULTS / "literature_findings.json"

REQUIRED_FIELDS = (
    "exact_problem",
    "exact_intervention_point",
    "closest_method",
    "same_information_condition",
    "same_benchmark",
    "same_backbone_family",
    "direct_overlap",
    "difference_substantial_or_cosmetic",
    "novelty_status",
    "papers",
)

VALID_STATUS = {"OPEN_SPACE", "ADJACENT_CROWDED", "DIRECTLY_OWNED", "UNCLEAR"}


def candidates_needing_audit() -> list[str]:
    path = paths.RESULTS / "headroom_table.csv"
    if not path.exists() or path.stat().st_size < 5:
        return []
    table = pd.read_csv(path)
    return table[table.passes_headroom_screen].candidate_id.tolist()


def build() -> dict:
    needed = candidates_needing_audit()
    findings = json.loads(FINDINGS.read_text(encoding="utf-8")) if FINDINGS.exists() else {}
    audit = {}
    for candidate_id in needed:
        entry = findings.get(candidate_id)
        if entry is None:
            audit[candidate_id] = {
                "novelty_status": "UNCLEAR",
                "summary": "no literature entry was recorded for this candidate",
                "papers": [],
                "feasibility_points": 1,
            }
            continue
        missing = [field for field in REQUIRED_FIELDS if field not in entry]
        if missing:
            raise SystemExit(f"literature entry for {candidate_id} is missing {missing}")
        if entry["novelty_status"] not in VALID_STATUS:
            raise SystemExit(f"invalid novelty_status for {candidate_id}: {entry['novelty_status']}")
        audit[candidate_id] = entry
    return audit


def render(audit: dict) -> str:
    lines = [
        "# Literature and novelty audit",
        "",
        "Searched only for candidates that cleared the headroom screen, as Section 22 requires.",
        "Papers are cited as Title (year/venue); arXiv preprints are marked as such and kept",
        "separate from peer-reviewed work.",
        "",
    ]
    if not audit:
        lines += [
            "No candidate cleared the headroom screen, so no novelty search was performed.",
            "",
            "This is a deliberate ordering: searching the literature for a gap that has not yet",
            "shown recoverable headroom would invite fitting a story to whatever turns up.",
            "",
        ]
        return "\n".join(lines)
    for candidate_id, entry in audit.items():
        lines += [
            f"## {candidate_id}",
            "",
            f"- **novelty status**: {entry['novelty_status']}",
            f"- exact problem: {entry.get('exact_problem', '')}",
            f"- exact intervention point: {entry.get('exact_intervention_point', '')}",
            f"- closest method: {entry.get('closest_method', '')}",
            f"- same information condition: {entry.get('same_information_condition', '')}",
            f"- same benchmark: {entry.get('same_benchmark', '')}",
            f"- same backbone family: {entry.get('same_backbone_family', '')}",
            f"- direct overlap: {entry.get('direct_overlap', '')}",
            f"- difference substantial or cosmetic: {entry.get('difference_substantial_or_cosmetic', '')}",
            "",
            "### Papers consulted",
            "",
        ]
        for paper in entry.get("papers", []):
            venue = paper.get("venue", "arXiv preprint")
            lines.append(f"- {paper['title']} ({paper.get('year')}/{venue}) — {paper.get('relevance', '')}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    audit = build()
    (paths.RESULTS / "literature_audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    (paths.RESULTS / "literature_audit.md").write_text(render(audit), encoding="utf-8")
    print(f"audited {len(audit)} candidate(s)")
    for candidate_id, entry in audit.items():
        print(f"  {candidate_id}: {entry['novelty_status']}")


if __name__ == "__main__":
    main()
