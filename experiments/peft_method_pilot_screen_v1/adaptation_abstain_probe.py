"""Read-only probe: does any stored cell prefer refusing adaptation?

Direction A assumes a rule that decides *whether* to adapt would pay off. The
stored Study14 selection archive already scores seven policies, F0 among them,
on four (source, block) cells. This recomputes the F0-relative loss of each
policy from that archive. Writes nothing into the study.
"""
import csv
import json
from pathlib import Path

ROOT = Path("E:/CODING/proj/mltimeseries")
SOURCE = ROOT / "results/peft_selection_regret_v1/selected_results.csv"

rows = [r for r in csv.DictReader(SOURCE.open()) if r["procedure"] == "SORT"]
selectors = sorted({r["selector"] for r in rows})
cells = sorted({r["cell"] for r in rows})

scores, relative = {}, {}
for cell in cells:
    baseline = float(next(r for r in rows if r["cell"] == cell and r["selector"] == "F0")["score"])
    scores[cell], relative[cell] = {}, {}
    for selector in selectors:
        match = [r for r in rows if r["cell"] == cell and r["selector"] == selector]
        if not match:
            continue
        value = float(match[0]["score"])
        scores[cell][selector] = value
        relative[cell][selector] = 100.0 * (value - baseline) / baseline

adapting = [s for s in selectors if s != "F0"]
report = {
    "source_csv": str(SOURCE.relative_to(ROOT)),
    "procedure": "SORT",
    "cells": cells,
    "selectors": selectors,
    "score": scores,
    "loss_vs_f0_pct": relative,
    "cells_where_f0_is_best": [c for c in cells if all(relative[c][s] > 0 for s in adapting if s in relative[c])],
    "cells_where_fixed_low_beats_f0": [c for c in cells if relative[c].get("FIXED_LOW", 0.0) < 0],
    "cells_where_validation_all_beats_f0": [c for c in cells if relative[c].get("ALL_V", 0.0) < 0],
    "note": "One optimizer seed (12000) on development data from Studies 12 and 13. Four cells is not a"
            " sample from which to generalise; it is enough to check whether the premise of direction A"
            " holds at all in the data this repository already has.",
}
print(json.dumps(report, indent=1))
