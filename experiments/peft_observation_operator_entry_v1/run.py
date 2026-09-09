"""Execute the frozen exact-conditioning audit through the external CPU guard."""

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "2"

from datetime import datetime, timezone
import json
from pathlib import Path
import time

from . import model, prepare


def run(root=prepare.ROOT):
    started = time.perf_counter()
    root = Path(root).resolve()
    contract = prepare.validate(root)
    study, output = root / "runs" / prepare.STUDY, root / "results" / prepare.STUDY
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Preserve existing complete or partial numerical outputs")
    result = model.diagnostics()
    if result.get("passed") is not True:
        raise AssertionError("Do not interpret failed numerical correctness checks")
    if (result.get("settings") != contract["settings"] or
            result.get("verdict") != "EXISTING_LINEAR_CONDITIONING_SUFFICIENT" or
            not result.get("checks") or not all(value is True for value in result["checks"].values())):
        raise AssertionError("The numerical audit did not execute the fixed settings and checks")
    if prepare.validate(root) != contract:
        raise AssertionError("Protected source or parent evidence changed during the audit")
    record = {"completed": True, "study": prepare.STUDY, "finished_at_utc": datetime.now(timezone.utc).isoformat(),
              "contract_sha256": prepare.sha(study / "contract.json"), "plan_sha256": contract["plan_sha256"],
              "source_hashes": contract["source_hashes"], "settings": contract["settings"],
              "decision": "EXISTING_LINEAR_CONDITIONING_SUFFICIENT", "new_training_updates": 0,
              "new_method_demonstrated": False, "diagnostics": result,
              "scope": contract["scope"], "wall_seconds_before_output": time.perf_counter() - started}
    prepare.write(output / "result.json", record)
    prepare.write(study / "completed.json", {"completed": True, "result_sha256": prepare.sha(output / "result.json"),
                                             "contract_sha256": record["contract_sha256"], "new_training_updates": 0})
    return {"completed": True, "decision": record["decision"], "output": str(output)}


if __name__ == "__main__":
    print(json.dumps(run()))
