#!/usr/bin/env bash
# Holdout leg: train the specialist suite on the fresh tasks, score everything with
# fev, then compute effects and the verdict. Ends with the decision, not with a
# finished job.
set -u

PY=".venv-tsfm/Scripts/python.exe"
PYS=".venv-tsfm-specialist/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
LOG="runs/tsfm_long_horizon_specialist_closure_v1/logs"
mkdir -p "$LOG"

step() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$LOG/driver.log"
  "$@" >> "$LOG/$name.log" 2>&1
  local code=$?
  echo "--- $name exit=$code" | tee -a "$LOG/driver.log"
  return $code
}

# One process per task. Fitting several AutoGluon suites in a single interpreter
# let resident memory climb until the host ran out mid-sweep; a fresh process per
# task bounds the peak. Completed (task, seed) pairs are skipped from the forecast
# cache, so an interrupted run resumes instead of refitting.
TASKS=$($PY -u -c "
import json
spec = json.load(open('results/tsfm_long_horizon_specialist_closure_v1/long_horizon_tasks.json'))
print(' '.join(spec['fresh_holdout_tasks']))
")
for task in $TASKS; do
  short=$(echo "$task" | sed 's/.*:://')
  step "train_holdout_${short}" $PYS -u -m \
    experiments.tsfm_long_horizon_specialist_closure_v1.src.train_specialists \
    --tasks "$task" --tag "_holdout_${short}" || exit 1
done

step merge_holdout_manifest $PY -u -c "
import pandas as pd, pathlib
R = pathlib.Path('results/tsfm_long_horizon_specialist_closure_v1')
parts = sorted(p for p in R.glob('specialist_manifest_holdout*.csv') if p.name != 'specialist_manifest_holdout.csv')
frames = [pd.read_csv(p) for p in parts]
pd.concat(frames, ignore_index=True).to_csv(R / 'specialist_manifest_holdout.csv', index=False)
print('holdout manifest rows:', sum(len(f) for f in frames))
" || exit 1

step evaluate_all $PY -u -m experiments.tsfm_long_horizon_specialist_closure_v1.src.evaluate_specialists \
  --pattern "S_*.npz" --out specialist_scores.csv || exit 1

step evaluate_components $PY -u -m experiments.tsfm_long_horizon_specialist_closure_v1.src.evaluate_specialists \
  --pattern "COMPONENT_*.npz" --out specialist_component_results.csv || exit 1

step merge_manifest $PY -u -c "
import pandas as pd, pathlib
R = pathlib.Path('results/tsfm_long_horizon_specialist_closure_v1')
frames = [pd.read_csv(R / f'specialist_manifest_{s}.csv') for s in ('development', 'holdout')]
pd.concat(frames, ignore_index=True).to_csv(R / 'specialist_manifest.csv', index=False)
print('merged manifest rows:', sum(len(f) for f in frames))
" || exit 1

step compare $PY -u -m experiments.tsfm_long_horizon_specialist_closure_v1.src.compare || exit 1
step verdict $PY -u -m experiments.tsfm_long_horizon_specialist_closure_v1.src.verdict || exit 1

echo "=== HOLDOUT LEG COMPLETE ===" | tee -a "$LOG/driver.log"
tail -40 "$LOG/compare.log" | tee -a "$LOG/driver.log"
tail -30 "$LOG/verdict.log" | tee -a "$LOG/driver.log"
