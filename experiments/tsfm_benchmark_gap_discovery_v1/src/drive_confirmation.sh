#!/usr/bin/env bash
# Confirmation driver. Runs only after candidate_probe_spec.json exists and is
# hashed: the holdout stays sealed until the candidates and their probes are
# frozen. Ends with the verdict, not with a finished job.
set -u

PY=".venv-tsfm/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
export HF_HOME="E:/CODING/proj/mltimeseries/runs/tsfm_benchmark_gap_discovery_v1/hf_cache"
LOG_DIR="runs/tsfm_benchmark_gap_discovery_v1/logs"
RESULTS="results/tsfm_benchmark_gap_discovery_v1"
mkdir -p "$LOG_DIR"

if [ ! -f "$RESULTS/candidate_probe_spec.sha256" ]; then
  echo "HARD STOP: candidate probe spec is not frozen; the holdout stays closed" | tee -a "$LOG_DIR/driver.log"
  exit 1
fi

step() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$LOG_DIR/driver.log"
  "$@" >> "$LOG_DIR/$name.log" 2>&1
  local code=$?
  echo "--- $name exit=$code" | tee -a "$LOG_DIR/driver.log"
  return $code
}

MODELS="chronos-2 tirex-2 timesfm-3.0 chronos-2-synth seasonal-naive linear-ar-specialist"

# One process per (model, track): several foundation models in one interpreter let
# resident memory climb across tasks until the host ran out during discovery.
run_track() {
  local track="$1"
  for model in $MODELS; do
    step "confirm_track${track}_${model}" $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.run_models       --split confirmation --track "$track" --models "$model"
  done
}

run_track U
run_track M
run_track C

step descriptors_confirmation $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.descriptors --splits confirmation || exit 1
step evaluate2 $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.evaluate || exit 1
step confirmation $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.confirm_candidates || exit 1
step characterisation_confirmation $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.oracle_exploitability --split confirmation || exit 1

echo "=== CONFIRMATION COMPLETE ===" | tee -a "$LOG_DIR/driver.log"
tail -20 "$LOG_DIR/confirmation.log" | tee -a "$LOG_DIR/driver.log"
