#!/usr/bin/env bash
# Discovery driver: finish the secondary tracks, then analyse, then stop at the
# freeze. It does not open the confirmation split - that needs the candidate spec
# to exist and be hashed first.
#
# Each (model, track) pair runs in its own process. Holding several foundation
# models in one long-lived interpreter let resident memory accumulate across
# tasks until the host ran out; a fresh process per model bounds the peak to one
# model and releases everything on exit. This is an infrastructure choice only -
# batching and device placement do not change any forecast.
set -u

PY=".venv-tsfm/Scripts/python.exe"
export PYTHONIOENCODING=utf-8
export HF_HOME="E:/CODING/proj/mltimeseries/runs/tsfm_benchmark_gap_discovery_v1/hf_cache"
LOG_DIR="runs/tsfm_benchmark_gap_discovery_v1/logs"
mkdir -p "$LOG_DIR"

MODELS="chronos-2 tirex-2 timesfm-3.0 chronos-2-synth seasonal-naive linear-ar-specialist"

expect_cells() {
  # Refuse to analyse a partly finished sweep: a missing SeasonalNaive row would
  # silently blank every relative_to_naive on that task.
  local track="$1" expected="$2"
  local have
  have=$(ls runs/tsfm_benchmark_gap_discovery_v1/predictions/*track${track}*.json 2>/dev/null | wc -l)
  if [ "$have" -lt "$expected" ]; then
    echo "HARD STOP: track ${track} has ${have}/${expected} cells" | tee -a "$LOG_DIR/driver.log"
    exit 1
  fi
  echo "track ${track}: ${have}/${expected} cells present" | tee -a "$LOG_DIR/driver.log"
}

step() {
  local name="$1"; shift
  echo "=== $name ===" | tee -a "$LOG_DIR/driver.log"
  "$@" >> "$LOG_DIR/$name.log" 2>&1
  local code=$?
  echo "--- $name exit=$code" | tee -a "$LOG_DIR/driver.log"
  return $code
}

run_track() {
  local track="$1"
  for model in $MODELS; do
    step "track${track}_${model}" $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.run_models \
      --split discovery --track "$track" --models "$model"
  done
}

run_track M
run_track C

expect_cells U 72
expect_cells M 24
expect_cells C 18

step descriptors $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.descriptors --splits discovery || exit 1
step evaluate $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.evaluate || exit 1
step fairness $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.fairness || exit 1
step failure_map $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.failure_map || exit 1
step freeze_candidates $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.make_candidates || exit 1
step probes $PY -u -m experiments.tsfm_benchmark_gap_discovery_v1.src.run_probes || exit 1

echo "=== DISCOVERY ANALYSIS COMPLETE ===" | tee -a "$LOG_DIR/driver.log"
tail -25 "$LOG_DIR/failure_map.log" | tee -a "$LOG_DIR/driver.log"
tail -25 "$LOG_DIR/freeze_candidates.log" | tee -a "$LOG_DIR/driver.log"
tail -30 "$LOG_DIR/probes.log" | tee -a "$LOG_DIR/driver.log"
