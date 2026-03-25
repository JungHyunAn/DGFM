#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_all.sh <task_name> <seed>
# Example:
#   CUDA_VISIBLE_DEVICES=1 ./run_all.sh door 2000

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <task_name> <seed>"
  exit 1
fi

TASK_NAME="$1"
SEED="$2"
SLEEP_TIME=600   # ← sleep duration between runs

###############################
# Dataset / results / N list
###############################
case "$TASK_NAME" in
  door)
    DATASET_PATH="Robot_simulation/heuristic_dataset/door_dataset_10000.hdf5"
    RESULTS_PATH="Robot_simulation/eval_results/door"
    N_LIST=(1000 500 250)
    ;;
  nut)
    DATASET_PATH="Robot_simulation/heuristic_dataset/nut_dataset_10000.hdf5"
    RESULTS_PATH="Robot_simulation/eval_results/nut"
    N_LIST=(1000 500 250)
    ;;
  two_arm)
    DATASET_PATH="Robot_simulation/heuristic_dataset/two_arm_dataset_10000.hdf5"
    RESULTS_PATH="Robot_simulation/eval_results/two_arm"
    N_LIST=(2000 1000 500)
    ;;
  *)
    echo "Error: task_name must be one of {door, nut, two_arm}."
    exit 1
    ;;
esac

echo "Task       : $TASK_NAME"
echo "Seed       : $SEED"
echo "Dataset    : $DATASET_PATH"
echo "Results    : $RESULTS_PATH"
echo "N list     : ${N_LIST[*]}"
echo

########################################
# Helper to run one FM configuration
########################################
run_fm() {
  local FM_TYPE="$1"    # UniformFM / ShiftedFM / DGFM
  local MF_ORDER="$2"   # "" or 2 or 4

  echo "==========================================="
  echo " Running FM_type=$FM_TYPE  mf_order=${MF_ORDER:-none}"
  echo "==========================================="

  local VAL_PERIOD
  local MAX_EPOCHS
  local WARMUP_STEPS
  local EXTRA_FLAGS=()

  if [[ "$FM_TYPE" == "UniformFM" || "$FM_TYPE" == "ShiftedFM" || "$FM_TYPE" == "LFM" || "$FM_TYPE" == "GFM" || "$FM_TYPE" == "GMM" ]]; then
    VAL_PERIOD=30
    MAX_EPOCHS=3000
    WARMUP_STEPS=600
  else
    # DGFM
    if [[ "$MF_ORDER" == "2" ]]; then
      VAL_PERIOD=10
      MAX_EPOCHS=1000
      WARMUP_STEPS=200
    elif [[ "$MF_ORDER" == "4" ]]; then
      VAL_PERIOD=6
      MAX_EPOCHS=600
      WARMUP_STEPS=120
    else
      echo "Error: DGFM requires mf_order 2 or 4."
      exit 1
    fi
    EXTRA_FLAGS=(--mf "$MF_ORDER")
  fi

  for N in "${N_LIST[@]}"; do
    local BATCH_SIZE
    if [[ "$TASK_NAME" == "two_arm" ]]; then
      BATCH_SIZE=$((N / 4))
    else
      BATCH_SIZE=$((N / 2))
    fi

    echo "---- Running N=$N (batch_size=$BATCH_SIZE) ----"

    python -m Robot_simulation.run_eval \
      --FM_type "$FM_TYPE" \
      --N "$N" \
      --dataset_path "$DATASET_PATH" \
      --task_name "$TASK_NAME" \
      --results_path "$RESULTS_PATH" \
      --device cuda \
      --val_period "$VAL_PERIOD" \
      --batch_size "$BATCH_SIZE" \
      --max_epochs "$MAX_EPOCHS" \
      --warmup_steps "$WARMUP_STEPS" \
      --seed "$SEED" \
      "${EXTRA_FLAGS[@]}"

    echo ">> Sleeping $SLEEP_TIME seconds before next N..."
    sleep $SLEEP_TIME
    echo
  done
}

########################################
# Full execution chain:
########################################

# run_fm "GFM" ""

# run_fm "LFM" ""

# run_fm "GMM" ""

# run_fm "DGFM" "2"

# run_fm "DGFM" "4"

run_fm "UniformFM" ""

run_fm "ShiftedFM" ""



echo "==========================================="
echo "        ALL RUNS FINISHED"
echo "==========================================="
