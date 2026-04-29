#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"

CONDA_ENV="${CONDA_ENV:-softgroup-cu124}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data/nubzuki_multiscan_trainval_npy_x2_revised}"
TEST_SPLIT="${TEST_SPLIT:-val}"
WORK_DIR="${WORK_DIR:-$REPO_DIR/work_dirs/softgroup_nubzuki_x2_revised}"
CKPT_PATH="${CKPT_PATH:-$REPO_DIR/checkpoints/best.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/eval_outputs/softgroup_nubzuki_x2_revised}"
VISUALIZE="${VISUALIZE:-0}"
VIS_LIMIT="${VIS_LIMIT:-8}"
VIS_MAX_POINTS="${VIS_MAX_POINTS:-50000}"
VIS_POINT_SIZE="${VIS_POINT_SIZE:-3.0}"
VIS_VIEWS="${VIS_VIEWS:-front,back,left,right,top,bottom}"
METRICS_FILE="${METRICS_FILE:-metrics.json}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-}}"

TEST_DATA_DIR="$DATA_ROOT/$TEST_SPLIT"

if [[ ! -d "$TEST_DATA_DIR" ]]; then
  echo "Evaluation split directory not found: $TEST_DATA_DIR" >&2
  exit 1
fi

if [[ ! -f "$CKPT_PATH" ]]; then
  if [[ -f "$WORK_DIR/latest.pth" ]]; then
    CKPT_PATH="$WORK_DIR/latest.pth"
  else
    echo "Checkpoint not found: $CKPT_PATH" >&2
    exit 1
  fi
fi

mkdir -p "$OUTPUT_DIR"

cmd=(
  conda run -n "$CONDA_ENV"
  python "$REPO_DIR/evaluate.py"
  --test-data-dir "$TEST_DATA_DIR"
  --ckpt-path "$CKPT_PATH"
  --output-dir "$OUTPUT_DIR"
  --metrics-file "$METRICS_FILE"
  --vis-limit "$VIS_LIMIT"
  --vis-max-points "$VIS_MAX_POINTS"
  --vis-point-size "$VIS_POINT_SIZE"
  --vis-views "$VIS_VIEWS"
)

if [[ "$VISUALIZE" == "1" ]]; then
  cmd+=(--visualize)
fi

echo "Evaluating SoftGroup model"
echo "  repo_dir: $REPO_DIR"
echo "  conda_env: $CONDA_ENV"
echo "  test_data_dir: $TEST_DATA_DIR"
echo "  checkpoint: $CKPT_PATH"
echo "  output_dir: $OUTPUT_DIR"
echo "  visualize: $VISUALIZE"

if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
  echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES_VALUE"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${cmd[@]}"
else
  "${cmd[@]}"
fi

echo "Evaluation finished: $OUTPUT_DIR"
