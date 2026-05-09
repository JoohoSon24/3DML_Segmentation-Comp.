#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="${CONDA_ENV:-3d-seg}"
DATA_ROOT="${DATA_ROOT:-$REPO_DIR/data/nubzuki_multiscan_trainval_npy}"
TRAIN_PREFIX="${TRAIN_PREFIX:-train}"
VAL_PREFIX="${VAL_PREFIX:-val}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_DIR/configs/softgroup/softgroup_nubzuki.yaml}"
WORK_DIR="${WORK_DIR:-$REPO_DIR/work_dirs/softgroup_nubzuki_x2_revised}"
CHECKPOINT_OUT="${CHECKPOINT_OUT:-$REPO_DIR/checkpoints/best.pth}"
RESUME_PATH="${RESUME_PATH:-}"
SKIP_VALIDATE="${SKIP_VALIDATE:-0}"
EPOCHS="${EPOCHS:-}"
STEP_EPOCH="${STEP_EPOCH:-}"
SAVE_FREQ="${SAVE_FREQ:-}"
BATCH_SIZE="${BATCH_SIZE:-}"
TRAIN_NUM_WORKERS="${TRAIN_NUM_WORKERS:-}"
TEST_NUM_WORKERS="${TEST_NUM_WORKERS:-}"
LR="${LR:-}"
CUDA_VISIBLE_DEVICES_VALUE="${CUDA_VISIBLE_DEVICES_VALUE:-${CUDA_VISIBLE_DEVICES:-}}"

if [[ ! -d "$DATA_ROOT/$TRAIN_PREFIX" ]]; then
  echo "Training split directory not found: $DATA_ROOT/$TRAIN_PREFIX" >&2
  exit 1
fi

if [[ ! -d "$DATA_ROOT/$VAL_PREFIX" ]]; then
  echo "Validation split directory not found: $DATA_ROOT/$VAL_PREFIX" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p "$WORK_DIR"
mkdir -p "$(dirname "$CHECKPOINT_OUT")"

cmd=(
  conda run --no-capture-output -n "$CONDA_ENV"
  python "$REPO_DIR/train.py"
  --config "$CONFIG_PATH"
  --data-root "$DATA_ROOT"
  --train-prefix "$TRAIN_PREFIX"
  --val-prefix "$VAL_PREFIX"
  --work-dir "$WORK_DIR"
)

if [[ -n "$RESUME_PATH" ]]; then
  cmd+=(--resume "$RESUME_PATH")
fi

if [[ "$SKIP_VALIDATE" == "1" ]]; then
  cmd+=(--skip-validate)
fi

if [[ -n "$EPOCHS" ]]; then
  cmd+=(--epochs "$EPOCHS")
fi

if [[ -n "$STEP_EPOCH" ]]; then
  cmd+=(--step-epoch "$STEP_EPOCH")
fi

if [[ -n "$SAVE_FREQ" ]]; then
  cmd+=(--save-freq "$SAVE_FREQ")
fi

if [[ -n "$BATCH_SIZE" ]]; then
  cmd+=(--batch-size "$BATCH_SIZE")
fi

if [[ -n "$TRAIN_NUM_WORKERS" ]]; then
  cmd+=(--train-num-workers "$TRAIN_NUM_WORKERS")
fi

if [[ -n "$TEST_NUM_WORKERS" ]]; then
  cmd+=(--test-num-workers "$TEST_NUM_WORKERS")
fi

if [[ -n "$LR" ]]; then
  cmd+=(--lr "$LR")
fi

echo "Training SoftGroup model"
echo "  repo_dir: $REPO_DIR"
echo "  conda_env: $CONDA_ENV"
echo "  data_root: $DATA_ROOT"
echo "  train_prefix: $TRAIN_PREFIX"
echo "  val_prefix: $VAL_PREFIX"
echo "  config_path: $CONFIG_PATH"
echo "  work_dir: $WORK_DIR"
echo "  checkpoint_out: $CHECKPOINT_OUT"

if [[ -n "$CUDA_VISIBLE_DEVICES_VALUE" ]]; then
  echo "  cuda_visible_devices: $CUDA_VISIBLE_DEVICES_VALUE"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_VALUE" "${cmd[@]}"
else
  "${cmd[@]}"
fi

SOURCE_CHECKPOINT="$WORK_DIR/best.pth"
if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  SOURCE_CHECKPOINT="$WORK_DIR/latest.pth"
fi

if [[ ! -f "$SOURCE_CHECKPOINT" ]]; then
  echo "Training finished, but no packaged checkpoint was found in: $WORK_DIR" >&2
  exit 1
fi

cp -f "$SOURCE_CHECKPOINT" "$CHECKPOINT_OUT"
echo "Packaged checkpoint to: $CHECKPOINT_OUT"

SUMMARY_FILE="$WORK_DIR/training_summary.json"
if [[ -f "$SUMMARY_FILE" ]]; then
  echo "Training summary: $SUMMARY_FILE"
fi
