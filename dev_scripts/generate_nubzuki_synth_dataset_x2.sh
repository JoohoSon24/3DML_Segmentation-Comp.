#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

CONDA_ENV="${CONDA_ENV:-3d-seg}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_DIR/data/data/object_instance_segmentation}"
MESH_PATH="${MESH_PATH:-$REPO_DIR/assets/sample.glb}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_DIR/data/nubzuki_multiscan_trainval_npy}"
SPLITS="${SPLITS:-train,val}"
VARIANTS_PER_SCENE="${VARIANTS_PER_SCENE:-2}"
SEED="${SEED:-20260428}"
DEBUG_GLB="${DEBUG_GLB:-0}"
DEBUG_UP_AXIS="${DEBUG_UP_AXIS:-y}"
OVERWRITE="${OVERWRITE:-0}"

if [[ ! -d "$SOURCE_ROOT" ]]; then
  echo "Source root not found: $SOURCE_ROOT" >&2
  exit 1
fi

if [[ ! -f "$MESH_PATH" ]]; then
  echo "Mesh path not found: $MESH_PATH" >&2
  exit 1
fi

cmd=(
  conda run --no-capture-output -n "$CONDA_ENV"
  python "$REPO_DIR/dataset_tools/generate_synthetic_dataset.py"
  --source-root "$SOURCE_ROOT"
  --mesh-path "$MESH_PATH"
  --output-dir "$OUTPUT_DIR"
  --splits "$SPLITS"
  --variants-per-scene "$VARIANTS_PER_SCENE"
  --seed "$SEED"
  --debug-up-axis "$DEBUG_UP_AXIS"
)

if [[ "$DEBUG_GLB" == "1" ]]; then
  cmd+=(--debug-glb)
fi

if [[ "$OVERWRITE" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "Generating synthetic dataset"
echo "  repo_dir: $REPO_DIR"
echo "  conda_env: $CONDA_ENV"
echo "  source_root: $SOURCE_ROOT"
echo "  output_dir: $OUTPUT_DIR"
echo "  splits: $SPLITS"
echo "  variants_per_scene: $VARIANTS_PER_SCENE"
echo "  seed: $SEED"
echo "  debug_glb: $DEBUG_GLB"

"${cmd[@]}"

echo "Dataset generation finished: $OUTPUT_DIR"
