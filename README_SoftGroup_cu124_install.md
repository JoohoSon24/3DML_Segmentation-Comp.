# SoftGroup CUDA 12.4 Installation Guide

This guide is for the challenge-local SoftGroup copy inside:

```text
3DML_Segmentation-Comp./
```

It installs the exact environment used for this submission, including the local
SoftGroup CUDA/C++ extension build.

## Quick Install Flow

From the challenge repo root, the installation path is:

1. Create and activate the conda environment.
2. Install the pinned PyTorch and CUDA 12.4 stack.
3. Install Python dependencies.
4. Install the sparsehash system dependency.
5. Build the local SoftGroup extension.

The smoke tests are intentionally separated into their own section at the end.

## Validated Local Environment

The environment validated locally for this project is:

- Python `3.10`
- PyTorch `2.5.1`
- TorchVision `0.20.1`
- TorchAudio `2.5.1`
- CUDA runtime `12.4`
- CUDA toolkit `12.4`
- `spconv-cu124==2.3.8`
- `numpy==2.0.1`
- `scipy==1.15.3`
- `pyyaml==6.0.3`
- `munch==4.0.0`
- `tensorboardX==2.6.5`
- `tqdm==4.67.3`

## Installation

All commands below assume you are working from the challenge repo root:

```bash
cd /path/to/3DML_Segmentation-Comp.
```

### 1. Create the conda environment

```bash
conda create -n softgroup-cu124 python=3.10 -y
conda activate softgroup-cu124
conda config --env --set channel_priority strict
```

### 2. Install PyTorch and the CUDA 12.4 toolkit

```bash
conda install -y \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  pytorch-cuda=12.4 cuda-toolkit=12.4 "mkl<2024.1" \
  -c pytorch -c nvidia
```

Set the build paths needed by the SoftGroup extension:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
```

Optional quick check:

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda runtime", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
PY
```

Expected:

- `torch 2.5.1`
- CUDA runtime `12.4`
- CUDA available `True`

### 3. Install Python dependencies

```bash
python -m pip install -r requirements-softgroup-cu124.txt
```

### 4. Install the system build dependency

SoftGroup ops use sparsehash headers.

```bash
sudo apt-get update
sudo apt-get install -y libsparsehash-dev
```

If `sudo` is unavailable, install `libsparsehash-dev` through the system package
manager or ask the environment maintainer to provide it.

### 5. Build the local SoftGroup CUDA extension

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
python -m pip install -e . --no-build-isolation
```

This builds and installs the local extension as:

```python
softgroup.ops.ops
```

The generated binary should appear under `softgroup/ops/`, typically with a name
like:

```text
ops.cpython-310-x86_64-linux-gnu.so
```

## Post-Install Verification

These checks are not part of the installation itself. They are optional sanity
checks for confirming the environment was built correctly.

### 1. Import check

Run from the challenge repo root:

```bash
python - <<'PY'
import torch
import spconv.pytorch as spconv
import softgroup
from softgroup.ops import ops
from softgroup.model import SoftGroup

print("torch", torch.__version__)
print("softgroup", softgroup.__file__)
print("ops", ops.__file__)
print("ok")
PY
```

Expected:

- `softgroup` path points inside the submitted `3DML_Segmentation-Comp./softgroup`
- `ops` imports successfully

## Smoke Tests

These are optional functional checks after installation. They are helpful for TA
verification, but they are not required to finish the environment setup.

### 1. Python compile check

```bash
python -m py_compile dataset.py evaluate.py model.py train.py tools/train.py tools/test.py tools/eval_nubzuki.py
```

### 2. Train initialization smoke test

```bash
python train.py \
  --config configs/softgroup/softgroup_nubzuki.yaml \
  --epochs 0 \
  --work-dir /tmp/nubzuki_train_smoke \
  --skip-validate
```

### 3. Evaluation smoke test

Run this only after a valid checkpoint exists at `checkpoints/best.pth`.

```bash
python evaluate.py \
  --test-data-dir data/nubzuki_multiscan_trainval_npy/val \
  --ckpt-path checkpoints/best.pth \
  --output-dir /tmp/nubzuki_eval_smoke
```

## Submission Notes

- The four intended Python entry points are:
  - `dataset.py`
  - `evaluate.py`
  - `model.py`
  - `train.py`
- No bash wrapper is required.
- `evaluate.py` loads the model through `model.initialize_model()`.
- The checkpoint should be placed at:

```text
checkpoints/best.pth
```

## Troubleshooting

### `ModuleNotFoundError: softgroup.ops.ops`

The CUDA extension was not built or is not on the import path. Rerun:

```bash
python -m pip install -e . --no-build-isolation
```

from the challenge repo root.

### `CUDA_HOME environment variable is not set`

Run:

```bash
export CUDA_HOME="$CONDA_PREFIX"
```

### `fatal error: nv/target: No such file or directory`

Run:

```bash
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
```

### `ImportError: undefined symbol iJIT_NotifyEvent`

This is usually an MKL/Intel runtime issue. Reinstall the pinned PyTorch stack:

```bash
conda install -y \
  pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  pytorch-cuda=12.4 cuda-toolkit=12.4 "mkl<2024.1" \
  -c pytorch -c nvidia
```
