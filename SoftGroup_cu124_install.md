# SoftGroup Environment Setup (CUDA 12.4 Only)

This guide creates a **clean conda environment** for SoftGroup + spconv **locked to CUDA 12.4**.
It avoids mixing CUDA versions, which can break spconv builds and runtime.

Important compatibility note:
- The local [SoftGroup README](/home/ubuntu/JW/seg/SoftGroup/README.md:20) says the refactored code supports `pytorch 1.11` and `spconv 2.1`.
- A CUDA 12.4 stack is therefore **outside the upstream-tested matrix** for this repo.
- We can still construct a CUDA 12.4 environment, but if SoftGroup or spconv fails later, the next problem will be framework compatibility rather than Conda packaging.

## Assumptions
- You have a GPU driver compatible with CUDA 12.4.
- You can use `conda`.
- You can use the system CUDA 12.4 install (or a matching `pytorch-cuda=12.4` runtime).

## Recommended Environment
### 1) Create and activate a new env
```bash
conda create -n softgroup-cu124 python=3.10 -y
conda activate softgroup-cu124
conda config --env --set channel_priority strict
```

### 2) Install PyTorch with CUDA 12.4
Use the official PyTorch and NVIDIA channels with the CUDA 12.4 runtime.
Also pin MKL below `2024.1` to avoid the `iJIT_NotifyEvent` import issue seen with some Conda solves:
```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 "mkl<2024.1" -c pytorch -c nvidia
```

For building local CUDA extensions such as SoftGroup's ops, install the CUDA 12.4 toolkit too:
```bash
conda install -y cuda-toolkit=12.4 -c nvidia
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
```

Verify CUDA runtime version used by PyTorch:
```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda runtime', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
PY
```
Expected: `torch.version.cuda` shows `12.4`.

### 3) Install spconv for CUDA 12.4
Try a prebuilt wheel first. If it fails, build from source.

**Option A: Prebuilt wheel (if available)**
```bash
pip install spconv-cu124
```

**Option B: Build from source (CUDA 12.4)**
```bash
pip install spconv
```
If this fails, build spconv from source with your CUDA 12.4 toolkit installed and `CUDA_HOME` set.

### 4) Install SoftGroup dependencies
```bash
cd /home/ubuntu/JW/seg/SoftGroup
pip install -r requirements.txt
```

### 5) System build requirement (one-time)
SoftGroup build requires sparsehash headers:
```bash
sudo apt-get install -y libsparsehash-dev
```
If you do not have sudo, install the package via your system admin or skip and report the error.

### 6) Build and install SoftGroup
```bash
cd /home/ubuntu/JW/seg/SoftGroup
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
python -m pip install -e . --no-build-isolation
```

Why this command:
- `setup.py build_ext develop` eventually delegates to editable install machinery that may use isolated build environments.
- In this repo, that isolated build path can fail with `ModuleNotFoundError: No module named 'torch'`.
- `pip install -e . --no-build-isolation` keeps the build inside the active env where `torch` is already installed.

### 7) Sanity check imports
```bash
python - <<'PY'
import torch
import spconv.pytorch as spconv
from softgroup.model import SoftGroup
print('ok', torch.__version__)
PY
```

## Notes
- **Do not install any other CUDA version** in this environment.
- If `spconv` fails to compile, re-check:
  - `torch.version.cuda == 12.4`
  - `nvcc --version` reports CUDA 12.4
  - `CUDA_HOME` points to CUDA 12.4
- If build fails with `CUDA_HOME environment variable is not set`, install `cuda-toolkit=12.4` in the env and set `CUDA_HOME="$CONDA_PREFIX"`.
- If build fails with `fatal error: nv/target: No such file or directory`, export:
```bash
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
```
- This is needed because Conda's CUDA headers place `nv/target` under `targets/x86_64-linux/include`, while some builds only include `$CONDA_PREFIX/include`.
- If editable install fails with `ModuleNotFoundError: No module named 'torch'`, use:
```bash
python -m pip install -e . --no-build-isolation
```

## Troubleshooting
### ImportError: undefined symbol `iJIT_NotifyEvent`
This is usually a Conda runtime-linking issue around MKL / Intel ITT, not a fully corrupted env.

Check whether `libittnotify.so` is missing:
```bash
conda activate softgroup-cu124
find "$CONDA_PREFIX" -type f -name 'libittnotify.so*'
```

If nothing is found, the most reliable fix is to downgrade MKL and refresh the PyTorch solve:
```bash
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 "mkl<2024.1" -c pytorch -c nvidia
```

If you want to repair the current env in place, try:
```bash
conda activate softgroup-cu124
conda install -y "mkl<2024.1"
```

Then rerun:
```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda runtime', torch.version.cuda)
print('cuda available', torch.cuda.is_available())
PY
```

If the symbol error still remains after downgrading MKL, recreate the env from scratch with the pinned install command above. That is usually faster and cleaner than fighting a partially solved Conda state.

## Optional: Add project utilities
If you want to run the dataset tools in this env:
```bash
pip install numpy trimesh matplotlib
```

---

If you want me to execute these steps, tell me the environment name you want to use.
