# SoftGroup Environment Setup (CUDA 12.4 Toolkit + pytorch 2.7.1 using cu126)

This guide creates a **clean conda environment** for SoftGroup + spconv **locked to CUDA 12.4 toolkit**.

Important compatibility note:
- The local SoftGroup README states that the refactored code supports pytorch 1.11 and spconv 2.1. Our team has confirmed that it builds successfully with pytorch 2.7.1, but runtime correctness still needs verification.

**Requirements**
- NVIDIA GPU driver compatible with CUDA 12.x (verified with CUDA 12.4 driver)
- `conda` (Miniconda or Anaconda)
- CUDA 12.4 toolkit installed into the conda environment via `conda install -y cuda-toolkit=12.4 -c nvidia` (used as `CUDA_HOME` for compiling extensions like `spconv` and `SoftGroup`)

**Note**
- PyTorch is installed from the `cu126` wheel, which bundles its own CUDA 12.6 runtime. All source-compiled extensions are built against the conda-installed CUDA 12.4 toolkit. The runtime/toolkit minor mismatch within CUDA 12.x is binary-compatible (CUDA minor version compatibility), and this setup has been verified to work on a CUDA 12.4 driver.

## Recommended Environment
### 1) Create and activate an environment
Env settings are almost identical to the one written in Seg Comp's given README.md. 
```bash
conda create -n 3d-seg python=3.10 -y
conda activate 3d-seg
```

### 2) Install PyTorch (cu126 wheel) and CUDA 12.4 toolkit
Install PyTorch from the official cu126 index. Also install numpy, scipy, tqdm, matplotlib for downstream use, and pin MKL below 2024.1 to avoid the iJIT_NotifyEvent import issue:
```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install numpy scipy tqdm matplotlib
conda install "mkl<2024.1"
```

For building local CUDA extensions such as SoftGroup's ops, install the CUDA 12.4 toolkit into the env:
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
Expected: `torch.version.cuda` shows `12.6`. This is fine — the PyTorch wheel bundles its own CUDA 12.6 runtime, while source-built extensions will use the conda-installed CUDA 12.4 toolkit. The runtime/toolkit minor mismatch within CUDA 12.x is binary-compatible (CUDA minor version compatibility).

### 3) Install spconv (cu126 wheel or CUDA 12.4 source build)
Try a prebuilt wheel first. If it fails, build from source.

**Option A: Prebuilt wheel (matches the cu126 runtime)**
```bash
pip install spconv-cu126
```

**Option B: Build from source (uses CUDA 12.4 toolkit via CUDA_HOME)**
```bash
pip install spconv
```
This will compile against the conda-installed CUDA 12.4 toolkit (the CUDA_HOME set in step 2).

### 4) Install SoftGroup dependencies
```bash
cd /home/ubuntu/cs479/3DML_Segmentation-Comp./third_party/SoftGroup
pip install -r requirements.txt
```

### 5) System build requirement (one-time)
SoftGroup build requires sparsehash headers, so install via conda
```bash
conda install -y -c conda-forge sparsehash
```

### 6) Build and install SoftGroup
```bash
cd /home/ubuntu/cs479/3DML_Segmentation-Comp./third_party/SoftGroup
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
- **Do not install any other CUDA version** in this environment. The only toolkit is CUDA 12.4 (conda-installed); the 12.6 runtime comes bundled inside the PyTorch wheel and is not a separate install.
- If `spconv` fails to compile, re-check:
  - `nvcc --version` reports CUDA 12.4
  - `CUDA_HOME` points to CUDA 12.4 toolkit
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
conda activate 3d-seg
find "$CONDA_PREFIX" -type f -name 'libittnotify.so*'
```

If nothing is found, downgrade MKL:
```bash
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
pip install trimesh
```

---