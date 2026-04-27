# SoftGroup Migration And Integration Status

Last updated: 2026-04-27

## Scope

Goal: keep the SoftGroup-based Nubzuki solution runnable from
`3DML_Segmentation-Comp.` through the official-style entry points:

- `dataset.py`
- `evaluate.py`
- `model.py`
- `train.py`

No bash scripts should be required in the final runtime path. After TA
clarification, the repository should include an end-to-end environment and build
guide because SoftGroup depends on a custom CUDA extension.

## Current Integration Status

### Completed

- Essential SoftGroup code is copied under `softgroup/`.
- `softgroup/__init__.py` makes the copied tree a real local package, avoiding
  accidental imports from sibling `seg/SoftGroup`.
- Nubzuki data loading is registered through `softgroup/data/nubzuki.py`.
- `softgroup/data/__init__.py` is Nubzuki-only and no longer imports missing
  upstream datasets.
- `model.py` implements the official challenge interface:
  - `initialize_model(ckpt_path, device, ...)`
  - `run_inference(model, features, ...) -> [B, N]`
- `model.py` converts challenge features `[B, 9, N]` into the SoftGroup
  voxelized batch format and decodes SoftGroup masks back into point-wise
  challenge instance IDs.
- `train.py` is now a Python entry point for SoftGroup training. It supports
  useful overrides such as:
  - `--epochs`
  - `--work-dir`
  - `--data-root`
  - `--batch-size`
  - `--lr`
- Bash workflow scripts were removed.
- Build/install support is present at the challenge root:
  - `setup.py`
  - `requirements-softgroup-cu124.txt`
- The copied compiled op is still retained:
  - `softgroup/ops/ops.cpython-310-x86_64-linux-gnu.so`
- `SoftGroup_cu124_install.md` now documents the full CUDA/Python/PyTorch/
  dependency/build sequence for the challenge-local SoftGroup copy.

## Important Runtime Constraint

SoftGroup requires custom CUDA operators at runtime. The op import is:

```python
from softgroup.ops import ops
```

So the compiled `.so` is necessary. A prebuilt `.so` can work locally, but the
submission should also include the build path so the TA environment can recreate
it if needed.

Current intended behavior:

1. Python imports local `softgroup/` from the repo root.
2. `softgroup.ops` imports the bundled `.so`.
3. `evaluate.py` calls `model.py`, which calls SoftGroup.

If the bundled `.so` is ABI-compatible with the official environment, evaluation
can run directly. If not, the provided `setup.py` build path should rebuild the
extension as `softgroup.ops.ops`.

## Environment Memo

The official environment is not fully disclosed, but the TA response says we
should provide an end-to-end installation guide for the current environment.
That guide is `SoftGroup_cu124_install.md`.

Locally verified environment:

- Python `3.10`
- PyTorch `2.5.1`
- CUDA runtime `12.4`
- `spconv-cu124==2.3.8`
- `numpy==2.0.1`
- `scipy==1.15.3`
- `pyyaml==6.0.3`
- `munch==4.0.0`
- `tensorboardX==2.6.5`
- `tqdm==4.67.3`

Expected challenge assumption:

- CUDA is available.
- If dependencies are not already available, follow `SoftGroup_cu124_install.md`.
- The repo is run from the challenge root.

## Current Functional Checks

All checks below were run from:

```bash
/home/ubuntu/JW/seg/3DML_Segmentation-Comp.
```

## Clean Environment Test

On 2026-04-27, a new environment was created specifically for the install/build
test:

```bash
softgroup-cu124_v2
```

The following guide steps were tested successfully:

1. Create a fresh Python 3.10 conda environment.
2. Install the pinned PyTorch/CUDA 12.4 stack.
3. Install `requirements-softgroup-cu124.txt`.
4. Build the challenge-local CUDA extension with:

```bash
python -m pip install -e . --no-build-isolation
```

Verified in `softgroup-cu124_v2`:

- `torch 2.5.1`
- `torch.version.cuda == 12.4`
- `torch.cuda.is_available() == True`
- `nvcc --version` reports CUDA `12.4`
- `softgroup` imports from `3DML_Segmentation-Comp./softgroup`
- `softgroup.ops.ops` imports from the locally built `.so`
- Python compile check passes for `dataset.py`, `evaluate.py`, `model.py`, and
  `train.py`
- `train.py --epochs 0 --skip-validate` initializes model/data/trainer cleanly
- a one-scene, one-epoch tiny training smoke passed and exercised forward/backward
  through the custom CUDA ops
- a one-scene `evaluate.py` smoke passed with the existing trained checkpoint

Import and local package check:

```bash
conda run -n softgroup-cu124 python -c "import softgroup; from softgroup.ops import ops; print(softgroup.__file__); print(ops.__file__)"
```

Observed:

- `softgroup` resolves inside `3DML_Segmentation-Comp./softgroup`.
- `ops` resolves to `softgroup/ops/ops.cpython-310-x86_64-linux-gnu.so`.

Python compile check:

```bash
conda run -n softgroup-cu124 python -m py_compile dataset.py evaluate.py model.py train.py tools/train.py tools/test.py tools/eval_nubzuki.py
```

Dataset check:

```bash
conda run -n softgroup-cu124 python -c "from dataset import InstancePointCloudDataset; ds=InstancePointCloudDataset('data/nubzuki_multiscan_trainval_npy/val', split='all'); item=ds[0]; print(len(ds), item['features'].shape, item['instance_labels'].shape)"
```

Observed:

- 42 validation scenes load through `dataset.py`.

Model interface single-scene check:

```bash
conda run -n softgroup-cu124 python -c "import torch; from dataset import InstancePointCloudDataset; from model import initialize_model, run_inference; ckpt='/home/ubuntu/JW/seg/SoftGroup/work_dirs/nubzuki_multiscan_trainval_softgroup_2/latest.pth'; device=torch.device('cuda'); ds=InstancePointCloudDataset('data/nubzuki_multiscan_trainval_npy/val', split='all'); features=ds[0]['features'].unsqueeze(0).to(device); model=initialize_model(ckpt, device=device); pred=run_inference(model, features); print(pred.shape, int(pred.max()))"
```

Observed:

- output shape `[1, N]`
- nonzero predicted instance IDs

Official evaluator check:

```bash
conda run -n softgroup-cu124 python evaluate.py \
  --test-data-dir data/nubzuki_multiscan_trainval_npy/val \
  --ckpt-path /home/ubuntu/JW/seg/SoftGroup/work_dirs/nubzuki_multiscan_trainval_softgroup_2/latest.pth \
  --output-dir /tmp/jw_nubzuki_eval_smoke
```

Observed on 42 local validation scenes:

- model parameter count: `30,837,323`
- F1@0.25: `0.9739`
- F1@0.50: `0.9652`

Training initialization check without bash:

```bash
conda run -n softgroup-cu124 python train.py \
  --config configs/softgroup/softgroup_nubzuki.yaml \
  --epochs 0 \
  --work-dir /tmp/jw_nubzuki_train_smoke \
  --skip-validate
```

Observed:

- model initialized
- train dataset loaded
- val dataset loaded
- trainer setup completed
- exited cleanly because `--epochs 0`

## Entry Point Status

### `dataset.py`

Functional for challenge-format `.npy` data. It returns:

- `features`: `[9, N]`
- `instance_labels`: `[N]`
- `scene_path`

### `model.py`

Functional for SoftGroup inference through the official interface. Requires:

- CUDA
- importable bundled SoftGroup op
- compatible checkpoint

### `evaluate.py`

Functional with `model.py`. It successfully ran on the local 42-scene val split.

### `train.py`

Functional as a Python entry point. It delegates to the copied SoftGroup trainer
without requiring bash scripts. It supports CLI overrides for short smoke tests
and regular training.

## Remaining Gaps

- The selected trained checkpoint is still outside the challenge repo:
  - `/home/ubuntu/JW/seg/SoftGroup/work_dirs/nubzuki_multiscan_trainval_softgroup_2/latest.pth`
- For submission-style execution, place the selected checkpoint at:
  - `checkpoints/best.pth`
- `.pth` files are ignored by `.gitignore`, so checkpoint packaging must be
  handled explicitly.
- The bundled `.so` can be a portability risk if the official Python, PyTorch,
  and CUDA ABI differ. This is why `setup.py` and the install guide include the
  rebuild path.
- There is no CPU fallback.

## Bottom Line

The current repo no longer depends on bash scripts for the intended runtime path.
The required SoftGroup op is bundled and imported directly, and an end-to-end
build guide is provided in case the op must be rebuilt in the TA environment.
The official-style entry points are functional locally, with checkpoint packaging
remaining as the main practical submission item.
