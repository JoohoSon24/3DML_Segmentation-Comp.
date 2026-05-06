# SoftGroup Integration Plan

## Goal

Use the local `third_party/SoftGroup` implementation to test whether a full 3D instance-segmentation pipeline can outperform the current placeholder baseline for the Nubzuki challenge, while keeping a path toward the official challenge interface in `model.py` and `evaluate.py`.

I see two targets:

1. **Capability test**
   Train and validate SoftGroup locally on our synthesized Nubzuki scenes as quickly as possible.
2. **Challenge-compatible integration**
   Make the resulting model callable through the official interface:
   - `initialize_model(ckpt_path, device, ...)`
   - `run_inference(model, features, ...) -> [B, N]`

The first target should come first. It is lower risk, and it tells us whether the architecture is worth fully wiring into the submission repo.

## Current Development Status

### What is already working

- The challenge rules and expected evaluation format are clear from `Read/README.md`.
- Synthetic dataset generation is already much further along than model development:
  - `dataset_tools/generate_synthetic_dataset.py` can synthesize fused `.npy` scenes by inserting `assets/sample.glb` into MultiScan OIS scenes.
  - The generator already logs manifests and supports debug `.glb` and `.ply` exports.
  - The intended final training artifact is already aligned with the challenge repo format:
    - `xyz`
    - `rgb`
    - `normal`
    - `instance_labels`
- The generator already assumes the `softgroup-cu124` environment for geometry tooling.
- The local MultiScan OIS dataset is present at:
  - `../data/object_instance_segmentation`
- Available local scene counts are:
  - `train`: 174 `.pth`
  - `val`: 42 `.pth`
  - `test`: 41 `.pth`

### What is still mostly a stub

- `model.py` is not a real model yet.
  - `DummyModel` has no implementation.
  - `run_inference()` currently returns all zeros.
  - `initialize_model()` is still placeholder-level code.
- `train.py` is only a scaffold and does not yet implement a real loss or checkpoint flow.

### SoftGroup readiness in this workspace

- The SoftGroup repo is already cloned at:
  - `third_party/SoftGroup`
- The requested environment exists:
  - `3d-seg2`
- Confirmed usable packages in that env:
  - `python 3.10.19`
  - `torch 2.7.1`
  - `spconv-cu126`
  - `numpy 2.2.6`
  - `scipy 1.15.3`
  - `trimesh 4.12.2`
- `import softgroup` succeeds in `softgroup-cu126`.
- `from softgroup.ops import ops` also succeeds, which means the custom compiled SoftGroup ops are already available in the current env.

My current conclusion is that the environment and third-party code are ready enough for integration work. The missing pieces are mostly data-contract wiring, config, and challenge-interface adaptation.

## Main Contract Mismatches We Need To Resolve

This is the most important part. Simply copying the SoftGroup model files into the challenge repo will not be enough because the data, labels, and inference contracts are different.

### 1. File format mismatch

Current challenge repo:

- Trains/evaluates on `.npy` dicts containing:
  - `xyz`
  - `rgb`
  - `normal`
  - `instance_labels`

SoftGroup custom dataset flow:

- Expects per-scene `.pth` files containing, effectively:
  - `xyz`
  - `rgb`
  - `semantic_label`
  - `instance_label`

Implication:

- We need a conversion step from our synthesized `.npy` scenes to SoftGroup-style `.pth` scenes.

### 2. Label convention mismatch

Challenge label convention:

- background = `0`
- object instances = `1..K`

SoftGroup label convention:

- semantic labels are required
- instance background / ignored points should be `-100`
- valid instance ids should be `0..K-1`

For our single-category challenge, the clean mapping is:

- `semantic_label = 0` for background
- `semantic_label = 1` for Nubzuki points
- `instance_label = -100` for background
- `instance_label = challenge_instance_id - 1` for positive points

This mapping is mandatory. If we keep background as `0` in SoftGroup instance labels, offset supervision and grouping targets will be wrong.

### 3. RGB range mismatch

SoftGroup custom dataset guidance expects RGB to be normalized to `[-1, 1]`.

Current challenge `.npy` data stores RGB as:

- `uint8 [0, 255]` in the synthetic artifacts I inspected

Implication:

- The converter must turn RGB into `float32` in `[-1, 1]` for SoftGroup training.

### 4. Semantic labels do not currently exist in the challenge `.npy`

Our current challenge artifacts only carry instance labels.

Implication:

- We must derive semantic labels during conversion.
- For the first SoftGroup integration, binary semantics are enough:
  - background = 0
  - Nubzuki = 1

### 5. Split handling is different

Current challenge `dataset.py`:

- recursively scans all `.npy` files
- randomizes them
- makes its own `train/val/test` split

SoftGroup:

- expects authored split directories such as `train/` and `val/`
- uses `prefix` and `suffix` from config

Implication:

- For SoftGroup training, we should not rely on the challenge loader’s split logic.
- We should write explicit split directories for SoftGroup and point configs directly at them.

### 6. Inference interface mismatch

Challenge evaluator expects:

- `run_inference(model, features)` where `features` is `[B, 9, N]`

SoftGroup expects:

- voxelized batch dict inputs with fields such as:
  - `coords`
  - `coords_float`
  - `feats`
  - `semantic_labels`
  - `instance_labels`
  - voxel maps

Implication:

- We need an adapter/wrapper for final challenge compatibility.
- That wrapper must build the SoftGroup-style inference batch from the challenge `features` tensor.

### 7. Prediction format mismatch

Challenge evaluator wants:

- one pointwise instance-id vector per scene
- IDs in `0..100`

SoftGroup inference produces:

- a list of predicted instances
- each with a class id, confidence, and mask

Implication:

- We need post-processing that converts SoftGroup’s predicted mask list into one non-overlapping pointwise label vector.
- The easiest first version is:
  - sort instances by confidence
  - paste masks in descending score order
  - assign contiguous ids `1..K`
  - leave untouched points as `0`

### 8. Coordinate convention decision

This is the most subtle contract.

The challenge `dataset.py` normalizes each scene by:

- subtracting the centroid
- dividing by max radius

SoftGroup normally groups in metric coordinate space after voxelization.

That means we must choose one consistent convention:

1. **Train SoftGroup on challenge-normalized coordinates**
   - Best if the final goal is to run through the official challenge `run_inference(features)` interface.
2. **Train SoftGroup on raw synthetic coordinates**
   - Easier for standalone SoftGroup experiments, but creates a mismatch when later plugging into the official evaluator.

My recommendation is to normalize coordinates for SoftGroup training in the same way the challenge loader does. That keeps training and final challenge inference aligned.

## What We Need To Add Or Modify

### A. Data conversion and dataset preparation

We need a dedicated conversion script, for example:

- `seg/3DML_Segmentation-Comp./dataset_tools/convert_npy_to_softgroup_pth.py`

Suggested input:

- generated challenge-format dataset under something like:
  - `seg/3DML_Segmentation-Comp./data/train`
  - `seg/3DML_Segmentation-Comp./data/val`
  - `seg/3DML_Segmentation-Comp./data/test`

Suggested output:

- `seg/SoftGroup/dataset/nubzuki/train/*.pth`
- `seg/SoftGroup/dataset/nubzuki/val/*.pth`
- `seg/SoftGroup/dataset/nubzuki/test/*.pth`

Per scene, the converter should:

1. Load challenge `.npy`.
2. Convert coordinates to the chosen convention.
   - Recommended: apply the same centering + radius normalization as `dataset.py`.
3. Convert RGB from `[0, 255]` to `[-1, 1]`.
4. Create binary `semantic_label`.
5. Convert instance labels:
   - background `0` -> `-100`
   - object ids `1..K` -> `0..K-1`
6. Save a SoftGroup-ready `.pth`.

The converter should also optionally compute dataset statistics, especially:

- object point-count mean for the positive class

We will need that for `grouping_cfg.class_numpoint_mean`.

### B. SoftGroup dataset registration

SoftGroup does not currently have a dataset entry for this challenge.

We should add a new dataset class, likely something like:

- `seg/SoftGroup/softgroup/data/nubzuki.py`

This can probably inherit from `CustomDataset` with minimal overrides.

Expected responsibilities:

- define:
  - `CLASSES = ('nubzuki',)`
- optionally support unlabeled test data later if needed
- otherwise reuse `CustomDataset.load()` if we store the `.pth` in the expected tuple form

We would also need to register it in:

- `seg/SoftGroup/softgroup/data/__init__.py`

### C. SoftGroup config for the Nubzuki task

We should add a dedicated config, for example:

- `seg/SoftGroup/configs/softgroup/softgroup_nubzuki.yaml`

First-pass settings should look roughly like this:

- `semantic_classes: 2`
- `instance_classes: 1`
- `ignore_label: -100`
- `grouping_cfg.ignore_classes: [0]`
- `grouping_cfg.class_numpoint_mean: [-1., <mean_points_for_nubzuki>]`
- `pretrain: ''`
- `fixed_modules: []`

Important notes:

- We should **not** use the official HAIS pretrained checkpoint for challenge submission work.
  - The challenge rules say no pretrained network.
- SoftGroup README uses HAIS pretraining for some official datasets, but our challenge integration should start from scratch.
- If training from scratch is unstable, we can still do a two-stage process inside our own data only:
  1. train semantic / offset backbone on the Nubzuki data
  2. continue full SoftGroup training from that checkpoint

That is still self-contained and does not violate the no-external-pretrain rule.

### D. Training path

For the capability test, the shortest useful path is:

1. Prepare converted SoftGroup dataset.
2. Add dataset class and config.
3. Run a single-scene overfit test.
4. Run a short train/val experiment with `seg/SoftGroup/tools/train.py`.
5. Validate predictions with `seg/SoftGroup/tools/test.py`.

This should happen before trying to replace the challenge `model.py`.

Reason:

- If SoftGroup does not train cleanly on our data, there is no value in spending time on the final adapter yet.

### E. Challenge-side wrapper for `model.py`

If the capability test looks promising, then we bridge it into the official challenge interface.

Likely changes:

- replace the current placeholder contents of `seg/3DML_Segmentation-Comp./model.py`
- or keep that file small and move the real wrapper logic into a new helper module

The wrapper needs to:

1. Load SoftGroup config + checkpoint in `initialize_model()`.
2. In `run_inference()`:
   - take `features [B, 9, N]`
   - use:
     - `features[:, 0:3, :]` as coords
     - `features[:, 3:6, :]` as RGB
   - ignore normals in the first version
   - convert RGB from `[0, 1]` to `[-1, 1]`
   - build the SoftGroup voxelized batch structure
   - run SoftGroup forward pass
   - decode predicted instances into a single `[B, N]` pointwise label tensor

This is the bridge that makes SoftGroup compatible with the official evaluator.

### F. Packaging and submission structure

For local experiments, the current setup is okay because:

- `softgroup-cu124` already imports `softgroup`
- the compiled ops already work

But for final submission, relying on a sibling repo outside the challenge directory is risky.

We should plan for one of these before submission:

1. Put SoftGroup under the challenge repo, for example under:
   - `3DML_Segmentation-Comp./third_party/SoftGroup`
2. Or copy only the required SoftGroup package files into the challenge repo.

Important point:

- copying only the network file is not enough
- SoftGroup also depends on:
  - dataset utilities
  - custom ops
  - voxelization / grouping logic
  - config
  - checkpoint loading utilities

### G. Build and dependency contract

In the current machine state, SoftGroup is already usable.

What is already satisfied:

- `softgroup-cu124` exists
- `spconv-cu124` exists
- `softgroup.ops` imports

What still matters for portability:

- if we rebuild on another machine, we may need:
  - `python setup.py build_ext develop` inside `seg/SoftGroup`
- the final submission environment must support the same custom op build path

This is a submission risk item, not a current blocker.

## Recommended Next Move

This is the sequence I would follow.

### Phase 1: Lock the data contract

1. Generate the real synthesized dataset into authored split folders if not already done.
2. Write the `.npy -> SoftGroup .pth` converter.
3. Make the converter follow the challenge normalization convention.
4. Compute `class_numpoint_mean` from the positive instances in the training split.

This phase is mandatory.

### Phase 2: Get SoftGroup training on the dataset

1. Add `nubzuki.py` dataset class in SoftGroup.
2. Register it in `softgroup/data/__init__.py`.
3. Add `softgroup_nubzuki.yaml`.
4. Run:
   - one-scene overfit
   - short train/val experiment
5. Inspect predicted masks from `tools/test.py`.

This phase tells us whether SoftGroup is actually promising on our synthesized data.

### Phase 3: Bridge to the challenge evaluator

1. Build the `model.py` wrapper around SoftGroup.
2. Convert SoftGroup instance predictions into challenge pointwise IDs.
3. Run the official `evaluate.py` on a local validation subset.
4. Check:
   - output IDs are in `0..100`
   - runtime is acceptable
   - memory use is acceptable

### Phase 4: Package for final submission

1. Decide how SoftGroup code will live inside the challenge submission tree.
2. Remove any dependency on external sibling paths.
3. Document all external code and citations in the write-up.

## Likely Minimal File Additions

If we proceed, I expect at least these new files:

- `seg/3DML_Segmentation-Comp./dataset_tools/convert_npy_to_softgroup_pth.py`
- `seg/SoftGroup/softgroup/data/nubzuki.py`
- `seg/SoftGroup/configs/softgroup/softgroup_nubzuki.yaml`

And likely changes to:

- `seg/SoftGroup/softgroup/data/__init__.py`
- `seg/3DML_Segmentation-Comp./model.py`

## Main Risks To Watch

- **Training / inference coordinate mismatch**
  - If SoftGroup is trained on raw coordinates but challenge inference uses normalized coordinates, grouping behavior will drift.
- **Wrong background instance encoding**
  - Background must become `-100` for SoftGroup training, not `0`.
- **No-pretrain rule**
  - We should not use the official HAIS pretrained checkpoint for challenge work.
- **Random split behavior in the current challenge loader**
  - Good for quick baselines, bad for exact SoftGroup train/val control.
- **Packaging risk**
  - Local env works now, but final submission must be self-contained.

## Bottom Line

My current understanding is:

- the data-generation side is already strong enough to support a real model experiment
- the SoftGroup repo and environment are already available locally
- the missing work is mostly conversion, dataset registration, config, and an inference adapter

So the next practical move is **not** to copy SoftGroup wholesale into `model.py` yet.

The next practical move is:

1. convert the synthesized challenge dataset into SoftGroup-ready `.pth`
2. add a minimal Nubzuki dataset/config inside `seg/SoftGroup`
3. verify SoftGroup can train and infer on our data
4. only then wire it into the official challenge `model.py`

That order gives the fastest feedback with the lowest integration risk.
