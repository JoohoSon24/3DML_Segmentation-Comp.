# Dataset Generation Progress

## Scope
- Goal: synthesize new training scenes by inserting `assets/sample.glb` into MultiScan object-instance-segmentation `.pth` scenes.
- Final training artifact: fused point-cloud `.npy` files with keys `xyz`, `rgb`, `normal`, `instance_labels`.
- Debug artifacts: `.glb` for mesh visualization and `.ply` for point-cloud visualization.

## Implemented Files
- `dataset_tools/generate_synthetic_dataset.py`
  - Main synthetic dataset generator.
- `dataset_tools/export_npy_to_ply.py`
  - Debug exporter from fused `.npy` point clouds to `.ply`.
- `dataset_tools/export_pth_to_glb.py`
  - Utility to inspect original MultiScan `.pth` scenes as meshes.
- `Read/dataset_generation_plan.md`
  - Planning note updated with the current augmentation and placement policy.

## Current Pipeline
1. Load one MultiScan OIS scene from `.pth` using `torch.load(..., weights_only=False)`.
2. Read `xyz`, `rgb`, `normal`, and `faces`, then build a `trimesh` scene mesh for debugging.
3. Estimate scene point spacing from nearest-neighbor distances in the source point cloud.
4. Load `assets/sample.glb` once, bake vertex colors, and normalize it to a bottom-centered local frame.
5. For each synthetic variant of a scene:
   - sample a random object count in `1..5`,
   - for each object, sample anisotropic xyz scaling in `(0.5, 1.5)`,
   - sample independent xyz rotations in `(-180, 180)` degrees,
   - rescale again so the transformed object diagonal is in `0.025..0.2` of the scene diagonal,
   - jitter object colors,
   - place the object either on the scene or on top of a previously inserted object.
6. Convert each transformed object mesh into a point cloud.
7. Fuse background scene points and inserted object points into one `.npy`.
8. Keep all original scene points as background with `instance_labels = 0`.
9. Assign inserted objects contiguous positive labels `1..k` in insertion order.
10. Write one `manifest.jsonl` entry per synthesized scene with seed, transforms, support parent, bounds, and point-count metadata.

## Placement Policy In Code
- The first inserted object is placed on the scene.
- Later objects may be supported either by the scene or by a previously inserted object.
- Scene support anchors are sampled from scene points, biased toward upward-facing and top-region points.
- Object support anchors are sampled from the top band of the parent object.
- XY jitter is applied around the chosen support point to encourage variety.
- Placement is permissive by design:
  - overhang is allowed,
  - stack height is not limited,
  - pairwise collision rules are intentionally weak.
- One extra conservative rule was added during debugging:
  - placements are rejected if the object protrudes outside the scene XY bounds,
  - placements are rejected if the object bottom drops below the scene floor margin,
  - placements are rejected if too many existing scene points intrude into the inserted object hull above the support-contact band.
- This intrusion check was added to reduce obvious wall clipping.

## Point Sampling Strategy
- Initial surface-sampling results looked too cloudy and too dense compared with MultiScan.
- The current implementation therefore treats MultiScan OIS as closer to mesh-vertex sampling than random dense surface sampling.
- Inserted object points are now generated from the transformed mesh vertices, then voxel-thinned using a spacing derived from the source scene point spacing.
- A small random spacing multiplier is still used so outcomes are not identical across objects.
- Current object sampling also applies:
  - small normal perturbation,
  - very light per-point color noise,
  - broad point-count safeguards.
- Geometric position jitter is currently disabled to avoid the previous cloudy appearance.

## Output Format
- Synthesized `.npy` files contain exactly:
  - `xyz`: `float32 [N, 3]`
  - `rgb`: `uint8 [N, 3]`
  - `normal`: `float32 [N, 3]`
  - `instance_labels`: `int32 [N]`
- No `faces` are written in the synthesized `.npy`, because the training target is a fused point cloud.
- Split directories are written under `data/train`, `data/val`, and optionally `data/test`.

## Debug Workflow
- Mesh-side debugging:
  - `export_pth_to_glb.py` exports original `.pth` scenes to mesh formats.
  - `generate_synthetic_dataset.py --debug-glb` exports combined scene + inserted meshes.
- Point-cloud debugging:
  - `export_npy_to_ply.py` exports synthesized `.npy` files to `.ply`.
  - `.ply` supports `rgb` coloring or `instance` coloring.
- Axis convention:
  - both debug GLB export and PLY export now support Y-up conversion,
  - current defaults are Y-up for both tools.
- Important distinction:
  - `.glb` is for visualization/debugging only,
  - `.npy` is the actual dataset artifact used for training,
  - `.ply` is a debug view generated from `.npy`.

## CLI Status
- `generate_synthetic_dataset.py` currently supports:
  - `--source-root`
  - `--mesh-path`
  - `--output-dir`
  - `--splits`
  - `--variants-per-scene`
  - `--seed`
  - `--debug-glb`
  - `--debug-up-axis`
  - `--overwrite`
- `export_npy_to_ply.py` currently supports:
  - `--input`
  - `--out-dir`
  - `--color-by`
  - `--format`
  - `--up-axis`
- `export_pth_to_glb.py` currently supports:
  - `--input`
  - `--out-dir`
  - `--ext`
  - `--color-by`
  - `--up-axis`

## Current Debug Outputs
- Single-scene smoke-test outputs were generated under `dataset_tools/synthesized_demo*`.
- Eight-scene debug batches were generated under:
  - `dataset_tools/debug_batch_8`
  - `dataset_tools/debug_batch_8_ply`
  - `dataset_tools/debug_batch_8_yup`
  - `dataset_tools/debug_batch_8_yup_ply`
- The stale Z-up `debug_batch_8` GLBs were regenerated, so the current batch outputs are aligned with the Y-up debug convention.

## Current Assessment
- Working:
  - end-to-end `.pth` -> synthesized `.npy` generation,
  - stacked object placement,
  - debug `.glb` export,
  - debug `.ply` export,
  - manifest logging,
  - Y-up debug export,
  - basic wall-clipping avoidance.
- Still under refinement:
  - inserted object point density still does not yet blend perfectly with the MultiScan background in visualization,
  - background points appear more regular than inserted objects, so the object sampling model still needs further tuning to better match the source distribution,
  - realism is improved relative to the earlier cloudy surface sampling, but not yet fully satisfactory.

## Practical Usage Note
- The existing dataset loader in this repo recursively scans `.npy` files and makes its own split logic.
- To train on one authored synthetic split directly, point the dataset root at a specific folder such as `data/train` and use `split=\"all\"`.
