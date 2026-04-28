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
   - sample an internal layout mode from `scene_only`, `mixed`, or `stack_heavy`,
   - for each object, sample anisotropic xyz scaling in `(0.5, 1.5)`,
   - sample independent xyz rotations in `(-180, 180)` degrees,
   - rescale again so the transformed object diagonal is in `0.025..0.2` of the scene diagonal,
   - apply global HSV color jitter with hue coverage over the full circle,
   - place the object either on the scene or on top of a previously inserted object using scene support or parent-AABB-top support.
6. Convert each transformed object mesh into a point cloud.
7. Fuse background scene points and inserted object points into one `.npy`.
8. Keep all original scene points as background with `instance_labels = 0`.
9. Assign inserted objects contiguous positive labels `1..k` in insertion order.
10. If a scene variant fails to realize the requested object count or required stacked placements for its layout mode, resynthesize the whole scene with a fresh RNG state.
11. Write one `manifest.jsonl` entry per synthesized scene with seed, layout mode, transforms, support metadata, bounds, color jitter metadata, and point-count metadata.

## Placement Policy In Code
- The first inserted object is placed on the scene.
- Later objects may be supported either by the scene or by a previously inserted object, depending on the sampled layout mode.
- The current fixed layout-mode mix is:
  - `scene_only`: 35%
  - `mixed`: 45%
  - `stack_heavy`: 20%
- `mixed` scenes are expected to realize at least one stacked placement when feasible.
- `stack_heavy` scenes are expected to realize at least two stacked placements when the scene has at least three inserted objects.
- Scene support anchors are sampled from scene points, biased toward upward-facing and top-region points.
- Object support now uses the parent object's AABB top plane rather than top-band mesh vertices.
- Child placement on a parent object follows three fixed styles:
  - `centered`
  - `edge_overhang`
  - `corner_overhang`
- Partial overhang is allowed, including visibly floating support within the parent AABB footprint.
- Stacked placements must keep positive parent-child XY support overlap, with a minimum of 10% of the child footprint.
- Chain stacking is allowed: an inserted object may support another inserted object.
- Pairwise collision handling is now penetration-only:
  - touching is allowed,
  - parent-child support contact is allowed,
  - true 3D AABB penetration is rejected.
- Conservative placement guards are still kept:
  - placements are rejected if the object protrudes outside the scene XY bounds,
  - placements are rejected if the object bottom drops below the scene floor margin,
  - placements are rejected if too many existing scene points intrude into the inserted object hull above the support-contact band.
- The intrusion check is intentionally looser for stacked placements so valid support contact is not treated as wall clipping.

## Color Augmentation Policy
- The earlier per-channel RGB gain/offset jitter has been replaced.
- Inserted meshes now receive global HSV jitter:
  - hue shift sampled over the full 360-degree circle,
  - modest saturation scaling,
  - modest value scaling,
  - minimum saturation floor so even low-saturation colors rotate visibly.
- Very light per-point color noise is still applied after point sampling.

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

## Manifest Metadata
- `manifest.jsonl` now records scene-level layout metadata such as:
  - `layout_mode`
  - `scene_retry_index`
  - `stacked_object_count`
- Each object entry now records additional placement-debug metadata such as:
  - `placement_style`
  - `support_xy_overlap_ratio`
  - `color_hue_shift_deg`
  - `color_sat_scale`
  - `color_val_scale`
- The old `color_gain` and `color_offset` manifest fields are no longer used.

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
- Recent revised-generator debug batches were generated under:
  - `dataset_tools/debug_revised_glb_samples`
  - `dataset_tools/debug_revised_glb_stack_focus`
  - `dataset_tools/debug_revised_ply_stack_focus_rgb`
  - `dataset_tools/debug_revised_ply_stack_focus_instance`
- The stack-focused batch is especially useful for visually checking:
  - multi-level stacking,
  - edge and corner overhangs,
  - point-cloud separation under instance coloring,
  - color diversity under RGB coloring.

## Current Assessment
- Working:
  - end-to-end `.pth` -> synthesized `.npy` generation,
  - layout-mode-controlled scene diversification,
  - stacked object placement using parent-AABB-top support,
  - edge and corner overhang placement,
  - penetration-only object-object collision handling,
  - full-hue HSV object color jitter,
  - debug `.glb` export,
  - debug `.ply` export,
  - manifest logging,
  - Y-up debug export,
  - scene-level retry logic to keep requested object counts,
  - basic wall-clipping avoidance.
- Still under refinement:
  - inserted object point density still does not yet blend perfectly with the MultiScan background in visualization,
  - background points appear more regular than inserted objects, so the object sampling model still needs further tuning to better match the source distribution,
  - stacked point clouds now look substantially better than the earlier contact-rejecting version, but visual realism is still not fully matched to the source scans,
  - corner-overhang and very-low-overlap cases are now generated, so those difficult cases should continue to be monitored in both `.glb` and `.ply`.

## Practical Usage Note
- The existing dataset loader in this repo recursively scans `.npy` files and makes its own split logic.
- To train on one authored synthetic split directly, point the dataset root at a specific folder such as `data/train` and use `split=\"all\"`.
