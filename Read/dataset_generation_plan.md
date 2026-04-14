# Dataset Generation Plan (MultiScan OIS + Synthetic GLB Insertions)

## Goals
- Export MultiScan `object_instance_segmentation/*.pth` samples into a VSCode-viewable 3D format (`.glb` preferred).
- Visualize and validate synthesized insertions by overlaying transformed `sample.glb` meshes inside the MultiScan scene mesh.
- Produce a combined mesh + combined pointcloud with updated instance labels for one inserted object category and multiple inserted instances per scene.

## Current Data Sources
- MultiScan object instance segmentation samples in `.pth` with keys: `xyz`, `rgb`, `normal`, `faces`, `sem_labels`, `instance_ids`, `inst2obj`, `inst2obj_id`
- External mesh to place: `seg/3DML_Segmentation-Comp./assets/sample.glb`

## Labeling Rule (Single Object Type)
- We segment only one object type (the inserted `sample.glb`).
- All original MultiScan points are treated as background with `instance_labels = 0`.
- Each inserted object instance gets a positive ID by insertion order: `1, 2, 3, ...`.
- We do not require `sem_labels`, `inst2obj`, or `inst2obj_id` in the final synthesized dataset.

## Synthesis Policy

### Per-Object Augmentations
- Anisotropic scaling: independently scale the x-, y-, and z-axes with factors sampled from `(0.5, 1.5)`.
- Affine rotation: rotate independently around the x-, y-, and z-axes with angles sampled from `(-180, 180)` degrees.
- Color map jittering: perturb the inserted mesh colors after geometry transforms and before export / point sampling.

### Scene-Level Placement Rules
- Insert a random number of objects per scene, sampled from `1` to `5`.
- For each inserted object, choose a target scale ratio from `0.025` to `0.2` of the scene diagonal.
- Placement should support both:
  - direct placement on the scene structure, and
  - placement on top of previously inserted objects.
- Stacked placement is determined using bounding-box alignment, and partial overhang is allowed.
- Instance IDs are assigned in insertion order regardless of whether an object is grounded on the scene or stacked on another object.

## Phase 1: Exporter for `.pth` to `.glb`
1. Load a `.pth` sample.
2. Build a triangle mesh from `xyz` + `faces`.
3. Assign vertex colors from `rgb`.
4. Export as `.glb` (also allow `.ply` or `.stl` if needed).
5. Verify VSCode 3D viewer can open the exported file.

## Phase 2: Transform + Place `sample.glb` in MultiScan Scene
1. Load `sample.glb` mesh.
2. Sample the number of insertions for the scene in the range `1..5`.
3. For each inserted object:
   - sample anisotropic scale factors in `(0.5, 1.5)` for x/y/z,
   - sample independent x/y/z rotations in `(-180, 180)` degrees,
   - apply color map jittering,
   - rescale the transformed object so its size falls within `0.025..0.2` of the scene diagonal,
   - place the object either on the scene or on top of an already placed object using bounding-box alignment with partial overhang allowed.
4. Ensure each transformed mesh is in the same coordinate system as MultiScan `xyz`.
5. Export combined scene for visualization:
   - MultiScan mesh
   - All transformed `sample.glb` insertions

## Phase 3: Pointcloud Fusion + Labeling
1. Sample points from each transformed `sample.glb` mesh (configurable density).
2. Merge sampled points into MultiScan pointcloud (`xyz`, `rgb`, `normal`).
3. Assign `instance_labels = 0` for original MultiScan points.
4. Assign `instance_labels = k` for each inserted object (k starts at 1, increments per insertion).

## Phase 4: Final Outputs
- Combined mesh export in `.glb` (or `.ply`) for quick visualization.
- Combined pointcloud export as a simple dict with: `xyz`, `rgb`, `normal`, `instance_labels`, and `faces` if mesh reconstruction is needed.
- Compatibility note: `dataset.py` and `visualize.py` in this repo already consume `.npy` dicts with `xyz`, `rgb`, `normal`, `instance_labels`.

## Validation Checks
- Visual inspection in VSCode 3D viewer.
- Sanity checks on label ranges and instance counts.
- Confirm inserted object vertices / sampled points are correctly labeled.
- Confirm per-scene insertion counts stay within `1..5`.
- Confirm inserted object scales stay within the configured scene-diagonal ratio range.
- Confirm stacked placements respect bounding-box support rules while allowing partial overhang.

## Next Implementation Tasks
1. Implement `.pth` -> `.glb` exporter script for OIS samples.
2. Implement multi-object `sample.glb` augmentation + placement logic.
3. Implement stacked placement with bounding-box alignment and partial overhang.
4. Implement point sampling + label assignment pipeline.
5. Run an end-to-end test on a single scene and inspect the exported visualization.
