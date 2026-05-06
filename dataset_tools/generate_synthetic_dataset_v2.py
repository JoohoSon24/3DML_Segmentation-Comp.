#!/usr/bin/env python3
"""Generate synthetic point-cloud scenes by inserting `sample.glb` into MultiScan OIS scenes.

This script reads MultiScan object-instance-segmentation `.pth` files, inserts one to five
randomized copies of the reference object mesh, samples noisy object point clouds with
instance labels, and writes fused `.npy` scenes using the format consumed by this repo.

Examples:
  python dataset_tools/generate_synthetic_dataset.py
  python dataset_tools/generate_synthetic_dataset.py --splits train --variants-per-scene 3
  python dataset_tools/generate_synthetic_dataset.py --debug-glb --seed 123

Important note about the existing loader:
  `dataset.py` recursively finds all `.npy` files under the given root and then creates its
  own train/val/test split. If you want to use the generated split directories as authored,
  point `data_dir` at a specific split directory such as `data/train` and use `split="all"`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import trimesh
    from trimesh.transformations import euler_matrix
except Exception as exc:  # pragma: no cover
    print(
        f"Failed to import trimesh: {exc}\n"
        "Run this script inside an environment that has trimesh installed, such as `softgroup-cu124`.",
        file=sys.stderr,
    )
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    print(
        f"Failed to import scipy.spatial.cKDTree: {exc}\n"
        "This script expects scipy to be installed in the active environment.",
        file=sys.stderr,
    )
    sys.exit(1)


ALLOWED_SPLITS = ("train", "val", "test")
MIN_INSERTIONS = 1
MAX_INSERTIONS = 5
ANISOTROPIC_SCALE_RANGE = (0.5, 1.5)
ROTATION_RANGE_DEG = (-180.0, 180.0)
SCENE_DIAG_RATIO_RANGE = (0.025, 0.2)
SCENE_SPACING_SAMPLE_SIZE = 5000
OBJECT_SAMPLE_COUNT_RANGE = (3000, 26000)
OBJECT_POSITION_JITTER_RATIO = 0.0
NORMAL_JITTER_RANGE = (0.0, 0.003)
COLOR_GAIN_RANGE = (0.85, 1.15)
COLOR_OFFSET_RANGE = (-18.0, 18.0)
COLOR_POINT_STD = 0.5
SCENE_SUPPORT_NORMAL_Z = 0.35
SCENE_SUPPORT_TOP_QUANTILE = 0.55
OBJECT_TOP_BAND_RATIO = 0.15
STACK_ON_OBJECT_PROB = 0.65
MAX_PLACEMENT_ATTEMPTS = 64
SCENE_MARGIN_RATIO = 0.01
SUPPORT_CONTACT_HEIGHT_RATIO = 0.06
SUPPORT_CONTACT_SCENE_RATIO = 0.008
COLLISION_QUERY_MARGIN_RATIO = 0.01
MAX_INTRUSION_POINTS = 12
DEBUG_Y_UP_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


@dataclass
class SceneData:
    xyz: np.ndarray
    rgb: np.ndarray
    normal: np.ndarray
    faces: np.ndarray
    mesh: trimesh.Trimesh
    bounds: np.ndarray
    scene_diag: float
    point_spacing: float
    support_up_idx: np.ndarray
    support_top_idx: np.ndarray
    support_any_top_idx: np.ndarray


@dataclass
class PlacedObject:
    instance_id: int
    mesh: trimesh.Trimesh
    bounds: np.ndarray
    support_type: str
    support_parent: int | None
    anchor: np.ndarray
    jitter_xy: np.ndarray
    rotation_deg: np.ndarray
    anisotropic_scale: np.ndarray
    global_scale: float
    diag_ratio: float
    sample_count: int
    final_object_count: int
    color_gain: np.ndarray
    color_offset: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_source_root() -> Path:
    return _repo_root().parent / "multiscan" / "dataset" / "object_instance_segmentation"


def _default_mesh_path() -> Path:
    return _repo_root() / "assets" / "sample.glb"


def _default_output_dir() -> Path:
    return _repo_root() / "data" / "synth_v3"


def _torch_load(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu")


def _prepare_rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb shape [N, 3], got {rgb.shape}")
    rgb = rgb.astype(np.float32)
    if rgb.size and float(np.max(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _normalize_vectors(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    return np.divide(vec, np.maximum(norms, 1e-8), out=np.zeros_like(vec), where=norms > 1e-8)


def _build_scene_mesh(xyz: np.ndarray, faces: np.ndarray, rgb: np.ndarray, normal: np.ndarray) -> trimesh.Trimesh:
    mesh = trimesh.Trimesh(
        vertices=np.asarray(xyz, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        vertex_normals=np.asarray(normal, dtype=np.float32),
        process=False,
    )
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    mesh.visual.vertex_colors = np.concatenate([rgb.astype(np.uint8), alpha], axis=1)
    return mesh


def _estimate_scene_point_spacing(xyz: np.ndarray) -> float:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape[0] <= 1:
        return 0.01

    rng = np.random.default_rng(0)
    sample_size = min(SCENE_SPACING_SAMPLE_SIZE, xyz.shape[0])
    sample_idx = rng.choice(xyz.shape[0], size=sample_size, replace=False)
    query_pts = xyz[sample_idx]

    tree = cKDTree(xyz)
    distances, _ = tree.query(query_pts, k=2)
    nn = np.asarray(distances[:, 1], dtype=np.float64)
    finite = nn[np.isfinite(nn) & (nn > 1e-8)]
    if finite.size == 0:
        bounds = np.stack([xyz.min(axis=0), xyz.max(axis=0)], axis=0)
        diag = float(np.linalg.norm(bounds[1] - bounds[0]))
        return max(diag / 512.0, 1e-3)
    return float(np.median(finite))


def _load_scene_data(path: Path) -> SceneData:
    data = _torch_load(path)
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = _prepare_rgb_uint8(data["rgb"])
    normal = _normalize_vectors(np.asarray(data["normal"], dtype=np.float32))
    faces = np.asarray(data["faces"], dtype=np.int64)
    bounds = np.stack([xyz.min(axis=0), xyz.max(axis=0)], axis=0).astype(np.float32)
    scene_diag = float(np.linalg.norm(bounds[1] - bounds[0]))
    scene_mesh = _build_scene_mesh(xyz=xyz, faces=faces, rgb=rgb, normal=normal)
    point_spacing = _estimate_scene_point_spacing(xyz)

    z_values = xyz[:, 2]
    top_threshold = float(np.quantile(z_values, SCENE_SUPPORT_TOP_QUANTILE))
    support_up_mask = normal[:, 2] >= SCENE_SUPPORT_NORMAL_Z
    support_top_mask = z_values >= top_threshold

    support_up_idx = np.flatnonzero(support_up_mask)
    support_top_idx = np.flatnonzero(support_up_mask & support_top_mask)
    support_any_top_idx = np.flatnonzero(support_top_mask)

    return SceneData(
        xyz=xyz,
        rgb=rgb,
        normal=normal,
        faces=faces,
        mesh=scene_mesh,
        bounds=bounds,
        scene_diag=scene_diag,
        point_spacing=point_spacing,
        support_up_idx=support_up_idx,
        support_top_idx=support_top_idx,
        support_any_top_idx=support_any_top_idx,
    )


def _load_object_mesh(path: Path) -> tuple[trimesh.Trimesh, trimesh.Trimesh]:
    mesh = trimesh.load(path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected mesh at {path}, got {type(mesh).__name__}")

    mesh = mesh.copy()
    mesh.visual = mesh.visual.to_color()
    mesh = _bottom_center_mesh(mesh)
    hull = mesh.convex_hull.copy()
    return mesh, hull


def _bottom_center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh = mesh.copy()
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    vertices = np.asarray(mesh.vertices, dtype=np.float64).copy()
    vertices[:, 0] -= center_xy[0]
    vertices[:, 1] -= center_xy[1]
    vertices[:, 2] -= bounds[0, 2]
    mesh.vertices = vertices
    return mesh


def _parse_splits(raw: str) -> list[str]:
    text = (raw or "").strip().lower()
    if text in {"", "all"}:
        return list(ALLOWED_SPLITS)
    splits = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in ALLOWED_SPLITS:
            raise ValueError(f"Unsupported split: {part}. Expected one of {ALLOWED_SPLITS} or 'all'.")
        if part not in splits:
            splits.append(part)
    if not splits:
        raise ValueError("No valid splits provided.")
    return splits


def _iter_split_files(source_root: Path, split: str) -> list[Path]:
    split_dir = source_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")
    return sorted(split_dir.glob("*.pth"))


def _prepare_output_root(output_dir: Path, splits: Sequence[str], debug_glb: bool, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if overwrite:
        manifest_path = output_dir / "manifest.jsonl"
        if manifest_path.exists():
            manifest_path.unlink()

        for split in splits:
            split_dir = output_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)
            if debug_glb:
                debug_dir = output_dir / "debug_glb" / split
                if debug_dir.exists():
                    shutil.rmtree(debug_dir)

    for split in splits:
        (output_dir / split).mkdir(parents=True, exist_ok=True)
        if debug_glb:
            (output_dir / "debug_glb" / split).mkdir(parents=True, exist_ok=True)


def _scene_support_anchor(scene: SceneData, rng: np.random.Generator) -> np.ndarray:
    if len(scene.support_top_idx) > 0 and rng.random() < 0.7:
        idx = int(rng.choice(scene.support_top_idx))
    elif len(scene.support_up_idx) > 0 and rng.random() < 0.85:
        idx = int(rng.choice(scene.support_up_idx))
    elif len(scene.support_any_top_idx) > 0:
        idx = int(rng.choice(scene.support_any_top_idx))
    else:
        idx = int(rng.integers(0, scene.xyz.shape[0]))
    return scene.xyz[idx].astype(np.float64)


def _object_support_anchor(parent_mesh: trimesh.Trimesh, rng: np.random.Generator) -> np.ndarray:
    vertices = np.asarray(parent_mesh.vertices, dtype=np.float64)
    z_max = float(vertices[:, 2].max())
    z_min = float(vertices[:, 2].min())
    band = max((z_max - z_min) * OBJECT_TOP_BAND_RATIO, 1e-6)
    top_idx = np.flatnonzero(vertices[:, 2] >= (z_max - band))
    if len(top_idx) == 0:
        idx = int(np.argmax(vertices[:, 2]))
    else:
        idx = int(rng.choice(top_idx))
    return vertices[idx]


def _randomized_object_mesh(
    base_mesh: trimesh.Trimesh,
    base_hull: trimesh.Trimesh,
    scene_diag: float,
    rng: np.random.Generator,
) -> tuple[trimesh.Trimesh, trimesh.Trimesh, np.ndarray, np.ndarray, float, float]:
    mesh = base_mesh.copy()
    hull = base_hull.copy()
    anisotropic_scale = rng.uniform(ANISOTROPIC_SCALE_RANGE[0], ANISOTROPIC_SCALE_RANGE[1], size=3)
    mesh.apply_scale(anisotropic_scale)
    hull.apply_scale(anisotropic_scale)

    rotation_deg = rng.uniform(ROTATION_RANGE_DEG[0], ROTATION_RANGE_DEG[1], size=3)
    rotation = euler_matrix(*np.deg2rad(rotation_deg))
    mesh.apply_transform(rotation)
    hull.apply_transform(rotation)

    current_diag = float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if current_diag <= 1e-8:
        current_diag = 1.0
    diag_ratio = float(rng.uniform(SCENE_DIAG_RATIO_RANGE[0], SCENE_DIAG_RATIO_RANGE[1]))
    target_diag = diag_ratio * float(scene_diag)
    global_scale = float(target_diag / current_diag)
    mesh.apply_scale(global_scale)
    hull.apply_scale(global_scale)

    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    center_xy = 0.5 * (bounds[0, :2] + bounds[1, :2])
    recenter = np.array([-center_xy[0], -center_xy[1], -bounds[0, 2]], dtype=np.float64)
    mesh.apply_translation(recenter)
    hull.apply_translation(recenter)
    return (
        mesh,
        hull,
        anisotropic_scale.astype(np.float64),
        rotation_deg.astype(np.float64),
        global_scale,
        diag_ratio,
    )


def _apply_color_jitter_to_mesh(
    mesh: trimesh.Trimesh,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()
    rgb = colors[:, :3].astype(np.float32)
    gain = rng.uniform(COLOR_GAIN_RANGE[0], COLOR_GAIN_RANGE[1], size=3).astype(np.float32)
    offset = rng.uniform(COLOR_OFFSET_RANGE[0], COLOR_OFFSET_RANGE[1], size=3).astype(np.float32)
    rgb = rgb * gain[None, :] + offset[None, :]
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    colors[:, :3] = rgb
    mesh.visual.vertex_colors = colors
    return gain.astype(np.float64), offset.astype(np.float64)


def _fits_scene_xy(bounds: np.ndarray, scene_bounds: np.ndarray, margin: float) -> bool:
    return bool(
        bounds[0, 0] >= scene_bounds[0, 0] - margin
        and bounds[1, 0] <= scene_bounds[1, 0] + margin
        and bounds[0, 1] >= scene_bounds[0, 1] - margin
        and bounds[1, 1] <= scene_bounds[1, 1] + margin
    )


def _aabb_overlaps(bounds_a: np.ndarray, bounds_b: np.ndarray) -> bool:
    bounds_a = np.asarray(bounds_a, dtype=np.float64)
    bounds_b = np.asarray(bounds_b, dtype=np.float64)
    if bounds_a.shape != (2, 3) or bounds_b.shape != (2, 3):
        raise ValueError(
            f"Expected AABB bounds with shape [2, 3], got {bounds_a.shape} and {bounds_b.shape}"
        )
    separated = np.any(bounds_a[1] < bounds_b[0]) or np.any(bounds_b[1] < bounds_a[0])
    return not separated


def _collides_with_placed_objects(bounds: np.ndarray, placed_objects: Sequence[PlacedObject]) -> bool:
    for placed_obj in placed_objects:
        if _aabb_overlaps(bounds, placed_obj.bounds):
            return True
    return False


def _convex_hull_planes(hull: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    normals = np.asarray(hull.face_normals, dtype=np.float64)
    face_points = np.asarray(hull.vertices[hull.faces[:, 0]], dtype=np.float64)
    offsets = np.einsum("ij,ij->i", normals, face_points)
    return normals, offsets


def _count_scene_intrusions(
    scene: SceneData,
    hull: trimesh.Trimesh,
    bounds: np.ndarray,
    support_type: str,
) -> int:
    object_height = float(max(bounds[1, 2] - bounds[0, 2], 1e-6))
    query_margin = max(COLLISION_QUERY_MARGIN_RATIO * scene.scene_diag, 0.02 * object_height)
    lower = bounds[0] - query_margin
    upper = bounds[1] + query_margin

    nearby_mask = np.all(scene.xyz >= lower[None, :], axis=1) & np.all(scene.xyz <= upper[None, :], axis=1)
    if not np.any(nearby_mask):
        return 0

    nearby = scene.xyz[nearby_mask].astype(np.float64)
    support_clearance = max(
        SUPPORT_CONTACT_HEIGHT_RATIO * object_height,
        SUPPORT_CONTACT_SCENE_RATIO * scene.scene_diag,
    )
    if support_type == "object":
        support_clearance *= 0.5
    nearby = nearby[nearby[:, 2] > float(bounds[0, 2] + support_clearance)]
    if nearby.shape[0] == 0:
        return 0

    normals, offsets = _convex_hull_planes(hull)
    chunk_size = 512
    total_inside = 0
    for start in range(0, nearby.shape[0], chunk_size):
        chunk = nearby[start : start + chunk_size]
        signed = chunk @ normals.T - offsets[None, :]
        inside = np.all(signed <= 1e-5, axis=1)
        total_inside += int(np.count_nonzero(inside))
        if total_inside >= MAX_INTRUSION_POINTS:
            return total_inside
    return total_inside


def _place_object(
    base_mesh: trimesh.Trimesh,
    base_hull: trimesh.Trimesh,
    scene: SceneData,
    placed_objects: Sequence[PlacedObject],
    instance_id: int,
    rng: np.random.Generator,
) -> tuple[PlacedObject, dict[str, np.ndarray | float]]:
    margin = SCENE_MARGIN_RATIO * float(scene.scene_diag)

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        mesh, hull, anisotropic_scale, rotation_deg, global_scale, diag_ratio = _randomized_object_mesh(
            base_mesh=base_mesh,
            base_hull=base_hull,
            scene_diag=scene.scene_diag,
            rng=rng,
        )
        color_gain, color_offset = _apply_color_jitter_to_mesh(mesh, rng=rng)

        support_type = "scene"
        support_parent = None
        if placed_objects and rng.random() < STACK_ON_OBJECT_PROB:
            support_parent_obj = placed_objects[int(rng.integers(0, len(placed_objects)))]
            anchor = _object_support_anchor(support_parent_obj.mesh, rng=rng)
            support_type = "object"
            support_parent = support_parent_obj.instance_id
        else:
            anchor = _scene_support_anchor(scene, rng=rng)

        half_extent_xy = 0.5 * np.maximum(mesh.bounds[1, :2] - mesh.bounds[0, :2], 1e-6)
        jitter_limit = 0.98 * half_extent_xy
        jitter_xy = rng.uniform(-jitter_limit, jitter_limit)

        translation = np.array([anchor[0], anchor[1], anchor[2]], dtype=np.float64)
        translation[:2] += jitter_xy
        mesh.apply_translation(translation)
        hull.apply_translation(translation)
        bounds = np.asarray(mesh.bounds, dtype=np.float64)

        if not _fits_scene_xy(bounds=bounds, scene_bounds=scene.bounds, margin=margin):
            continue
        if bounds[0, 2] < float(scene.bounds[0, 2] - margin):
            continue
        if _count_scene_intrusions(scene=scene, hull=hull, bounds=bounds, support_type=support_type) >= MAX_INTRUSION_POINTS:
            continue
        if _collides_with_placed_objects(bounds=bounds, placed_objects=placed_objects):
            continue

        sampled = _sample_object_points(
            mesh=mesh,
            rng=rng,
        )

        return PlacedObject(
            instance_id=instance_id,
            mesh=mesh,
            bounds=bounds.copy(),
            support_type=support_type,
            support_parent=support_parent,
            anchor=anchor.astype(np.float64),
            jitter_xy=jitter_xy.astype(np.float64),
            rotation_deg=rotation_deg,
            anisotropic_scale=anisotropic_scale,
            global_scale=global_scale,
            diag_ratio=diag_ratio,
            sample_count=int(sampled["sample_count"]),
            final_object_count=int(sampled["xyz"].shape[0]),
            color_gain=color_gain,
            color_offset=color_offset,
        ), sampled

    raise RuntimeError(f"Failed to place object {instance_id} after {MAX_PLACEMENT_ATTEMPTS} attempts.")


def _sample_object_points(
    mesh: trimesh.Trimesh,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    count = int(rng.integers(OBJECT_SAMPLE_COUNT_RANGE[0], OBJECT_SAMPLE_COUNT_RANGE[1] + 1))

    # Sample random points on triangle surfaces (barycentric sampling)
    seed_val = int(rng.integers(0, 2**31))
    points, face_indices = trimesh.sample.sample_surface(mesh, count, seed=seed_val)
    points = np.asarray(points, dtype=np.float32)
    face_indices = np.asarray(face_indices)

    # Smooth normals via barycentric interpolation
    tris = np.asarray(mesh.triangles[face_indices], dtype=np.float32)
    bary = trimesh.triangles.points_to_barycentric(tris, points)
    bary = np.clip(bary, 0.0, None)
    bary = bary / np.maximum(bary.sum(axis=1, keepdims=True), 1e-8)
    fi = mesh.faces[face_indices]
    vn = np.asarray(mesh.vertex_normals, dtype=np.float32)
    normals = _normalize_vectors(
        bary[:, 0:1] * vn[fi[:, 0]] + bary[:, 1:2] * vn[fi[:, 1]] + bary[:, 2:3] * vn[fi[:, 2]]
    )

    # Color via barycentric interpolation
    vc = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32)
    colors = (bary[:, 0:1] * vc[fi[:, 0]] + bary[:, 1:2] * vc[fi[:, 1]] + bary[:, 2:3] * vc[fi[:, 2]])

    normal_jitter = float(rng.uniform(NORMAL_JITTER_RANGE[0], NORMAL_JITTER_RANGE[1]))
    if normal_jitter > 0.0:
        normals = normals + rng.normal(0.0, normal_jitter, size=normals.shape).astype(np.float32)
        normals = _normalize_vectors(normals)

    if COLOR_POINT_STD > 0.0:
        colors = colors + rng.normal(0.0, COLOR_POINT_STD, size=colors.shape).astype(np.float32)
    colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)

    return {
        "xyz": points,
        "normal": normals,
        "rgb": colors,
        "sample_count": count,
    }


def _voxel_downsample_indices(
    points: np.ndarray,
    voxel_size: float,
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)

    min_corner = points.min(axis=0)
    cell = np.floor((points - min_corner[None, :]) / voxel_size).astype(np.int64)
    _, unique_pos = np.unique(cell, axis=0, return_index=True)
    keep = np.sort(unique_pos)
    return keep.astype(np.int64)


def _synthesize_scene(
    scene_path: Path,
    base_mesh: trimesh.Trimesh,
    base_hull: trimesh.Trimesh,
    scene_seed: int,
    debug_glb_path: Path | None,
    debug_up_axis: str,
) -> tuple[dict[str, np.ndarray], dict]:
    rng = np.random.default_rng(scene_seed)
    scene = _load_scene_data(scene_path)

    target_insertions = int(rng.integers(MIN_INSERTIONS, MAX_INSERTIONS + 1))
    placed_objects: list[PlacedObject] = []
    sampled_objects: list[dict[str, np.ndarray | float]] = []

    instance_id = 1
    total_attempt_rounds = 0
    max_total_attempt_rounds = MAX_INSERTIONS * MAX_PLACEMENT_ATTEMPTS * 2

    while len(placed_objects) < target_insertions and total_attempt_rounds < max_total_attempt_rounds:
        total_attempt_rounds += 1
        try:
            placed_obj, sampled = _place_object(
                base_mesh=base_mesh,
                base_hull=base_hull,
                scene=scene,
                placed_objects=placed_objects,
                instance_id=instance_id,
                rng=rng,
            )
        except RuntimeError:
            if len(placed_objects) == 0:
                continue
            break

        placed_objects.append(placed_obj)
        sampled_objects.append(sampled)
        instance_id += 1

    if len(placed_objects) == 0:
        raise RuntimeError(f"Unable to synthesize any object placements for scene: {scene_path}")

    fused_xyz = [scene.xyz.astype(np.float32)]
    fused_rgb = [scene.rgb.astype(np.uint8)]
    fused_normal = [scene.normal.astype(np.float32)]
    fused_instance = [np.zeros((scene.xyz.shape[0],), dtype=np.int32)]

    object_manifest = []
    for placed_obj, sampled in zip(placed_objects, sampled_objects):
        obj_xyz = np.asarray(sampled["xyz"], dtype=np.float32)
        obj_rgb = np.asarray(sampled["rgb"], dtype=np.uint8)
        obj_normal = np.asarray(sampled["normal"], dtype=np.float32)
        obj_labels = np.full((obj_xyz.shape[0],), placed_obj.instance_id, dtype=np.int32)

        fused_xyz.append(obj_xyz)
        fused_rgb.append(obj_rgb)
        fused_normal.append(obj_normal)
        fused_instance.append(obj_labels)

        bounds = np.asarray(placed_obj.bounds, dtype=np.float64)
        object_manifest.append(
            {
                "instance_id": placed_obj.instance_id,
                "support_type": placed_obj.support_type,
                "support_parent": placed_obj.support_parent,
                "anchor": placed_obj.anchor.tolist(),
                "jitter_xy": placed_obj.jitter_xy.tolist(),
                "anisotropic_scale": placed_obj.anisotropic_scale.tolist(),
                "rotation_deg": placed_obj.rotation_deg.tolist(),
                "global_scale": placed_obj.global_scale,
                "diag_ratio": placed_obj.diag_ratio,
                "mesh_bounds_min": bounds[0].tolist(),
                "mesh_bounds_max": bounds[1].tolist(),
                "mesh_diagonal": float(np.linalg.norm(bounds[1] - bounds[0])),
                "sample_count": placed_obj.sample_count,
                "final_point_count": placed_obj.final_object_count,
                "color_gain": placed_obj.color_gain.tolist(),
                "color_offset": placed_obj.color_offset.tolist(),
            }
        )

    output = {
        "xyz": np.concatenate(fused_xyz, axis=0).astype(np.float32),
        "rgb": np.concatenate(fused_rgb, axis=0).astype(np.uint8),
        "normal": np.concatenate(fused_normal, axis=0).astype(np.float32),
        "instance_labels": np.concatenate(fused_instance, axis=0).astype(np.int32),
    }

    if debug_glb_path is not None:
        debug_scene = trimesh.Scene()
        debug_scene.add_geometry(_mesh_for_debug_export(scene.mesh, up_axis=debug_up_axis), geom_name="scene")
        for placed_obj in placed_objects:
            debug_scene.add_geometry(
                _mesh_for_debug_export(placed_obj.mesh, up_axis=debug_up_axis),
                geom_name=f"obj_{placed_obj.instance_id:02d}",
            )
        debug_glb_path.parent.mkdir(parents=True, exist_ok=True)
        debug_scene.export(debug_glb_path)

    manifest_entry = {
        "source_scene": str(scene_path.resolve()),
        "effective_seed": int(scene_seed),
        "scene_diag": float(scene.scene_diag),
        "scene_point_spacing": float(scene.point_spacing),
        "scene_bounds_min": scene.bounds[0].tolist(),
        "scene_bounds_max": scene.bounds[1].tolist(),
        "requested_object_count": int(target_insertions),
        "placed_object_count": int(len(placed_objects)),
        "objects": object_manifest,
    }
    return output, manifest_entry


def _variant_output_name(source_path: Path, variant_idx: int, variants_per_scene: int) -> str:
    stem = source_path.stem
    if variants_per_scene <= 1:
        return f"{stem}.npy"
    return f"{stem}_v{variant_idx:02d}.npy"


def _write_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _mesh_for_debug_export(mesh: trimesh.Trimesh, up_axis: str) -> trimesh.Trimesh:
    axis = (up_axis or "y").strip().lower()
    mesh = mesh.copy()
    if axis == "z":
        return mesh
    if axis == "y":
        mesh.apply_transform(DEBUG_Y_UP_TRANSFORM)
        return mesh
    raise ValueError(f"Unsupported debug up axis: {up_axis}. Use 'y' or 'z'.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate permissive synthetic point-cloud datasets.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=_default_source_root(),
        help="Root directory containing MultiScan object_instance_segmentation split folders.",
    )
    parser.add_argument(
        "--mesh-path",
        type=Path,
        default=_default_mesh_path(),
        help="Path to the object mesh to insert.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Output directory that will receive split subdirectories and manifest.jsonl.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,val",
        help="Comma-separated split list from {train,val,test}. Use 'all' for every split.",
    )
    parser.add_argument(
        "--variants-per-scene",
        type=int,
        default=1,
        help="How many synthetic variants to create for each source scene (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional base seed. If omitted, a fresh random seed is generated.",
    )
    parser.add_argument(
        "--debug-glb",
        action="store_true",
        help="Also export combined debug GLB files under output_dir/debug_glb.",
    )
    parser.add_argument(
        "--debug-up-axis",
        default="y",
        choices=["y", "z"],
        help="Up axis for exported debug GLBs (default: y). Use z to preserve source coordinates.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete targeted split outputs and manifest before regeneration.",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    if args.variants_per_scene < 1:
        parser.error("--variants-per-scene must be >= 1")

    splits = _parse_splits(args.splits)
    source_root = args.source_root.resolve()
    mesh_path = args.mesh_path.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.jsonl"

    if not source_root.exists():
        parser.error(f"Source root does not exist: {source_root}")
    if not mesh_path.is_file():
        parser.error(f"Mesh path does not exist: {mesh_path}")

    session_seed = (
        int(args.seed)
        if args.seed is not None
        else int(np.random.default_rng().integers(0, np.iinfo(np.int64).max))
    )
    seed_rng = np.random.default_rng(session_seed)

    _prepare_output_root(output_dir=output_dir, splits=splits, debug_glb=args.debug_glb, overwrite=args.overwrite)
    base_mesh, base_hull = _load_object_mesh(mesh_path)

    print(f"Using source root: {source_root}")
    print(f"Using mesh: {mesh_path}")
    print(f"Writing outputs to: {output_dir}")
    print(f"Using base seed: {session_seed}")

    total_written = 0
    for split in splits:
        files = _iter_split_files(source_root=source_root, split=split)
        if not files:
            print(f"No .pth files found for split: {split}", file=sys.stderr)
            continue

        print(f"[{split}] Found {len(files)} source scenes")
        for scene_path in files:
            for variant_idx in range(args.variants_per_scene):
                out_name = _variant_output_name(
                    source_path=scene_path,
                    variant_idx=variant_idx,
                    variants_per_scene=args.variants_per_scene,
                )
                out_path = output_dir / split / out_name
                if out_path.exists() and not args.overwrite:
                    print(f"Skipping existing output: {out_path}")
                    continue

                debug_glb_path = None
                if args.debug_glb:
                    debug_glb_path = output_dir / "debug_glb" / split / out_name.replace(".npy", ".glb")

                scene_seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))
                synthesized, manifest_entry = _synthesize_scene(
                    scene_path=scene_path,
                    base_mesh=base_mesh,
                    base_hull=base_hull,
                    scene_seed=scene_seed,
                    debug_glb_path=debug_glb_path,
                    debug_up_axis=args.debug_up_axis,
                )

                out_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(out_path, synthesized)

                manifest_entry.update(
                    {
                        "split": split,
                        "variant_index": int(variant_idx),
                        "output_path": str(out_path.resolve()),
                        "base_seed": int(session_seed),
                    }
                )
                _write_manifest(manifest_path=manifest_path, entry=manifest_entry)
                total_written += 1

        print(f"[{split}] Completed")

    print(f"Wrote {total_written} synthesized scene files")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
