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
POINT_COUNT_RANGE = (2048, 32768)
SCENE_SPACING_SAMPLE_SIZE = 5000
MIN_OBJECT_POINTS = 512
OBJECT_SAMPLE_COUNT_RANGE = (3000, 26000)
SPACING_RELAXATION_FACTOR = 0.8
MAX_SPACING_RELAXATION_STEPS = 4
OBJECT_POSITION_JITTER_RATIO = 0.0
NORMAL_JITTER_RANGE = (0.0, 0.003)
HSV_SAT_SCALE_RANGE = (0.8, 1.2)
HSV_VAL_SCALE_RANGE = (0.85, 1.15)
HSV_MIN_SATURATION = 0.18
COLOR_POINT_STD = 0.5
SCENE_SUPPORT_NORMAL_Z = 0.35
SCENE_SUPPORT_TOP_QUANTILE = 0.55
LAYOUT_MODE_NAMES = ("scene_only", "mixed", "stack_heavy")
LAYOUT_MODE_PROBS = np.array((0.35, 0.45, 0.20), dtype=np.float64)
LAYOUT_MODE_BASE_STACK_PROBS = {
    "scene_only": 0.0,
    "mixed": 0.55,
    "stack_heavy": 0.85,
}
LAYOUT_MODE_EXTRA_STACK_PROBS = {
    "scene_only": 0.0,
    "mixed": 0.25,
    "stack_heavy": 0.55,
}
STACK_STYLE_NAMES = ("centered", "edge_overhang", "corner_overhang")
STACK_STYLE_PROBS = np.array((0.40, 0.40, 0.20), dtype=np.float64)
STACK_CHILD_MIN_OVERLAP_RATIO = 0.10
STACK_CENTER_JITTER_RATIO = 0.30
STACK_EDGE_OVERLAP_RATIO_RANGE = (0.20, 0.60)
STACK_CORNER_OVERLAP_RATIO_RANGE = (0.35, 0.70)
STACK_STYLE_SAMPLE_ATTEMPTS = 20
MAX_PLACEMENT_ATTEMPTS = 64
SCENE_SYNTHESIS_MAX_RETRIES = 5
SCENE_MARGIN_RATIO = 0.01
SUPPORT_CONTACT_HEIGHT_RATIO = 0.06
SUPPORT_CONTACT_SCENE_RATIO = 0.008
OBJECT_SUPPORT_CLEARANCE_RATIO = 0.10
COLLISION_QUERY_MARGIN_RATIO = 0.01
MAX_INTRUSION_POINTS = 12
AABB_PENETRATION_EPS_RATIO = 1e-5
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
    placement_style: str
    support_xy_overlap_ratio: float
    anchor: np.ndarray
    jitter_xy: np.ndarray
    rotation_deg: np.ndarray
    anisotropic_scale: np.ndarray
    global_scale: float
    diag_ratio: float
    sample_count: int
    final_object_count: int
    color_hue_shift_deg: float
    color_sat_scale: float
    color_val_scale: float


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_source_root() -> Path:
    return _repo_root().parent / "multiscan" / "dataset" / "object_instance_segmentation"


def _default_mesh_path() -> Path:
    return _repo_root() / "assets" / "sample.glb"


def _default_output_dir() -> Path:
    return _repo_root() / "data"


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


def _sample_weighted_choice(options: Sequence[str], probabilities: np.ndarray, rng: np.random.Generator) -> str:
    index = int(rng.choice(len(options), p=probabilities))
    return str(options[index])


def _sample_layout_mode(target_insertions: int, rng: np.random.Generator) -> str:
    if target_insertions < 2:
        return "scene_only"
    return _sample_weighted_choice(LAYOUT_MODE_NAMES, LAYOUT_MODE_PROBS, rng=rng)


def _desired_stack_count(layout_mode: str, target_insertions: int) -> int:
    if layout_mode == "scene_only" or target_insertions < 2:
        return 0
    if layout_mode == "mixed":
        return 1
    return 2 if target_insertions >= 3 else 1


def _support_mode_order(
    layout_mode: str,
    placed_count: int,
    target_insertions: int,
    stacked_count: int,
    rng: np.random.Generator,
) -> tuple[str, ...]:
    if placed_count == 0:
        return ("scene",)
    if layout_mode == "scene_only":
        return ("scene",)

    desired_stack_count = _desired_stack_count(layout_mode=layout_mode, target_insertions=target_insertions)
    remaining_objects = target_insertions - placed_count
    remaining_stack_needed = max(desired_stack_count - stacked_count, 0)

    if remaining_stack_needed > 0 and remaining_objects <= remaining_stack_needed:
        preferred = "object"
    else:
        prefer_prob = (
            LAYOUT_MODE_BASE_STACK_PROBS[layout_mode]
            if remaining_stack_needed > 0
            else LAYOUT_MODE_EXTRA_STACK_PROBS[layout_mode]
        )
        preferred = "object" if rng.random() < prefer_prob else "scene"

    alternate = "scene" if preferred == "object" else "object"
    return (preferred, alternate)


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


def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    rgb = np.clip(rgb, 0.0, 255.0) / 255.0
    maxc = rgb.max(axis=1)
    minc = rgb.min(axis=1)
    delta = maxc - minc

    hsv = np.zeros_like(rgb, dtype=np.float32)
    hsv[:, 2] = maxc

    nonzero = maxc > 1e-8
    hsv[nonzero, 1] = delta[nonzero] / maxc[nonzero]

    colorful = delta > 1e-8
    if np.any(colorful):
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]

        mask_r = colorful & (maxc == r)
        mask_g = colorful & (maxc == g)
        mask_b = colorful & (maxc == b)

        hsv[mask_r, 0] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6.0
        hsv[mask_g, 0] = ((b[mask_g] - r[mask_g]) / delta[mask_g]) + 2.0
        hsv[mask_b, 0] = ((r[mask_b] - g[mask_b]) / delta[mask_b]) + 4.0
        hsv[colorful, 0] = (hsv[colorful, 0] / 6.0) % 1.0
    return hsv


def _hsv_to_rgb_uint8(hsv: np.ndarray) -> np.ndarray:
    hsv = np.asarray(hsv, dtype=np.float32)
    h = np.mod(hsv[:, 0], 1.0)
    s = np.clip(hsv[:, 1], 0.0, 1.0)
    v = np.clip(hsv[:, 2], 0.0, 1.0)

    scaled = h * 6.0
    sector = np.floor(scaled).astype(np.int32)
    frac = scaled - sector

    p = v * (1.0 - s)
    q = v * (1.0 - frac * s)
    t = v * (1.0 - (1.0 - frac) * s)

    rgb = np.empty((h.shape[0], 3), dtype=np.float32)
    idx = sector % 6

    mask = idx == 0
    rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=1)
    mask = idx == 1
    rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=1)
    mask = idx == 2
    rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=1)
    mask = idx == 3
    rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=1)
    mask = idx == 4
    rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=1)
    mask = idx == 5
    rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=1)

    return np.clip(np.rint(rgb * 255.0), 0.0, 255.0).astype(np.uint8)


def _apply_color_jitter_to_mesh(
    mesh: trimesh.Trimesh,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8).copy()
    rgb = colors[:, :3]
    if rgb.shape[0] == 0:
        return 0.0, 1.0, 1.0

    hsv = _rgb_to_hsv(rgb)
    hue_shift_deg = float(rng.uniform(0.0, 360.0))
    sat_scale = float(rng.uniform(HSV_SAT_SCALE_RANGE[0], HSV_SAT_SCALE_RANGE[1]))
    val_scale = float(rng.uniform(HSV_VAL_SCALE_RANGE[0], HSV_VAL_SCALE_RANGE[1]))

    hsv[:, 0] = np.mod(hsv[:, 0] + np.float32(hue_shift_deg / 360.0), 1.0)
    hsv[:, 1] = np.clip(np.maximum(hsv[:, 1], HSV_MIN_SATURATION) * sat_scale, 0.0, 1.0)
    hsv[:, 2] = np.clip(hsv[:, 2] * val_scale, 0.0, 1.0)

    colors[:, :3] = _hsv_to_rgb_uint8(hsv)
    mesh.visual.vertex_colors = colors
    return hue_shift_deg, sat_scale, val_scale


def _fits_scene_xy(bounds: np.ndarray, scene_bounds: np.ndarray, margin: float) -> bool:
    return bool(
        bounds[0, 0] >= scene_bounds[0, 0] - margin
        and bounds[1, 0] <= scene_bounds[1, 0] + margin
        and bounds[0, 1] >= scene_bounds[0, 1] - margin
        and bounds[1, 1] <= scene_bounds[1, 1] + margin
    )


def _aabb_overlap_lengths(bounds_a: np.ndarray, bounds_b: np.ndarray) -> np.ndarray:
    bounds_a = np.asarray(bounds_a, dtype=np.float64)
    bounds_b = np.asarray(bounds_b, dtype=np.float64)
    if bounds_a.shape != (2, 3) or bounds_b.shape != (2, 3):
        raise ValueError(
            f"Expected AABB bounds with shape [2, 3], got {bounds_a.shape} and {bounds_b.shape}"
        )
    lower = np.maximum(bounds_a[0], bounds_b[0])
    upper = np.minimum(bounds_a[1], bounds_b[1])
    return np.maximum(0.0, upper - lower)


def _penetration_epsilon(scene_diag: float) -> float:
    return max(AABB_PENETRATION_EPS_RATIO * float(scene_diag), 1e-6)


def _aabb_penetrates(bounds_a: np.ndarray, bounds_b: np.ndarray, eps: float) -> bool:
    overlaps = _aabb_overlap_lengths(bounds_a, bounds_b)
    return bool(np.all(overlaps > eps))


def _xy_footprint_area(bounds: np.ndarray) -> float:
    bounds = np.asarray(bounds, dtype=np.float64)
    extent = np.maximum(bounds[1, :2] - bounds[0, :2], 1e-8)
    return float(extent[0] * extent[1])


def _xy_overlap_area(bounds_a: np.ndarray, bounds_b: np.ndarray) -> float:
    overlaps = _aabb_overlap_lengths(bounds_a, bounds_b)
    return float(overlaps[0] * overlaps[1])


def _support_xy_overlap_ratio(child_bounds: np.ndarray, parent_bounds: np.ndarray) -> float:
    child_area = _xy_footprint_area(child_bounds)
    if child_area <= 1e-8:
        return 0.0
    return float(_xy_overlap_area(child_bounds, parent_bounds) / child_area)


def _max_support_xy_overlap_ratio(parent_bounds: np.ndarray, child_bounds: np.ndarray) -> float:
    parent_bounds = np.asarray(parent_bounds, dtype=np.float64)
    child_bounds = np.asarray(child_bounds, dtype=np.float64)
    parent_extent = np.maximum(parent_bounds[1, :2] - parent_bounds[0, :2], 1e-8)
    child_extent = np.maximum(child_bounds[1, :2] - child_bounds[0, :2], 1e-8)
    return float(min(1.0, parent_extent[0] / child_extent[0]) * min(1.0, parent_extent[1] / child_extent[1]))


def _sample_support_parent(
    child_bounds: np.ndarray,
    placed_objects: Sequence[PlacedObject],
    rng: np.random.Generator,
) -> PlacedObject | None:
    candidates: list[PlacedObject] = []
    weights = []
    for placed_obj in placed_objects:
        support_capacity = _max_support_xy_overlap_ratio(parent_bounds=placed_obj.bounds, child_bounds=child_bounds)
        if support_capacity >= STACK_CHILD_MIN_OVERLAP_RATIO:
            candidates.append(placed_obj)
            weights.append(support_capacity)

    if not candidates:
        return None

    weight_array = np.asarray(weights, dtype=np.float64)
    weight_array = weight_array / np.maximum(weight_array.sum(), 1e-8)
    index = int(rng.choice(len(candidates), p=weight_array))
    return candidates[index]


def _sample_overlap_fraction(max_fraction: float, fraction_range: tuple[float, float], rng: np.random.Generator) -> float:
    max_fraction = float(max(max_fraction, 0.0))
    if max_fraction <= 1e-6:
        return 0.0
    upper = min(float(fraction_range[1]), max_fraction)
    lower = min(float(fraction_range[0]), upper)
    if upper <= lower:
        return upper
    return float(rng.uniform(lower, upper))


def _edge_center_from_overlap(
    parent_min: float,
    parent_max: float,
    child_extent: float,
    overlap_extent: float,
    sign: float,
) -> float:
    if sign >= 0.0:
        child_min = parent_max - overlap_extent
        return float(child_min + 0.5 * child_extent)
    child_max = parent_min + overlap_extent
    return float(child_max - 0.5 * child_extent)


def _sample_object_support_pose(
    child_bounds: np.ndarray,
    parent_bounds: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, float]:
    child_bounds = np.asarray(child_bounds, dtype=np.float64)
    parent_bounds = np.asarray(parent_bounds, dtype=np.float64)

    parent_min_xy = parent_bounds[0, :2]
    parent_max_xy = parent_bounds[1, :2]
    parent_center_xy = 0.5 * (parent_min_xy + parent_max_xy)
    parent_extent_xy = np.maximum(parent_max_xy - parent_min_xy, 1e-8)
    child_extent_xy = np.maximum(child_bounds[1, :2] - child_bounds[0, :2], 1e-8)
    parent_top_z = float(parent_bounds[1, 2])
    anchor = np.array([parent_center_xy[0], parent_center_xy[1], parent_top_z], dtype=np.float64)

    for _ in range(STACK_STYLE_SAMPLE_ATTEMPTS):
        placement_style = _sample_weighted_choice(STACK_STYLE_NAMES, STACK_STYLE_PROBS, rng=rng)
        center_xy = parent_center_xy.copy()

        if placement_style == "centered":
            jitter_scale = np.minimum(parent_extent_xy, child_extent_xy) * STACK_CENTER_JITTER_RATIO
            center_xy = center_xy + rng.uniform(-jitter_scale, jitter_scale)
        elif placement_style == "edge_overhang":
            axis = int(rng.integers(0, 2))
            sign = 1.0 if rng.random() < 0.5 else -1.0
            primary_fraction = _sample_overlap_fraction(
                max_fraction=min(1.0, parent_extent_xy[axis] / child_extent_xy[axis]),
                fraction_range=STACK_EDGE_OVERLAP_RATIO_RANGE,
                rng=rng,
            )
            center_xy[axis] = _edge_center_from_overlap(
                parent_min=parent_min_xy[axis],
                parent_max=parent_max_xy[axis],
                child_extent=child_extent_xy[axis],
                overlap_extent=primary_fraction * child_extent_xy[axis],
                sign=sign,
            )
            other_axis = 1 - axis
            other_jitter = np.minimum(parent_extent_xy[other_axis], child_extent_xy[other_axis]) * (
                STACK_CENTER_JITTER_RATIO + 0.10
            )
            center_xy[other_axis] = center_xy[other_axis] + float(rng.uniform(-other_jitter, other_jitter))
        else:
            signs = np.where(rng.random(2) < 0.5, -1.0, 1.0)
            for axis in range(2):
                overlap_fraction = _sample_overlap_fraction(
                    max_fraction=min(1.0, parent_extent_xy[axis] / child_extent_xy[axis]),
                    fraction_range=STACK_CORNER_OVERLAP_RATIO_RANGE,
                    rng=rng,
                )
                center_xy[axis] = _edge_center_from_overlap(
                    parent_min=parent_min_xy[axis],
                    parent_max=parent_max_xy[axis],
                    child_extent=child_extent_xy[axis],
                    overlap_extent=overlap_fraction * child_extent_xy[axis],
                    sign=float(signs[axis]),
                )

        translation = np.array([center_xy[0], center_xy[1], parent_top_z], dtype=np.float64)
        candidate_bounds = child_bounds + translation[None, :]
        overlap_ratio = _support_xy_overlap_ratio(child_bounds=candidate_bounds, parent_bounds=parent_bounds)
        if overlap_ratio >= STACK_CHILD_MIN_OVERLAP_RATIO:
            jitter_xy = center_xy - parent_center_xy
            return anchor, jitter_xy.astype(np.float64), translation, placement_style, float(overlap_ratio)

    raise RuntimeError("Failed to sample a valid AABB-top object support pose.")


def _validate_object_support(child_bounds: np.ndarray, parent_bounds: np.ndarray, scene_diag: float) -> float | None:
    eps = _penetration_epsilon(scene_diag)
    if abs(float(child_bounds[0, 2] - parent_bounds[1, 2])) > eps:
        return None
    if _aabb_penetrates(child_bounds, parent_bounds, eps=eps):
        return None
    overlap_ratio = _support_xy_overlap_ratio(child_bounds=child_bounds, parent_bounds=parent_bounds)
    if overlap_ratio < STACK_CHILD_MIN_OVERLAP_RATIO:
        return None
    return float(overlap_ratio)


def _collides_with_placed_objects(
    bounds: np.ndarray,
    placed_objects: Sequence[PlacedObject],
    scene_diag: float,
    support_parent: int | None,
) -> bool:
    eps = _penetration_epsilon(scene_diag)
    for placed_obj in placed_objects:
        if support_parent is not None and placed_obj.instance_id == support_parent:
            if _aabb_penetrates(bounds, placed_obj.bounds, eps=eps):
                return True
            continue
        if _aabb_penetrates(bounds, placed_obj.bounds, eps=eps):
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
        support_clearance = max(support_clearance, OBJECT_SUPPORT_CLEARANCE_RATIO * object_height)
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
    support_modes: Sequence[str],
    rng: np.random.Generator,
) -> tuple[PlacedObject, dict[str, np.ndarray | float]]:
    margin = SCENE_MARGIN_RATIO * float(scene.scene_diag)
    support_modes = tuple(dict.fromkeys(support_modes))
    attempts_per_mode = max(1, MAX_PLACEMENT_ATTEMPTS // max(len(support_modes), 1))
    extra_attempts = MAX_PLACEMENT_ATTEMPTS % max(len(support_modes), 1)

    for mode_index, requested_support in enumerate(support_modes):
        mode_attempts = attempts_per_mode + (1 if mode_index < extra_attempts else 0)
        for _ in range(mode_attempts):
            mesh, hull, anisotropic_scale, rotation_deg, global_scale, diag_ratio = _randomized_object_mesh(
                base_mesh=base_mesh,
                base_hull=base_hull,
                scene_diag=scene.scene_diag,
                rng=rng,
            )
            child_bounds_local = np.asarray(mesh.bounds, dtype=np.float64).copy()
            color_hue_shift_deg, color_sat_scale, color_val_scale = _apply_color_jitter_to_mesh(mesh, rng=rng)

            support_type = requested_support
            support_parent = None
            support_xy_overlap_ratio = 0.0

            if support_type == "object":
                support_parent_obj = _sample_support_parent(
                    child_bounds=child_bounds_local,
                    placed_objects=placed_objects,
                    rng=rng,
                )
                if support_parent_obj is None:
                    continue
                try:
                    anchor, jitter_xy, translation, placement_style, support_xy_overlap_ratio = _sample_object_support_pose(
                        child_bounds=child_bounds_local,
                        parent_bounds=support_parent_obj.bounds,
                        rng=rng,
                    )
                except RuntimeError:
                    continue
                support_parent = support_parent_obj.instance_id
            else:
                support_type = "scene"
                placement_style = "scene_support"
                anchor = _scene_support_anchor(scene, rng=rng)
                half_extent_xy = 0.5 * np.maximum(child_bounds_local[1, :2] - child_bounds_local[0, :2], 1e-6)
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
            if support_type == "object":
                support_parent_obj = next(obj for obj in placed_objects if obj.instance_id == support_parent)
                validated_overlap_ratio = _validate_object_support(
                    child_bounds=bounds,
                    parent_bounds=support_parent_obj.bounds,
                    scene_diag=scene.scene_diag,
                )
                if validated_overlap_ratio is None:
                    continue
                support_xy_overlap_ratio = float(validated_overlap_ratio)
            if _count_scene_intrusions(scene=scene, hull=hull, bounds=bounds, support_type=support_type) >= MAX_INTRUSION_POINTS:
                continue
            if _collides_with_placed_objects(
                bounds=bounds,
                placed_objects=placed_objects,
                scene_diag=scene.scene_diag,
                support_parent=support_parent,
            ):
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
                placement_style=placement_style,
                support_xy_overlap_ratio=float(support_xy_overlap_ratio),
                anchor=anchor.astype(np.float64),
                jitter_xy=jitter_xy.astype(np.float64),
                rotation_deg=rotation_deg,
                anisotropic_scale=anisotropic_scale,
                global_scale=global_scale,
                diag_ratio=diag_ratio,
                sample_count=int(sampled["sample_count"]),
                final_object_count=int(sampled["xyz"].shape[0]),
                color_hue_shift_deg=float(color_hue_shift_deg),
                color_sat_scale=float(color_sat_scale),
                color_val_scale=float(color_val_scale),
            ), sampled

    raise RuntimeError(
        f"Failed to place object {instance_id} after {MAX_PLACEMENT_ATTEMPTS} attempts with support_modes={support_modes}."
    )


def _sample_object_points(
    mesh: trimesh.Trimesh,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float]:
    count = int(rng.integers(OBJECT_SAMPLE_COUNT_RANGE[0], OBJECT_SAMPLE_COUNT_RANGE[1] + 1))

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


def _synthesize_scene_layout(
    scene: SceneData,
    base_mesh: trimesh.Trimesh,
    base_hull: trimesh.Trimesh,
    target_insertions: int,
    layout_mode: str,
    rng: np.random.Generator,
) -> tuple[list[PlacedObject], list[dict[str, np.ndarray | float]], int]:
    placed_objects: list[PlacedObject] = []
    sampled_objects: list[dict[str, np.ndarray | float]] = []
    stacked_count = 0

    for instance_id in range(1, target_insertions + 1):
        support_modes = _support_mode_order(
            layout_mode=layout_mode,
            placed_count=len(placed_objects),
            target_insertions=target_insertions,
            stacked_count=stacked_count,
            rng=rng,
        )
        placed_obj, sampled = _place_object(
            base_mesh=base_mesh,
            base_hull=base_hull,
            scene=scene,
            placed_objects=placed_objects,
            instance_id=instance_id,
            support_modes=support_modes,
            rng=rng,
        )
        placed_objects.append(placed_obj)
        sampled_objects.append(sampled)
        if placed_obj.support_type == "object":
            stacked_count += 1

    desired_stack_count = _desired_stack_count(layout_mode=layout_mode, target_insertions=target_insertions)
    if stacked_count < desired_stack_count:
        raise RuntimeError(
            f"Layout mode {layout_mode} only produced {stacked_count} stacked placements out of {desired_stack_count}."
        )
    return placed_objects, sampled_objects, stacked_count


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
    layout_mode = _sample_layout_mode(target_insertions=target_insertions, rng=rng)

    placed_objects: list[PlacedObject] = []
    sampled_objects: list[dict[str, np.ndarray | float]] = []
    stacked_count = 0
    effective_seed = int(scene_seed)
    retry_index = 0
    last_error: RuntimeError | None = None

    for retry_index in range(SCENE_SYNTHESIS_MAX_RETRIES):
        effective_seed = int(rng.integers(0, np.iinfo(np.int32).max))
        attempt_rng = np.random.default_rng(effective_seed)
        try:
            placed_objects, sampled_objects, stacked_count = _synthesize_scene_layout(
                scene=scene,
                base_mesh=base_mesh,
                base_hull=base_hull,
                target_insertions=target_insertions,
                layout_mode=layout_mode,
                rng=attempt_rng,
            )
            break
        except RuntimeError as exc:
            last_error = exc
    else:
        raise RuntimeError(
            f"Unable to synthesize {target_insertions} objects for scene {scene_path} "
            f"after {SCENE_SYNTHESIS_MAX_RETRIES} attempts in layout_mode={layout_mode}: {last_error}"
        )

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
                "placement_style": placed_obj.placement_style,
                "support_xy_overlap_ratio": placed_obj.support_xy_overlap_ratio,
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
                "color_hue_shift_deg": placed_obj.color_hue_shift_deg,
                "color_sat_scale": placed_obj.color_sat_scale,
                "color_val_scale": placed_obj.color_val_scale,
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
        "effective_seed": int(effective_seed),
        "scene_diag": float(scene.scene_diag),
        "scene_point_spacing": float(scene.point_spacing),
        "scene_bounds_min": scene.bounds[0].tolist(),
        "scene_bounds_max": scene.bounds[1].tolist(),
        "layout_mode": layout_mode,
        "scene_retry_index": int(retry_index),
        "requested_object_count": int(target_insertions),
        "placed_object_count": int(len(placed_objects)),
        "stacked_object_count": int(stacked_count),
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
