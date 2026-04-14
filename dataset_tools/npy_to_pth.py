#!/usr/bin/env python3
"""Convert challenge-format `.npy` scenes into SoftGroup-ready `.pth` scenes.

This keeps the challenge `.npy` files as the source of truth and writes a parallel
SoftGroup dataset under `seg/SoftGroup/dataset/nubzuki/<split>/*.pth`.

The conversion applies the same coordinate normalization currently used by the challenge
loader: subtract scene centroid and divide by max radius.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except Exception as exc:  # pragma: no cover
    print(f"Failed to import yaml: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    print(f"Failed to import scipy.spatial.cKDTree: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_SPLITS = ("train", "val", "test")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _softgroup_root() -> Path:
    return _repo_root().parent / "SoftGroup"


def _default_output_root() -> Path:
    return _softgroup_root() / "dataset" / "nubzuki"


def _default_config_path() -> Path:
    return _softgroup_root() / "configs" / "softgroup" / "softgroup_nubzuki.yaml"


def _default_reference_pth_root() -> Path:
    return _repo_root().parent / "multiscan" / "dataset" / "object_instance_segmentation"


def _load_npy_dict(path: Path) -> dict:
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        out = loaded.item()
        if not isinstance(out, dict):
            raise ValueError(f"Expected dict-like scalar array in {path}")
        return out
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    raise ValueError(f"Unsupported data format in {path}")


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape [N, 3], got {xyz.shape}")
    centroid = np.mean(xyz, axis=0, dtype=np.float64)
    xyz = xyz - centroid.astype(np.float32)
    radius = float(np.max(np.sqrt(np.sum(xyz**2, axis=1, dtype=np.float64))))
    if radius > 1e-8:
        xyz = xyz / np.float32(radius)
    return xyz.astype(np.float32)


def _estimate_spacing_metrics(xyz: np.ndarray, sample_size: int = 5000, seed: int = 0) -> dict[str, float]:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected xyz shape [N, 3], got {xyz.shape}")
    if xyz.shape[0] < 2:
        return {"min": 0.0, "p05": 0.0, "p10": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(xyz.shape[0], size=min(sample_size, xyz.shape[0]), replace=False)
    query_pts = xyz[sample_idx]
    tree = cKDTree(xyz)
    distances, _ = tree.query(query_pts, k=2)
    nn = np.asarray(distances[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 1e-8)]
    if nn.size == 0:
        return {"min": 0.0, "p05": 0.0, "p10": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": float(nn.min()),
        "p05": float(np.quantile(nn, 0.05)),
        "p10": float(np.quantile(nn, 0.10)),
        "median": float(np.median(nn)),
        "mean": float(nn.mean()),
        "max": float(nn.max()),
    }


def _load_reference_xyz(pth_path: Path) -> np.ndarray:
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        data = torch.load(pth_path, map_location="cpu")
    return np.asarray(data["xyz"], dtype=np.float32)


def _collect_reference_spacing_target(
    reference_root: Path,
    splits: list[str],
    spacing_metric: str,
    sample_size: int,
    max_scenes: int,
) -> tuple[float, dict]:
    scene_metrics = []
    for split in splits:
        split_dir = reference_root / split
        if not split_dir.is_dir():
            continue
        split_files = sorted(p for p in split_dir.glob("*.pth") if p.is_file())[:max_scenes]
        for pth_path in split_files:
            xyz = _load_reference_xyz(pth_path)
            metrics = _estimate_spacing_metrics(xyz, sample_size=sample_size, seed=0)
            if metrics[spacing_metric] > 0:
                scene_metrics.append(metrics)
    if not scene_metrics:
        raise FileNotFoundError(
            f"Could not compute reference spacing from {reference_root} for splits {splits}"
        )
    target_spacing = float(np.median([m[spacing_metric] for m in scene_metrics]))
    aggregate = {
        "target_metric": spacing_metric,
        "scene_count": int(len(scene_metrics)),
        "target_spacing": target_spacing,
        "median_min": float(np.median([m["min"] for m in scene_metrics])),
        "median_p05": float(np.median([m["p05"] for m in scene_metrics])),
        "median_p10": float(np.median([m["p10"] for m in scene_metrics])),
        "median_median": float(np.median([m["median"] for m in scene_metrics])),
    }
    return target_spacing, aggregate


def _prepare_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"Expected rgb shape [N, 3], got {rgb.shape}")
    if rgb.size == 0:
        return rgb.astype(np.float32)
    if float(np.max(rgb)) <= 1.0:
        rgb = rgb * 2.0 - 1.0
    else:
        rgb = rgb / 127.5 - 1.0
    return np.clip(rgb, -1.0, 1.0).astype(np.float32)


def _prepare_normal(normal: np.ndarray) -> np.ndarray:
    normal = np.asarray(normal, dtype=np.float32)
    if normal.ndim != 2 or normal.shape[1] != 3:
        raise ValueError(f"Expected normal shape [N, 3], got {normal.shape}")
    return normal.astype(np.float32)


def _prepare_labels(instance_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    instance_labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    semantic_label = (instance_labels > 0).astype(np.int64)
    softgroup_instance = np.where(instance_labels > 0, instance_labels - 1, -100).astype(np.int64)
    return semantic_label, softgroup_instance


def _validate_lengths(scene_id: str, arrays: dict) -> int:
    lengths = {k: int(v.shape[0]) for k, v in arrays.items()}
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"{scene_id}: mismatched point counts {lengths}")
    return next(iter(unique_lengths))


def _iter_npy_files(split_dir: Path) -> list[Path]:
    return sorted(p for p in split_dir.glob("*.npy") if p.is_file())


def _compute_instance_point_stats(instance_label: np.ndarray) -> list[int]:
    positives = instance_label[instance_label >= 0]
    if positives.size == 0:
        return []
    out = []
    for instance_id in np.unique(positives):
        out.append(int(np.sum(positives == instance_id)))
    return out


def _build_config(foreground_instance_point_mean: float) -> dict:
    return {
        "model": {
            "channels": 32,
            "num_blocks": 7,
            "semantic_classes": 2,
            "instance_classes": 1,
            "sem2ins_classes": [],
            "semantic_only": False,
            "ignore_label": -100,
            "with_coords": True,
            "grouping_cfg": {
                "score_thr": 0.2,
                "radius": 0.03,
                "mean_active": 300,
                "class_numpoint_mean": [-1.0, float(foreground_instance_point_mean)],
                "npoint_thr": 0.05,
                "ignore_classes": [0],
            },
            "instance_voxel_cfg": {
                "scale": 128,
                "spatial_shape": 20,
            },
            "train_cfg": {
                "max_proposal_num": 200,
                "pos_iou_thr": 0.5,
            },
            "test_cfg": {
                "x4_split": False,
                "cls_score_thr": 0.001,
                "mask_score_thr": -0.5,
                "min_npoint": 50,
                "eval_tasks": ["instance"],
            },
            "fixed_modules": [],
        },
        "data": {
            "train": {
                "type": "nubzuki",
                "data_root": "dataset/nubzuki",
                "prefix": "train",
                "suffix": ".pth",
                "training": True,
                "repeat": 1,
                "voxel_cfg": {
                    "scale": 128,
                    "spatial_shape": [128, 512],
                    "max_npoint": 250000,
                    "min_npoint": 2000,
                },
            },
            "test": {
                "type": "nubzuki",
                "data_root": "dataset/nubzuki",
                "prefix": "val",
                "suffix": ".pth",
                "training": False,
                "with_label": True,
                "voxel_cfg": {
                    "scale": 128,
                    "spatial_shape": [128, 512],
                    "max_npoint": 250000,
                    "min_npoint": 2000,
                },
            },
        },
        "dataloader": {
            "train": {
                "batch_size": 2,
                "num_workers": 4,
            },
            "test": {
                "batch_size": 1,
                "num_workers": 1,
            },
        },
        "optimizer": {
            "type": "Adam",
            "lr": 0.001,
        },
        "fp16": False,
        "epochs": 96,
        "step_epoch": 48,
        "save_freq": 4,
        "pretrain": "",
        "work_dir": "",
    }


def _write_config(config_path: Path, foreground_instance_point_mean: float) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = _build_config(foreground_instance_point_mean=foreground_instance_point_mean)
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def convert_split(
    input_root: Path,
    output_root: Path,
    split: str,
    spacing_target: float | None,
    spacing_metric: str,
    spacing_sample_size: int,
) -> dict:
    split_dir = input_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Split directory does not exist: {split_dir}")

    files = _iter_npy_files(split_dir)
    if not files:
        raise FileNotFoundError(f"No .npy files found under: {split_dir}")

    out_split_dir = output_root / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    instance_point_sizes: list[int] = []
    total_points = 0
    spacing_metric_values: list[float] = []
    scale_factors: list[float] = []

    for npy_path in files:
        scene_id = npy_path.stem
        data = _load_npy_dict(npy_path)
        xyz = _normalize_xyz(np.asarray(data["xyz"], dtype=np.float32))
        spacing_metrics = _estimate_spacing_metrics(xyz, sample_size=spacing_sample_size, seed=0)
        scene_spacing = float(spacing_metrics[spacing_metric])
        scale_factor = 1.0
        if spacing_target is not None and scene_spacing > 1e-8:
            scale_factor = float(spacing_target / scene_spacing)
            xyz = (xyz * np.float32(scale_factor)).astype(np.float32)
        rgb = _prepare_rgb(data["rgb"])
        normal = _prepare_normal(data["normal"])
        semantic_label, instance_label = _prepare_labels(data["instance_labels"])
        n_points = _validate_lengths(
            scene_id,
            {
                "xyz": xyz,
                "rgb": rgb,
                "normal": normal,
                "semantic_label": semantic_label,
                "instance_label": instance_label,
            },
        )
        total_points += n_points
        instance_point_sizes.extend(_compute_instance_point_stats(instance_label))
        spacing_metric_values.append(scene_spacing)
        scale_factors.append(scale_factor)

        payload = {
            "xyz": xyz,
            "rgb": rgb,
            "normal": normal,
            "semantic_label": semantic_label.astype(np.int64),
            "instance_label": instance_label.astype(np.int64),
            "scene_id": scene_id,
        }
        torch.save(payload, out_split_dir / f"{scene_id}.pth")

    point_mean = float(np.mean(instance_point_sizes)) if instance_point_sizes else 1.0
    point_min = int(min(instance_point_sizes)) if instance_point_sizes else 0
    point_max = int(max(instance_point_sizes)) if instance_point_sizes else 0

    return {
        "scene_count": len(files),
        "total_points": int(total_points),
        "foreground_instance_count": int(len(instance_point_sizes)),
        "foreground_instance_point_mean": point_mean,
        "foreground_instance_point_min": point_min,
        "foreground_instance_point_max": point_max,
        "spacing_metric": spacing_metric,
        "spacing_metric_mean_before_rescale": float(np.mean(spacing_metric_values)) if spacing_metric_values else 0.0,
        "spacing_metric_median_before_rescale": float(np.median(spacing_metric_values)) if spacing_metric_values else 0.0,
        "scale_factor_mean": float(np.mean(scale_factors)) if scale_factors else 1.0,
        "scale_factor_median": float(np.median(scale_factors)) if scale_factors else 1.0,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert challenge .npy scenes into SoftGroup .pth scenes.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing split subdirectories with .npy scenes.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_default_output_root(),
        help=f"Output root for converted .pth scenes (default: {_default_output_root()})",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=_default_config_path(),
        help=f"Path to write the generated SoftGroup config (default: {_default_config_path()})",
    )
    parser.add_argument(
        "--reference-pth-root",
        type=Path,
        default=_default_reference_pth_root(),
        help=f"Root of reference MultiScan .pth scenes for spacing-based rescale (default: {_default_reference_pth_root()})",
    )
    parser.add_argument(
        "--reference-splits",
        nargs="+",
        default=["train", "val"],
        help="Reference splits used to compute target point spacing (default: train val)",
    )
    parser.add_argument(
        "--reference-max-scenes",
        type=int,
        default=32,
        help="Maximum number of reference scenes to sample for target spacing (default: 32)",
    )
    parser.add_argument(
        "--spacing-sample-size",
        type=int,
        default=5000,
        help="Sample size for nearest-neighbor spacing estimation (default: 5000)",
    )
    parser.add_argument(
        "--spacing-metric",
        choices=["min", "p05", "p10", "median"],
        default="p05",
        help="Nearest-neighbor spacing metric used for density-based rescale (default: p05)",
    )
    parser.add_argument(
        "--fixed-spacing-target",
        type=float,
        default=None,
        help="Fixed spacing target for density-based rescale. If set, reference .pth scenes are not consulted.",
    )
    parser.add_argument(
        "--disable-spacing-rescale",
        action="store_true",
        help="Disable density-based rescaling and keep only challenge-normalized coordinates.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to convert (default: train val test)",
    )

    args = parser.parse_args(argv)
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config_path.resolve()

    spacing_target = None
    reference_spacing = None
    if not args.disable_spacing_rescale:
        if args.fixed_spacing_target is not None:
            spacing_target = float(args.fixed_spacing_target)
            reference_spacing = {
                "mode": "fixed",
                "target_metric": args.spacing_metric,
                "target_spacing": spacing_target,
            }
        else:
            spacing_target, reference_spacing = _collect_reference_spacing_target(
                reference_root=args.reference_pth_root.resolve(),
                splits=args.reference_splits,
                spacing_metric=args.spacing_metric,
                sample_size=args.spacing_sample_size,
                max_scenes=args.reference_max_scenes,
            )

    split_stats = {}
    for split in args.splits:
        split_stats[split] = convert_split(
            input_root=input_root,
            output_root=output_root,
            split=split,
            spacing_target=spacing_target,
            spacing_metric=args.spacing_metric,
            spacing_sample_size=args.spacing_sample_size,
        )

    mean_source_split = "train" if "train" in split_stats else args.splits[0]
    foreground_instance_point_mean = float(
        split_stats[mean_source_split]["foreground_instance_point_mean"]
    )

    stats = {
        "dataset_name": "nubzuki",
        "input_root": str(input_root),
        "output_root": str(output_root),
        "config_path": str(config_path),
        "splits": split_stats,
        "foreground_instance_point_mean": foreground_instance_point_mean,
        "foreground_instance_point_mean_source_split": mean_source_split,
        "spacing_rescale_enabled": bool(not args.disable_spacing_rescale),
        "reference_spacing": reference_spacing,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    _write_config(config_path=config_path, foreground_instance_point_mean=foreground_instance_point_mean)

    print(f"Converted splits: {', '.join(args.splits)}")
    print(f"SoftGroup dataset root: {output_root}")
    print(f"Stats written to: {output_root / 'stats.json'}")
    print(f"Config written to: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
