#!/usr/bin/env python3
"""Survey the characteristic point spacing of raw MultiScan OIS point clouds.

This script measures nearest-neighbor spacing statistics directly from the raw
MultiScan `object_instance_segmentation/<split>/*.pth` scenes. The goal is to
estimate a stable "characteristic unit" for the dataset before any challenge
loader normalization pushes each scene into a unit sphere.

Why this matters:
- `dataset.py` normalizes every scene independently.
- SoftGroup expects distances with a meaningful metric scale for voxelization
  and radius-based grouping.
- If the raw dataset uses a fairly stable sampling density, we can measure that
  density once and use it as the global target when rescaling normalized scenes.

The script reports:
- Per-scene raw spacing metrics (`p05`, `p10`, `median`, ...)
- Optional normalized-spacing metrics after simulating the unit-sphere step
- Aggregate summaries for `train`, `val`, `test`, and `all`
- Recommended global spacing targets based on the scene-median of each metric
"""

from __future__ import annotations

import argparse
import csv
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
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    print(f"Failed to import scipy.spatial.cKDTree: {exc}", file=sys.stderr)
    sys.exit(1)


DEFAULT_SPLITS = ("train", "val", "test")
SPACING_METRIC_NAMES = ("min", "p01", "p05", "p10", "median", "mean", "p90", "max")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _default_input_root() -> Path:
    return _repo_root().parent / "multiscan" / "dataset" / "object_instance_segmentation"


def _load_scene_xyz(path: Path) -> np.ndarray:
    try:
        data = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # pragma: no cover
        data = torch.load(path, map_location="cpu")
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"{path}: expected xyz shape [N, 3], got {xyz.shape}")
    return xyz


def _normalize_xyz(xyz: np.ndarray) -> tuple[np.ndarray, float]:
    xyz = np.asarray(xyz, dtype=np.float32)
    centroid = np.mean(xyz, axis=0, dtype=np.float64)
    centered = xyz - centroid.astype(np.float32)
    radius = float(np.max(np.sqrt(np.sum(centered**2, axis=1, dtype=np.float64))))
    if radius > 1e-8:
        centered = centered / np.float32(radius)
    return centered.astype(np.float32), radius


def _estimate_spacing_metrics(xyz: np.ndarray, sample_size: int, seed: int) -> dict[str, float]:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape[0] < 2:
        return {name: 0.0 for name in SPACING_METRIC_NAMES}

    query_size = min(int(sample_size), int(xyz.shape[0]))
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(xyz.shape[0], size=query_size, replace=False)
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz[sample_idx], k=2)
    nn = np.asarray(distances[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 1e-8)]
    if nn.size == 0:
        return {name: 0.0 for name in SPACING_METRIC_NAMES}

    return {
        "min": float(nn.min()),
        "p01": float(np.quantile(nn, 0.01)),
        "p05": float(np.quantile(nn, 0.05)),
        "p10": float(np.quantile(nn, 0.10)),
        "median": float(np.median(nn)),
        "mean": float(nn.mean()),
        "p90": float(np.quantile(nn, 0.90)),
        "max": float(nn.max()),
    }


def _value_summary(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {
            "min": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "mean": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "min": float(arr.min()),
        "p10": float(np.quantile(arr, 0.10)),
        "median": float(np.median(arr)),
        "mean": float(arr.mean()),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(arr.max()),
    }


def _iter_scene_files(split_dir: Path, max_scenes: int | None) -> list[Path]:
    files = sorted(p for p in split_dir.glob("*.pth") if p.is_file())
    if max_scenes is not None and max_scenes > 0:
        return files[:max_scenes]
    return files


def _make_scene_record(
    split: str,
    path: Path,
    sample_size: int,
    seed: int,
    include_normalized: bool,
) -> dict[str, float | int | str]:
    xyz_raw = _load_scene_xyz(path)
    xyz_norm, centroid_radius = _normalize_xyz(xyz_raw)
    bbox_min = xyz_raw.min(axis=0)
    bbox_max = xyz_raw.max(axis=0)
    bbox_extent = bbox_max - bbox_min

    raw_metrics = _estimate_spacing_metrics(xyz_raw, sample_size=sample_size, seed=seed)
    record: dict[str, float | int | str] = {
        "split": split,
        "scene_id": path.stem,
        "path": str(path),
        "num_points": int(xyz_raw.shape[0]),
        "centroid_radius": float(centroid_radius),
        "bbox_extent_x": float(bbox_extent[0]),
        "bbox_extent_y": float(bbox_extent[1]),
        "bbox_extent_z": float(bbox_extent[2]),
        "bbox_diag": float(np.linalg.norm(bbox_extent.astype(np.float64))),
    }
    for metric_name, value in raw_metrics.items():
        record[f"raw_{metric_name}"] = value

    if include_normalized:
        norm_metrics = _estimate_spacing_metrics(xyz_norm, sample_size=sample_size, seed=seed)
        for metric_name, value in norm_metrics.items():
            record[f"normalized_{metric_name}"] = value
        p05_norm = float(norm_metrics["p05"])
        p10_norm = float(norm_metrics["p10"])
        record["radius_times_normalized_p05"] = float(centroid_radius * p05_norm)
        record["radius_times_normalized_p10"] = float(centroid_radius * p10_norm)

    return record


def _aggregate_records(records: list[dict[str, float | int | str]], include_normalized: bool) -> dict:
    summary = {
        "scene_count": len(records),
        "num_points": _value_summary([float(r["num_points"]) for r in records]),
        "centroid_radius": _value_summary([float(r["centroid_radius"]) for r in records]),
        "bbox_diag": _value_summary([float(r["bbox_diag"]) for r in records]),
        "raw_spacing": {},
        "recommended_targets": {},
    }
    for metric_name in SPACING_METRIC_NAMES:
        values = [float(r[f"raw_{metric_name}"]) for r in records]
        summary["raw_spacing"][metric_name] = _value_summary(values)
        summary["recommended_targets"][metric_name] = float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0

    if include_normalized:
        normalized_spacing = {}
        for metric_name in SPACING_METRIC_NAMES:
            values = [float(r[f"normalized_{metric_name}"]) for r in records]
            normalized_spacing[metric_name] = _value_summary(values)
        summary["normalized_spacing"] = normalized_spacing
        summary["radius_times_normalized_p05"] = _value_summary(
            [float(r["radius_times_normalized_p05"]) for r in records]
        )
        summary["radius_times_normalized_p10"] = _value_summary(
            [float(r["radius_times_normalized_p10"]) for r in records]
        )
    return summary


def _print_console_summary(report: dict, include_normalized: bool) -> None:
    print("MultiScan characteristic spacing survey")
    print(f"Input root: {report['input_root']}")
    print(f"Splits: {', '.join(report['splits'])}")
    print(f"Sample size per scene: {report['sample_size']}")
    print(f"Seed: {report['seed']}")
    print()
    for split_name in list(report["aggregates"].keys()):
        summary = report["aggregates"][split_name]
        print(f"[{split_name}] scenes={summary['scene_count']}")
        print(
            "  recommended raw targets:"
            f" p05={summary['recommended_targets']['p05']:.6f}"
            f" p10={summary['recommended_targets']['p10']:.6f}"
            f" median={summary['recommended_targets']['median']:.6f}"
        )
        print(
            "  centroid radius:"
            f" median={summary['centroid_radius']['median']:.6f}"
            f" mean={summary['centroid_radius']['mean']:.6f}"
        )
        if include_normalized:
            print(
                "  normalized p05:"
                f" median={summary['normalized_spacing']['p05']['median']:.6f}"
                f" mean={summary['normalized_spacing']['p05']['mean']:.6f}"
            )
        print()


def _write_csv(path: Path, records: list[dict[str, float | int | str]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def build_report(
    input_root: Path,
    splits: Iterable[str],
    sample_size: int,
    seed: int,
    include_normalized: bool,
    max_scenes_per_split: int | None,
) -> dict:
    split_names = list(splits)
    per_scene: list[dict[str, float | int | str]] = []
    records_by_split: dict[str, list[dict[str, float | int | str]]] = {}

    for split in split_names:
        split_dir = input_root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"Split directory does not exist: {split_dir}")
        files = _iter_scene_files(split_dir, max_scenes=max_scenes_per_split)
        if not files:
            raise FileNotFoundError(f"No .pth files found under: {split_dir}")
        split_records = [
            _make_scene_record(
                split=split,
                path=path,
                sample_size=sample_size,
                seed=seed,
                include_normalized=include_normalized,
            )
            for path in files
        ]
        records_by_split[split] = split_records
        per_scene.extend(split_records)

    aggregates = {split: _aggregate_records(records_by_split[split], include_normalized) for split in split_names}
    aggregates["all"] = _aggregate_records(per_scene, include_normalized)

    return {
        "input_root": str(input_root),
        "splits": split_names,
        "sample_size": int(sample_size),
        "seed": int(seed),
        "include_normalized": bool(include_normalized),
        "max_scenes_per_split": None if max_scenes_per_split is None else int(max_scenes_per_split),
        "scene_count": int(len(per_scene)),
        "aggregates": aggregates,
        "per_scene": per_scene,
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Survey the characteristic nearest-neighbor spacing of MultiScan OIS scenes."
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=_default_input_root(),
        help=f"Root containing train/val/test raw MultiScan .pth scenes (default: {_default_input_root()})",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to survey (default: train val test)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="How many points to query per scene for nearest-neighbor spacing (default: 5000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for point subsampling (default: 0)",
    )
    parser.add_argument(
        "--max-scenes-per-split",
        type=int,
        default=None,
        help="Optional cap on the number of scenes read from each split.",
    )
    parser.add_argument(
        "--skip-normalized",
        action="store_true",
        help="Skip the extra pass that simulates unit-sphere normalization and reports normalized spacing.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the full report as JSON.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to write per-scene metrics as CSV.",
    )

    args = parser.parse_args(argv)
    report = build_report(
        input_root=args.input_root.resolve(),
        splits=args.splits,
        sample_size=args.sample_size,
        seed=args.seed,
        include_normalized=not args.skip_normalized,
        max_scenes_per_split=args.max_scenes_per_split,
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if args.output_csv is not None:
        _write_csv(args.output_csv, report["per_scene"])

    _print_console_summary(report, include_normalized=not args.skip_normalized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
