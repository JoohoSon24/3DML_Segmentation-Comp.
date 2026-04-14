#!/usr/bin/env python3
"""Export point-cloud `.npy` / `.npz` scenes to `.ply` for quick debugging.

Supported input structure:
  - scalar `.npy` files that store a dict
  - `.npz` files with array keys

Expected keys:
  - `xyz`: [N, 3]
  - `rgb`: [N, 3] (optional if `--color-by instance`)
  - `normal`: [N, 3] (optional, zero-filled if missing)
  - `instance_labels`: [N] (optional; falls back to `is_mesh` if present)

Examples:
  python export_npy_to_ply.py --input assets/test_0000.npy
  python export_npy_to_ply.py --input dataset_tools/synthesized_demo/train --color-by instance
  python export_npy_to_ply.py --input /path/to/data --out-dir /tmp/debug_ply --format ascii
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

import numpy as np


BACKGROUND_COLOR = np.array([180, 180, 180], dtype=np.uint8)
Y_UP_ROTATION = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float32,
)


def _load_npy_dict(file_path: str) -> dict:
    loaded = np.load(file_path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    raise ValueError(f"Unsupported npy/npz structure: {file_path}")


def _iter_input_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path) and input_path.endswith((".npy", ".npz")):
        return [input_path]
    if os.path.isdir(input_path):
        out: List[str] = []
        for root, _, files in os.walk(input_path):
            for file_name in files:
                if file_name.endswith((".npy", ".npz")):
                    out.append(os.path.join(root, file_name))
        return sorted(out)
    return []


def _default_out_path(input_path: str, out_dir: str | None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0] + ".ply"
    if out_dir:
        return os.path.join(out_dir, base)
    return os.path.join(os.path.dirname(input_path), base)


def _prepare_xyz(data: dict, file_path: str) -> np.ndarray:
    xyz = np.asarray(data.get("xyz"))
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"`xyz` must have shape [N, 3] in {file_path}, got {xyz.shape}")
    return np.asarray(xyz, dtype=np.float32)


def _prepare_normals(data: dict, n_points: int) -> np.ndarray:
    normal = data.get("normal")
    if normal is None:
        return np.zeros((n_points, 3), dtype=np.float32)
    normal = np.asarray(normal)
    if normal.ndim != 2 or normal.shape != (n_points, 3):
        return np.zeros((n_points, 3), dtype=np.float32)
    return np.asarray(normal, dtype=np.float32)


def _prepare_rgb(data: dict, n_points: int) -> np.ndarray:
    rgb = data.get("rgb")
    if rgb is None:
        return np.tile(BACKGROUND_COLOR[None, :], (n_points, 1))

    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape != (n_points, 3):
        return np.tile(BACKGROUND_COLOR[None, :], (n_points, 1))

    rgb = rgb.astype(np.float32)
    if rgb.size and float(np.max(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return np.clip(np.rint(rgb), 0, 255).astype(np.uint8)


def _prepare_instance_labels(data: dict, n_points: int) -> np.ndarray:
    if "instance_labels" in data:
        labels = np.asarray(data["instance_labels"], dtype=np.int32).reshape(-1)
    elif "is_mesh" in data:
        labels = np.asarray(data["is_mesh"], dtype=np.int32).reshape(-1)
    else:
        labels = np.zeros((n_points,), dtype=np.int32)

    if labels.shape[0] != n_points:
        raise ValueError(f"Instance label count {labels.shape[0]} does not match xyz count {n_points}")
    return labels.astype(np.int32, copy=False)


def _color_for_id(inst_id: int, salt: int = 7919) -> np.ndarray:
    if inst_id <= 0:
        return BACKGROUND_COLOR.copy()
    rng = np.random.default_rng(int(inst_id) * int(salt))
    return np.clip(rng.uniform(0.2, 1.0, size=3) * 255.0, 0, 255).astype(np.uint8)


def _colorize_instance_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int32).reshape(-1)
    colors = np.tile(BACKGROUND_COLOR[None, :], (labels.shape[0], 1)).astype(np.uint8)
    for inst_id in np.unique(labels):
        if int(inst_id) <= 0:
            continue
        colors[labels == inst_id] = _color_for_id(int(inst_id))
    return colors


def _transform_to_up_axis(xyz: np.ndarray, normals: np.ndarray, up_axis: str) -> tuple[np.ndarray, np.ndarray]:
    axis = (up_axis or "y").strip().lower()
    if axis == "z":
        return xyz, normals
    if axis == "y":
        return (
            (xyz @ Y_UP_ROTATION.T).astype(np.float32),
            (normals @ Y_UP_ROTATION.T).astype(np.float32),
        )
    raise ValueError(f"Unsupported up axis: {up_axis}. Use 'y' or 'z'.")


def _build_export_arrays(
    data: dict,
    file_path: str,
    color_by: str,
    up_axis: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xyz = _prepare_xyz(data, file_path=file_path)
    n_points = xyz.shape[0]
    normals = _prepare_normals(data, n_points=n_points)
    instance_labels = _prepare_instance_labels(data, n_points=n_points)

    if color_by == "rgb":
        colors = _prepare_rgb(data, n_points=n_points)
    elif color_by == "instance":
        colors = _colorize_instance_labels(instance_labels)
    else:
        raise ValueError(f"Unsupported color mode: {color_by}")

    xyz, normals = _transform_to_up_axis(xyz=xyz, normals=normals, up_axis=up_axis)
    return xyz, normals, colors, instance_labels


def _ply_header(n_vertices: int, ascii_format: bool) -> bytes:
    fmt = "ascii 1.0" if ascii_format else "binary_little_endian 1.0"
    header_lines = [
        "ply",
        f"format {fmt}",
        "comment generated by export_npy_to_ply.py",
        f"element vertex {n_vertices}",
        "property float x",
        "property float y",
        "property float z",
        "property float nx",
        "property float ny",
        "property float nz",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "property int instance_label",
        "end_header",
    ]
    return ("\n".join(header_lines) + "\n").encode("ascii")


def _write_ply(
    out_path: str,
    xyz: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    instance_labels: np.ndarray,
    ascii_format: bool,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if ascii_format:
        with open(out_path, "wb") as f:
            f.write(_ply_header(n_vertices=xyz.shape[0], ascii_format=True))
            payload = np.concatenate(
                [
                    xyz.astype(np.float32),
                    normals.astype(np.float32),
                    colors.astype(np.uint8).astype(np.float32),
                    instance_labels.astype(np.int32).reshape(-1, 1).astype(np.float32),
                ],
                axis=1,
            )
            np.savetxt(
                f,
                payload,
                fmt="%.6f %.6f %.6f %.6f %.6f %.6f %d %d %d %d",
            )
        return

    vertex_dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("instance_label", "<i4"),
        ]
    )
    vertex_data = np.empty(xyz.shape[0], dtype=vertex_dtype)
    vertex_data["x"] = xyz[:, 0]
    vertex_data["y"] = xyz[:, 1]
    vertex_data["z"] = xyz[:, 2]
    vertex_data["nx"] = normals[:, 0]
    vertex_data["ny"] = normals[:, 1]
    vertex_data["nz"] = normals[:, 2]
    vertex_data["red"] = colors[:, 0]
    vertex_data["green"] = colors[:, 1]
    vertex_data["blue"] = colors[:, 2]
    vertex_data["instance_label"] = instance_labels

    with open(out_path, "wb") as f:
        f.write(_ply_header(n_vertices=xyz.shape[0], ascii_format=False))
        vertex_data.tofile(f)


def export_one(file_path: str, out_path: str, color_by: str, ascii_format: bool, up_axis: str) -> None:
    data = _load_npy_dict(file_path)
    xyz, normals, colors, instance_labels = _build_export_arrays(
        data,
        file_path=file_path,
        color_by=color_by,
        up_axis=up_axis,
    )
    _write_ply(
        out_path=out_path,
        xyz=xyz,
        normals=normals,
        colors=colors,
        instance_labels=instance_labels,
        ascii_format=ascii_format,
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export point-cloud npy/npz scenes to PLY.")
    parser.add_argument("--input", required=True, help="Path to a .npy/.npz file or directory")
    parser.add_argument("--out-dir", default=None, help="Output directory for batch mode")
    parser.add_argument(
        "--color-by",
        default="rgb",
        choices=["rgb", "instance"],
        help="Point coloring mode in the exported PLY (default: rgb)",
    )
    parser.add_argument(
        "--format",
        default="binary",
        choices=["binary", "ascii"],
        help="PLY encoding format (default: binary)",
    )
    parser.add_argument(
        "--up-axis",
        default="y",
        choices=["y", "z"],
        help="Axis to treat as up in the exported PLY (default: y). Use z to preserve source coordinates.",
    )

    args = parser.parse_args(argv)
    files = _iter_input_files(args.input)
    if not files:
        print(f"No .npy/.npz files found at: {args.input}", file=sys.stderr)
        return 1

    ascii_format = args.format == "ascii"

    if os.path.isfile(args.input) and args.out_dir is None:
        out_path = _default_out_path(args.input, None)
        export_one(args.input, out_path, color_by=args.color_by, ascii_format=ascii_format, up_axis=args.up_axis)
        print(f"Exported: {out_path}")
        return 0

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "export_ply")

    os.makedirs(out_dir, exist_ok=True)
    for file_path in files:
        out_path = _default_out_path(file_path, out_dir)
        export_one(file_path, out_path, color_by=args.color_by, ascii_format=ascii_format, up_axis=args.up_axis)

    print(f"Exported {len(files)} files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
