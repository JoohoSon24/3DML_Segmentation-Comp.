#!/usr/bin/env python3
"""Export MultiScan OIS .pth samples to a VSCode-viewable mesh (GLB/PLY/STL).

Examples:
  python export_pth_to_glb.py --input /path/to/scene_00000_00.pth
  python export_pth_to_glb.py --input /path/to/object_instance_segmentation/train --out-dir /tmp/ois_glb --color-by instance
"""

import argparse
import os
import sys
from typing import Iterable, List

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    print(f"Failed to import torch: {exc}", file=sys.stderr)
    sys.exit(1)

try:
    import trimesh
except Exception as exc:  # pragma: no cover
    print(f"Failed to import trimesh: {exc}", file=sys.stderr)
    sys.exit(1)


BACKGROUND_COLOR = np.array([180, 180, 180, 255], dtype=np.uint8)
Y_UP_TRANSFORM = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


def _color_for_id(inst_id: int, salt: int = 7919) -> np.ndarray:
    rng = np.random.default_rng(int(inst_id) * int(salt))
    rgb = rng.uniform(0.2, 1.0, size=3)
    return (rgb * 255.0).astype(np.uint8)


def _colorize_labels(labels: np.ndarray, salt: int = 7919) -> np.ndarray:
    labels = labels.astype(np.int64).reshape(-1)
    colors = np.zeros((len(labels), 4), dtype=np.uint8)
    colors[:] = BACKGROUND_COLOR
    for lab in np.unique(labels):
        if int(lab) < 0:
            continue
        col = _color_for_id(int(lab), salt=salt)
        colors[labels == lab, :3] = col
        colors[labels == lab, 3] = 255
    return colors


def _prepare_vertex_colors(rgb: np.ndarray) -> np.ndarray:
    if rgb is None:
        return None
    rgb = np.asarray(rgb)
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        return None
    if rgb.max() <= 1.0:
        rgb = (rgb * 255.0)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
    return np.concatenate([rgb, alpha], axis=1)


def _iter_pth_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path) and input_path.endswith(".pth"):
        return [input_path]
    if os.path.isdir(input_path):
        out: List[str] = []
        for root, _, files in os.walk(input_path):
            for f in files:
                if f.endswith(".pth"):
                    out.append(os.path.join(root, f))
        return sorted(out)
    return []


def _default_out_path(input_path: str, out_dir: str | None, ext: str) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0] + f".{ext}"
    if out_dir:
        return os.path.join(out_dir, base)
    return os.path.join(os.path.dirname(input_path), base)


def _build_mesh(data: dict, color_by: str) -> trimesh.Trimesh:
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    faces = np.asarray(data["faces"], dtype=np.int64)
    normals = None
    if "normal" in data:
        normals = np.asarray(data["normal"], dtype=np.float32)

    mesh = trimesh.Trimesh(vertices=xyz, faces=faces, vertex_normals=normals, process=False)

    if color_by == "rgb":
        colors = _prepare_vertex_colors(data.get("rgb"))
        if colors is not None:
            mesh.visual.vertex_colors = colors
    elif color_by == "instance":
        labels = np.asarray(data.get("instance_ids", np.full((xyz.shape[0],), -1)))
        mesh.visual.vertex_colors = _colorize_labels(labels, salt=7919)
    elif color_by == "semantic":
        labels = np.asarray(data.get("sem_labels", np.full((xyz.shape[0],), -1)))
        mesh.visual.vertex_colors = _colorize_labels(labels, salt=15485863)

    return mesh


def _apply_up_axis(mesh: trimesh.Trimesh, up_axis: str) -> trimesh.Trimesh:
    axis = (up_axis or "y").strip().lower()
    if axis == "z":
        return mesh
    if axis == "y":
        mesh = mesh.copy()
        mesh.apply_transform(Y_UP_TRANSFORM)
        return mesh
    raise ValueError(f"Unsupported up axis: {up_axis}. Use 'y' or 'z'.")


def export_one(pth_path: str, out_path: str, color_by: str, up_axis: str) -> None:
    data = torch.load(pth_path, map_location="cpu")
    mesh = _build_mesh(data, color_by=color_by)
    mesh = _apply_up_axis(mesh, up_axis=up_axis)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    mesh.export(out_path)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export MultiScan OIS .pth to GLB/PLY/STL")
    parser.add_argument("--input", required=True, help="Path to .pth file or directory")
    parser.add_argument("--out-dir", default=None, help="Output directory (for batch mode)")
    parser.add_argument(
        "--ext",
        default="glb",
        choices=["glb", "gltf", "ply", "stl"],
        help="Output mesh format (default: glb)",
    )
    parser.add_argument(
        "--color-by",
        default="rgb",
        choices=["rgb", "instance", "semantic"],
        help="Vertex coloring mode (default: rgb)",
    )
    parser.add_argument(
        "--up-axis",
        default="y",
        choices=["y", "z"],
        help="Axis to treat as up in the exported mesh (default: y). Use z to preserve source coordinates.",
    )

    args = parser.parse_args(argv)
    files = _iter_pth_files(args.input)
    if not files:
        print(f"No .pth files found at: {args.input}", file=sys.stderr)
        return 1

    if os.path.isfile(args.input) and args.out_dir is None:
        out_path = _default_out_path(args.input, None, args.ext)
        export_one(args.input, out_path, color_by=args.color_by, up_axis=args.up_axis)
        print(f"Exported: {out_path}")
        return 0

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), f"export_{args.ext}")

    os.makedirs(out_dir, exist_ok=True)
    for p in files:
        out_path = _default_out_path(p, out_dir, args.ext)
        export_one(p, out_path, color_by=args.color_by, up_axis=args.up_axis)
    print(f"Exported {len(files)} files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
