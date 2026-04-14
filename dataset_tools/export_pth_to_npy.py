#!/usr/bin/env python3
"""Convert MultiScan OIS .pth samples to .npy dicts usable by visualize.py.

Examples:
  python export_pth_to_npy.py --input /path/to/scene_00000_00.pth
  python export_pth_to_npy.py --input /path/to/object_instance_segmentation/train --out-dir /tmp/ois_npy
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


def _default_out_path(input_path: str, out_dir: str | None) -> str:
    base = os.path.splitext(os.path.basename(input_path))[0] + ".npy"
    if out_dir:
        return os.path.join(out_dir, base)
    return os.path.join(os.path.dirname(input_path), base)


def _convert_instance_labels(data: dict, mode: str, offset: int) -> np.ndarray:
    n = len(data["xyz"])
    if mode == "background":
        return np.zeros((n,), dtype=np.int64)

    if mode == "from_instance_ids":
        inst = np.asarray(data.get("instance_ids", np.full((n,), -1)), dtype=np.int64)
        out = np.where(inst < 0, 0, inst + 1 + offset)
        return out.astype(np.int64)

    if mode == "from_sem_labels":
        sem = np.asarray(data.get("sem_labels", np.full((n,), -1)), dtype=np.int64)
        out = np.where(sem < 0, 0, sem + 1 + offset)
        return out.astype(np.int64)

    raise ValueError(f"Unsupported label mode: {mode}")


def export_one(pth_path: str, out_path: str, mode: str, offset: int, keep_faces: bool) -> None:
    data = torch.load(pth_path, map_location="cpu")
    xyz = np.asarray(data["xyz"], dtype=np.float32)
    rgb = np.asarray(data["rgb"], dtype=np.float32)
    normal = np.asarray(data["normal"], dtype=np.float32)
    instance_labels = _convert_instance_labels(data, mode=mode, offset=offset)

    out = {
        "xyz": xyz,
        "rgb": rgb,
        "normal": normal,
        "instance_labels": instance_labels,
    }
    if keep_faces and "faces" in data:
        out["faces"] = np.asarray(data["faces"], dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    np.save(out_path, out)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert MultiScan OIS .pth to .npy dict")
    parser.add_argument("--input", required=True, help="Path to .pth file or directory")
    parser.add_argument("--out-dir", default=None, help="Output directory (for batch mode)")
    parser.add_argument(
        "--label-mode",
        default="from_instance_ids",
        choices=["from_instance_ids", "from_sem_labels", "background"],
        help="How to generate instance_labels (default: from_instance_ids)",
    )
    parser.add_argument(
        "--instance-offset",
        type=int,
        default=0,
        help="Offset added to positive instance IDs (default: 0)",
    )
    parser.add_argument(
        "--keep-faces",
        action="store_true",
        help="Include faces in the output dict",
    )

    args = parser.parse_args(argv)
    files = _iter_pth_files(args.input)
    if not files:
        print(f"No .pth files found at: {args.input}", file=sys.stderr)
        return 1

    if os.path.isfile(args.input) and args.out_dir is None:
        out_path = _default_out_path(args.input, None)
        export_one(args.input, out_path, args.label_mode, args.instance_offset, args.keep_faces)
        print(f"Exported: {out_path}")
        return 0

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(os.path.abspath(args.input)), "export_npy")

    os.makedirs(out_dir, exist_ok=True)
    for p in files:
        out_path = _default_out_path(p, out_dir)
        export_one(p, out_path, args.label_mode, args.instance_offset, args.keep_faces)
    print(f"Exported {len(files)} files to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
