from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .custom import CustomDataset


def _load_npy_dict(path: str):
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


def _estimate_spacing_metric(xyz: np.ndarray, metric: str = "p05", sample_size: int = 5000) -> float:
    xyz = np.asarray(xyz, dtype=np.float32)
    if xyz.shape[0] < 2:
        return 0.0
    sample_size = min(int(sample_size), int(xyz.shape[0]))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(xyz.shape[0], size=sample_size, replace=False)
    query_pts = xyz[sample_idx]
    tree = cKDTree(xyz)
    distances, _ = tree.query(query_pts, k=2)
    nn = np.asarray(distances[:, 1], dtype=np.float64)
    nn = nn[np.isfinite(nn) & (nn > 1e-8)]
    if nn.size == 0:
        return 0.0
    if metric == "min":
        return float(nn.min())
    if metric == "p05":
        return float(np.quantile(nn, 0.05))
    if metric == "p10":
        return float(np.quantile(nn, 0.10))
    if metric == "median":
        return float(np.median(nn))
    raise ValueError(f"Unsupported spacing metric: {metric}")


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


def _prepare_labels(instance_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    instance_labels = np.asarray(instance_labels, dtype=np.int64).reshape(-1)
    semantic_label = (instance_labels > 0).astype(np.int64)
    softgroup_instance = np.where(instance_labels > 0, instance_labels - 1, -100).astype(np.int64)
    return semantic_label, softgroup_instance


class NubzukiDataset(CustomDataset):

    CLASSES = ("nubzuki", )
    NYU_ID = (1, )

    def __init__(
        self,
        aug_prob=1.0,
        use_normalized_coords=True,
        fixed_spacing_target=None,
        spacing_metric="p05",
        spacing_sample_size=5000,
        **kwargs,
    ):
        self.aug_prob = float(aug_prob)
        self.use_normalized_coords = bool(use_normalized_coords)
        self.fixed_spacing_target = (
            None if fixed_spacing_target in (None, "", "null") else float(fixed_spacing_target)
        )
        self.spacing_metric = spacing_metric
        self.spacing_sample_size = int(spacing_sample_size)
        super().__init__(**kwargs)

    def load(self, filename):
        data = _load_npy_dict(filename)
        xyz = np.asarray(data["xyz"], dtype=np.float32)
        if self.use_normalized_coords:
            xyz = _normalize_xyz(xyz)
        if self.fixed_spacing_target is not None:
            scene_spacing = _estimate_spacing_metric(
                xyz,
                metric=self.spacing_metric,
                sample_size=self.spacing_sample_size,
            )
            if scene_spacing > 1e-8:
                xyz = (xyz * np.float32(self.fixed_spacing_target / scene_spacing)).astype(np.float32)
        rgb = _prepare_rgb(data["rgb"])
        if not self.with_label:
            dummy_semantic = np.zeros(xyz.shape[0], dtype=np.int64)
            dummy_instance = np.full(xyz.shape[0], -100, dtype=np.int64)
            return xyz, rgb, dummy_semantic, dummy_instance
        semantic_label, instance_label = _prepare_labels(data["instance_labels"])
        return xyz, rgb, semantic_label, instance_label

    def getInstanceInfo(self, xyz, instance_label, semantic_label):
        instance_num, instance_pointnum, instance_cls, pt_offset_label = super().getInstanceInfo(
            xyz, instance_label, semantic_label)
        instance_cls = [x - 1 if x != -100 else x for x in instance_cls]
        return instance_num, instance_pointnum, instance_cls, pt_offset_label

    def transform_train(self, xyz, rgb, semantic_label, instance_label, aug_prob=1.0):
        return super().transform_train(
            xyz,
            rgb,
            semantic_label,
            instance_label,
            aug_prob=self.aug_prob,
        )
