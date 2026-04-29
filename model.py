from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from munch import Munch
from scipy.spatial import cKDTree

from softgroup.data.nubzuki import (_load_npy_dict as _load_raw_scene_dict, _normalize_xyz,
                                    _prepare_labels, _prepare_rgb)
from softgroup.model import SoftGroup
from softgroup.ops import voxelization_idx
from softgroup.util import rle_decode

MAX_ALLOWED_INSTANCE_ID = 100


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _default_config_candidates() -> list[Path]:
    root = _repo_root()
    return [
        root / "configs" / "softgroup" / "nubzuki_multiscan_trainval_softgroup_2.yaml",
        root / "configs" / "softgroup" / "softgroup_nubzuki.yaml",
    ]


def _find_config(config_path: str | os.PathLike[str] | None, ckpt_path: str | os.PathLike[str]) -> Path:
    candidates: list[Path] = []
    if config_path:
        candidates.append(Path(config_path))

    ckpt = Path(ckpt_path)
    if ckpt.parent.is_dir():
        candidates.extend(sorted(ckpt.parent.glob("*.yaml")))

    candidates.extend(_default_config_candidates())
    for candidate in candidates:
        candidate = candidate.expanduser()
        if not candidate.is_absolute():
            candidate = _repo_root() / candidate
        if candidate.is_file():
            return candidate

    checked = ", ".join(str(x) for x in candidates)
    raise FileNotFoundError(f"Could not find a SoftGroup config. Checked: {checked}")


def _load_config(path: Path) -> Munch:
    with open(path, "r", encoding="utf-8") as f:
        cfg = Munch.fromDict(yaml.safe_load(f))
    cfg.model.test_cfg.eval_tasks = ["instance"]
    return cfg


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if any(key.startswith("module.") for key in state_dict):
        return {key.replace("module.", "", 1): value for key, value in state_dict.items()}
    return state_dict


def _load_softgroup_weights(model: nn.Module, ckpt_path: str | os.PathLike[str], device: torch.device) -> None:
    del device
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "net" in checkpoint:
            state_dict = checkpoint["net"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {ckpt_path}")

    state_dict = _strip_module_prefix(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[model.py] Missing checkpoint keys: {len(missing)}")
    if unexpected:
        print(f"[model.py] Unexpected checkpoint keys: {len(unexpected)}")


def _estimate_spacing_metric(xyz: np.ndarray, metric: str = "p05", sample_size: int = 5000) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    sample_size = min(int(sample_size), int(xyz.shape[0]))
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(xyz.shape[0], size=sample_size, replace=False)
    tree = cKDTree(xyz)
    distances, _ = tree.query(xyz[sample_idx], k=2)
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


def _rgb_to_softgroup(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.float32)
    if rgb.size == 0:
        return rgb
    if float(np.max(rgb)) <= 1.0:
        rgb = rgb * 2.0 - 1.0
    else:
        rgb = rgb / 127.5 - 1.0
    return np.clip(rgb, -1.0, 1.0).astype(np.float32)


def _softgroup_test_rotation(xyz: np.ndarray) -> np.ndarray:
    theta = 0.35 * math.pi
    matrix = np.array(
        [
            [math.cos(theta), math.sin(theta), 0.0],
            [-math.sin(theta), math.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return np.matmul(np.asarray(xyz, dtype=np.float64), matrix)


def _maybe_restore_spacing(xyz: np.ndarray, data_cfg: Munch) -> np.ndarray:
    fixed_spacing_target = getattr(data_cfg, "fixed_spacing_target", None)
    if fixed_spacing_target in (None, "", "null"):
        return xyz
    spacing = _estimate_spacing_metric(
        xyz,
        metric=getattr(data_cfg, "spacing_metric", "p05"),
        sample_size=getattr(data_cfg, "spacing_sample_size", 5000),
    )
    if spacing <= 1e-8:
        return xyz
    return (xyz * np.float32(float(fixed_spacing_target) / spacing)).astype(np.float32)


def _decode_instances_to_pointwise(pred_instances: list[dict[str, Any]], n_points: int) -> np.ndarray:
    pointwise = np.zeros((n_points,), dtype=np.int64)
    next_instance_id = 1
    ordered = sorted(pred_instances, key=lambda item: float(item["conf"]), reverse=True)
    for instance in ordered:
        if next_instance_id > MAX_ALLOWED_INSTANCE_ID:
            break
        mask = rle_decode(instance["pred_mask"]).astype(bool)
        if mask.shape[0] != n_points:
            raise ValueError(f"Predicted mask length mismatch: expected {n_points}, got {mask.shape[0]}")
        paste = np.logical_and(mask, pointwise == 0)
        if np.any(paste):
            pointwise[paste] = next_instance_id
            next_instance_id += 1
    return pointwise


def _get_cropped_inst_label(instance_label: np.ndarray, valid_idxs: np.ndarray) -> np.ndarray:
    instance_label = instance_label[valid_idxs]
    j = 0
    while j < instance_label.max():
        if len(np.where(instance_label == j)[0]) == 0:
            instance_label[instance_label == instance_label.max()] = j
        j += 1
    return instance_label


def _get_instance_info(
    xyz: np.ndarray,
    instance_label: np.ndarray,
    semantic_label: np.ndarray,
) -> tuple[int, list[int], list[int], np.ndarray]:
    pt_mean = np.ones((xyz.shape[0], 3), dtype=np.float32) * -100.0
    instance_pointnum = []
    instance_cls = []
    instance_num = max(int(instance_label.max()) + 1, 0)
    for instance_id in range(instance_num):
        inst_idx = np.where(instance_label == instance_id)
        xyz_inst = xyz[inst_idx]
        pt_mean[inst_idx] = xyz_inst.mean(0)
        instance_pointnum.append(inst_idx[0].size)
        cls_idx = inst_idx[0][0]
        instance_cls.append(int(semantic_label[cls_idx]))
    pt_offset_label = pt_mean - xyz
    return instance_num, instance_pointnum, instance_cls, pt_offset_label


class SoftGroupChallengeModel(nn.Module):
    """Challenge interface wrapper around the copied SoftGroup implementation."""

    def __init__(self, cfg: Munch, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.network = SoftGroup(**cfg.model).to(device)

    def _build_single_batch(self, scene_features: torch.Tensor, scan_id: str) -> dict[str, Any]:
        data_cfg = getattr(self.cfg.data, "test", Munch())
        voxel_cfg = data_cfg.voxel_cfg

        features_np = scene_features.detach().float().cpu().numpy()
        if features_np.shape[0] < 6:
            raise ValueError(f"Expected at least xyz+rgb channels [>=6, N], got {features_np.shape}")

        xyz = np.ascontiguousarray(features_np[0:3, :].T, dtype=np.float32)
        rgb = _rgb_to_softgroup(features_np[3:6, :].T)
        xyz = _maybe_restore_spacing(xyz, data_cfg)

        xyz_middle = _softgroup_test_rotation(xyz)
        xyz_voxel = xyz_middle * float(voxel_cfg.scale)
        xyz_voxel = xyz_voxel - xyz_voxel.min(0)

        n_points = xyz.shape[0]
        coord = torch.from_numpy(xyz_voxel).long()
        coord_float = torch.from_numpy(xyz_middle.astype(np.float32)).float()
        feat = torch.from_numpy(rgb).float()
        semantic_label = torch.zeros(n_points, dtype=torch.long)
        instance_label = torch.full((n_points,), -100, dtype=torch.long)
        pt_offset_label = torch.full((n_points, 3), -100.0, dtype=torch.float32) - coord_float

        coords = torch.cat([coord.new_zeros((coord.size(0), 1)), coord], dim=1).contiguous()
        batch_idxs = coords[:, 0].int()
        spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 1)

        return {
            "scan_ids": [scan_id],
            "coords": coords,
            "batch_idxs": batch_idxs,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "coords_float": coord_float,
            "feats": feat,
            "semantic_labels": semantic_label,
            "instance_labels": instance_label,
            "instance_pointnum": torch.zeros((0,), dtype=torch.int),
            "instance_cls": torch.zeros((0,), dtype=torch.long),
            "pt_offset_labels": pt_offset_label,
            "spatial_shape": spatial_shape,
            "batch_size": 1,
        }

    def _build_single_batch_from_scene_path(
        self,
        scene_path: str | os.PathLike[str],
        scan_id: str,
    ) -> dict[str, Any]:
        data_cfg = getattr(self.cfg.data, "test", Munch())
        voxel_cfg = data_cfg.voxel_cfg

        raw = _load_raw_scene_dict(os.fspath(scene_path))
        xyz = np.asarray(raw["xyz"], dtype=np.float32)
        if getattr(data_cfg, "use_normalized_coords", False):
            xyz = _normalize_xyz(xyz)
        xyz = _maybe_restore_spacing(xyz, data_cfg)
        rgb = _prepare_rgb(raw["rgb"])

        if "instance_labels" in raw:
            semantic_label, instance_label = _prepare_labels(raw["instance_labels"])
        else:
            semantic_label = np.zeros(xyz.shape[0], dtype=np.int64)
            instance_label = np.full(xyz.shape[0], -100, dtype=np.int64)

        xyz_middle = _softgroup_test_rotation(xyz)
        xyz_voxel = xyz_middle * float(voxel_cfg.scale)
        xyz_voxel = xyz_voxel - xyz_voxel.min(0)
        valid_idxs = np.ones(xyz_voxel.shape[0], dtype=bool)
        instance_label = _get_cropped_inst_label(instance_label, valid_idxs)
        inst_num, inst_pointnum, inst_cls, pt_offset_label = _get_instance_info(
            xyz_middle,
            instance_label.astype(np.int32),
            semantic_label,
        )

        coord = torch.from_numpy(xyz_voxel).long()
        coord_float = torch.from_numpy(xyz_middle).to(torch.float32)
        feat = torch.from_numpy(rgb).float()
        semantic_label_t = torch.from_numpy(semantic_label)
        instance_label_t = torch.from_numpy(instance_label)
        instance_pointnum_t = torch.tensor(inst_pointnum, dtype=torch.int)
        instance_cls_t = torch.tensor(
            [cls - 1 if cls != -100 else cls for cls in inst_cls],
            dtype=torch.long,
        )
        pt_offset_label_t = torch.from_numpy(pt_offset_label).float()

        coords = torch.cat([coord.new_zeros((coord.size(0), 1)), coord], dim=1).contiguous()
        batch_idxs = coords[:, 0].int()
        spatial_shape = np.clip(coords.max(0)[0][1:].numpy() + 1, voxel_cfg.spatial_shape[0], None)
        voxel_coords, v2p_map, p2v_map = voxelization_idx(coords, 1)

        return {
            "scan_ids": [scan_id],
            "coords": coords,
            "batch_idxs": batch_idxs,
            "voxel_coords": voxel_coords,
            "p2v_map": p2v_map,
            "v2p_map": v2p_map,
            "coords_float": coord_float,
            "feats": feat,
            "semantic_labels": semantic_label_t,
            "instance_labels": instance_label_t,
            "instance_pointnum": instance_pointnum_t,
            "instance_cls": instance_cls_t,
            "pt_offset_labels": pt_offset_label_t,
            "spatial_shape": spatial_shape,
            "batch_size": 1,
        }

    @torch.inference_mode()
    def predict_instances(
        self,
        features: torch.Tensor,
        scene_paths: str | os.PathLike[str] | list[str | os.PathLike[str]] | None = None,
    ) -> torch.Tensor:
        if features.ndim != 3:
            raise ValueError(f"Expected features shape [B, C, N], got {tuple(features.shape)}")
        if not torch.cuda.is_available():
            raise RuntimeError("SoftGroup inference requires CUDA and the compiled SoftGroup CUDA ops.")

        if scene_paths is None:
            resolved_scene_paths: list[str | os.PathLike[str] | None] = [None] * features.shape[0]
        elif isinstance(scene_paths, (str, os.PathLike)):
            resolved_scene_paths = [scene_paths]
        else:
            resolved_scene_paths = list(scene_paths)
        if len(resolved_scene_paths) != features.shape[0]:
            raise ValueError(
                f"Expected {features.shape[0]} scene paths, got {len(resolved_scene_paths)}"
            )

        self.network.eval()
        predictions = []
        for batch_idx in range(features.shape[0]):
            scan_id = f"challenge_{batch_idx:04d}"
            if resolved_scene_paths[batch_idx] is not None:
                scene_path = resolved_scene_paths[batch_idx]
                scan_id = Path(os.fspath(scene_path)).stem
                batch = self._build_single_batch_from_scene_path(scene_path, scan_id=scan_id)
            else:
                batch = self._build_single_batch(features[batch_idx], scan_id=scan_id)
            result = self.network(batch)
            pointwise = _decode_instances_to_pointwise(result["pred_instances"], features.shape[2])
            predictions.append(torch.from_numpy(pointwise))
        return torch.stack(predictions, dim=0).to(device=features.device, dtype=torch.long)


def initialize_model(
    ckpt_path: str,
    device: torch.device,
    in_channels: int = 9,
    num_classes: int = 2,
    config_path: str | None = None,
    **_: Any,
) -> nn.Module:
    del in_channels, num_classes
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device.index)
    elif torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device.index)
    else:
        raise RuntimeError("SoftGroup requires CUDA; no CUDA device is available.")

    cfg = _load_config(_find_config(config_path, ckpt_path))
    model = SoftGroupChallengeModel(cfg, device=device)
    _load_softgroup_weights(model.network, ckpt_path, device=device)
    model.eval()
    return model


def run_inference(model: nn.Module, features: torch.Tensor, **kwargs: Any) -> torch.Tensor:
    """Return per-point instance labels [B, N], with 0 as background."""
    if not hasattr(model, "predict_instances"):
        raise TypeError("initialize_model() must return a SoftGroupChallengeModel-compatible object.")
    scene_paths = kwargs.pop("scene_path", None)
    if scene_paths is None:
        scene_paths = kwargs.pop("scene_paths", None)
    return model.predict_instances(features, scene_paths=scene_paths)
