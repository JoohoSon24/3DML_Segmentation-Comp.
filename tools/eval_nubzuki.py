import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from munch import Munch
from scipy.optimize import linear_sum_assignment

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from softgroup.data import build_dataloader, build_dataset
from softgroup.model import SoftGroup
from softgroup.util import get_root_logger, load_checkpoint, rle_decode
from tqdm import tqdm

MAX_ALLOWED_INSTANCE_ID = 100


def get_args():
    parser = argparse.ArgumentParser("SoftGroup Nubzuki evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--npy-root", type=str, required=True, help="Original challenge .npy dataset root")
    parser.add_argument("--split", type=str, default="val", help="Split name to evaluate")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to store predictions and metrics")
    return parser.parse_args()


def _load_npy_dict(path: str):
    loaded = np.load(path, allow_pickle=True)
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        return loaded.item()
    if isinstance(loaded, np.lib.npyio.NpzFile):
        return {k: loaded[k] for k in loaded.files}
    raise ValueError(f"Unsupported data format in {path}")


def _labels_to_masks(labels: np.ndarray):
    ids = [int(x) for x in np.unique(labels) if int(x) > 0]
    masks = [(labels == i) for i in ids]
    return ids, masks


def _pairwise_iou_masks(pred_masks: list, gt_masks: list):
    k = len(pred_masks)
    m = len(gt_masks)
    if k == 0 or m == 0:
        return np.zeros((k, m), dtype=np.float32)
    iou = np.zeros((k, m), dtype=np.float32)
    for i in range(k):
        pm = pred_masks[i]
        for j in range(m):
            gm = gt_masks[j]
            inter = np.logical_and(pm, gm).sum()
            union = np.logical_or(pm, gm).sum()
            iou[i, j] = (inter / union) if union > 0 else 0.0
    return iou


def _hungarian_match(iou_mat: np.ndarray):
    if iou_mat.size == 0:
        return (
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    cost = 1.0 - iou_mat
    row_ind, col_ind = linear_sum_assignment(cost)
    matched_ious = iou_mat[row_ind, col_ind].astype(np.float32)
    return row_ind.astype(np.int64), col_ind.astype(np.int64), matched_ious


def _tp_fp_fn_from_matched(matched_ious: np.ndarray, num_pred: int, num_gt: int, thr: float):
    tp = int(np.sum(matched_ious >= float(thr)))
    fp = int(num_pred - tp)
    fn = int(num_gt - tp)
    return tp, fp, fn


def _prf(tp: int, fp: int, fn: int):
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _decode_instances_to_pointwise(pred_instances: list, n_points: int) -> np.ndarray:
    pointwise = np.zeros((n_points,), dtype=np.int64)
    next_instance_id = 1
    ordered_instances = sorted(pred_instances, key=lambda item: float(item["conf"]), reverse=True)
    for instance in ordered_instances:
        if next_instance_id > MAX_ALLOWED_INSTANCE_ID:
            break
        mask = rle_decode(instance["pred_mask"]).astype(bool)
        if mask.shape[0] != n_points:
            raise ValueError(
                f"Predicted mask length mismatch: expected {n_points}, got {mask.shape[0]}"
            )
        paste = np.logical_and(mask, pointwise == 0)
        if not np.any(paste):
            continue
        pointwise[paste] = next_instance_id
        next_instance_id += 1
    return pointwise


def _load_gt_instance(npy_root: str, split: str, scan_id: str) -> np.ndarray:
    split_root = osp.join(npy_root, split)
    scene_path = osp.join(split_root, f"{scan_id}.npy")
    if not osp.exists(scene_path):
        raise FileNotFoundError(f"Could not find ground-truth .npy for scan {scan_id}: {scene_path}")
    data = _load_npy_dict(scene_path)
    return np.asarray(data["instance_labels"], dtype=np.int64)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    pred_save_dir = osp.join(args.output_dir, "predictions")
    os.makedirs(pred_save_dir, exist_ok=True)

    logger = get_root_logger()
    cfg_txt = open(args.config, "r", encoding="utf-8").read()
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))
    cfg.data.test.data_root = args.npy_root
    cfg.data.test.prefix = args.split
    cfg.data.test.suffix = ".npy"
    cfg.data.test.training = False
    cfg.data.test.with_label = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model = SoftGroup(**cfg.model).to(device)
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)
    model.eval()
    total_params = int(sum(p.numel() for p in model.parameters()))

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, dist=False, **cfg.dataloader.test)

    agg = {0.25: {"tp": 0, "fp": 0, "fn": 0}, 0.50: {"tp": 0, "fp": 0, "fn": 0}}
    per_scene_metrics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Nubzuki F1"):
            result = model(batch)
            scan_id = result["scan_id"]
            gt_instance = _load_gt_instance(args.npy_root, args.split, scan_id)
            pred_instance = _decode_instances_to_pointwise(
                pred_instances=result["pred_instances"],
                n_points=gt_instance.shape[0],
            )

            np.save(osp.join(pred_save_dir, f"{scan_id}_pred.npy"), pred_instance)

            _, pred_masks = _labels_to_masks(pred_instance)
            _, gt_masks = _labels_to_masks(gt_instance)
            iou_mat = _pairwise_iou_masks(pred_masks, gt_masks)
            _, _, matched_ious = _hungarian_match(iou_mat)

            tp25, fp25, fn25 = _tp_fp_fn_from_matched(matched_ious, len(pred_masks), len(gt_masks), 0.25)
            _, _, f1_25 = _prf(tp25, fp25, fn25)
            tp50, fp50, fn50 = _tp_fp_fn_from_matched(matched_ious, len(pred_masks), len(gt_masks), 0.50)
            _, _, f1_50 = _prf(tp50, fp50, fn50)

            agg[0.25]["tp"] += tp25
            agg[0.25]["fp"] += fp25
            agg[0.25]["fn"] += fn25
            agg[0.50]["tp"] += tp50
            agg[0.50]["fp"] += fp50
            agg[0.50]["fn"] += fn50

            per_scene_metrics.append(
                {
                    "scene": scan_id,
                    "num_gt_instances": int(len(gt_masks)),
                    "num_pred_instances": int(len(pred_masks)),
                    "f1_25": float(f1_25),
                    "f1_50": float(f1_50),
                }
            )

    _, _, f1_25 = _prf(agg[0.25]["tp"], agg[0.25]["fp"], agg[0.25]["fn"])
    _, _, f1_50 = _prf(agg[0.50]["tp"], agg[0.50]["fp"], agg[0.50]["fn"])

    metrics = {
        "evaluated_at": datetime.utcnow().isoformat() + "Z",
        "config_path": osp.abspath(args.config),
        "checkpoint_path": osp.abspath(args.checkpoint),
        "npy_root": osp.abspath(args.npy_root),
        "split": args.split,
        "num_scenes": len(per_scene_metrics),
        "model_param_count": total_params,
        "max_allowed_instance_id": MAX_ALLOWED_INSTANCE_ID,
        "instance_f1_25": float(f1_25),
        "instance_f1_50": float(f1_50),
        "instance_f1_by_threshold": {
            "0.25": float(f1_25),
            "0.50": float(f1_50),
        },
    }

    metrics_path = osp.join(args.output_dir, "metrics.json")
    per_scene_path = osp.join(args.output_dir, "metrics_per_scene.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(per_scene_path, "w", encoding="utf-8") as f:
        json.dump(per_scene_metrics, f, indent=2)

    logger.info(f"Instance F1 -> @25: {f1_25:.4f}, @50: {f1_50:.4f}")
    logger.info(f"Predictions saved to: {pred_save_dir}")
    logger.info(f"Metrics saved to: {metrics_path}")
    logger.info(f"Per-scene metrics saved to: {per_scene_path}")


if __name__ == "__main__":
    main()
