from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml

from tools.train import main as softgroup_train_main


def _set_if_present(mapping: dict[str, Any], key: str, value: Any) -> None:
    if value is not None:
        mapping[key] = value


def _write_config_with_overrides(args: argparse.Namespace, config_path: Path) -> Path:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.data_root is not None:
        cfg["data"]["train"]["data_root"] = args.data_root
        cfg["data"]["test"]["data_root"] = args.data_root
    _set_if_present(cfg["data"]["train"], "prefix", args.train_prefix)
    _set_if_present(cfg["data"]["test"], "prefix", args.val_prefix)
    _set_if_present(cfg["dataloader"]["train"], "batch_size", args.batch_size)
    _set_if_present(cfg["dataloader"]["train"], "num_workers", args.train_num_workers)
    _set_if_present(cfg["dataloader"]["test"], "num_workers", args.test_num_workers)
    _set_if_present(cfg["optimizer"], "lr", args.lr)
    _set_if_present(cfg, "epochs", args.epochs)
    _set_if_present(cfg, "step_epoch", args.step_epoch)
    _set_if_present(cfg, "save_freq", args.save_freq)

    if args.work_dir is not None:
        cfg["work_dir"] = args.work_dir

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="nubzuki_train_", delete=False) as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
        return Path(f.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Nubzuki SoftGroup model.")
    parser.add_argument(
        "--config",
        default="configs/softgroup/softgroup_nubzuki.yaml",
        help="SoftGroup YAML config to train from.",
    )
    parser.add_argument("--work-dir", default=None, help="Optional output work directory.")
    parser.add_argument("--resume", default=None, help="Optional checkpoint to resume from.")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation during training.")

    parser.add_argument("--data-root", default=None, help="Override train/val .npy dataset root.")
    parser.add_argument("--train-prefix", default=None, help="Override training split directory name.")
    parser.add_argument("--val-prefix", default=None, help="Override validation split directory name.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of training epochs.")
    parser.add_argument("--step-epoch", type=int, default=None, help="Override LR schedule step epoch.")
    parser.add_argument("--save-freq", type=int, default=None, help="Override checkpoint save frequency.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override training batch size.")
    parser.add_argument("--train-num-workers", type=int, default=None, help="Override training workers.")
    parser.add_argument("--test-num-workers", type=int, default=None, help="Override validation workers.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).resolve().parent / config_path

    forwarded_config = _write_config_with_overrides(args, config_path)
    forwarded = ["tools/train.py", str(forwarded_config)]
    if args.work_dir:
        forwarded.extend(["--work_dir", args.work_dir])
    if args.resume:
        forwarded.extend(["--resume", args.resume])
    if args.skip_validate:
        forwarded.append("--skip_validate")

    sys.argv = forwarded
    softgroup_train_main()


if __name__ == "__main__":
    main()
