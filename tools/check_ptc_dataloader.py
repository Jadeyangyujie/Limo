#!/usr/bin/env python3
"""Focused checks for PTC label loading and dataloader safety behavior."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from limo.src.dataset.limo_datset import MissionDataset  # noqa: E402


DEFAULT_DATASET_ROOT = Path("/home/robot-device/yangyujie/BEV_LIMO/LIMO_DATASET")


def find_default_mission(dataset_root: Path) -> str:
    candidates = sorted(path.parent.name for path in dataset_root.glob("*/ptc_labels"))
    if not candidates:
        raise FileNotFoundError(f"No */ptc_labels directories found under {dataset_root}")
    return candidates[0]


def base_ptc_config(**overrides):
    cfg = {
        "enabled": True,
        "label_group": "ptc_labels",
        "risk_key": "risk_map",
        "valid_key": "valid_mask",
        "allow_missing_labels": False,
        "validate_metadata": True,
        "allow_metadata_mismatch": False,
        "x_min": -4.0,
        "x_max": 4.0,
        "y_min": -4.0,
        "y_max": 4.0,
        "resolution": 0.04,
    }
    cfg.update(overrides)
    return cfg


def make_dataset(
    dataset_root: Path,
    mission: str,
    dataset_type: str,
    ptc: dict,
) -> MissionDataset:
    transform = transforms.Compose(
        [
            transforms.Resize((308, 476)),
            transforms.ToTensor(),
        ]
    )
    return MissionDataset(
        dataset_type,
        dataset_root,
        mission,
        transform,
        with_side_cams=False,
        ptc=ptc,
    )


def expect_runtime_error(fn, label: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        print(f"[OK] {label}: {exc}")
        return
    raise AssertionError(f"{label} should have raised RuntimeError")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--mission")
    parser.add_argument("--dataset-type", default="geo", choices=("geo", "tel"))
    parser.add_argument("--sample-index", type=int, default=0)
    args = parser.parse_args()

    dataset_root = args.dataset_root.expanduser()
    mission = args.mission or find_default_mission(dataset_root)
    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] mission={mission}")
    print(f"[INFO] dataset_type={args.dataset_type}")

    ds = make_dataset(
        dataset_root,
        mission,
        args.dataset_type,
        base_ptc_config(),
    )
    sample = ds[args.sample_index]
    risk = sample["risk_map"]
    valid = sample["valid_mask"]
    if risk.ndim != 3 or valid.ndim != 3:
        raise AssertionError(f"expected [1,H,W] maps, got {risk.shape}, {valid.shape}")
    if float(sample["ptc_missing_label"]) != 0.0:
        raise AssertionError("existing sample should not be marked missing")
    print(
        "[OK] existing sample loaded: "
        f"image_id={int(sample['image_id'])}, "
        f"risk_shape={tuple(risk.shape)}, "
        f"valid_nonzero={int(torch.count_nonzero(valid))}, "
        f"empty_valid={float(sample['ptc_empty_valid_mask'])}"
    )

    missing_image_id = max(ds.ptc_image_id_to_row.keys()) + 1000000
    expect_runtime_error(
        lambda: ds._load_ptc_label(missing_image_id),
        "missing image_id fail-fast",
    )

    ds_allow = make_dataset(
        dataset_root,
        mission,
        args.dataset_type,
        base_ptc_config(allow_missing_labels=True),
    )
    missing = ds_allow._load_ptc_label(missing_image_id)
    if float(missing["ptc_missing_label"]) != 1.0:
        raise AssertionError("allow_missing_labels=true should mark missing labels")
    if int(torch.count_nonzero(missing["valid_mask"])) != 0:
        raise AssertionError("missing label fallback should have zero valid mask")
    print(
        "[OK] missing label fallback only when explicitly allowed: "
        f"missing_count={ds_allow.missing_ptc_label_count}, "
        f"empty_valid_count={ds_allow.empty_valid_mask_count}"
    )

    expect_runtime_error(
        lambda: make_dataset(
            dataset_root,
            mission,
            args.dataset_type,
            base_ptc_config(x_max=4.04),
        ),
        "metadata/config extent mismatch fail-fast",
    )

    print("[OK] PTC dataloader checks passed")


if __name__ == "__main__":
    main()
