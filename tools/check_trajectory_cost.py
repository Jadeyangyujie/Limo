#!/usr/bin/env python3
"""Focused checks for PTC trajectory-cost sampling and losses."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from limo.training.trajectory_cost import (  # noqa: E402
    path_xy_to_grid_cell_center,
    sample_map_at_path,
    trajectory_cost_regularization,
)


def assert_close(value: float, expected: float, tol: float = 1e-5) -> None:
    if abs(value - expected) > tol:
        raise AssertionError(f"expected {expected}, got {value}")


def check_boundaries(device: torch.device) -> None:
    H = W = 200
    risk = torch.arange(H * W, dtype=torch.float32, device=device).view(1, 1, H, W)
    pts = torch.tensor(
        [[[-4.0, 0.0], [3.98, 0.0], [4.0, 0.0], [0.0, 4.0]]],
        dtype=torch.float32,
        device=device,
    )
    _, in_bounds = path_xy_to_grid_cell_center(
        pts,
        x_min=-4.0,
        y_min=-4.0,
        resolution=0.04,
        H=H,
        W=W,
    )
    sampled, sampled_in_bounds = sample_map_at_path(
        risk,
        pts,
        x_min=-4.0,
        y_min=-4.0,
        resolution=0.04,
    )

    expected = torch.tensor([[True, True, False, False]], device=device)
    if not torch.equal(in_bounds, expected):
        raise AssertionError(f"unexpected in_bounds: {in_bounds}")
    if not torch.equal(sampled_in_bounds.bool(), expected):
        raise AssertionError(f"unexpected sampled in_bounds: {sampled_in_bounds}")
    if sampled[0, 0].item() == 0.0 or sampled[0, 1].item() == 0.0:
        raise AssertionError("in-bound samples looked like zero padding")


def check_invalid_relative_loss(device: torch.device) -> None:
    H = W = 10
    risk = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device)
    valid = torch.zeros_like(risk)
    valid[:, :, 0, 0] = 1.0
    gt = torch.tensor([[[0.5, 0.5, 0.0]]], dtype=torch.float32, device=device)
    pred = torch.tensor([[[5.5, 5.5, 0.0]]], dtype=torch.float32, device=device)

    _, logs = trajectory_cost_regularization(
        pred,
        gt,
        risk,
        valid,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        resolution=1.0,
        lambda_risk=0.0,
        lambda_oor=0.0,
        lambda_invalid=1.0,
        lambda_smooth=0.0,
    )
    assert_close(float(logs["ptc/pred_valid_ratio"]), 0.0)
    if float(logs["ptc/gt_valid_ratio"]) <= 0.0:
        raise AssertionError("gt_valid_ratio should be positive")
    if float(logs["ptc/invalid_rel_loss"]) <= 0.0:
        raise AssertionError("invalid_rel_loss should be positive")
    assert_close(float(logs["ptc/risk_rel_loss"]), 0.0)


def check_relative_oor(device: torch.device) -> None:
    H = W = 10
    risk = torch.zeros(1, 1, H, W, dtype=torch.float32, device=device)
    valid = torch.ones_like(risk)
    gt = torch.tensor([[[11.0, 5.0, 0.0]]], dtype=torch.float32, device=device)
    pred_less_oor = torch.tensor([[[10.5, 5.0, 0.0]]], dtype=torch.float32, device=device)
    pred_more_oor = torch.tensor([[[12.0, 5.0, 0.0]]], dtype=torch.float32, device=device)

    _, logs_less = trajectory_cost_regularization(
        pred_less_oor,
        gt,
        risk,
        valid,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        resolution=1.0,
        lambda_risk=0.0,
        lambda_oor=1.0,
        lambda_invalid=0.0,
        lambda_smooth=0.0,
        oor_mode="relative",
    )
    _, logs_more = trajectory_cost_regularization(
        pred_more_oor,
        gt,
        risk,
        valid,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        resolution=1.0,
        lambda_risk=0.0,
        lambda_oor=1.0,
        lambda_invalid=0.0,
        lambda_smooth=0.0,
        oor_mode="relative",
    )
    assert_close(float(logs_less["ptc/oor_loss"]), 0.0)
    if float(logs_more["ptc/oor_loss"]) <= 0.0:
        raise AssertionError("relative OOR loss should be positive when pred is worse")


def check_safety_margin(device: torch.device) -> None:
    H = W = 10
    risk = torch.ones(1, 1, H, W, dtype=torch.float32, device=device) * 0.25
    valid = torch.ones_like(risk)
    path = torch.tensor([[[5.5, 5.5, 0.0]]], dtype=torch.float32, device=device)
    _, logs = trajectory_cost_regularization(
        path,
        path,
        risk,
        valid,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        resolution=1.0,
        lambda_risk=1.0,
        lambda_oor=0.0,
        lambda_invalid=0.0,
        lambda_smooth=0.0,
        safety_margin=0.1,
    )
    if float(logs["ptc/risk_rel_loss"]) <= 0.0:
        raise AssertionError("positive safety_margin should produce positive risk loss at equal cost")


def check_gradient(device: torch.device) -> None:
    H = W = 10
    row_risk = torch.arange(H, dtype=torch.float32, device=device).view(1, 1, H, 1)
    risk = row_risk.expand(1, 1, H, W) / float(H - 1)
    valid = torch.ones_like(risk)
    pred = torch.tensor([[[5.5, 5.5, 0.0]]], dtype=torch.float32, device=device)
    pred.requires_grad_(True)
    gt = torch.tensor([[[0.5, 5.5, 0.0]]], dtype=torch.float32, device=device)

    loss, logs = trajectory_cost_regularization(
        pred,
        gt,
        risk,
        valid,
        x_min=0.0,
        x_max=10.0,
        y_min=0.0,
        y_max=10.0,
        resolution=1.0,
        lambda_risk=1.0,
        lambda_oor=0.0,
        lambda_invalid=0.0,
        lambda_smooth=0.0,
    )
    if float(logs["ptc/risk_rel_loss"]) <= 0.0:
        raise AssertionError("risk loss should be positive in gradient check")
    loss.backward()
    grad_norm = pred.grad[..., :2].norm().item()
    if grad_norm <= 0.0:
        raise AssertionError("risk loss should have nonzero xy gradient")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)

    check_boundaries(device)
    check_invalid_relative_loss(device)
    check_relative_oor(device)
    check_safety_margin(device)
    check_gradient(device)
    print("[OK] trajectory cost checks passed")


if __name__ == "__main__":
    main()
