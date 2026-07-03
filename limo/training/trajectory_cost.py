"""PTC-LIMO trajectory cost regularization utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def path_xy_to_grid_index_aligned(
    path_xy: torch.Tensor,
    x_min: float,
    y_min: float,
    resolution: float,
    H: int,
    W: int,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    swap_xy: bool = False,
) -> torch.Tensor:
    """Convert robot-frame xy path points to grid_sample coordinates.

    The map layout is index-aligned with dataset_builder:
    height/row corresponds to x and width/column corresponds to y.
    """
    if path_xy.ndim != 3 or path_xy.shape[-1] < 2:
        raise ValueError(f"path_xy must have shape [B, N, >=2], got {path_xy.shape}")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    if swap_xy:
        x = path_xy[..., 1]
        y = path_xy[..., 0]
    else:
        x = path_xy[..., 0]
        y = path_xy[..., 1]

    i = (x - x_min) / resolution
    j = (y - y_min) / resolution

    u = 2.0 * j / max(int(W) - 1, 1) - 1.0
    v = 2.0 * i / max(int(H) - 1, 1) - 1.0

    if flip_y_axis:
        u = -u
    if flip_x_axis:
        v = -v

    return torch.stack([u, v], dim=-1).unsqueeze(2)


def sample_map_at_path(
    map_tensor: torch.Tensor,
    path_xy: torch.Tensor,
    x_min: float,
    y_min: float,
    resolution: float,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    swap_xy: bool = False,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """Sample a [B, 1, H, W] map at [B, N, 2] robot-frame path points."""
    if map_tensor.ndim != 4 or map_tensor.shape[1] != 1:
        raise ValueError(f"map_tensor must have shape [B, 1, H, W], got {map_tensor.shape}")
    if path_xy.ndim != 3 or path_xy.shape[-1] < 2:
        raise ValueError(f"path_xy must have shape [B, N, >=2], got {path_xy.shape}")
    if map_tensor.shape[0] != path_xy.shape[0]:
        raise ValueError(
            f"batch size mismatch: map_tensor has {map_tensor.shape[0]}, "
            f"path_xy has {path_xy.shape[0]}"
        )

    _, _, H, W = map_tensor.shape
    grid = path_xy_to_grid_index_aligned(
        path_xy,
        x_min=x_min,
        y_min=y_min,
        resolution=resolution,
        H=H,
        W=W,
        flip_x_axis=flip_x_axis,
        flip_y_axis=flip_y_axis,
        swap_xy=swap_xy,
    )
    sampled = F.grid_sample(
        map_tensor.float(),
        grid.float(),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.squeeze(1).squeeze(-1)


def masked_path_cost(
    risk_values: torch.Tensor,
    valid_values: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average sampled risk over valid samples."""
    valid_clamped = valid_values.clamp(0.0, 1.0)
    cost = (risk_values * valid_clamped).sum(dim=1) / (
        valid_clamped.sum(dim=1) + eps
    )
    valid_ratio = valid_clamped.mean(dim=1)
    return cost, valid_ratio


def out_of_roi_loss(
    path_xy: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Continuous out-of-ROI penalty plus a discrete ratio metric."""
    x = path_xy[..., 0]
    y = path_xy[..., 1]
    oor = (
        F.relu(torch.as_tensor(x_min, device=x.device, dtype=x.dtype) - x)
        + F.relu(x - torch.as_tensor(x_max, device=x.device, dtype=x.dtype))
        + F.relu(torch.as_tensor(y_min, device=y.device, dtype=y.dtype) - y)
        + F.relu(y - torch.as_tensor(y_max, device=y.device, dtype=y.dtype))
    )
    oor_loss = oor.mean()
    oor_mask = (x < x_min) | (x > x_max) | (y < y_min) | (y > y_max)
    oor_ratio = oor_mask.float().mean()
    return oor_loss, oor_ratio


def smoothness_loss(path_xy: torch.Tensor) -> torch.Tensor:
    """Second-difference smoothness penalty for xy paths."""
    if path_xy.shape[1] < 3:
        return path_xy[..., :2].sum() * 0.0
    second_diff = path_xy[:, 2:] - 2.0 * path_xy[:, 1:-1] + path_xy[:, :-2]
    return torch.norm(second_diff, dim=-1).mean()


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def trajectory_cost_regularization(
    pred_path: torch.Tensor,
    gt_path: torch.Tensor,
    risk_map: torch.Tensor,
    valid_mask: torch.Tensor,
    x_min: float = -4.0,
    x_max: float = 4.0,
    y_min: float = -4.0,
    y_max: float = 4.0,
    resolution: float = 0.04,
    lambda_risk: float = 0.03,
    lambda_oor: float = 0.01,
    lambda_smooth: float = 0.002,
    margin: float = 0.0,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    swap_xy: bool = False,
    sampling_mode: str = "bilinear",
    align_corners: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute PTC trajectory cost regularization.

    The predicted branch stays differentiable through bilinear grid_sample.
    The GT branch is detached and used as a relative privileged baseline.
    """
    pred_xy = pred_path[..., :2]
    gt_xy = gt_path[..., :2].detach()

    risk_map = risk_map.float()
    valid_mask = valid_mask.float()

    sample_kwargs = {
        "x_min": x_min,
        "y_min": y_min,
        "resolution": resolution,
        "flip_x_axis": flip_x_axis,
        "flip_y_axis": flip_y_axis,
        "swap_xy": swap_xy,
        "mode": sampling_mode,
        "padding_mode": "zeros",
        "align_corners": align_corners,
    }

    pred_risk = sample_map_at_path(risk_map, pred_xy, **sample_kwargs)
    pred_valid = sample_map_at_path(valid_mask, pred_xy, **sample_kwargs)
    gt_risk = sample_map_at_path(risk_map, gt_xy, **sample_kwargs).detach()
    gt_valid = sample_map_at_path(valid_mask, gt_xy, **sample_kwargs).detach()

    pred_cost, pred_valid_ratio = masked_path_cost(pred_risk, pred_valid)
    gt_cost, gt_valid_ratio = masked_path_cost(gt_risk, gt_valid)
    gt_cost = gt_cost.detach()
    gt_valid_ratio = gt_valid_ratio.detach()

    risk_gap = pred_cost - gt_cost
    risk_rel_loss = F.relu(risk_gap + margin).mean()
    oor_loss, oor_ratio = out_of_roi_loss(pred_xy, x_min, x_max, y_min, y_max)
    smooth_loss = smoothness_loss(pred_xy)

    risk_term = lambda_risk * risk_rel_loss
    oor_term = lambda_oor * oor_loss
    smooth_term = lambda_smooth * smooth_loss
    total = risk_term + oor_term + smooth_term

    logs = {
        "ptc/risk_rel_loss": risk_rel_loss.detach(),
        "ptc/oor_loss": oor_loss.detach(),
        "ptc/smooth_loss": smooth_loss.detach(),
        "ptc/risk_term": risk_term.detach(),
        "ptc/oor_term": oor_term.detach(),
        "ptc/smooth_term": smooth_term.detach(),
        "ptc/pred_risk_cost": pred_cost.mean().detach(),
        "ptc/gt_risk_cost": gt_cost.mean().detach(),
        "ptc/risk_gap": risk_gap.mean().detach(),
        "ptc/pred_valid_ratio": pred_valid_ratio.mean().detach(),
        "ptc/gt_valid_ratio": gt_valid_ratio.mean().detach(),
        "ptc/oor_ratio": oor_ratio.detach(),
    }
    return total, logs


def trajectory_cost_regularization_from_config(
    pred_path: torch.Tensor,
    gt_path: torch.Tensor,
    risk_map: torch.Tensor,
    valid_mask: torch.Tensor,
    config: Any,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Convenience wrapper that reads PTC settings from a dict-like config."""
    sampling = _cfg_value(config, "sampling", None)
    coordinate = str(_cfg_value(sampling, "coordinate", "index_aligned"))
    sampling_mode = str(_cfg_value(sampling, "mode", "bilinear"))
    align_corners = bool(_cfg_value(sampling, "align_corners", True))

    if coordinate != "index_aligned":
        raise ValueError(f"PTC sampling.coordinate must be 'index_aligned', got {coordinate!r}")
    if sampling_mode != "bilinear":
        raise ValueError(f"PTC sampling.mode must be 'bilinear', got {sampling_mode!r}")
    if not align_corners:
        raise ValueError("PTC sampling.align_corners must be true for index-aligned sampling")

    return trajectory_cost_regularization(
        pred_path=pred_path,
        gt_path=gt_path,
        risk_map=risk_map,
        valid_mask=valid_mask,
        x_min=float(_cfg_value(config, "x_min", -4.0)),
        x_max=float(_cfg_value(config, "x_max", 4.0)),
        y_min=float(_cfg_value(config, "y_min", -4.0)),
        y_max=float(_cfg_value(config, "y_max", 4.0)),
        resolution=float(_cfg_value(config, "resolution", 0.04)),
        lambda_risk=float(_cfg_value(config, "lambda_risk", 0.03)),
        lambda_oor=float(_cfg_value(config, "lambda_oor", 0.01)),
        lambda_smooth=float(_cfg_value(config, "lambda_smooth", 0.002)),
        margin=float(_cfg_value(config, "margin", 0.0)),
        flip_x_axis=bool(_cfg_value(config, "flip_x_axis", False)),
        flip_y_axis=bool(_cfg_value(config, "flip_y_axis", False)),
        swap_xy=bool(_cfg_value(config, "swap_xy", False)),
        sampling_mode=sampling_mode,
        align_corners=align_corners,
    )
