"""PTC-LIMO trajectory cost regularization utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _cfg_value(config: Any, key: str, default: Any) -> Any:
    if config is None:
        return default
    if isinstance(config, dict):
        return config.get(key, default)
    getter = getattr(config, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(config, key, default)


def _map_extent(
    *,
    x_min: float,
    y_min: float,
    resolution: float,
    H: int,
    W: int,
) -> tuple[float, float, float, float]:
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    map_x_min = float(x_min)
    map_y_min = float(y_min)
    map_x_max = map_x_min + int(H) * float(resolution)
    map_y_max = map_y_min + int(W) * float(resolution)
    return map_x_min, map_x_max, map_y_min, map_y_max


def path_xy_to_grid_cell_center(
    path_xy: torch.Tensor,
    *,
    x_min: float,
    y_min: float,
    resolution: float,
    H: int,
    W: int,
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    swap_xy: bool = False,
    clamp_grid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert robot-frame xy path points to grid_sample coordinates.

    The map layout is index-aligned with dataset_builder: height/row
    corresponds to x and width/column corresponds to y. A map with H rows and W
    columns covers half-open extents [x_min, x_min + H * resolution) and
    [y_min, y_min + W * resolution). Coordinates are converted to cell-center
    indices before grid_sample.
    """
    if path_xy.ndim != 3 or path_xy.shape[-1] < 2:
        raise ValueError(f"path_xy must have shape [B, N, >=2], got {path_xy.shape}")

    if swap_xy:
        x = path_xy[..., 1]
        y = path_xy[..., 0]
    else:
        x = path_xy[..., 0]
        y = path_xy[..., 1]

    map_x_min, map_x_max, map_y_min, map_y_max = _map_extent(
        x_min=x_min,
        y_min=y_min,
        resolution=resolution,
        H=H,
        W=W,
    )

    in_bounds = (
        (x >= map_x_min)
        & (x < map_x_max)
        & (y >= map_y_min)
        & (y < map_y_max)
    )

    i = (x - map_x_min) / resolution - 0.5
    j = (y - map_y_min) / resolution - 0.5

    if clamp_grid:
        i = i.clamp(0.0, float(max(int(H) - 1, 0)))
        j = j.clamp(0.0, float(max(int(W) - 1, 0)))

    u = 2.0 * j / max(int(W) - 1, 1) - 1.0
    v = 2.0 * i / max(int(H) - 1, 1) - 1.0

    if flip_y_axis:
        u = -u
    if flip_x_axis:
        v = -v

    grid = torch.stack([u, v], dim=-1).unsqueeze(2)
    return grid, in_bounds


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
    """Backward-compatible wrapper returning only grid coordinates."""
    grid, _ = path_xy_to_grid_cell_center(
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
    return grid


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
    padding_mode: str = "border",
    align_corners: bool = True,
    clamp_grid: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a [B, 1, H, W] map at [B, N, 2] robot-frame path points.

    Returns sampled values and a float in-bounds mask. Coordinates outside the
    half-open map extent are clamped for stable border sampling, then marked
    invalid by the returned mask.
    """
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
    grid, in_bounds = path_xy_to_grid_cell_center(
        path_xy,
        x_min=x_min,
        y_min=y_min,
        resolution=resolution,
        H=H,
        W=W,
        flip_x_axis=flip_x_axis,
        flip_y_axis=flip_y_axis,
        swap_xy=swap_xy,
        clamp_grid=clamp_grid,
    )
    sampled = F.grid_sample(
        map_tensor.float(),
        grid.float(),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )
    return sampled.squeeze(1).squeeze(-1), in_bounds.to(
        device=sampled.device, dtype=sampled.dtype
    )


def masked_path_cost(
    risk_values: torch.Tensor,
    valid_values: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Average sampled risk over valid samples."""
    valid_clamped = valid_values.clamp(0.0, 1.0)
    valid_sum = valid_clamped.sum(dim=1)
    cost = (risk_values * valid_clamped).sum(dim=1) / (valid_sum + eps)
    valid_ratio = valid_clamped.mean(dim=1)
    return cost, valid_ratio, valid_sum


def continuous_oor_distance(
    path_xy: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    """Continuous distance outside a half-open ROI."""
    x = path_xy[..., 0]
    y = path_xy[..., 1]
    return (
        F.relu(torch.as_tensor(x_min, device=x.device, dtype=x.dtype) - x)
        + F.relu(x - torch.as_tensor(x_max, device=x.device, dtype=x.dtype))
        + F.relu(torch.as_tensor(y_min, device=y.device, dtype=y.dtype) - y)
        + F.relu(y - torch.as_tensor(y_max, device=y.device, dtype=y.dtype))
    )


def out_of_roi_ratio(
    path_xy: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> torch.Tensor:
    """Ratio of points outside a half-open ROI."""
    x = path_xy[..., 0]
    y = path_xy[..., 1]
    oor_mask = (x < x_min) | (x >= x_max) | (y < y_min) | (y >= y_max)
    return oor_mask.float().mean()


def out_of_roi_loss(
    path_xy: torch.Tensor,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Absolute continuous out-of-ROI penalty plus a discrete ratio metric."""
    oor = continuous_oor_distance(path_xy, x_min, x_max, y_min, y_max)
    return oor.mean(), out_of_roi_ratio(path_xy, x_min, x_max, y_min, y_max)


def smoothness_loss(path_xy: torch.Tensor) -> torch.Tensor:
    """Second-difference smoothness penalty for xy paths."""
    if path_xy.shape[1] < 3:
        return path_xy[..., :2].sum() * 0.0
    second_diff = path_xy[:, 2:] - 2.0 * path_xy[:, 1:-1] + path_xy[:, :-2]
    return torch.norm(second_diff, dim=-1).mean()


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
    lambda_invalid: float = 0.005,
    lambda_smooth: float = 0.002,
    safety_margin: float = 0.0,
    valid_margin: float = 0.0,
    oor_mode: str = "relative",
    flip_x_axis: bool = False,
    flip_y_axis: bool = False,
    swap_xy: bool = False,
    sampling_mode: str = "bilinear",
    align_corners: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute PTC trajectory cost regularization.

    The predicted branch stays differentiable through bilinear grid_sample.
    The GT branch is detached and used as a relative privileged baseline.
    Positive safety_margin means pred must be safer than gt by this amount.
    """
    if pred_path.shape[:2] != gt_path.shape[:2]:
        raise ValueError(
            f"pred_path and gt_path must share [B,N], got {pred_path.shape} and {gt_path.shape}"
        )

    pred_xy = pred_path[..., :2]
    gt_xy = gt_path[..., :2].detach()

    risk_map = risk_map.float()
    valid_mask = valid_mask.float()
    _, _, H, W = risk_map.shape
    map_x_min, map_x_max, map_y_min, map_y_max = _map_extent(
        x_min=x_min,
        y_min=y_min,
        resolution=resolution,
        H=H,
        W=W,
    )

    sample_kwargs = {
        "x_min": x_min,
        "y_min": y_min,
        "resolution": resolution,
        "flip_x_axis": flip_x_axis,
        "flip_y_axis": flip_y_axis,
        "swap_xy": swap_xy,
        "mode": sampling_mode,
        "padding_mode": "border",
        "align_corners": align_corners,
        "clamp_grid": True,
    }

    pred_risk, pred_in_bounds = sample_map_at_path(risk_map, pred_xy, **sample_kwargs)
    pred_valid_sampled, _ = sample_map_at_path(valid_mask, pred_xy, **sample_kwargs)
    gt_risk, gt_in_bounds = sample_map_at_path(risk_map, gt_xy, **sample_kwargs)
    gt_valid_sampled, _ = sample_map_at_path(valid_mask, gt_xy, **sample_kwargs)

    pred_valid = pred_valid_sampled.clamp(0.0, 1.0) * pred_in_bounds
    gt_valid = (gt_valid_sampled.clamp(0.0, 1.0) * gt_in_bounds).detach()
    gt_risk = gt_risk.detach()

    pred_cost, pred_valid_ratio, pred_valid_sum = masked_path_cost(pred_risk, pred_valid)
    gt_cost, gt_valid_ratio, gt_valid_sum = masked_path_cost(gt_risk, gt_valid)
    gt_cost = gt_cost.detach()
    gt_valid_ratio = gt_valid_ratio.detach()
    gt_valid_sum = gt_valid_sum.detach()

    risk_gap = pred_cost - gt_cost
    risk_rel_loss = F.relu(risk_gap + safety_margin).mean()

    valid_gap = pred_valid_ratio - gt_valid_ratio
    invalid_rel_loss = F.relu(gt_valid_ratio - pred_valid_ratio + valid_margin).mean()

    pred_oor = continuous_oor_distance(
        pred_xy, map_x_min, map_x_max, map_y_min, map_y_max
    )
    gt_oor = continuous_oor_distance(
        gt_xy, map_x_min, map_x_max, map_y_min, map_y_max
    ).detach()
    pred_oor_dist = pred_oor.mean(dim=1)
    gt_oor_dist = gt_oor.mean(dim=1)

    if oor_mode == "relative":
        oor_loss = F.relu(pred_oor_dist - gt_oor_dist).mean()
    elif oor_mode == "absolute":
        oor_loss = pred_oor_dist.mean()
    else:
        raise ValueError(f"PTC oor_mode must be 'relative' or 'absolute', got {oor_mode!r}")

    pred_oor_ratio = out_of_roi_ratio(
        pred_xy, map_x_min, map_x_max, map_y_min, map_y_max
    )
    gt_oor_ratio = out_of_roi_ratio(
        gt_xy, map_x_min, map_x_max, map_y_min, map_y_max
    ).detach()
    smooth_loss = smoothness_loss(pred_xy)

    risk_term = lambda_risk * risk_rel_loss
    oor_term = lambda_oor * oor_loss
    invalid_term = lambda_invalid * invalid_rel_loss
    smooth_term = lambda_smooth * smooth_loss
    total = risk_term + oor_term + invalid_term + smooth_term

    logs = {
        "ptc/risk_rel_loss": risk_rel_loss.detach(),
        "ptc/oor_loss": oor_loss.detach(),
        "ptc/invalid_rel_loss": invalid_rel_loss.detach(),
        "ptc/smooth_loss": smooth_loss.detach(),
        "ptc/risk_term": risk_term.detach(),
        "ptc/oor_term": oor_term.detach(),
        "ptc/invalid_term": invalid_term.detach(),
        "ptc/smooth_term": smooth_term.detach(),
        "ptc/pred_risk_cost": pred_cost.mean().detach(),
        "ptc/gt_risk_cost": gt_cost.mean().detach(),
        "ptc/risk_gap": risk_gap.mean().detach(),
        "ptc/pred_valid_ratio": pred_valid_ratio.mean().detach(),
        "ptc/gt_valid_ratio": gt_valid_ratio.mean().detach(),
        "ptc/valid_gap": valid_gap.mean().detach(),
        "ptc/pred_valid_sum": pred_valid_sum.mean().detach(),
        "ptc/gt_valid_sum": gt_valid_sum.mean().detach(),
        "ptc/pred_oor_dist": pred_oor_dist.mean().detach(),
        "ptc/gt_oor_dist": gt_oor_dist.mean().detach(),
        "ptc/pred_oor_ratio": pred_oor_ratio.detach(),
        "ptc/gt_oor_ratio": gt_oor_ratio.detach(),
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
    coordinate = str(_cfg_value(sampling, "coordinate", "cell_center"))
    sampling_mode = str(_cfg_value(sampling, "mode", "bilinear"))
    align_corners = bool(_cfg_value(sampling, "align_corners", True))

    if coordinate not in {"cell_center", "index_aligned"}:
        raise ValueError(
            "PTC sampling.coordinate must be 'cell_center' "
            f"(or legacy 'index_aligned'), got {coordinate!r}"
        )
    if sampling_mode != "bilinear":
        raise ValueError(f"PTC sampling.mode must be 'bilinear', got {sampling_mode!r}")
    if not align_corners:
        raise ValueError("PTC sampling.align_corners must be true for cell-center sampling")

    # Backward compatibility: old configs used margin. Its actual semantics are
    # a safety margin, not a tolerance.
    safety_margin = _cfg_value(config, "safety_margin", None)
    if safety_margin is None:
        safety_margin = _cfg_value(config, "margin", 0.0)

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
        lambda_invalid=float(_cfg_value(config, "lambda_invalid", 0.005)),
        lambda_smooth=float(_cfg_value(config, "lambda_smooth", 0.002)),
        safety_margin=float(safety_margin),
        valid_margin=float(_cfg_value(config, "valid_margin", 0.0)),
        oor_mode=str(_cfg_value(config, "oor_mode", "relative")),
        flip_x_axis=bool(_cfg_value(config, "flip_x_axis", False)),
        flip_y_axis=bool(_cfg_value(config, "flip_y_axis", False)),
        swap_xy=bool(_cfg_value(config, "swap_xy", False)),
        sampling_mode=sampling_mode,
        align_corners=align_corners,
    )
