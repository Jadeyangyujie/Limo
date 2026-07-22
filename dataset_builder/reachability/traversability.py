from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from dataset_builder.mppi_planner.mppi_planner import max_pool


@dataclass
class TraversabilityResult:
    score: np.ndarray
    risk: np.ndarray
    reachability_risk: np.ndarray
    known_raw: np.ndarray
    known_trav: np.ndarray
    known_reachability: np.ndarray
    trav_cost: np.ndarray


def _fill_unknown_with_nearest(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill unknown context without changing the center-cell known mask.

    The traversability CNN has a three-cell receptive-field radius. Passing
    NaNs into its convolutions makes every nearby output NaN, which silently
    behaves like unknown-mask dilation. Nearest-neighbour filling is used only
    as CNN context; the original center-cell unknown mask is restored on the
    output risk map.
    """
    elevation_array = np.asarray(elevation, dtype=np.float32)
    if elevation_array.ndim != 2:
        raise ValueError(
            f"elevation must be a 2-D array, got shape {elevation_array.shape}"
        )
    known_center = np.isfinite(elevation_array)
    if known_center.all():
        return elevation_array, known_center
    if not known_center.any():
        return np.zeros_like(elevation_array), known_center

    nearest_indices = ndimage.distance_transform_edt(
        ~known_center,
        return_distances=False,
        return_indices=True,
    )
    filled = elevation_array[tuple(nearest_indices)]
    return filled.astype(np.float32, copy=False), known_center


@torch.no_grad()
def compute_reachability_risk(
    elevation: np.ndarray,
    filter_model: torch.nn.Module,
    *,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Compute center-defined reachability risk without unknown dilation.

    Unknown elevation is nearest-filled only while evaluating the CNN. The
    map boundary receives three replicated context cells to avoid the model's
    native zero-padding cliff. After inference, precisely the original
    unknown center cells are restored to NaN.
    """
    filled_elevation, known_center = _fill_unknown_with_nearest(elevation)
    if not known_center.any():
        return np.full(known_center.shape, np.nan, dtype=np.float32)

    dev = torch.device(device)
    model = filter_model.to(dev).eval()
    elev = torch.as_tensor(filled_elevation, dtype=torch.float32, device=dev)

    context = 3
    padded_elev = F.pad(
        elev.unsqueeze(0).unsqueeze(0),
        (context, context, context, context),
        mode="replicate",
    ).squeeze(0)
    score_t = model(padded_elev)[context:-context, context:-context]
    risk_t = 1.0 - score_t

    risk = risk_t.detach().cpu().numpy().astype(np.float32, copy=False)
    risk[~known_center] = np.nan
    return risk


@torch.no_grad()
def compute_limo_traversability(
    elevation: np.ndarray,
    filter_model: torch.nn.Module,
    cfg,
    device: str | torch.device = "cpu",
) -> TraversabilityResult:
    """Reproduce the current MPPI traversability preprocessing.

    ``score`` is the raw TraversabilityFilter output (higher is safer).
    ``risk`` is ``1 - score`` after the same optional fatal-cell max pooling
    and border invalidation used by MPPIObjective._compute_traversability.
    ``reachability_risk`` uses nearest-filled unknown context plus replicated
    map-boundary context, then restores only original unknown center cells to
    NaN. It skips artificial MPPI border invalidation, avoiding both implicit
    unknown dilation and false rectangular edge rims.
    ``trav_cost`` exactly reproduces the MPPI thresholded output.
    """
    dev = torch.device(device)
    elev = torch.as_tensor(elevation, dtype=torch.float32, device=dev)
    model = filter_model.to(dev).eval()

    score_t = model(elev.unsqueeze(0))
    risk_t = 1.0 - score_t
    original_nan = torch.isnan(risk_t)

    risk_t = risk_t.clone()
    risk_t[original_nan] = 0.0
    risk_t = max_pool(risk_t, int(cfg.fatal_cells_buffer))
    risk_t[original_nan] = torch.nan
    # MPPI's original score/risk path above remains unchanged.
    reachability_risk = compute_reachability_risk(
        elevation,
        model,
        device=dev,
    )

    border = int(cfg.border_cells)
    if border > 0:
        risk_t[:, -border:] = torch.nan
        risk_t[:, :border] = torch.nan
        risk_t[-border:, :] = torch.nan
        risk_t[:border, :] = torch.nan

    cost_t = risk_t.clone()
    finite = torch.isfinite(cost_t)
    values = cost_t[finite]
    original = values.clone()
    slope = float(cfg.risky_value) / (float(cfg.risky_th) - float(cfg.safe_th))
    values[original < float(cfg.safe_th)] = 0.0
    cautious = (original >= float(cfg.safe_th)) & (original <= float(cfg.risky_th))
    values[cautious] = (original[cautious] - float(cfg.safe_th)) * slope
    values[(original > float(cfg.risky_th)) & (original < float(cfg.fatal_th))] = float(
        cfg.risky_value
    )
    values[original >= float(cfg.fatal_th)] = float(cfg.fatal_value)
    cost_t[finite] = values

    score = score_t.detach().cpu().numpy().astype(np.float32, copy=False)
    risk = risk_t.detach().cpu().numpy().astype(np.float32, copy=False)
    cost = cost_t.detach().cpu().numpy().astype(np.float32, copy=False)
    known_raw = np.isfinite(elevation)
    known_trav = np.isfinite(risk)
    known_reachability = known_raw & np.isfinite(reachability_risk)
    return TraversabilityResult(
        score=score,
        risk=risk,
        reachability_risk=reachability_risk,
        known_raw=known_raw,
        known_trav=known_trav,
        known_reachability=known_reachability,
        trav_cost=cost,
    )
