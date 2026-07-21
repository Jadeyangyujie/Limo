from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from dataset_builder.mppi_planner.mppi_planner import max_pool


@dataclass
class TraversabilityResult:
    score: np.ndarray
    risk: np.ndarray
    known_raw: np.ndarray
    known_trav: np.ndarray
    trav_cost: np.ndarray


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
    return TraversabilityResult(
        score=score,
        risk=risk,
        known_raw=known_raw,
        known_trav=known_trav,
        trav_cost=cost,
    )
