"""Structured supervision for relative-motion trajectory prediction."""

from typing import Dict, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from limo.src.models.components.se2_dynamics import path_to_body_motion


class StructuredTrajectoryLoss(nn.Module):
    def __init__(
        self,
        dt: float = 0.1,
        xy_weight: float = 1.0,
        yaw_weight: float = 0.2,
        motion_weight: float = 0.5,
        endpoint_weight: float = 2.0,
        progress_weight: float = 0.5,
        smooth_weight: float = 0.1,
        motion_scales: Sequence[float] = (1.0, 0.2, 0.8),
        progress_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if len(motion_scales) != 3 or any(scale <= 0 for scale in motion_scales):
            raise ValueError("motion_scales must contain three positive values")

        self.dt = dt
        self.xy_weight = xy_weight
        self.yaw_weight = yaw_weight
        self.motion_weight = motion_weight
        self.endpoint_weight = endpoint_weight
        self.progress_weight = progress_weight
        self.smooth_weight = smooth_weight
        self.progress_eps = progress_eps
        self.register_buffer(
            "motion_scales", torch.tensor(motion_scales, dtype=torch.float32)
        )

    @staticmethod
    def _cumulative_xy_distance(path: torch.Tensor) -> torch.Tensor:
        origin = torch.zeros_like(path[:, :1, :2])
        points = torch.cat([origin, path[..., :2]], dim=1)
        segment_lengths = torch.linalg.vector_norm(
            points[:, 1:] - points[:, :-1], dim=-1
        )
        return torch.cumsum(segment_lengths, dim=1)

    def compute_components(
        self,
        pred_path: torch.Tensor,
        target_path: torch.Tensor,
        pred_motion: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if pred_path.shape != target_path.shape:
            raise ValueError(
                "pred_path and target_path must have the same shape, "
                f"got {tuple(pred_path.shape)} and {tuple(target_path.shape)}"
            )
        if pred_motion.shape != target_path.shape:
            raise ValueError(
                "pred_motion and target_path must have the same shape, "
                f"got {tuple(pred_motion.shape)} and {tuple(target_path.shape)}"
            )

        target_motion = path_to_body_motion(target_path, self.dt)
        scales = self.motion_scales.to(
            device=pred_motion.device, dtype=pred_motion.dtype
        )

        xy = F.smooth_l1_loss(pred_path[..., :2], target_path[..., :2])
        yaw = (1.0 - torch.cos(pred_path[..., 2] - target_path[..., 2])).mean()
        motion = F.smooth_l1_loss(
            pred_motion / scales, target_motion / scales
        )
        endpoint = F.smooth_l1_loss(
            pred_path[:, -1, :2], target_path[:, -1, :2]
        )

        pred_progress = self._cumulative_xy_distance(pred_path)
        target_progress = self._cumulative_xy_distance(target_path)
        teacher_total = target_progress[:, -1:].clamp_min(self.progress_eps)
        progress = F.smooth_l1_loss(
            pred_progress / teacher_total, target_progress / teacher_total
        )

        if pred_motion.shape[1] > 1:
            pred_change = pred_motion[:, 1:] - pred_motion[:, :-1]
            target_change = target_motion[:, 1:] - target_motion[:, :-1]
            smooth = F.smooth_l1_loss(
                pred_change / scales, target_change / scales
            )
        else:
            smooth = pred_motion.sum() * 0.0

        total = (
            self.xy_weight * xy
            + self.yaw_weight * yaw
            + self.motion_weight * motion
            + self.endpoint_weight * endpoint
            + self.progress_weight * progress
            + self.smooth_weight * smooth
        )
        return {
            "total": total,
            "xy": xy,
            "yaw": yaw,
            "motion": motion,
            "endpoint": endpoint,
            "progress": progress,
            "smooth": smooth,
        }

    def forward(
        self,
        pred_path: torch.Tensor,
        target_path: torch.Tensor,
        pred_motion: torch.Tensor,
    ) -> torch.Tensor:
        return self.compute_components(pred_path, target_path, pred_motion)["total"]
