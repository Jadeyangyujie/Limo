"""Relative-motion LiMo network with an SE(2) integration output adapter."""

from typing import Dict, Tuple

import torch
from torch import nn

from limo.src.models.components.se2_dynamics import body_motion_to_path


class RelativeLimoNet(nn.Module):
    """Interpret a LiMo decoder's outputs as [vx, vy, wz] commands."""

    def __init__(self, motion_net: nn.Module, dt: float = 0.1) -> None:
        super().__init__()
        self.motion_net = motion_net
        self.dt = dt

    def setup(self) -> None:
        setup = getattr(self.motion_net, "setup", None)
        if setup is not None:
            setup()

    def predict_motion(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        motion = self.motion_net(batch)
        if motion.ndim != 3 or motion.shape[-1] != 3:
            raise ValueError(
                "motion_net must return [B, N, 3], "
                f"got {tuple(motion.shape)}"
            )
        return motion

    def forward_with_motion(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        motion = self.predict_motion(batch)
        return body_motion_to_path(motion, self.dt), motion

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        path, _ = self.forward_with_motion(batch)
        return path
