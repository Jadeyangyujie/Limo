"""Differentiable conversions between local SE(2) paths and body motions."""

import torch


def wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] without breaking autograd."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _initial_pose_like(sequence: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        sequence.shape[0], 1, 3, device=sequence.device, dtype=sequence.dtype
    )


def path_to_body_motion(path: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Convert local-frame SE(2) waypoints to pre-step body-frame velocities.

    The current robot pose is the implicit origin [0, 0, 0]. Thus an N-point
    future path maps to N commands, including the command from the current pose
    to the first waypoint.
    """
    if path.ndim != 3 or path.shape[-1] != 3:
        raise ValueError(f"path must have shape [B, N, 3], got {tuple(path.shape)}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    previous = torch.cat([_initial_pose_like(path), path[:, :-1]], dim=1)
    delta_xy = path[..., :2] - previous[..., :2]
    previous_yaw = previous[..., 2]
    cos_yaw = torch.cos(previous_yaw)
    sin_yaw = torch.sin(previous_yaw)

    vx = (cos_yaw * delta_xy[..., 0] + sin_yaw * delta_xy[..., 1]) / dt
    vy = (-sin_yaw * delta_xy[..., 0] + cos_yaw * delta_xy[..., 1]) / dt
    wz = wrap_angle(path[..., 2] - previous_yaw) / dt
    return torch.stack([vx, vy, wz], dim=-1)


def body_motion_to_path(motion: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Integrate body-frame velocities into local SE(2) waypoints.

    Translation at step t is rotated by the yaw before applying that step's
    angular velocity, matching the teacher MPPI rollout.
    """
    if motion.ndim != 3 or motion.shape[-1] != 3:
        raise ValueError(
            f"motion must have shape [B, N, 3], got {tuple(motion.shape)}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    yaw_delta = motion[..., 2] * dt
    yaw_unwrapped = torch.cumsum(yaw_delta, dim=1)
    pre_step_yaw = torch.cat(
        [torch.zeros_like(yaw_unwrapped[:, :1]), yaw_unwrapped[:, :-1]], dim=1
    )

    body_delta = motion[..., :2] * dt
    cos_yaw = torch.cos(pre_step_yaw)
    sin_yaw = torch.sin(pre_step_yaw)
    delta_x = cos_yaw * body_delta[..., 0] - sin_yaw * body_delta[..., 1]
    delta_y = sin_yaw * body_delta[..., 0] + cos_yaw * body_delta[..., 1]
    xy = torch.cumsum(torch.stack([delta_x, delta_y], dim=-1), dim=1)

    return torch.cat([xy, wrap_angle(yaw_unwrapped).unsqueeze(-1)], dim=-1)
