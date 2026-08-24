from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch import nn

from limo.src.models.components.relative_limo_net import RelativeLimoNet
from limo.src.models.components.se2_dynamics import (
    body_motion_to_path,
    path_to_body_motion,
    wrap_angle,
)
from limo.src.models.limo_model import LimoModel
from limo.src.models.structured_trajectory_loss import StructuredTrajectoryLoss


DT = 0.1
HORIZON = 50


def _constant_motion(vx: float, vy: float, wz: float) -> torch.Tensor:
    motion = torch.tensor([vx, vy, wz], dtype=torch.float32)
    return motion.view(1, 1, 3).expand(2, HORIZON, 3).clone()


@pytest.mark.parametrize(
    "motion",
    [
        _constant_motion(0.8, 0.0, 0.0),  # straight
        _constant_motion(0.0, 0.2, 0.0),  # lateral
        _constant_motion(0.0, 0.0, 0.6),  # rotation
        _constant_motion(0.8, 0.0, 0.6),  # arc
        _constant_motion(0.0, 0.0, 0.0),  # stationary
        _constant_motion(0.2, 0.0, 1.0),  # crosses +pi and wraps to -pi
        _constant_motion(0.2, 0.0, -1.0),  # crosses -pi and wraps to +pi
    ],
    ids=[
        "straight",
        "lateral",
        "rotation",
        "arc",
        "stationary",
        "positive_pi_crossing",
        "negative_pi_crossing",
    ],
)
def test_path_motion_path_round_trip(motion: torch.Tensor) -> None:
    path = body_motion_to_path(motion, DT)
    recovered_motion = path_to_body_motion(path, DT)
    recovered_path = body_motion_to_path(recovered_motion, DT)

    xy_error = (recovered_path[..., :2] - path[..., :2]).abs().max()
    yaw_error = wrap_angle(recovered_path[..., 2] - path[..., 2]).abs().max()
    assert xy_error.item() < 1e-5
    assert yaw_error.item() < 1e-5
    assert (recovered_motion - motion).abs().max().item() < 1e-5


class _LearnableMotionNet(nn.Module):
    def __init__(self, path_length: int = HORIZON) -> None:
        super().__init__()
        self.motion = nn.Parameter(torch.randn(1, path_length, 3) * 0.05)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.motion.expand(batch["goal"].shape[0], -1, -1)


def test_forward_shape_and_single_batch_backward() -> None:
    motion_net = _LearnableMotionNet()
    net = RelativeLimoNet(motion_net=motion_net, dt=DT)
    loss_fn = StructuredTrajectoryLoss(dt=DT)
    batch = {
        "image_front": torch.randn(3, 3, 28, 42),
        "goal": torch.randn(3, 3),
        "path": body_motion_to_path(_constant_motion(0.5, 0.0, 0.1)[:1], DT)
        .expand(3, -1, -1)
        .clone(),
    }

    pred_path, pred_motion = net.forward_with_motion(batch)
    assert net(batch).shape == (3, HORIZON, 3)
    assert pred_path.shape == (3, HORIZON, 3)

    loss = loss_fn(pred_path, batch["path"], pred_motion)
    loss.backward()
    assert torch.isfinite(loss)
    assert motion_net.motion.grad is not None
    assert torch.isfinite(motion_net.motion.grad).all()


def test_exact_teacher_prediction_has_near_zero_loss() -> None:
    loss_fn = StructuredTrajectoryLoss(dt=DT)
    target_motion = _constant_motion(0.6, 0.1, 0.3)
    target_path = body_motion_to_path(target_motion, DT)
    components = loss_fn.compute_components(
        target_path, target_path, target_motion
    )

    assert components["total"].item() < 1e-7
    for name, value in components.items():
        assert value.item() < 1e-7, name


def test_shortened_path_increases_endpoint_and_progress_losses() -> None:
    loss_fn = StructuredTrajectoryLoss(dt=DT)
    target_motion = _constant_motion(0.8, 0.0, 0.0)
    target_path = body_motion_to_path(target_motion, DT)
    exact = loss_fn.compute_components(target_path, target_path, target_motion)

    short_motion = target_motion * 0.5
    short_path = body_motion_to_path(short_motion, DT)
    shortened = loss_fn.compute_components(short_path, target_path, short_motion)

    assert shortened["endpoint"] > exact["endpoint"]
    assert shortened["progress"] > exact["progress"]
    assert shortened["endpoint"].item() > 0.0
    assert shortened["progress"].item() > 0.0


def test_original_limo_config_still_instantiates() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_dir = repo_root / "limo" / "configs"
    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="train",
            overrides=[f"paths.root_dir={repo_root}", "model=limo"],
        )
    model = instantiate(cfg.model)
    assert isinstance(model, LimoModel)
    assert model.net.path_length == HORIZON
