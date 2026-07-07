#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import site
import sys
from pathlib import Path


MODULES = [
    "torch",
    "torchvision",
    "lightning",
    "hydra",
    "omegaconf",
    "torchmetrics",
    "safetensors",
    "wandb",
    "zarr",
    "PIL",
    "cv2",
    "yaml",
    "numpy",
    "matplotlib",
    "tqdm",
    "huggingface_hub",
    "FastGeodis",
]


def check_imports() -> None:
    missing = [name for name in MODULES if importlib.util.find_spec(name) is None]
    if missing:
        raise SystemExit(f"Missing Python modules: {', '.join(missing)}")
    print("Python imports: OK")


def check_user_site() -> None:
    user_paths = [p for p in sys.path if ".local" in p]
    print(f"User site enabled: {site.ENABLE_USER_SITE}")
    if user_paths:
        raise SystemExit("User site packages are visible; source scripts/limo_env.sh first.")


def check_torch() -> None:
    import torch

    print(f"Python: {sys.executable}")
    print(f"Torch: {torch.__version__}, CUDA build: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}, devices: {torch.cuda.device_count()}")


def check_fastgeodis() -> None:
    import FastGeodis
    import torch

    image = torch.zeros(1, 1, 32, 32, dtype=torch.float32)
    mask = torch.ones_like(image)
    mask[..., 16, 16] = 0
    out = FastGeodis.generalised_geodesic2d(image, mask, 1e10, 0.5, 2)
    if out.shape != image.shape or not torch.isfinite(out).all():
        raise SystemExit("FastGeodis smoke test failed.")
    print("FastGeodis: OK")


def check_repo_imports() -> None:
    import dataset_builder.src.build_paths  # noqa: F401
    import limo.src.train  # noqa: F401

    print("Repo imports: OK")


def check_model() -> None:
    from limo.src.models.components.limo_net import LimoNet

    model = LimoNet()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LimoNet: OK, total params={total}, trainable params={trainable}")


def check_dataset_builder_planner(repo_root: Path) -> None:
    import torch
    from omegaconf import OmegaConf

    from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIPlanner

    cfg = OmegaConf.load(repo_root / "dataset_builder" / "configs" / "build.yaml")
    cfg.mppi.population_size = 32
    cfg.mppi.num_iterations = 2
    cfg.mppi.horizon = 12

    n_cells = int(cfg.map_size * 2 / cfg.map_resolution)
    elevation = torch.zeros(n_cells, n_cells, dtype=torch.float32)
    origin = torch.tensor([-cfg.map_size, -cfg.map_size], dtype=torch.float32)
    gridmap = GridMap2D(
        elevation=elevation,
        resolution=float(cfg.map_resolution),
        origin_xy=origin,
    )

    planner = MPPIPlanner(cfg.mppi, "cpu")
    start = torch.zeros(3, dtype=torch.float32)
    goal = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
    states = planner.plan(gridmap, start, goal)

    expected = (int(cfg.mppi.horizon), 3)
    if tuple(states.shape) != expected or not torch.isfinite(states).all():
        raise SystemExit("Dataset builder planner smoke test failed.")
    print(f"Dataset builder planner: OK, states shape={tuple(states.shape)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-model", action="store_true", help="skip DINO/LimoNet load")
    parser.add_argument(
        "--planner-smoke",
        action="store_true",
        help="run a small CPU MPPI/FastGeodis dataset_builder smoke test",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    print(f"Repo root: {repo_root}")

    check_user_site()
    check_imports()
    check_torch()
    check_fastgeodis()
    check_repo_imports()
    if not args.skip_model:
        check_model()
    if args.planner_smoke:
        check_dataset_builder_planner(repo_root)
    print("Environment check: OK")


if __name__ == "__main__":
    main()
