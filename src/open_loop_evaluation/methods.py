from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .common import ALGORITHM_ROOT, EVALUATION_ROOT
from .data import MissionStore, Sample
from .metrics import OpenLoopPathMetric
from .models import LimoPredictor

from dataset_builder.mppi_planner.mppi_planner import GridMap2D, MPPIPlanner  # noqa: E402

# 5.0 in paper, but we allow longer straight-line paths here since we evaluate them in a more forgiving way (see `evaluate_path` in metrics.py)
STRAIGHT_LINE_MAX_LENGTH_M = 25.0 

"""“方法注册 + 路径生成”模块
    当前有哪些可评测方法？
    每个方法需要什么资源？
    给一个 sample 这个方法怎么产出一条轨迹 path
    ---> 后面的 metrics.py 会拿这里生成的 path 去算 collision / reach / success / SPL"""
# 把 LiMO、直线路径、真实轨迹、几何规划器
# 统一包装成同一种接口：
# Sample -> path
"""cli.py
  用户选择 methods
      |
      v
available_methods()
  得到 MethodSpec
      |
      v
evaluate.py
  创建 MethodRunner
      |
      v
MethodRunner.predict_many()
  生成 paths
      |
      v
metrics.py
  计算 collision / reach / success / SPL"""

@dataclass(frozen=True)
class MethodSpec:
    """MethodSpec 是一个“方法说明书”"""
    id: str       # 程序内部用的唯一 ID。例如： "limo_D_tel"、"straight_line"、"real_world"、"geometric_planner"
    label: str    # 用于展示的名字。例如： "Trained on D_TEL"、"Straight-Line Paths"、"Real-World Paths"、"Geometric Planner"
    kind: str     # 方法类型，决定了这个方法是怎么生成路径的。例如： "limo"、"straight_line"、"real_world"、"geometric_planner"
    weights_path: Path | None = None  # LiMO 模型才需要权重路径。直线、真实轨迹、几何规划器都不需要 LiMO 权重，所以默认是 None
    model_arch: str = "front"  # LiMO 模型结构。front=只用前视相机；side_cams=三相机；side_cams_history_depth=三相机+前视历史
    image_subdir: str = "images"  # mission 下的图像目录；去畸变模型使用 images_undistorted_308x476
    side_cam_use_nearest: bool = False  # 按 front timestamp 为左右相机找最近同步帧
    side_cam_time_tolerance_s: float = 0.02
    front_history_size: int = 0
    front_history_stride: int = 1
    front_history_pose_yaw_offset_rad: float = -np.pi / 2
    front_history_pose_tolerance_s: float | None = None
    allowed_datasets: tuple[str, ...] = ("D_TEL", "D_GEO", "D_AUG") # 这个方法适用于哪些数据集？默认都适用，但 "real_world" 只适用于 "D_TEL"


def available_methods(
    bev_limo_a_weights: Path | None = None,
    ptc_limo_weights: Path | None = None,
    limo_d_tel_weights: Path | None = None,
    limo_d_aug_weights: Path | None = None,
    limo_side_cams_weights: Path | None = None,
    limo_image_subdir: str | None = None,
    ptc_limo_image_subdir: str | None = None,
    limo_d_tel_image_subdir: str | None = None,
    limo_d_aug_image_subdir: str | None = None,
    limo_side_cams_image_subdir: str | None = None,
) -> dict[str, MethodSpec]:
    """返回一个字典，列出当前系统支持的所有 method"""
    weights_root = EVALUATION_ROOT / "weights" / "LIMO"
    default_limo_image_subdir = "images_undistorted_308x476"
    limo_d_tel_image_subdir = (
        limo_d_tel_image_subdir or limo_image_subdir or default_limo_image_subdir
    )
    limo_d_aug_image_subdir = (
        limo_d_aug_image_subdir or limo_image_subdir or default_limo_image_subdir
    )
    limo_side_cams_image_subdir = (
        limo_side_cams_image_subdir or limo_image_subdir or default_limo_image_subdir
    )
    ptc_limo_image_subdir = ptc_limo_image_subdir or limo_image_subdir or "images"
    default_bev_limo_a_weights = (
        ALGORITHM_ROOT
        / "logs"
        / "train"
        / "runs"
        / "2026-06-25_21-29-28"
        / "weights"
        / "last.safetensors"
    )
    return {
        "bev_limo_a": MethodSpec(
            id="bev_limo_a",
            label="BEV-LIMO-A",
            kind="limo",
            weights_path=bev_limo_a_weights or default_bev_limo_a_weights,
            model_arch="bev_limo_a",
            image_subdir="images",
        ),
        "ptc_limo": MethodSpec(
            id="ptc_limo",
            label="PTC-LIMO",
            kind="limo",
            weights_path=(
                ptc_limo_weights
                or ALGORITHM_ROOT
                / "logs"
                / "train"
                / "runs"
                / "2026-07-02_19-17-34"
                / "weights"
                / "last.safetensors"
            ),
            model_arch="front",
            image_subdir=ptc_limo_image_subdir,
        ),
        "limo_D_tel": MethodSpec(
            id="limo_D_tel",
            label="Trained on D_TEL",
            kind="limo", # 用 D_TEL 训练的模型
            weights_path=(
                limo_d_tel_weights
                or weights_root / "limo_trained_on_D_tel.safetensors"
            ),
            image_subdir=limo_d_tel_image_subdir,
        ),
        "limo_D_aug": MethodSpec(
            id="limo_D_aug",
            label="Trained on D_AUG",
            kind="limo", # 用 D_AUG 训练的权重
            weights_path=(
                limo_d_aug_weights
                or weights_root / "limo_trained_on_D_aug.safetensors"
            ),
            image_subdir=limo_d_aug_image_subdir,
        ),
        "limo_side_cams": MethodSpec(
            id="limo_side_cams",
            label="LiMO Side Cams",
            kind="limo", # 使用前/左/右三路相机输入的 LiMO 权重
            weights_path=(
                limo_side_cams_weights
                or weights_root / "limo_with_side_cams.safetensors"
            ),
            model_arch="side_cams",
            image_subdir=limo_side_cams_image_subdir,
        ),
        "limo_side_cams_undistorted": MethodSpec(
            id="limo_side_cams_undistorted",
            label="LiMO Side Cams (Undistorted)",
            kind="limo",
            weights_path=weights_root / "grandtour_limo_side_cams_D_aug_undistort.safetensors",
            model_arch="side_cams",
            image_subdir="images_undistorted_308x476",
            allowed_datasets=("D_TEL", "D_AUG"),
        ),
        "limo_side_cams_sync_history_depth": MethodSpec(
            id="limo_side_cams_sync_history_depth",
            label="LiMO Side Cams Sync + Front History Depth",
            kind="limo",
            weights_path=(
                weights_root
                / "grandtour_limo_side_cams_sync_D_aug_front_history_depth.safetensors"
            ),
            model_arch="side_cams_history_depth",
            image_subdir="images_undistorted_308x476",
            side_cam_use_nearest=True,
            side_cam_time_tolerance_s=0.1, # 0.02
            front_history_size=4,
            front_history_stride=2,
            front_history_pose_yaw_offset_rad=0.0, # -np.pi / 2
            front_history_pose_tolerance_s=0.1,
            allowed_datasets=("D_TEL", "D_AUG"),
        ),
        "straight_line": MethodSpec(
            id="straight_line", # 直线路径 baseline。它不看图像、不看地图、不避障，只朝目标方向插值一条直线
            label="Straight-Line Paths",
            kind="straight_line",
        ),
        "real_world": MethodSpec(
            id="real_world", # 真实路径 baseline。它直接使用数据集里的 teleop reference path
            label="Real-World Paths",
            kind="real_world",
            allowed_datasets=("D_TEL",),
        ),
        "geometric_planner": MethodSpec(
            id="geometric_planner", # 几何规划器 baseline。它会读取 elevation map，然后用 MPPI planner 重新规划路径
            label="Geometric Planner",
            kind="geometric_planner",
        ),
    }


def straight_line_path(
    goal: np.ndarray,
    horizon: int = 50,
    max_length_m: float = STRAIGHT_LINE_MAX_LENGTH_M,
) -> np.ndarray:
    """输入一个目标位姿，输出一条直线路径：不看障碍物，所以可能直接穿过障碍，后面由 metrics 判断碰撞"""
    goal = np.asarray(goal, dtype=np.float32).copy()
    # 计算目标点到原点的平面距离
    dist = float(np.linalg.norm(goal[:2]))
    # 如果目标太远，就把目标的 x, y 缩放到最大距离内
    if dist > max_length_m and dist > 1e-6:
        scale = max_length_m / dist
        goal[0] *= scale
        goal[1] *= scale
    # 生成 50 个插值比例
    t = np.linspace(1.0 / horizon, 1.0, horizon, dtype=np.float32)
    path = np.zeros((horizon, 3), dtype=np.float32)
    path[:, 0] = goal[0] * t
    path[:, 1] = goal[1] * t
    path[:, 2] = goal[2] * t
    return path


class GeometricPlannerPredictor:
    """这个类包装了 MPPI 几何规划器。作用是：给一个 sample 根据地图和 goal 规划一条路径"""
    def __init__(self, metric: OpenLoopPathMetric, device: str = "cuda") -> None:
        self.metric = metric # metric：不是为了算指标，而是借用 metric.map_resolution 和 metric.origin_xy
        self.device = torch.device(device)
        # 与 OpenLoopPathMetric 共用配置，确保 CLI 覆盖的 footprint 同时作用于评测和规划。
        self.planner = MPPIPlanner(metric.mppi_cfg, str(self.device))
        self.start = torch.zeros(3, dtype=torch.float32, device=self.device)

    def predict(self, store: MissionStore, sample: Sample) -> np.ndarray:
        # Share the evaluator's elevation transform with D_TEL preparation and
        # the geometric baseline, so planning and collision use one frame.
        elevation = self.metric.elevation_for_sample(store, sample)
        gridmap = GridMap2D(
            elevation=torch.from_numpy(elevation).to(self.device),
            resolution=self.metric.map_resolution,
            origin_xy=self.metric.origin_xy,
        )
        goal = torch.tensor(sample.goal, dtype=torch.float32, device=self.device)
        return self.planner.plan(gridmap, self.start, goal).cpu().numpy().astype(np.float32)


class MethodRunner:
    """这个类把所有不同 kind 的方法统一起来。外部评测代码不需要写一堆 if method == ...，只要创建一个 runner然后调用 runner.predict_many(...)"""
    def __init__(
        self,
        spec: MethodSpec,
        dataset_root: Path,
        metric: OpenLoopPathMetric,
        device: str,
        torch_hub_dir: Path,
        batch_size: int,
    ) -> None:
        self.spec = spec
        self.metric = metric
        self.batch_size = batch_size
        self.limo: LimoPredictor | None = None
        self.geometric: GeometricPlannerPredictor | None = None
        if spec.kind == "limo":
            assert spec.weights_path is not None
            self.limo = LimoPredictor(
                weights_path=spec.weights_path,
                dataset_root=dataset_root,
                device=device,
                torch_hub_dir=torch_hub_dir,
                model_arch=spec.model_arch,
                image_subdir=spec.image_subdir,
                side_cam_use_nearest=spec.side_cam_use_nearest,
                side_cam_time_tolerance_s=spec.side_cam_time_tolerance_s,
                front_history_size=spec.front_history_size,
                front_history_stride=spec.front_history_stride,
                front_history_pose_yaw_offset_rad=spec.front_history_pose_yaw_offset_rad,
                front_history_pose_tolerance_s=spec.front_history_pose_tolerance_s,
            )
        elif spec.kind == "geometric_planner":
            self.geometric = GeometricPlannerPredictor(metric, device=device)

    def predict_many(self, store_by_mission: dict[str, MissionStore], samples: list[Sample]) -> list[np.ndarray]:
        """批量预测 输入一批 samples 返回一批 paths"""
        if self.spec.kind == "limo":
            assert self.limo is not None
            return self.limo.predict_batch(samples)
        paths = []
        for sample in samples:
            store = store_by_mission[sample.mission]
            paths.append(self.predict_one(store, sample))
        return paths

    def select_samples(self, samples: list[Sample]) -> tuple[list[Sample], list[dict[str, object]]]:
        """返回符合某个方法输入约束的样本，以及被排除样本的原因。"""
        if self.spec.kind != "limo":
            return samples, []
        assert self.limo is not None
        return self.limo.select_samples(samples)

    def predict_one(self, store: MissionStore, sample: Sample) -> np.ndarray:
        """单样本预测 输入一个 sample 返回一条 path"""
        if self.spec.kind == "straight_line":
            return straight_line_path(sample.goal)
        if self.spec.kind == "real_world":
            if sample.subset != "tel_eval":
                raise ValueError("Real-World Paths are only defined for prepared D_TEL samples.")
            return sample.reference_path
        if self.spec.kind == "geometric_planner":
            if sample.subset == "tel_eval" and store.has_group("tel_planner"):
                return np.asarray(store.groups["tel_planner"]["path"][sample.index], dtype=np.float32)
            if sample.subset == "geo_eval":
                return sample.reference_path
            assert self.geometric is not None
            return self.geometric.predict(store, sample)
        raise ValueError(f"Use predict_many for method kind {self.spec.kind}")
