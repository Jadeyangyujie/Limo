from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import zarr
from PIL import Image
from safetensors.torch import load_file
from torchvision import transforms

from .common import ALGORITHM_ROOT as _ALGORITHM_ROOT  # noqa: F401
from .data import Sample

from limo.src.models.components.limo_net import LimoNet as FrontLimoNet  # noqa: E402
from limo.src.models.components.limo_net_side_cams import LimoNet as SideCamsLimoNet  # noqa: E402

try:
    from limo.src.models.components.bev_limo_net_a import BevLimoNetA  # noqa: E402
except ModuleNotFoundError:  # This branch is absent in the PTC-LIMO checkout.
    BevLimoNetA = None

try:
    from limo.src.models.components.limo_net_side_cams_depth_history import (  # noqa: E402
        LimoNet as SideCamsHistoryDepthLimoNet,
    )
except ModuleNotFoundError:  # This branch is absent in some local LiMO checkouts.
    SideCamsHistoryDepthLimoNet = None


@contextmanager
def local_dinov2_hub(hub_dir: Path):
    """让官方 LimoNet 复用本地 DINOv2 torch hub 缓存，避免评测时访问网络。"""
    hub_repo = hub_dir / "facebookresearch_dinov2_main"
    if not hub_repo.exists():
        raise FileNotFoundError(f"DINOv2 torch hub cache not found: {hub_repo}")

    original_load = torch.hub.load
    old_torch_home = os.environ.get("TORCH_HOME")
    os.environ["TORCH_HOME"] = str(hub_dir.parent)

    def load_from_local(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/dinov2":
            kwargs["source"] = "local"
            kwargs.setdefault("pretrained", False)
            return original_load(str(hub_repo), model, *args, **kwargs)
        return original_load(repo_or_dir, model, *args, **kwargs)

    try:
        with patch("torch.hub.load", side_effect=load_from_local):
            yield
    finally:
        if old_torch_home is None:
            os.environ.pop("TORCH_HOME", None)
        else:
            os.environ["TORCH_HOME"] = old_torch_home


def _strip_prefix_if_present(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    if not state_dict or not all(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key[len(prefix) :]: value for key, value in state_dict.items()}


def load_limo_state_dict(weights_path: Path) -> dict[str, torch.Tensor]:
    """Load LiMO weights from SafeTensors or a Lightning checkpoint.

    PTC-LIMO keeps the original network for inference. Its preferred export is
    weights/*.safetensors containing pl_module.net.state_dict(), while .ckpt
    stores the same tensors under the Lightning state_dict with a "net." prefix.
    """
    suffix = weights_path.suffix.lower()
    if suffix == ".safetensors":
        state_dict = load_file(str(weights_path))
    elif suffix == ".ckpt":
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        if "state_dict" not in checkpoint:
            raise KeyError(f"Lightning checkpoint missing state_dict: {weights_path}")
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError(
            f"Unsupported LiMO weights format: {weights_path}. "
            "Expected .safetensors or .ckpt."
        )

    state_dict = _strip_prefix_if_present(dict(state_dict), "net.")
    state_dict = _strip_prefix_if_present(state_dict, "model.")
    return state_dict


def _nearest_indices(timestamps: np.ndarray, query_timestamps: np.ndarray) -> np.ndarray:
    """与训练 Dataset 完全一致的最近时间戳查找规则。"""
    right = np.searchsorted(timestamps, query_timestamps)
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    use_left = np.abs(query_timestamps - timestamps[left]) <= np.abs(
        timestamps[right] - query_timestamps
    )
    return np.where(use_left, left, right).astype(np.int64)


def _yaw_from_quat_xyzw(quaternions: np.ndarray) -> np.ndarray:
    x, y, z, w = (
        quaternions[:, 0],
        quaternions[:, 1],
        quaternions[:, 2],
        quaternions[:, 3],
    )
    return np.arctan2(
        2.0 * (w * z + x * y),
        1.0 - 2.0 * (y * y + z * z),
    )


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def _relative_se2(current: np.ndarray, history: np.ndarray) -> np.ndarray:
    """训练时的约定：历史帧相对当前帧、并表达在当前 robot frame。"""
    delta_xy_world = history[:2] - current[:2]
    c = np.cos(current[2])
    s = np.sin(current[2])
    return np.asarray(
        [
            c * delta_xy_world[0] + s * delta_xy_world[1],
            -s * delta_xy_world[0] + c * delta_xy_world[1],
            _wrap_angle(history[2] - current[2]),
        ],
        dtype=np.float32,
    )


@dataclass(frozen=True)
class TemporalMetadata:
    """一个 mission 的前视、侧视和 DLIO 同步查找表。"""

    front_timestamps: np.ndarray
    left_timestamps: np.ndarray | None
    right_timestamps: np.ndarray | None
    nearest_left_ids: np.ndarray | None
    nearest_right_ids: np.ndarray | None
    left_valid: np.ndarray | None
    right_valid: np.ndarray | None
    front_pose_valid: np.ndarray | None
    front_pose_se2: np.ndarray | None


class TemporalInputProvider:
    """复用训练集的三相机同步与前视历史输入构造逻辑。"""

    def __init__(
        self,
        dataset_root: Path,
        side_cam_use_nearest: bool,
        side_cam_time_tolerance_s: float,
        front_history_size: int,
        front_history_stride: int,
        front_history_pose_yaw_offset_rad: float,
        front_history_pose_tolerance_s: float | None,
    ) -> None:
        self.dataset_root = dataset_root
        self.side_cam_use_nearest = side_cam_use_nearest
        self.side_cam_time_tolerance_s = side_cam_time_tolerance_s
        self.front_history_size = front_history_size
        self.front_history_stride = front_history_stride
        self.front_history_pose_yaw_offset_rad = front_history_pose_yaw_offset_rad
        self.front_history_pose_tolerance_s = front_history_pose_tolerance_s
        self._metadata_by_mission: dict[str, TemporalMetadata] = {}

    def metadata_for(self, mission: str) -> TemporalMetadata:
        cached = self._metadata_by_mission.get(mission)
        if cached is not None:
            return cached

        mission_data = self.dataset_root / mission / "data"
        front_group = self._open_timestamp_group(mission_data / "hdr_front", mission, "hdr_front")
        front_timestamps = np.asarray(front_group["timestamp"], dtype=np.float64)

        left_timestamps = right_timestamps = None
        nearest_left_ids = nearest_right_ids = None
        left_valid = right_valid = None
        if self.side_cam_use_nearest:
            left_timestamps = self._timestamps_for(mission_data / "hdr_left", mission, "hdr_left")
            right_timestamps = self._timestamps_for(mission_data / "hdr_right", mission, "hdr_right")
            nearest_left_ids, left_valid = self._nearest_side_lookup(
                front_timestamps, left_timestamps
            )
            nearest_right_ids, right_valid = self._nearest_side_lookup(
                front_timestamps, right_timestamps
            )

        front_pose_valid = front_pose_se2 = None
        if self.front_history_size > 0:
            front_pose_valid, front_pose_se2 = self._front_pose_lookup(
                mission_data=mission_data,
                mission=mission,
                front_timestamps=front_timestamps,
            )

        metadata = TemporalMetadata(
            front_timestamps=front_timestamps,
            left_timestamps=left_timestamps,
            right_timestamps=right_timestamps,
            nearest_left_ids=nearest_left_ids,
            nearest_right_ids=nearest_right_ids,
            left_valid=left_valid,
            right_valid=right_valid,
            front_pose_valid=front_pose_valid,
            front_pose_se2=front_pose_se2,
        )
        self._metadata_by_mission[mission] = metadata
        return metadata

    def side_image_id(self, mission: str, camera: str, front_image_id: int) -> int | None:
        if not self.side_cam_use_nearest:
            return front_image_id
        metadata = self.metadata_for(mission)
        if camera == "hdr_left":
            ids, valid = metadata.nearest_left_ids, metadata.left_valid
        elif camera == "hdr_right":
            ids, valid = metadata.nearest_right_ids, metadata.right_valid
        else:
            raise ValueError(f"Unsupported side camera: {camera}")
        if ids is None or valid is None or not 0 <= front_image_id < len(ids):
            return None
        return int(ids[front_image_id]) if bool(valid[front_image_id]) else None

    def history_image_ids(self, current_image_id: int) -> list[int | None]:
        # Training implementation: [id - 6, id - 4, id - 2, id] for size=4, stride=2.
        return [
            image_id if image_id >= 0 else None
            for image_id in (
                current_image_id - offset * self.front_history_stride
                for offset in range(self.front_history_size - 1, -1, -1)
            )
        ]

    def history_pose_inputs(
        self, mission: str, current_image_id: int
    ) -> tuple[np.ndarray, np.ndarray] | None:
        metadata = self.metadata_for(mission)
        if metadata.front_pose_valid is None or metadata.front_pose_se2 is None:
            return None
        if not self._history_poses_are_valid(metadata, current_image_id):
            return None

        history_ids = self.history_image_ids(current_image_id)
        current_timestamp = float(metadata.front_timestamps[current_image_id])
        current_pose = metadata.front_pose_se2[current_image_id]
        dt = np.zeros(self.front_history_size, dtype=np.float32)
        delta_pose = np.zeros((self.front_history_size, 3), dtype=np.float32)
        for slot, image_id in enumerate(history_ids):
            if image_id is None:
                continue
            dt[slot] = float(metadata.front_timestamps[image_id] - current_timestamp)
            delta_pose[slot] = _relative_se2(current_pose, metadata.front_pose_se2[image_id])
        return dt, delta_pose

    def sample_exclusion_reason(self, sample: Sample) -> str | None:
        if self.side_cam_use_nearest:
            for camera in ("hdr_left", "hdr_right"):
                if self.side_image_id(sample.mission, camera, sample.image_id) is None:
                    return f"no synchronized {camera} frame within {self.side_cam_time_tolerance_s:.3f}s"
        if self.front_history_size > 0:
            if self.history_pose_inputs(sample.mission, sample.image_id) is None:
                tolerance = self.front_history_pose_tolerance_s
                tolerance_text = "the configured tolerance" if tolerance is None else f"{tolerance:.3f}s"
                return f"front history pose unavailable within {tolerance_text}"
        return None

    def _open_timestamp_group(self, path: Path, mission: str, name: str) -> zarr.hierarchy.Group:
        if not path.exists():
            raise FileNotFoundError(f"Missing {name} zarr group for {mission}: {path}")
        group = zarr.open_group(str(path), mode="r")
        if "timestamp" not in group:
            raise KeyError(f"Missing timestamp in {name} zarr group for {mission}: {path}")
        return group

    def _timestamps_for(self, path: Path, mission: str, name: str) -> np.ndarray:
        group = self._open_timestamp_group(path, mission, name)
        timestamps = np.asarray(group["timestamp"], dtype=np.float64)
        if len(timestamps) == 0:
            raise ValueError(f"Empty timestamp array for {name} in mission {mission}")
        return timestamps

    def _nearest_side_lookup(
        self, front_timestamps: np.ndarray, side_timestamps: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        nearest_ids = _nearest_indices(side_timestamps, front_timestamps)
        errors = np.abs(side_timestamps[nearest_ids] - front_timestamps)
        return nearest_ids, errors <= self.side_cam_time_tolerance_s

    def _front_pose_lookup(
        self,
        mission_data: Path,
        mission: str,
        front_timestamps: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        path = mission_data / "dlio_map_odometry"
        if not path.exists():
            raise FileNotFoundError(f"Missing DLIO zarr group for {mission}: {path}")
        group = zarr.open_group(str(path), mode="r")
        required = ("timestamp", "pose_pos", "pose_orien")
        missing = [name for name in required if name not in group]
        if missing:
            raise KeyError(f"Missing DLIO fields for {mission}: {', '.join(missing)}")

        dlio_timestamps = np.asarray(group["timestamp"], dtype=np.float64)
        dlio_positions = np.asarray(group["pose_pos"], dtype=np.float32)
        dlio_orientations = np.asarray(group["pose_orien"], dtype=np.float32)
        nearest_ids = _nearest_indices(dlio_timestamps, front_timestamps)
        errors = np.abs(dlio_timestamps[nearest_ids] - front_timestamps)
        if self.front_history_pose_tolerance_s is None:
            valid = np.ones(len(front_timestamps), dtype=bool)
        else:
            valid = errors <= self.front_history_pose_tolerance_s
        yaw = np.unwrap(
            _yaw_from_quat_xyzw(dlio_orientations)
            + self.front_history_pose_yaw_offset_rad
        ).astype(np.float32)
        poses = np.stack(
            [
                dlio_positions[nearest_ids, 0],
                dlio_positions[nearest_ids, 1],
                yaw[nearest_ids],
            ],
            axis=1,
        ).astype(np.float32)
        return valid, poses

    def _history_poses_are_valid(
        self, metadata: TemporalMetadata, current_image_id: int
    ) -> bool:
        valid = metadata.front_pose_valid
        if valid is None or not 0 <= current_image_id < len(valid):
            return False
        if not bool(valid[current_image_id]):
            return False
        return all(
            image_id is None or (image_id < len(valid) and bool(valid[image_id]))
            for image_id in self.history_image_ids(current_image_id)
        )


class LimoPredictor:
    def __init__(
        self,
        weights_path: Path,
        dataset_root: Path,
        device: str,
        torch_hub_dir: Path,
        model_arch: str = "front",
        image_size: tuple[int, int] = (308, 476),
        image_subdir: str = "images",
        side_cam_use_nearest: bool = False,
        side_cam_time_tolerance_s: float = 0.02,
        front_history_size: int = 0,
        front_history_stride: int = 1,
        front_history_pose_yaw_offset_rad: float = -np.pi / 2,
        front_history_pose_tolerance_s: float | None = None,
    ) -> None:
        self.dataset_root = dataset_root
        self.device = torch.device(device)
        self.model_arch = model_arch
        self.image_subdir = image_subdir
        self.front_history_size = front_history_size
        self.transform = transforms.Compose(
            [transforms.Resize(image_size), transforms.ToTensor()]
        )
        self.temporal_inputs = TemporalInputProvider(
            dataset_root=dataset_root,
            side_cam_use_nearest=side_cam_use_nearest,
            side_cam_time_tolerance_s=side_cam_time_tolerance_s,
            front_history_size=front_history_size,
            front_history_stride=front_history_stride,
            front_history_pose_yaw_offset_rad=front_history_pose_yaw_offset_rad,
            front_history_pose_tolerance_s=front_history_pose_tolerance_s,
        )
        with local_dinov2_hub(torch_hub_dir):
            if model_arch == "front":
                self.model = FrontLimoNet(pretrained=False, image_size=image_size)
            elif model_arch == "side_cams":
                self.model = SideCamsLimoNet(pretrained=False, image_size=image_size)
            elif model_arch == "bev_limo_a":
                if BevLimoNetA is None:
                    raise ModuleNotFoundError(
                        "bev_limo_net_a.py is not available in this PTC-LIMO checkout. "
                        "Use method 'ptc_limo' or another available LiMO method."
                    )
                self.model = BevLimoNetA(pretrained=False, image_size=image_size)
            elif model_arch == "side_cams_history_depth":
                if SideCamsHistoryDepthLimoNet is None:
                    raise ModuleNotFoundError(
                        "limo_net_side_cams_depth_history.py is not available in "
                        "this less-is-more checkout."
                    )
                self.model = SideCamsHistoryDepthLimoNet(
                    pretrained=False,
                    image_size=image_size,
                    use_front_history_depth=True,
                    front_history_memory_size=front_history_size,
                )
            else:
                raise ValueError(f"Unknown LiMO model_arch: {model_arch}")
        self.model.load_state_dict(load_limo_state_dict(weights_path))
        self.model.to(self.device).eval()
        self._image_cache: OrderedDict[tuple[str, str, int, str], torch.Tensor] = OrderedDict()
        self._image_cache_size = 128

    @property
    def uses_side_cams(self) -> bool:
        return self.model_arch in {"side_cams", "side_cams_history_depth"}

    @property
    def uses_front_history(self) -> bool:
        return self.front_history_size > 0

    def select_samples(self, samples: list[Sample]) -> tuple[list[Sample], list[dict[str, object]]]:
        """按训练时同步约束筛选历史模型的有效样本，并保留排除原因。"""
        if not self.temporal_inputs.side_cam_use_nearest and not self.uses_front_history:
            return samples, []
        kept: list[Sample] = []
        excluded: list[dict[str, object]] = []
        for sample in samples:
            reason = self.temporal_inputs.sample_exclusion_reason(sample)
            if reason is None:
                kept.append(sample)
            else:
                excluded.append(
                    {
                        "dataset": sample.dataset,
                        "subset": sample.subset,
                        "mission": sample.mission,
                        "sample_index": sample.index,
                        "source_index": "" if sample.source_index is None else sample.source_index,
                        "image_id": sample.image_id,
                        "reason": reason,
                    }
                )
        return kept, excluded

    @torch.inference_mode()
    def predict_batch(self, samples: list[Sample]) -> list[np.ndarray]:
        goals = []
        batch: dict[str, torch.Tensor] = {}
        fronts = []
        lefts = []
        rights = []
        histories = []
        history_dts = []
        history_delta_poses = []
        for sample in samples:
            fronts.append(self._image_tensor(sample, "hdr_front"))
            if self.uses_side_cams:
                left_id = self.temporal_inputs.side_image_id(
                    sample.mission, "hdr_left", sample.image_id
                )
                right_id = self.temporal_inputs.side_image_id(
                    sample.mission, "hdr_right", sample.image_id
                )
                if left_id is None or right_id is None:
                    raise ValueError(
                        f"Sample {sample.mission}/{sample.image_id} was not filtered for side-camera sync"
                    )
                lefts.append(self._image_tensor(sample, "hdr_left", left_id))
                rights.append(self._image_tensor(sample, "hdr_right", right_id))
            if self.uses_front_history:
                history, dt, delta_pose = self._front_history_tensors(sample, fronts[-1])
                histories.append(history)
                history_dts.append(dt)
                history_delta_poses.append(delta_pose)
            goals.append(torch.tensor(sample.goal, dtype=torch.float32))
        batch["image_front"] = torch.stack(fronts, dim=0).to(self.device)
        if self.uses_side_cams:
            batch["image_left"] = torch.stack(lefts, dim=0).to(self.device)
            batch["image_right"] = torch.stack(rights, dim=0).to(self.device)
        if self.uses_front_history:
            batch["image_front_history"] = torch.stack(histories, dim=0).to(self.device)
            batch["front_history_dt"] = torch.stack(history_dts, dim=0).to(self.device)
            batch["front_history_delta_pose"] = torch.stack(history_delta_poses, dim=0).to(
                self.device
            )
        goal_tensor = torch.stack(goals, dim=0).to(self.device)
        batch["goal"] = goal_tensor
        pred = self.model(batch)
        if isinstance(pred, dict):
            pred = pred["path"]
        return [path.detach().cpu().numpy().astype(np.float32) for path in pred]

    def _front_history_tensors(
        self, sample: Sample, current_image: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        history_ids = self.temporal_inputs.history_image_ids(sample.image_id)
        images = []
        for image_id in history_ids:
            if image_id is None:
                images.append(torch.zeros_like(current_image))
            elif image_id == sample.image_id:
                images.append(current_image)
            else:
                images.append(self._image_tensor(sample, "hdr_front", image_id))
        pose_inputs = self.temporal_inputs.history_pose_inputs(sample.mission, sample.image_id)
        if pose_inputs is None:
            raise ValueError(
                f"Sample {sample.mission}/{sample.image_id} was not filtered for history pose availability"
            )
        dt, delta_pose = pose_inputs
        return (
            torch.stack(images, dim=0),
            torch.from_numpy(dt),
            torch.from_numpy(delta_pose),
        )

    def _image_tensor(
        self, sample: Sample, camera: str, image_id: int | None = None
    ) -> torch.Tensor:
        image_id = sample.image_id if image_id is None else image_id
        key = (sample.mission, self.image_subdir, image_id, camera)
        cached = self._image_cache.get(key)
        if cached is not None:
            self._image_cache.move_to_end(key)
            return cached
        path = (
            self.dataset_root
            / sample.mission
            / self.image_subdir
            / camera
            / f"{image_id:06d}.jpeg"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {camera} image for LiMO evaluation: {path}"
            )
        tensor = self.transform(Image.open(path).convert("RGB"))
        self._image_cache[key] = tensor
        if len(self._image_cache) > self._image_cache_size:
            self._image_cache.popitem(last=False)
        return tensor
