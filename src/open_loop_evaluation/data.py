from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import zarr

"""把磁盘上的 mission / zarr 数据封装成 Python 对象，让 cli.py、prepare.py、evaluate.py 不需要直接关心底层路径和 zarr 细节"""

# teleop_paths：人工遥操作轨迹，后面对应 subset "tel"
TELEOP_GROUP = "teleop_paths"
# teleop_paths_evalution：prepare 后筛选出的 D_TEL 可达评测集，后面对应 subset "tel_eval"
TELEOP_EVAL_GROUP = "teleop_paths_evalution"
# teleop_paths_planner：与 teleop_paths_evalution 一一对应的 MPPI 规划路径，后面对应 subset "tel_planner"
TELEOP_PLANNER_GROUP = "teleop_paths_planner"
# geometric_paths：原始几何规划轨迹，后面对应 subset "geo"
GEOMETRIC_GROUP = "geometric_paths"
# geometric_paths_evalution：prepare 之后筛选出的 D_GEO 可达评测集，后面对应 subset "geo_eval"
GEOMETRIC_EVAL_GROUP = "geometric_paths_evalution"


@dataclass(frozen=True)
class MissionSplit:
    """表示一个 mission 在 split 表里的信息, 创建后不希望被修改"""
    mission: str
    short_name: str
    split: str


@dataclass(frozen=True)
class Sample:
    """每个 Sample 代表一条导航样本, 创建后不希望被修改"""
    dataset: str                      # 逻辑数据集名，比如 D_TEL、D_GEO、D_AUG
    subset: str                       # 底层来源，比如 tel、tel_eval、geo、geo_eval
    mission: str                      # 来自哪个 mission
    index: int                        # 在当前 zarr group 里的下标
    image_id: int                     # 对应的前视相机图片编号
    goal: np.ndarray                  # 目标位姿，一般是 [x, y, yaw]
    reference_path: np.ndarray        # 参考路径，一般形状是 [50, 3]
    goal_time: float                  # 这条样本的目标时间
    source_index: int | None = None   # 如果这个样本来自过滤后的 eval group，它记录原始 group 里的下标


class MissionStore:
    """MissionStore 是“一个 mission 目录”的数据访问对象。它会打开这个 mission 里面存在的 zarr group"""
    def __init__(self, mission_dir: Path, split: MissionSplit | None = None):
        # 初始化时先保存基本信息
        self.mission_dir = mission_dir
        self.mission = mission_dir.name
        self.split = split.split if split else "unknown"
        self.short_name = split.short_name if split else self.mission
        # 然后打开轨迹 group
        self.groups: dict[str, zarr.hierarchy.Group] = {}
        for subset, group_name in (
            ("tel", TELEOP_GROUP),
            ("tel_eval", TELEOP_EVAL_GROUP),
            ("tel_planner", TELEOP_PLANNER_GROUP),
            ("geo", GEOMETRIC_GROUP),
            ("geo_eval", GEOMETRIC_EVAL_GROUP),
        ):
            group_path = mission_dir / "data" / group_name
            if group_path.exists():
                self.groups[subset] = zarr.open_group(str(group_path), mode="r")

        # 打开 elevation map 用于碰撞检测和 GD 距离计算
        elev_path = mission_dir / "data" / "elevation_map"
        self.elevation_group = (
            zarr.open_group(str(elev_path), mode="r") if elev_path.exists() else None
        )
        self._elevation_index: dict[int, int] | None = None

    def has_group(self, subset: str) -> bool:
        """判断当前 mission 是否有某个 subset store.has_group("tel")"""
        return subset in self.groups

    def group_len(self, subset: str) -> int:
        """返回某个 subset 里有多少条路径样本"""
        group = self.groups.get(subset)
        return int(group["path"].shape[0]) if group is not None else 0

    def get_sample(self, dataset: str, subset: str, index: int) -> Sample:
        """把 zarr 里的一条数据读出来，包装成 Sample"""
        group = self.groups[subset]
        source_index = None
        if "source_index" in group:
            source_index = int(group["source_index"][index])
        return Sample(
            dataset=dataset,
            subset=subset,
            mission=self.mission,
            index=index,
            image_id=int(group["image_id"][index]),
            goal=np.asarray(group["goal"][index], dtype=np.float32),
            reference_path=np.asarray(group["path"][index], dtype=np.float32),
            goal_time=float(group["goal_time"][index]),
            source_index=source_index,
        )

    def has_elevation(self) -> bool:
        """判断当前 mission 是否有 elevation map"""
        return self.elevation_group is not None and "elevation" in self.elevation_group

    def elevation_len(self) -> int:
        """返回 elevation frame 数量"""
        if not self.has_elevation():
            return 0
        return int(self.elevation_group["elevation"].shape[0])

    def elevation_for_image(self, image_id: int) -> np.ndarray:
        """根据 image_id 找到对应的 elevation map"""
        if self.elevation_group is None:
            raise FileNotFoundError(f"Missing elevation_map for {self.mission}")
        # if elevation group 里有 image_id 数组: 说明 elevation 的数组下标不一定等于 image_id，所以需要建立映射
        if "image_id" in self.elevation_group:
            if self._elevation_index is None:
                ids = np.asarray(self.elevation_group["image_id"])
                self._elevation_index = {int(v): int(i) for i, v in enumerate(ids)}
            if image_id not in self._elevation_index:
                raise KeyError(f"No elevation frame for image_id={image_id}")
            elev_idx = self._elevation_index[image_id]
        else:
        # if elevation group 里没有 image_id 数组: 说明 elevation 的数组下标就是 image_id，可以直接使用
            elev_idx = image_id
        return np.asarray(self.elevation_group["elevation"][elev_idx], dtype=np.float32)


def load_mission_splits(path: Path) -> dict[str, MissionSplit]:
    """读取 mission split CSV 返回一个字典"""
    # {
    #     "2023-05-10-12-00-00": MissionSplit(
    #         mission="2023-05-10-12-00-00",
    #         short_name="mission_01",
    #         split="test",
    #     )
    # }
    splits: dict[str, MissionSplit] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            mission = row["Timestamp"]
            splits[mission] = MissionSplit(
                mission=mission,
                short_name=row.get("Short Name", mission),
                split=row.get("Split", "unknown"),
            )
    return splits


def load_mission_stores(dataset_root: Path, missions_csv: Path) -> list[MissionStore]:
    """把整个数据集目录加载成一组 MissionStore"""
    splits = load_mission_splits(missions_csv)
    mission_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    if not mission_dirs:
        raise FileNotFoundError(f"No mission directories under {dataset_root}")
    return [MissionStore(path, splits.get(path.name)) for path in mission_dirs]


def iter_samples(
    stores: Iterable[MissionStore],      # 所有 mission
    dataset: str,                        # 逻辑数据集名，例如 D_TEL
    subsets: list[str],                  # 要遍历哪些 subset，例如 ["tel_eval"] 或 ["geo_eval", "tel_eval"]
    max_samples: int | None = None,      # 最多取多少个样本，用于快速测试
) -> Iterable[Sample]:
    """样本迭代器。它会按照指定 subset 遍历所有 mission 的样本"""
    count = 0
    for subset in subsets:
        for store in stores:
            if not store.has_group(subset):
                continue
            for index in range(store.group_len(subset)):
                if max_samples is not None and count >= max_samples:
                    return
                count += 1
                yield store.get_sample(dataset, subset, index)


def dataset_subsets(dataset: str) -> list[str]:
    """定义逻辑数据集和底层 subset 的关系"""
    # D_TEL = filtered teleop eval paths
    # D_GEO = filtered geometric eval paths
    # D_AUG = filtered geometric eval paths + filtered teleop eval paths
    if dataset == "D_TEL":
        return ["tel_eval"]
    if dataset == "D_GEO":
        return ["geo_eval"]
    if dataset == "D_AUG":
        return ["geo_eval", "tel_eval"]
    raise ValueError(f"Unknown dataset: {dataset}")


def path_length_m(path: np.ndarray) -> float:
    """计算一条路径的二维长度 把每段长度加起来，就是路径总长度"""
    xy = np.concatenate([np.zeros((1, 2), dtype=np.float32), path[:, :2]], axis=0)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())


def summarize_group(stores: Iterable[MissionStore], subset: str) -> dict[str, float]:
    """统计某个 subset 在一批 stores 里的总体情况 样本数、路径总长度、总时间小时数、平均速度"""
    samples = 0
    length_m = 0.0
    time_s = 0.0
    for store in stores:
        if not store.has_group(subset):
            continue
        group = store.groups[subset]
        n = int(group["path"].shape[0])
        samples += n
        goal_times = np.asarray(group["goal_time"], dtype=np.float64)
        time_s += float(goal_times.sum())
        paths = group["path"]
        # 这里不是一次性读全部 path，而是每次读 2048 条，避免数据太大占满内存
        for start in range(0, n, 2048):
            chunk = np.asarray(paths[start : start + 2048], dtype=np.float32)
            origin = np.zeros((chunk.shape[0], 1, 2), dtype=np.float32)
            xy = np.concatenate([origin, chunk[:, :, :2]], axis=1)
            length_m += float(np.linalg.norm(np.diff(xy, axis=1), axis=2).sum())
    avg_vel = length_m / time_s if time_s > 0 else 0.0
    return {
        "samples": samples,
        "length_m": length_m,
        "time_h": time_s / 3600.0,
        "avg_vel_mps": avg_vel,
    }


def collect_dataset_stats(stores: list[MissionStore]) -> list[dict[str, Any]]:
    """生成 CLI 里打印的数据集统计表"""
    rows: list[dict[str, Any]] = []
    for dataset, subset in (
        ("D_TEL_RAW", "tel"),
        ("D_TEL_EVAL", "tel_eval"),
        ("D_TEL_PLANNER", "tel_planner"),
        ("D_GEO_RAW", "geo"),
        ("D_GEO_EVAL", "geo_eval"),
    ):
        for split in ("train", "test"):
            split_stores = [store for store in stores if store.split == split]
            stats = summarize_group(split_stores, subset)
            rows.append(
                {
                    "dataset": dataset,
                    "split": split,
                    "#samples": int(stats["samples"]),
                    "length_m": stats["length_m"],
                    "time_h": stats["time_h"],
                    "avg_vel_mps": stats["avg_vel_mps"],
                }
            )

    for split in ("train", "test"):
        split_stores = [store for store in stores if store.split == split]
        tel = summarize_group(split_stores, "tel_eval")
        geo_eval = summarize_group(split_stores, "geo_eval")
        rows.append(
            {
                "dataset": "D_AUG_EVAL",
                "split": split,
                "#samples": int(tel["samples"] + geo_eval["samples"]),
                "length_m": tel["length_m"] + geo_eval["length_m"],
                "time_h": tel["time_h"] + geo_eval["time_h"],
                "avg_vel_mps": (
                    (tel["length_m"] + geo_eval["length_m"])
                    / ((tel["time_h"] + geo_eval["time_h"]) * 3600.0)
                    if (tel["time_h"] + geo_eval["time_h"]) > 0
                    else 0.0
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                keys.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
