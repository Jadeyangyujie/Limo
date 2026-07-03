import csv
import re
import shutil
import tarfile
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any, Literal, Tuple

import numpy as np
import torch
import zarr
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset
from torchvision import transforms

from limo.src.dataset.zarr_v2 import ZarrV2Array
from limo.src.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(cfg, key, default)


def _ptc_enabled(ptc: Any) -> bool:
    return bool(_cfg_get(ptc, "enabled", False))


def parse_missions_csv(missions_csv: Path) -> dict[str, str]:
    """Parse missions CSV and return a dict mapping Timestamp to Split."""
    timestamp_to_split = {}
    with missions_csv.open("r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            timestamp = row.get("Timestamp", "").strip()
            split = row.get("Split", "").strip()
            if timestamp:
                timestamp_to_split[timestamp] = split
    return timestamp_to_split


def pull_missions_from_hf(
    missions: list[str], topics: list[str], dataset_folder: Path
) -> Path:
    allow_patterns = []
    for mission, topic in product(missions, topics):
        allow_patterns.append(f"{mission}/*{topic}*")

    log.info("Downloading missions from Hugging Face...")
    hf_data_cache = snapshot_download(
        repo_id="leggedrobotics/grand_tour_dataset",
        revision="refs/pr/6",  # REMOVE LATER
        allow_patterns=allow_patterns,
        repo_type="dataset",
    )

    log.info(f"Extraction missions from HF cache at {hf_data_cache}...")
    move_dataset(hf_data_cache, dataset_folder, allow_patterns=allow_patterns)
    return Path(dataset_folder)


def move_dataset(cache, dataset_folder, allow_patterns=["*"]):
    def convert_glob_patterns_to_regex(glob_patterns):
        regex_parts = []
        for pat in glob_patterns:
            # Escape regex special characters except for * and ?
            pat = re.escape(pat)
            # Convert escaped glob wildcards to regex equivalents
            pat = pat.replace(r"\*", ".*").replace(r"\?", ".")
            # Make sure it matches full paths
            regex_parts.append(f".*{pat}$")

        # Join with |
        combined = "|".join(regex_parts)
        return re.compile(combined)

    pattern = convert_glob_patterns_to_regex(allow_patterns)
    files = [f for f in Path(cache).rglob("*") if pattern.match(str(f))]
    tar_files = [f for f in files if f.suffix == ".tar"]

    for source_path in tar_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with tarfile.open(source_path, "r") as tar:
                tar.extractall(path=dest_path.parent)
        except tarfile.ReadError as e:
            log.error(f"Error opening or extracting tar file '{source_path}': {e}")
        except Exception as e:
            log.error(
                f"An unexpected error occurred while processing {source_path}: {e}"
            )

    other_files = [f for f in files if not f.suffix == ".tar" and f.is_file()]
    for source_path in other_files:
        dest_path = dataset_folder / source_path.relative_to(cache)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest_path)


class MissionDataset(Dataset):
    def __init__(
        self,
        dataset_type: Literal["tel", "geo", "aug"],
        dataset_folder: Path,
        mission_name: str,
        transform: transforms.Compose,
        with_side_cams: bool = False,
        ptc: Any | None = None,
    ):
        self.dataset_type = dataset_type
        self.dataset_folder = dataset_folder
        self.mission_name = mission_name
        self.transform = transform
        self.with_side_cams = with_side_cams
        self.ptc = ptc
        self.ptc_enabled = _ptc_enabled(ptc)
        self.ptc_missing_count = 0
        self.ptc_arrays: dict[str, ZarrV2Array] | None = None
        self.ptc_image_id_to_row: dict[int, int] = {}
        self.ptc_map_shape = self._default_ptc_map_shape()

        mission_dir = dataset_folder / mission_name
        if not mission_dir.exists():
            err = f"Mission dataset '{mission_name}' not found in {dataset_folder}"
            log.error(err)
            raise FileNotFoundError(err)

        if dataset_type == "tel":
            self.z = zarr.open_group(
                str(mission_dir / "data" / "teleop_paths"), mode="r"
            )
        elif dataset_type == "geo":
            self.z = zarr.open_group(
                str(mission_dir / "data" / "geometric_paths"), mode="r"
            )
        else:
            raise ValueError(f"Invalid dataset_type: {dataset_type}")

        if self.ptc_enabled:
            self._init_ptc_labels(mission_dir)

    def __len__(self):
        return len(self.z["path"])

    def _default_ptc_map_shape(self) -> tuple[int, int]:
        x_min = float(_cfg_get(self.ptc, "x_min", -4.0))
        x_max = float(_cfg_get(self.ptc, "x_max", 4.0))
        y_min = float(_cfg_get(self.ptc, "y_min", -4.0))
        y_max = float(_cfg_get(self.ptc, "y_max", 4.0))
        resolution = float(_cfg_get(self.ptc, "resolution", 0.04))
        height = max(int(round((x_max - x_min) / resolution)), 1)
        width = max(int(round((y_max - y_min) / resolution)), 1)
        return height, width

    def _image_id_at(self, idx: int) -> int:
        return int(np.asarray(self.z["image_id"][idx]).reshape(-1)[0])

    def _init_ptc_labels(self, mission_dir: Path) -> None:
        label_group = str(_cfg_get(self.ptc, "label_group", "ptc_labels"))
        risk_key = str(_cfg_get(self.ptc, "risk_key", "risk_map"))
        valid_key = str(_cfg_get(self.ptc, "valid_key", "valid_mask"))
        ptc_dir = mission_dir / label_group

        required_paths = {
            "image_id": ptc_dir / "image_id",
            "risk_map": ptc_dir / risk_key,
            "valid_mask": ptc_dir / valid_key,
        }
        missing = [name for name, path in required_paths.items() if not (path / ".zarray").exists()]
        if missing:
            log.warning(
                f"PTC labels enabled but mission '{self.mission_name}' is missing "
                f"{missing} under {ptc_dir}. Returning zero risk/valid maps."
            )
            return

        image_id_arr = ZarrV2Array(required_paths["image_id"])
        risk_arr = ZarrV2Array(required_paths["risk_map"])
        valid_arr = ZarrV2Array(required_paths["valid_mask"])

        if len(risk_arr.shape) != 3 or len(valid_arr.shape) != 3:
            log.warning(
                f"PTC labels for mission '{self.mission_name}' must be [N,H,W], "
                f"got risk={risk_arr.shape}, valid={valid_arr.shape}. Returning zeros."
            )
            return
        if risk_arr.shape != valid_arr.shape:
            log.warning(
                f"PTC risk/valid shape mismatch for mission '{self.mission_name}': "
                f"risk={risk_arr.shape}, valid={valid_arr.shape}. Returning zeros."
            )
            return

        ptc_ids = np.asarray(image_id_arr[:], dtype=np.int64).reshape(-1)
        self.ptc_image_id_to_row = {
            int(image_id): int(row) for row, image_id in enumerate(ptc_ids)
        }
        self.ptc_arrays = {
            "risk_map": risk_arr,
            "valid_mask": valid_arr,
        }
        self.ptc_map_shape = (int(risk_arr.shape[1]), int(risk_arr.shape[2]))
        log.info(
            f"Loaded PTC labels for mission '{self.mission_name}' from {ptc_dir}: "
            f"{len(self.ptc_image_id_to_row)} image ids, map shape={self.ptc_map_shape}"
        )

    def _zero_ptc_label(self) -> dict[str, torch.Tensor]:
        height, width = self.ptc_map_shape
        return {
            "risk_map": torch.zeros((1, height, width), dtype=torch.float32),
            "valid_mask": torch.zeros((1, height, width), dtype=torch.float32),
        }

    def _load_ptc_label(self, image_id: int) -> dict[str, torch.Tensor]:
        if self.ptc_arrays is None:
            return self._zero_ptc_label()

        row = self.ptc_image_id_to_row.get(int(image_id))
        if row is None:
            self.ptc_missing_count += 1
            if self.ptc_missing_count <= 5:
                log.warning(
                    f"PTC label missing for mission='{self.mission_name}', "
                    f"image_id={image_id}. Returning zero risk/valid maps."
                )
            return self._zero_ptc_label()

        risk = np.asarray(self.ptc_arrays["risk_map"][row])[0].astype(np.float32)
        valid = np.asarray(self.ptc_arrays["valid_mask"][row])[0].astype(np.float32)
        return {
            "risk_map": torch.from_numpy(risk[None]),
            "valid_mask": torch.from_numpy(valid[None]),
        }

    def load_image(self, topic: str, idx: int) -> Image.Image:
        image_id = self._image_id_at(idx)
        image_path = (
            self.dataset_folder
            / self.mission_name
            / "images"
            / topic
            / f"{image_id:06d}.jpeg"
        )
        if not image_path.exists():
            log.error(f"Image not found at {image_path}")
            raise FileNotFoundError(f"Image not found at {image_path}")
        return Image.open(image_path).convert("RGB")

    def __getitem__(self, idx):
        image_front = self.load_image("hdr_front", idx)
        image_front = self.transform(image_front)

        goal = torch.tensor(self.z["goal"][idx], dtype=torch.float32)
        path = torch.tensor(self.z["path"][idx], dtype=torch.float32)
        image_id = self._image_id_at(idx)

        batch = {
            "image_front": image_front,
            "goal": goal,
            "path": path,
            "image_id": torch.tensor(image_id, dtype=torch.long),
        }

        if self.ptc_enabled:
            batch.update(self._load_ptc_label(image_id))

        if self.with_side_cams:
            image_left = self.load_image("hdr_left", idx)
            image_left = self.transform(image_left)
            batch["image_left"] = image_left

            image_right = self.load_image("hdr_right", idx)
            image_right = self.transform(image_right)
            batch["image_right"] = image_right

        return batch


def get_mission_dataset(
    dataset_type: Literal["tel", "geo", "aug"],
    dataset_folder: Path,
    mission_name: str,
    transform: transforms.Compose,
    with_side_cams: bool = False,
    ptc: Any | None = None,
) -> Dataset:
    if dataset_type == "aug":
        geo_ds = MissionDataset(
            "geo", dataset_folder, mission_name, transform, with_side_cams, ptc
        )
        tel_ds = MissionDataset(
            "tel", dataset_folder, mission_name, transform, with_side_cams, ptc
        )
        return ConcatDataset([geo_ds, tel_ds])
    if dataset_type in ["tel", "geo"]:
        return MissionDataset(
            dataset_type, dataset_folder, mission_name, transform, with_side_cams, ptc
        )
    else:
        raise ValueError(f"Invalid dataset_type: {dataset_type}")


def get_dataset(
    dataset_type: Literal["tel", "geo", "aug"],
    dataset_folder: Path,
    missions_csv: Path,
    with_side_cams: bool = False,
    image_size: Tuple[int, int] = (308, 476),
    ptc: Any | None = None,
):
    missions = parse_missions_csv(missions_csv)

    transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )

    topics = ["hdr_front"]
    if with_side_cams:
        topics += ["hdr_left", "hdr_right"]
    if dataset_type in ["tel", "aug"]:
        topics.append("teleop_paths")
    if dataset_type in ["geo", "aug"]:
        topics.append("geometric_paths")
    if _ptc_enabled(ptc):
        topics.append(str(_cfg_get(ptc, "label_group", "ptc_labels")))

    grandtour_folder = dataset_folder 
    grandtour_folder.mkdir(parents=True, exist_ok=True)
    datset_dir = pull_missions_from_hf(list(missions.keys()), topics, grandtour_folder)

    datasets = defaultdict(list)
    for mission, split in missions.items():
        datasets[split].append(
            get_mission_dataset(
                dataset_type, datset_dir, mission, transform, with_side_cams, ptc
            )
        )

    splits: dict[str, Dataset] = dict()
    for split, ds_list in datasets.items():
        splits[split] = ConcatDataset(ds_list)
        log.info(f"Split '{split}' has {len(splits[split])} samples")
    return splits
