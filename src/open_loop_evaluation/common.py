from __future__ import annotations

import sys
import time
import os
from pathlib import Path
from typing import Iterable, Iterator, TypeVar

import yaml

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None


T = TypeVar("T")


def _discover_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "limo").is_dir() and (parent / "dataset_builder").is_dir():
            return parent
        if (parent / "less-is-more").is_dir():
            return parent
        if (parent / "algorithms" / "less-is-more").is_dir():
            return parent
    return path.parents[4]


REPO_ROOT = _discover_repo_root()
ALGORITHM_ROOT = (
    REPO_ROOT
    if (REPO_ROOT / "limo").is_dir() and (REPO_ROOT / "dataset_builder").is_dir()
    else REPO_ROOT / "less-is-more"
    if (REPO_ROOT / "less-is-more").is_dir()
    else REPO_ROOT / "algorithms" / "less-is-more"
)
EVALUATION_ROOT = (
    REPO_ROOT / "algorithms" / "evalution"
    if (REPO_ROOT / "algorithms" / "evalution").is_dir()
    else REPO_ROOT
)
DEFAULT_DATASET_ROOT = (
    ALGORITHM_ROOT / "data" / "dataset_builder" / "grandtour"
    if (ALGORITHM_ROOT / "data" / "dataset_builder" / "grandtour").exists()
    else EVALUATION_ROOT / "dataset" / "grandtour"
)
DEFAULT_OUTPUT_ROOT = EVALUATION_ROOT / "results" / "open_loop_evaluation"
DEFAULT_MISSIONS_CSV = (
    ALGORITHM_ROOT / "limo" / "configs" / "dataset" / "missions_split.csv"
)
DEFAULT_BUILD_CONFIG = ( # MPPI config
    ALGORITHM_ROOT / "dataset_builder" / "configs" / "build.yaml"
)
def _default_torch_hub_dir() -> Path:
    candidates = [
        REPO_ROOT / "tmp" / "torch-cache" / "hub",
        Path(os.environ.get("TORCH_HOME", Path.home() / ".cache" / "torch")) / "hub",
    ]
    for candidate in candidates:
        if (candidate / "facebookresearch_dinov2_main").exists():
            return candidate
    return candidates[-1]


DEFAULT_TORCH_HUB_DIR = _default_torch_hub_dir()


if str(ALGORITHM_ROOT) not in sys.path:
    sys.path.insert(0, str(ALGORITHM_ROOT))


def resolve_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def timestamp_run_id(prefix: str = "open-loop-eval") -> str:
    return f"{prefix}-{time.strftime('%Y%m%d-%H%M%S')}"


def progress(items: Iterable[T], total: int | None = None, desc: str = "") -> Iterable[T]:
    if tqdm is None:
        return items
    return tqdm(items, total=total, desc=desc, dynamic_ncols=True)


def batched(items: Iterable[T], batch_size: int) -> Iterator[list[T]]:
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
