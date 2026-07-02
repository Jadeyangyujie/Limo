#!/usr/bin/env python3
"""Generate PTC-LIMO privileged risk labels from existing BEV labels.

This script converts an existing per-image bev_labels_from_elevation zarr group
into a PTC training label group. It intentionally reuses the existing BEV
outputs and does not run TraversabilityFilter or regenerate raw risk from
elevation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import traceback
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any

import numcodecs
import numpy as np

import re
MISSION_RE = re.compile(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")

PTC_VERSION = "v1"
DEFAULT_INPUT_GROUP = "bev_labels_from_elevation"
DEFAULT_OUTPUT_GROUP = "ptc_labels"
DEFAULT_RISK_SOURCE = "limo_raw_risk"
DEFAULT_NORMALIZATION = "clip01"
DEFAULT_SUMMARY_OUT = (
    Path(__file__).resolve().parent / "ptc_labels_generation_summary.json"
)
EPS = 1e-6


def valid_source_text(risk_source: str) -> str:
    base = "isfinite(limo_trav_cost) & isfinite(limo_raw_risk)"
    if risk_source != "limo_raw_risk":
        base += f" & isfinite({risk_source})"
    return base + " & optional occupancy"


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PTC-LIMO labels from bev_labels_from_elevation."
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    mission_group = parser.add_mutually_exclusive_group(required=True)
    mission_group.add_argument("--mission", type=str)
    mission_group.add_argument("--all-missions", action="store_true")
    parser.add_argument("--input-group-name", default=DEFAULT_INPUT_GROUP)
    parser.add_argument("--output-group-name", default=DEFAULT_OUTPUT_GROUP)
    parser.add_argument("--risk-source", default=DEFAULT_RISK_SOURCE)
    parser.add_argument(
        "--normalization",
        default=DEFAULT_NORMALIZATION,
        choices=("clip01", "robust", "global_robust"),
    )
    parser.add_argument("--roi-x-min", default=-4.0, type=float)
    parser.add_argument("--roi-x-max", default=4.0, type=float)
    parser.add_argument("--roi-y-min", default=-4.0, type=float)
    parser.add_argument("--roi-y-max", default=4.0, type=float)
    parser.add_argument("--chunk-images", default=32, type=positive_int)
    parser.add_argument("--max-images", type=positive_int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--summary-out", default=DEFAULT_SUMMARY_OUT, type=Path)
    return parser.parse_args()


def as_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): as_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return as_jsonable(value.tolist())
    if isinstance(value, np.generic):
        return as_jsonable(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(as_jsonable(payload), f, indent=2, sort_keys=True)
        f.write("\n")


def dtype_to_zarr_str(dtype: np.dtype | str) -> str:
    return np.dtype(dtype).str


def zarr_fill_value(value: Any, dtype: np.dtype) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "NaN":
            return np.nan
        return value
    if dtype.kind in {"f", "c"} and isinstance(value, float) and math.isnan(value):
        return np.nan
    return value


def json_fill_value(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return "NaN"
    if isinstance(value, np.generic):
        return json_fill_value(value.item())
    return value


def default_compressor_config() -> dict[str, Any]:
    return {
        "id": "blosc",
        "cname": "lz4",
        "clevel": 5,
        "shuffle": 1,
        "blocksize": 0,
    }


class ZarrV2Array:
    """Small zarr v2 array reader for basic contiguous slicing."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        meta_path = self.path / ".zarray"
        if not meta_path.exists():
            raise FileNotFoundError(f"zarr array metadata missing: {meta_path}")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self.meta.get("zarr_format") != 2:
            raise ValueError(f"only zarr v2 arrays are supported: {self.path}")
        self.shape = tuple(int(x) for x in self.meta["shape"])
        self.chunks = tuple(int(x) for x in self.meta["chunks"])
        self.dtype = np.dtype(self.meta["dtype"])
        self.order = self.meta.get("order", "C")
        self.dimension_separator = self.meta.get("dimension_separator", ".")
        compressor_config = self.meta.get("compressor")
        self.compressor = (
            numcodecs.get_codec(compressor_config) if compressor_config else None
        )
        self.fill_value = zarr_fill_value(self.meta.get("fill_value"), self.dtype)

    def _chunk_name(self, coords: tuple[int, ...]) -> Path:
        if self.dimension_separator == "/":
            return self.path.joinpath(*(str(c) for c in coords))
        return self.path / ".".join(str(c) for c in coords)

    def _empty_chunk(self) -> np.ndarray:
        chunk = np.empty(self.chunks, dtype=self.dtype, order=self.order)
        fill = self.fill_value
        if fill is None:
            fill = 0
        chunk[...] = fill
        return chunk

    def _read_chunk(self, coords: tuple[int, ...]) -> np.ndarray:
        chunk_path = self._chunk_name(coords)
        if not chunk_path.exists():
            return self._empty_chunk()
        payload = chunk_path.read_bytes()
        if self.compressor is not None:
            payload = self.compressor.decode(payload)
        arr = np.frombuffer(payload, dtype=self.dtype)
        expected = int(np.prod(self.chunks))
        if arr.size != expected:
            raise ValueError(
                f"decoded chunk {chunk_path} has {arr.size} values, expected {expected}"
            )
        return arr.reshape(self.chunks, order=self.order)

    def __getitem__(self, index: Any) -> np.ndarray:
        slices = self._normalize_index(index)
        out_shape = tuple(s.stop - s.start for s in slices)
        out = np.empty(out_shape, dtype=self.dtype)
        if any(size == 0 for size in out_shape):
            return out

        chunk_ranges = [
            range(s.start // c, (s.stop - 1) // c + 1)
            for s, c in zip(slices, self.chunks)
        ]
        for coords in product(*chunk_ranges):
            chunk = self._read_chunk(tuple(coords))
            src_slices = []
            dst_slices = []
            for dim, coord in enumerate(coords):
                chunk_start = coord * self.chunks[dim]
                chunk_stop = chunk_start + self.chunks[dim]
                overlap_start = max(slices[dim].start, chunk_start)
                overlap_stop = min(slices[dim].stop, chunk_stop)
                src_slices.append(slice(overlap_start - chunk_start, overlap_stop - chunk_start))
                dst_slices.append(slice(overlap_start - slices[dim].start, overlap_stop - slices[dim].start))
            out[tuple(dst_slices)] = chunk[tuple(src_slices)]
        return out

    def _normalize_index(self, index: Any) -> tuple[slice, ...]:
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(self.shape):
            raise IndexError(f"too many indices for {self.path}")
        normalized: list[slice] = []
        for dim, item in enumerate(index):
            if isinstance(item, int):
                start = item if item >= 0 else self.shape[dim] + item
                normalized.append(slice(start, start + 1))
                continue
            if not isinstance(item, slice):
                raise TypeError(f"unsupported index type {type(item)!r} for {self.path}")
            start, stop, step = item.indices(self.shape[dim])
            if step != 1:
                raise ValueError("only step=1 slices are supported")
            normalized.append(slice(start, stop))
        for dim in range(len(normalized), len(self.shape)):
            normalized.append(slice(0, self.shape[dim]))
        return tuple(normalized)


class ZarrV2WritableArray:
    """Small zarr v2 array writer supporting region assignment."""

    def __init__(
        self,
        path: Path | str,
        shape: tuple[int, ...],
        chunks: tuple[int, ...],
        dtype: np.dtype | str,
        fill_value: Any = 0,
        compressor_config: dict[str, Any] | None = None,
    ) -> None:
        self.path = Path(path)
        self.shape = tuple(int(x) for x in shape)
        self.chunks = tuple(int(x) for x in chunks)
        self.dtype = np.dtype(dtype)
        self.order = "C"
        self.dimension_separator = "."
        self.fill_value = zarr_fill_value(fill_value, self.dtype)
        self.compressor_config = compressor_config or default_compressor_config()
        self.compressor = numcodecs.get_codec(self.compressor_config)
        self.path.mkdir(parents=True, exist_ok=True)
        meta = {
            "zarr_format": 2,
            "shape": list(self.shape),
            "chunks": list(self.chunks),
            "dtype": dtype_to_zarr_str(self.dtype),
            "compressor": self.compressor_config,
            "fill_value": json_fill_value(fill_value),
            "filters": None,
            "order": self.order,
        }
        write_json(self.path / ".zarray", meta)

    def __getitem__(self, index: Any) -> np.ndarray:
        return ZarrV2Array(self.path)[index]

    def __setitem__(self, index: Any, values: Any) -> None:
        slices = self._normalize_index(index)
        data = np.asarray(values, dtype=self.dtype)
        expected_shape = tuple(s.stop - s.start for s in slices)
        if data.shape != expected_shape:
            raise ValueError(
                f"assignment shape {data.shape} does not match target {expected_shape}"
            )
        self._write_region(slices, data)

    def write_all(self, values: Any) -> None:
        data = np.asarray(values, dtype=self.dtype)
        if data.shape != self.shape:
            raise ValueError(f"data shape {data.shape} does not match {self.shape}")
        self._write_region(tuple(slice(0, n) for n in self.shape), data)

    def _chunk_name(self, coords: tuple[int, ...]) -> Path:
        return self.path / ".".join(str(c) for c in coords)

    def _empty_chunk(self) -> np.ndarray:
        chunk = np.empty(self.chunks, dtype=self.dtype)
        fill = self.fill_value
        if fill is None:
            fill = 0
        chunk[...] = fill
        return chunk

    def _read_existing_chunk(self, coords: tuple[int, ...]) -> np.ndarray:
        chunk_path = self._chunk_name(coords)
        if not chunk_path.exists():
            return self._empty_chunk()
        return ZarrV2Array(self.path)._read_chunk(coords).copy()

    def _write_chunk(self, coords: tuple[int, ...], chunk: np.ndarray) -> None:
        chunk_path = self._chunk_name(coords)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.compressor.encode(np.ascontiguousarray(chunk))
        chunk_path.write_bytes(payload)

    def _write_region(self, slices: tuple[slice, ...], data: np.ndarray) -> None:
        if any(s.stop <= s.start for s in slices):
            return
        chunk_ranges = [
            range(s.start // c, (s.stop - 1) // c + 1)
            for s, c in zip(slices, self.chunks)
        ]
        for coords in product(*chunk_ranges):
            chunk = self._read_existing_chunk(tuple(coords))
            src_slices = []
            dst_slices = []
            for dim, coord in enumerate(coords):
                chunk_start = coord * self.chunks[dim]
                chunk_stop = chunk_start + self.chunks[dim]
                overlap_start = max(slices[dim].start, chunk_start)
                overlap_stop = min(slices[dim].stop, chunk_stop)
                dst_slices.append(slice(overlap_start - chunk_start, overlap_stop - chunk_start))
                src_slices.append(slice(overlap_start - slices[dim].start, overlap_stop - slices[dim].start))
            chunk[tuple(dst_slices)] = data[tuple(src_slices)]
            self._write_chunk(tuple(coords), chunk)

    def _normalize_index(self, index: Any) -> tuple[slice, ...]:
        return ZarrV2ArrayLike.normalize_index(index, self.shape, self.path)


class ZarrV2ArrayLike:
    @staticmethod
    def normalize_index(index: Any, shape: tuple[int, ...], path: Path) -> tuple[slice, ...]:
        if not isinstance(index, tuple):
            index = (index,)
        if len(index) > len(shape):
            raise IndexError(f"too many indices for {path}")
        normalized: list[slice] = []
        for dim, item in enumerate(index):
            if isinstance(item, int):
                start = item if item >= 0 else shape[dim] + item
                normalized.append(slice(start, start + 1))
                continue
            if not isinstance(item, slice):
                raise TypeError(f"unsupported index type {type(item)!r} for {path}")
            start, stop, step = item.indices(shape[dim])
            if step != 1:
                raise ValueError("only step=1 slices are supported")
            normalized.append(slice(start, stop))
        for dim in range(len(normalized), len(shape)):
            normalized.append(slice(0, shape[dim]))
        return tuple(normalized)


class ZarrV2Group:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)

    def __getitem__(self, key: str) -> ZarrV2Array:
        return ZarrV2Array(self.path / key)


class ZarrV2Attrs:
    def __init__(self, group_path: Path) -> None:
        self.group_path = group_path

    def update(self, values: dict[str, Any]) -> None:
        attrs_path = self.group_path / ".zattrs"
        existing: dict[str, Any] = {}
        if attrs_path.exists():
            existing = json.loads(attrs_path.read_text(encoding="utf-8"))
        existing.update(as_jsonable(values))
        write_json(attrs_path, existing)


class ZarrV2OutputGroup:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        write_json(self.path / ".zgroup", {"zarr_format": 2})
        write_json(self.path / ".zattrs", {})
        self.attrs = ZarrV2Attrs(self.path)
        self.arrays: dict[str, ZarrV2WritableArray] = {}

    def create_dataset(
        self,
        name: str,
        data: Any | None = None,
        shape: tuple[int, ...] | None = None,
        chunks: tuple[int, ...] | None = None,
        dtype: np.dtype | str | None = None,
        fill_value: Any = 0,
    ) -> ZarrV2WritableArray:
        if data is not None:
            arr_data = np.asarray(data, dtype=dtype)
            shape = arr_data.shape
            dtype = arr_data.dtype
        elif shape is None or dtype is None:
            raise ValueError("shape and dtype are required when data is None")
        if chunks is None:
            raise ValueError("chunks are required")
        writable = ZarrV2WritableArray(
            self.path / name,
            tuple(int(x) for x in shape),
            tuple(int(x) for x in chunks),
            np.dtype(dtype),
            fill_value=fill_value,
        )
        if data is not None:
            writable.write_all(arr_data)
        self.arrays[name] = writable
        return writable

    def __getitem__(self, key: str) -> ZarrV2WritableArray:
        if key not in self.arrays:
            self.arrays[key] = ZarrV2WritableArray(
                self.path / key,
                ZarrV2Array(self.path / key).shape,
                ZarrV2Array(self.path / key).chunks,
                ZarrV2Array(self.path / key).dtype,
            )
        return self.arrays[key]


class WarningCollector:
    def __init__(self, max_messages_per_key: int = 5) -> None:
        self.max_messages_per_key = max_messages_per_key
        self.messages: list[str] = []
        self.counts: dict[str, int] = {}

    def warn(self, key: str, message: str) -> None:
        count = self.counts.get(key, 0) + 1
        self.counts[key] = count
        if count <= self.max_messages_per_key:
            self.messages.append(message)

    def finalize(self) -> list[str]:
        result = list(self.messages)
        for key, count in sorted(self.counts.items()):
            if count > self.max_messages_per_key:
                result.append(
                    f"{key}: {count} warnings total; only first "
                    f"{self.max_messages_per_key} are listed"
                )
        return result


class SampledStats:
    """Streaming stats with bounded samples for percentile estimates."""

    def __init__(
        self,
        max_samples: int = 500_000,
        max_samples_per_add: int = 4096,
        seed: int = 12345,
    ) -> None:
        self.max_samples = max_samples
        self.max_samples_per_add = max_samples_per_add
        self.rng = np.random.default_rng(seed)
        self.count = 0
        self.sum = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf
        self.samples: list[np.ndarray] = []

    def add(self, values: Any) -> None:
        arr = np.asarray(values).reshape(-1)
        if arr.size == 0:
            return
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return
        arr64 = arr.astype(np.float64, copy=False)
        self.count += int(arr64.size)
        self.sum += float(np.sum(arr64))
        self.min_value = min(self.min_value, float(np.min(arr64)))
        self.max_value = max(self.max_value, float(np.max(arr64)))

        sample = arr.astype(np.float32, copy=False)
        if sample.size > self.max_samples_per_add:
            idx = np.linspace(
                0, sample.size - 1, self.max_samples_per_add, dtype=np.int64
            )
            sample = sample[idx]
        self.samples.append(np.asarray(sample, dtype=np.float32))
        self._trim_samples(force=False)

    def _trim_samples(self, force: bool) -> None:
        sample_count = sum(sample.size for sample in self.samples)
        if sample_count <= self.max_samples * 2 and not force:
            return
        if sample_count == 0:
            return
        combined = np.concatenate(self.samples)
        if combined.size > self.max_samples:
            idx = self.rng.choice(combined.size, size=self.max_samples, replace=False)
            combined = combined[idx]
        self.samples = [combined.astype(np.float32, copy=False)]

    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        self._trim_samples(force=True)
        sample = np.concatenate(self.samples) if self.samples else np.asarray([])
        if sample.size == 0:
            percentiles = {key: None for key in ("p1", "p50", "p95", "p99")}
        else:
            p1, p50, p95, p99 = np.percentile(sample, [1, 50, 95, 99])
            percentiles = {
                "p1": float(p1),
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99),
            }
        return {
            "count": int(self.count),
            "min": float(self.min_value),
            "mean": float(self.sum / self.count),
            **percentiles,
            "max": float(self.max_value),
            "sample_count": int(sample.size),
            "percentiles_estimated": bool(sample.size < self.count),
        }


def array_shape(arr: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in getattr(arr, "shape", ()))


def get_array(group: Any, key: str) -> Any | None:
    try:
        return group[key]
    except Exception:
        return None


def require_array(group: Any, key: str) -> Any:
    arr = get_array(group, key)
    if arr is None:
        raise KeyError(f"required array missing: {key}")
    return arr


def vector_chunk(num_frames: int, chunk_images: int) -> tuple[int]:
    return (min(max(chunk_images * 10, 1024), max(num_frames, 1)),)


def map_chunk(num_frames: int, height: int, width: int, chunk_images: int) -> tuple[int, int, int]:
    return (min(chunk_images, max(num_frames, 1)), height, width)


def finite_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    return {
        "count": int(values.size),
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def validate_shapes(
    num_frames: int,
    risk_shape: tuple[int, ...],
    raw_shape: tuple[int, ...],
    trav_shape: tuple[int, ...],
    image_id_shape: tuple[int, ...],
    timestamp_shape: tuple[int, ...],
) -> tuple[int, int]:
    if len(raw_shape) != 3:
        raise ValueError(f"limo_raw_risk must have shape [N,H,W], got {raw_shape}")
    if len(risk_shape) != 3:
        raise ValueError(f"risk source must have shape [N,H,W], got {risk_shape}")
    if len(trav_shape) != 3:
        raise ValueError(f"limo_trav_cost must have shape [N,H,W], got {trav_shape}")
    if raw_shape != trav_shape:
        raise ValueError(
            f"limo_raw_risk shape {raw_shape} != limo_trav_cost shape {trav_shape}"
        )
    if risk_shape != raw_shape:
        raise ValueError(f"risk source shape {risk_shape} != limo_raw_risk shape {raw_shape}")
    if image_id_shape != (raw_shape[0],):
        raise ValueError(f"image_id must have shape [{raw_shape[0]}], got {image_id_shape}")
    if timestamp_shape != (raw_shape[0],):
        raise ValueError(
            f"timestamp must have shape [{raw_shape[0]}], got {timestamp_shape}"
        )
    if num_frames > raw_shape[0]:
        raise ValueError(f"requested {num_frames} frames, source has {raw_shape[0]}")
    return raw_shape[1], raw_shape[2]


def build_base_masks(
    raw_risk: np.ndarray,
    trav_cost: np.ndarray,
    risk_values: np.ndarray,
    occupancy: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = (
        np.isfinite(raw_risk) & np.isfinite(trav_cost) & np.isfinite(risk_values)
    )
    valid_mask = finite_mask.copy()
    if occupancy is not None:
        valid_mask &= np.asarray(occupancy) > 0
    return finite_mask, valid_mask


def normalize_image(
    values: np.ndarray,
    valid_mask: np.ndarray,
    method: str,
    warnings: WarningCollector,
    source_row: int,
    global_range: tuple[float, float] | None,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    out = np.zeros(values.shape, dtype=np.float32)
    mask = valid_mask & np.isfinite(values)
    if not np.any(mask):
        return out

    valid_values = values[mask]
    if method == "clip01":
        out[mask] = np.clip(valid_values, 0.0, 1.0)
        return out

    if method == "global_robust":
        if global_range is None:
            warnings.warn(
                "global_robust_missing_range",
                "global_robust range unavailable; falling back to clip01",
            )
            out[mask] = np.clip(valid_values, 0.0, 1.0)
            return out
        p1, p99 = global_range
    else:
        p1, p99 = np.percentile(valid_values, [1, 99])

    if not np.isfinite(p1) or not np.isfinite(p99) or (p99 - p1) <= EPS:
        warnings.warn(
            f"{method}_degenerate",
            f"source_row={source_row}: {method} p99-p1 too small; "
            "falling back to clip01",
        )
        out[mask] = np.clip(valid_values, 0.0, 1.0)
        return out

    out[mask] = np.clip((valid_values - p1) / (p99 - p1), 0.0, 1.0)
    return out


def estimate_global_robust_range(
    source: Any,
    num_frames: int,
    chunk_images: int,
    risk_source: str,
    warnings: WarningCollector,
) -> tuple[float, float] | None:
    risk_arr = require_array(source, risk_source)
    raw_arr = require_array(source, "limo_raw_risk")
    trav_arr = require_array(source, "limo_trav_cost")
    occupancy_arr = get_array(source, "occupancy")
    stats = SampledStats()

    for start in range(0, num_frames, chunk_images):
        end = min(start + chunk_images, num_frames)
        risk_chunk = np.asarray(risk_arr[start:end], dtype=np.float32)
        raw_chunk = np.asarray(raw_arr[start:end], dtype=np.float32)
        trav_chunk = np.asarray(trav_arr[start:end], dtype=np.float32)
        occupancy_chunk = (
            np.asarray(occupancy_arr[start:end]) if occupancy_arr is not None else None
        )

        for offset in range(end - start):
            occupancy = occupancy_chunk[offset] if occupancy_chunk is not None else None
            _, valid_mask = build_base_masks(
                raw_chunk[offset],
                trav_chunk[offset],
                risk_chunk[offset],
                occupancy,
            )
            stats.add(risk_chunk[offset][valid_mask])

    finalized = stats.finalize()
    if finalized.get("count", 0) == 0:
        warnings.warn(
            "global_robust_no_values",
            "global_robust found no valid finite risk values; falling back to clip01",
        )
        return None
    p1 = finalized.get("p1")
    p99 = finalized.get("p99")
    if p1 is None or p99 is None or (float(p99) - float(p1)) <= EPS:
        warnings.warn(
            "global_robust_degenerate",
            f"global_robust p1={p1}, p99={p99}; falling back to clip01",
        )
        return None
    return float(p1), float(p99)


def create_output_group(
    output_path: Path,
    num_frames: int,
    height: int,
    width: int,
    args: argparse.Namespace,
    source: Any,
) -> Any:
    if output_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output group already exists: {output_path}. Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)

    output = ZarrV2OutputGroup(output_path)
    vchunk = vector_chunk(num_frames, args.chunk_images)
    mchunk = map_chunk(num_frames, height, width, args.chunk_images)

    image_id = np.asarray(require_array(source, "image_id")[:num_frames], dtype=np.int64)
    timestamp = np.asarray(require_array(source, "timestamp")[:num_frames], dtype=np.float64)
    output.create_dataset("image_id", data=image_id, chunks=vchunk)
    output.create_dataset("timestamp", data=timestamp, chunks=vchunk)
    output.create_dataset(
        "source_row", data=np.arange(num_frames, dtype=np.int64), chunks=vchunk
    )

    elevation_row = get_array(source, "elevation_row")
    if elevation_row is not None:
        output.create_dataset(
            "elevation_row",
            data=np.asarray(elevation_row[:num_frames], dtype=np.int64),
            chunks=vchunk,
        )

    frame_valid = get_array(source, "valid")
    if frame_valid is not None:
        output.create_dataset(
            "frame_valid",
            data=np.asarray(frame_valid[:num_frames], dtype=np.uint8),
            chunks=vchunk,
        )

    path_sample_count = get_array(source, "path_sample_count")
    if path_sample_count is not None:
        output.create_dataset(
            "path_sample_count",
            data=np.asarray(path_sample_count[:num_frames], dtype=np.int32),
            chunks=vchunk,
        )

    output.create_dataset(
        "risk_map",
        shape=(num_frames, height, width),
        chunks=mchunk,
        dtype=np.float16,
        fill_value=np.float16(0.0),
    )
    output.create_dataset(
        "valid_mask",
        shape=(num_frames, height, width),
        chunks=mchunk,
        dtype=np.uint8,
        fill_value=np.uint8(0),
    )
    output.create_dataset(
        "raw_risk",
        shape=(num_frames, height, width),
        chunks=mchunk,
        dtype=np.float16,
    )
    output.create_dataset(
        "trav_cost",
        shape=(num_frames, height, width),
        chunks=mchunk,
        dtype=np.float32,
    )
    output.create_dataset(
        "finite_mask",
        shape=(num_frames, height, width),
        chunks=mchunk,
        dtype=np.uint8,
        fill_value=np.uint8(0),
    )
    return output


def set_output_attrs(
    output: Any,
    args: argparse.Namespace,
    num_frames: int,
    height: int,
    width: int,
    global_range: tuple[float, float] | None,
) -> None:
    attrs = {
        "ptc_version": PTC_VERSION,
        "source_group": args.input_group_name,
        "risk_source": args.risk_source,
        "valid_source": valid_source_text(args.risk_source),
        "risk_meaning": "larger means riskier",
        "map_is_per_image": True,
        "sample_lookup_rule": (
            "build image_id_to_row from ptc_labels/image_id, then use path sample "
            "image_id to fetch risk_map/valid_mask"
        ),
        "orientation": "not_changed_from_bev_labels_from_elevation",
        "normalization": args.normalization,
        "roi_x_min": float(args.roi_x_min),
        "roi_x_max": float(args.roi_x_max),
        "roi_y_min": float(args.roi_y_min),
        "roi_y_max": float(args.roi_y_max),
        "num_frames": int(num_frames),
        "H": int(height),
        "W": int(width),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_risk_note": "limo_raw_risk = 1 - limo_score; larger means riskier",
        "trav_cost_note": (
            "limo_trav_cost is MPPI-style thresholded traversability cost; "
            "not used directly as v1 risk_map"
        ),
        "occupancy_note": (
            "occupancy from bev_labels_from_elevation is a known/finite elevation "
            "mask, not obstacle occupancy"
        ),
        "valid_note": "valid is a frame-level flag, not a per-pixel valid mask",
    }
    if global_range is not None:
        attrs["global_robust_p1"] = float(global_range[0])
        attrs["global_robust_p99"] = float(global_range[1])
    output.attrs.update(attrs)


def post_check_output(source: Any, output: Any, num_frames: int) -> dict[str, Any]:
    check_count = min(5, num_frames)
    checks: dict[str, Any] = {
        "checked_rows": int(check_count),
        "ok": True,
        "failures": [],
    }
    if check_count == 0:
        return checks

    src_image_id = np.asarray(require_array(source, "image_id")[:check_count])
    out_image_id = np.asarray(output["image_id"][:check_count])
    if not np.array_equal(src_image_id, out_image_id):
        checks["ok"] = False
        checks["failures"].append("image_id mismatch in first rows")

    src_timestamp = np.asarray(require_array(source, "timestamp")[:check_count])
    out_timestamp = np.asarray(output["timestamp"][:check_count])
    if not np.allclose(src_timestamp, out_timestamp, equal_nan=True):
        checks["ok"] = False
        checks["failures"].append("timestamp mismatch in first rows")

    risk_map = np.asarray(output["risk_map"][:check_count], dtype=np.float32)
    valid_mask = np.asarray(output["valid_mask"][:check_count], dtype=np.uint8)
    checks["risk_map_shape_prefix"] = list(risk_map.shape)
    checks["valid_mask_nonzero_counts"] = [
        int(np.count_nonzero(valid_mask[i])) for i in range(check_count)
    ]
    if any(count == 0 for count in checks["valid_mask_nonzero_counts"]):
        checks["ok"] = False
        checks["failures"].append("valid_mask is all zero for at least one checked row")
    if risk_map.size > 0 and (float(np.min(risk_map)) < -EPS or float(np.max(risk_map)) > 1.0 + EPS):
        checks["ok"] = False
        checks["failures"].append("risk_map values outside [0,1] in checked rows")
    invalid_values = risk_map[valid_mask == 0]
    if invalid_values.size > 0 and np.any(np.abs(invalid_values) > EPS):
        checks["ok"] = False
        checks["failures"].append("invalid risk_map area is not zero")
    return checks


def process_mission(args: argparse.Namespace, mission: str) -> dict[str, Any]:
    warnings = WarningCollector()
    mission_dir = args.dataset_root / mission
    input_path = mission_dir / args.input_group_name
    output_path = mission_dir / args.output_group_name
    roi = {
        "x_min": float(args.roi_x_min),
        "x_max": float(args.roi_x_max),
        "y_min": float(args.roi_y_min),
        "y_max": float(args.roi_y_max),
    }

    summary: dict[str, Any] = {
        "mission": mission,
        "input_group": args.input_group_name,
        "output_group": args.output_group_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "dry_run": bool(args.dry_run),
        "risk_source": args.risk_source,
        "normalization": args.normalization,
        "valid_source": valid_source_text(args.risk_source),
        "roi": roi,
        "ptc_version": PTC_VERSION,
        "ready_for_training": False,
        "warnings": [],
    }

    if not mission_dir.exists():
        raise FileNotFoundError(f"mission directory not found: {mission_dir}")
    if not input_path.exists():
        raise FileNotFoundError(f"input group not found: {input_path}")
    if output_path.exists() and args.dry_run and not args.overwrite:
        warnings.warn(
            "output_exists",
            f"output group already exists and would require --overwrite: {output_path}",
        )

    source = ZarrV2Group(input_path)
    image_id_arr = require_array(source, "image_id")
    timestamp_arr = require_array(source, "timestamp")
    raw_arr = require_array(source, "limo_raw_risk")
    trav_arr = require_array(source, "limo_trav_cost")
    risk_arr = require_array(source, args.risk_source)

    source_num_frames = array_shape(raw_arr)[0]
    num_frames = source_num_frames
    if args.max_images is not None:
        num_frames = min(num_frames, int(args.max_images))
    if num_frames <= 0:
        raise ValueError("no frames selected for processing")

    height, width = validate_shapes(
        num_frames,
        array_shape(risk_arr),
        array_shape(raw_arr),
        array_shape(trav_arr),
        array_shape(image_id_arr),
        array_shape(timestamp_arr),
    )

    occupancy_arr = get_array(source, "occupancy")
    frame_valid_arr = get_array(source, "valid")
    path_sample_count_arr = get_array(source, "path_sample_count")
    optional_arrays = {
        "elevation_row": get_array(source, "elevation_row") is not None,
        "occupancy": occupancy_arr is not None,
        "valid": frame_valid_arr is not None,
        "path_sample_count": path_sample_count_arr is not None,
        "limo_score": get_array(source, "limo_score") is not None,
    }
    if occupancy_arr is None:
        warnings.warn(
            "occupancy_missing",
            "occupancy is missing; valid_mask uses finite raw/trav/risk values only",
        )
    elif array_shape(occupancy_arr) != array_shape(raw_arr):
        raise ValueError(
            f"occupancy shape {array_shape(occupancy_arr)} != limo_raw_risk shape {array_shape(raw_arr)}"
        )
    if frame_valid_arr is not None and array_shape(frame_valid_arr) != (source_num_frames,):
        raise ValueError(
            f"valid shape {array_shape(frame_valid_arr)} != [{source_num_frames}]"
        )
    if path_sample_count_arr is not None and array_shape(path_sample_count_arr) != (source_num_frames,):
        raise ValueError(
            f"path_sample_count shape {array_shape(path_sample_count_arr)} != [{source_num_frames}]"
        )

    image_ids = np.asarray(image_id_arr[:num_frames], dtype=np.int64)
    image_id_unique_count = int(np.unique(image_ids).size)
    if image_id_unique_count == 0:
        warnings.warn("image_id_empty", "image_id unique count is zero")

    global_range: tuple[float, float] | None = None
    if args.normalization == "global_robust":
        global_range = estimate_global_robust_range(
            source, num_frames, args.chunk_images, args.risk_source, warnings
        )

    output = None
    if not args.dry_run:
        output = create_output_group(output_path, num_frames, height, width, args, source)
        set_output_attrs(output, args, num_frames, height, width, global_range)

    risk_stats = SampledStats()
    risk_all_stats = SampledStats()
    raw_stats = SampledStats()
    trav_stats = SampledStats()
    valid_ratios: list[float] = []
    empty_valid_frames = 0
    raw_out_of_range_count = 0
    raw_finite_count = 0

    for start in range(0, num_frames, args.chunk_images):
        end = min(start + args.chunk_images, num_frames)
        count = end - start
        raw_chunk = np.asarray(raw_arr[start:end], dtype=np.float32)
        if args.risk_source == "limo_raw_risk":
            risk_chunk = raw_chunk
        else:
            risk_chunk = np.asarray(risk_arr[start:end], dtype=np.float32)
        trav_chunk = np.asarray(trav_arr[start:end], dtype=np.float32)
        occupancy_chunk = (
            np.asarray(occupancy_arr[start:end]) if occupancy_arr is not None else None
        )

        finite_mask_chunk = (
            np.isfinite(raw_chunk) & np.isfinite(trav_chunk) & np.isfinite(risk_chunk)
        )
        valid_mask_chunk = finite_mask_chunk.copy()
        if occupancy_chunk is not None:
            valid_mask_chunk &= occupancy_chunk > 0

        if args.normalization == "clip01":
            risk_map_chunk = np.clip(risk_chunk, 0.0, 1.0)
            risk_map_chunk[~valid_mask_chunk] = 0.0
            risk_out = risk_map_chunk.astype(np.float16)
            valid_out = valid_mask_chunk.astype(np.uint8)
            finite_out = finite_mask_chunk.astype(np.uint8)
        elif args.normalization == "global_robust" and global_range is not None:
            p1, p99 = global_range
            risk_map_chunk = np.clip((risk_chunk - p1) / (p99 - p1), 0.0, 1.0)
            risk_map_chunk[~valid_mask_chunk] = 0.0
            risk_out = risk_map_chunk.astype(np.float16)
            valid_out = valid_mask_chunk.astype(np.uint8)
            finite_out = finite_mask_chunk.astype(np.uint8)
        else:
            risk_out = np.zeros((count, height, width), dtype=np.float16)
            valid_out = np.zeros((count, height, width), dtype=np.uint8)
            finite_out = np.zeros((count, height, width), dtype=np.uint8)
            risk_map_chunk = np.zeros((count, height, width), dtype=np.float32)
            for offset in range(count):
                source_row = start + offset
                valid_mask = valid_mask_chunk[offset]
                risk_map = normalize_image(
                    risk_chunk[offset],
                    valid_mask,
                    args.normalization,
                    warnings,
                    source_row,
                    global_range,
                )
                risk_map[~valid_mask] = 0.0
                risk_map_chunk[offset] = risk_map
                finite_out[offset] = finite_mask_chunk[offset].astype(np.uint8)
                valid_out[offset] = valid_mask.astype(np.uint8)
                risk_out[offset] = risk_map.astype(np.float16)

        valid_ratio_chunk = valid_mask_chunk.reshape(count, -1).mean(axis=1)
        valid_ratios.extend(float(x) for x in valid_ratio_chunk)
        empty_valid_frames += int(np.count_nonzero(valid_ratio_chunk == 0.0))

        risk_stats.add(risk_map_chunk[valid_mask_chunk])
        risk_all_stats.add(risk_map_chunk)
        raw_stats.add(raw_chunk)
        trav_stats.add(trav_chunk)

        finite_raw = raw_chunk[np.isfinite(raw_chunk)]
        raw_finite_count += int(finite_raw.size)
        if finite_raw.size > 0:
            raw_out_of_range_count += int(
                np.count_nonzero((finite_raw < -EPS) | (finite_raw > 1.0 + EPS))
            )

        if output is not None:
            output["risk_map"][start:end] = risk_out
            output["valid_mask"][start:end] = valid_out
            output["raw_risk"][start:end] = raw_chunk.astype(np.float16)
            output["trav_cost"][start:end] = trav_chunk.astype(np.float32)
            output["finite_mask"][start:end] = finite_out

    if raw_out_of_range_count > 0:
        ratio = raw_out_of_range_count / max(raw_finite_count, 1)
        warnings.warn(
            "raw_risk_out_of_range",
            f"{raw_out_of_range_count}/{raw_finite_count} finite limo_raw_risk "
            f"values ({ratio:.6f}) are outside [0,1] before normalization",
        )
    if empty_valid_frames > 0:
        warnings.warn(
            "empty_valid_frames",
            f"{empty_valid_frames}/{num_frames} frames have no valid pixels",
        )

    frame_valid_available = frame_valid_arr is not None
    num_invalid_frames: int | None = None
    if frame_valid_arr is not None:
        frame_valid = np.asarray(frame_valid_arr[:num_frames])
        num_invalid_frames = int(np.count_nonzero(frame_valid == 0))

    valid_ratio_arr = np.asarray(valid_ratios, dtype=np.float64)
    valid_ratio_stats = finite_stats(valid_ratio_arr)
    risk_map_stats = risk_stats.finalize()
    summary.update(
        {
            "status": "dry_run_ready" if args.dry_run else "generated",
            "source_num_frames": int(source_num_frames),
            "num_frames": int(num_frames),
            "H": int(height),
            "W": int(width),
            "map_shape": [int(num_frames), int(height), int(width)],
            "optional_arrays_present": optional_arrays,
            "image_id_unique_count": image_id_unique_count,
            "risk_map_stats": risk_map_stats,
            "risk_map_all_pixels_stats": risk_all_stats.finalize(),
            "valid_ratio": valid_ratio_stats,
            "raw_risk_original_stats": raw_stats.finalize(),
            "trav_cost_finite_stats": trav_stats.finalize(),
            "num_invalid_frames": num_invalid_frames,
            "frame_valid_available": frame_valid_available,
            "num_empty_valid_frames": int(empty_valid_frames),
            "global_robust_range": list(global_range) if global_range is not None else None,
        }
    )

    post_checks = {"ok": True, "checked_rows": 0, "failures": []}
    if output is not None:
        post_checks = post_check_output(source, output, num_frames)
    summary["post_checks"] = post_checks

    valid_median = valid_ratio_stats.get("p50")
    ready_for_training = (
        num_frames > 0
        and height > 0
        and width > 0
        and image_id_unique_count > 0
        and valid_median is not None
        and float(valid_median) > 0.05
        and bool(post_checks.get("ok", True))
    )
    summary["ready_for_training"] = bool(ready_for_training)
    summary["warnings"] = warnings.finalize()

    if output is not None:
        summary_path = output_path / "summary.json"
        write_json(summary_path, summary)
        summary["summary_json"] = str(summary_path)
    else:
        summary["summary_json"] = None

    return summary


def discover_missions(dataset_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in dataset_root.iterdir()
        if path.is_dir() and MISSION_RE.fullmatch(path.name)
    )


def failed_summary(
    args: argparse.Namespace, mission: str, exc: BaseException, status: str = "failed"
) -> dict[str, Any]:
    mission_dir = args.dataset_root / mission
    return {
        "mission": mission,
        "input_group": args.input_group_name,
        "output_group": args.output_group_name,
        "input_path": str(mission_dir / args.input_group_name),
        "output_path": str(mission_dir / args.output_group_name),
        "dry_run": bool(args.dry_run),
        "status": status,
        "ready_for_training": False,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "warnings": [],
    }


def build_global_summary(
    args: argparse.Namespace,
    missions: list[str],
    mission_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    ready = [s for s in mission_summaries if s.get("ready_for_training")]
    failed = [
        s
        for s in mission_summaries
        if s.get("status") in {"failed", "missing_input_group"}
        or bool(s.get("error"))
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(args.dataset_root),
        "missions_requested": missions,
        "missions_processed": len(mission_summaries),
        "missions_ready": len(ready),
        "missions_failed": len(failed),
        "input_group_name": args.input_group_name,
        "output_group_name": args.output_group_name,
        "risk_source": args.risk_source,
        "normalization": args.normalization,
        "roi": {
            "x_min": float(args.roi_x_min),
            "x_max": float(args.roi_x_max),
            "y_min": float(args.roi_y_min),
            "y_max": float(args.roi_y_max),
        },
        "dry_run": bool(args.dry_run),
        "summary_out": str(args.summary_out),
        "mission_summaries": mission_summaries,
        "recommended_next_step": (
            "Use ptc_labels/image_id to map each path sample image_id to "
            "ptc_labels/risk_map and ptc_labels/valid_mask during PTC training."
        ),
    }


def print_terminal_summary(summary: dict[str, Any]) -> None:
    roi = summary["roi"]
    suffix = " (dry-run, not written)" if summary.get("dry_run") else ""
    print("[PTC LABEL GENERATION SUMMARY]")
    print(f"missions_processed: {summary['missions_processed']}")
    print(f"missions_ready: {summary['missions_ready']}")
    print(f"missions_failed: {summary['missions_failed']}")
    print(f"output_group_name: {summary['output_group_name']}")
    print(f"risk_source: {summary['risk_source']}")
    print(f"normalization: {summary['normalization']}")
    print(
        "roi: "
        f"x=[{roi['x_min']}, {roi['x_max']}], "
        f"y=[{roi['y_min']}, {roi['y_max']}]"
    )
    print(f"summary_out: {summary['summary_out']}{suffix}")
    print(f"recommended_next_step: {summary['recommended_next_step']}")


def main() -> int:
    args = parse_args()
    args.dataset_root = args.dataset_root.expanduser()
    args.summary_out = args.summary_out.expanduser()

    if not args.dataset_root.exists():
        print(f"dataset root not found: {args.dataset_root}", file=sys.stderr)
        return 2

    missions = discover_missions(args.dataset_root) if args.all_missions else [args.mission]
    mission_summaries: list[dict[str, Any]] = []

    for mission in missions:
        try:
            mission_summaries.append(process_mission(args, mission))
        except FileNotFoundError as exc:
            status = "missing_input_group" if args.input_group_name in str(exc) else "failed"
            mission_summaries.append(failed_summary(args, mission, exc, status=status))
        except Exception as exc:
            mission_summaries.append(failed_summary(args, mission, exc))

    global_summary = build_global_summary(args, missions, mission_summaries)
    if not args.dry_run:
        write_json(args.summary_out, global_summary)
    print_terminal_summary(global_summary)

    return 1 if global_summary["missions_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
