"""Small zarr v2 array reader used by PTC label loading and debug tools."""

from __future__ import annotations

import json
import math
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

try:
    import numcodecs
except Exception:  # pragma: no cover - depends on runtime environment
    numcodecs = None


def _decode_fill_value(value: Any, dtype: np.dtype) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and value == "NaN":
        return np.nan
    if dtype.kind in {"f", "c"} and isinstance(value, float) and math.isnan(value):
        return np.nan
    return value


class ZarrV2Array:
    """Minimal zarr v2 reader for integer indexing and contiguous slicing."""

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
        self.fill_value = _decode_fill_value(self.meta.get("fill_value"), self.dtype)

        compressor_config = self.meta.get("compressor")
        if compressor_config is not None and numcodecs is None:
            raise ImportError(
                f"reading compressed zarr array requires numcodecs: {self.path}"
            )
        self.compressor = (
            numcodecs.get_codec(compressor_config) if compressor_config else None
        )

    def _chunk_name(self, coords: tuple[int, ...]) -> Path:
        if self.dimension_separator == "/":
            return self.path.joinpath(*(str(c) for c in coords))
        return self.path / ".".join(str(c) for c in coords)

    def _empty_chunk(self) -> np.ndarray:
        chunk = np.empty(self.chunks, dtype=self.dtype, order=self.order)
        fill = 0 if self.fill_value is None else self.fill_value
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
                src_slices.append(
                    slice(overlap_start - chunk_start, overlap_stop - chunk_start)
                )
                dst_slices.append(
                    slice(overlap_start - slices[dim].start, overlap_stop - slices[dim].start)
                )
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
                if start < 0 or start >= self.shape[dim]:
                    raise IndexError(f"index {item} is out of bounds for {self.path}")
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
