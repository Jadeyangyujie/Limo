from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class MapGeometry:
    """Coordinate contract used by the existing LIMO elevation maps.

    Array axis 0 is robot x (forward), array axis 1 is robot y (left).
    ``origin_xy`` is the world-coordinate anchor of index (0, 0), matching
    ``dataset_builder.mppi_planner.world_to_map_idx``.  An index anchor is
    converted back with ``origin + index * resolution``.  The displayed pixel
    extent extends half a cell around those anchors.

    The existing world-to-map operation is floor quantisation.  Consequently,
    index -> anchor -> index is exact, while an arbitrary world point maps to
    the lower grid anchor with an error in [0, resolution) on each axis.
    """

    height: int
    width: int
    resolution: float
    origin_xy: tuple[float, float]
    frame_id: str = "base"

    def world_to_map_idx(self, xy: np.ndarray | Iterable[float]) -> np.ndarray:
        points = np.asarray(xy, dtype=np.float64)
        origin = np.asarray(self.origin_xy, dtype=np.float64)
        return np.floor((points - origin) / self.resolution).astype(np.int64)

    def map_idx_to_world(self, ij: np.ndarray | Iterable[int]) -> np.ndarray:
        indices = np.asarray(ij, dtype=np.float64)
        origin = np.asarray(self.origin_xy, dtype=np.float64)
        return origin + indices * self.resolution

    def valid_indices(self, ij: np.ndarray) -> np.ndarray:
        indices = np.asarray(ij)
        return (
            (indices[..., 0] >= 0)
            & (indices[..., 0] < self.height)
            & (indices[..., 1] >= 0)
            & (indices[..., 1] < self.width)
        )

    @property
    def root_index(self) -> tuple[int, int]:
        ij = self.world_to_map_idx(np.array([0.0, 0.0]))
        return int(ij[0]), int(ij[1])

    @property
    def imshow_extent_yx(self) -> tuple[float, float, float, float]:
        """Matplotlib extent for data indexed [x, y], without rotating data."""
        ox, oy = self.origin_xy
        half = 0.5 * self.resolution
        return (
            oy - half,
            oy + (self.width - 1) * self.resolution + half,
            ox - half,
            ox + (self.height - 1) * self.resolution + half,
        )

    def fixed_cartesian_domain(self) -> np.ndarray:
        """Return the full rectangular grid support as the domain support."""
        return np.ones((self.height, self.width), dtype=bool)


def coordinate_validation_records(geometry: MapGeometry) -> list[dict]:
    named_points = [
        ("root", (0.0, 0.0)),
        ("front", (1.0, 0.0)),
        ("left", (0.0, 1.0)),
        ("right", (0.0, -1.0)),
    ]
    records: list[dict] = []
    for name, xy in named_points:
        world = np.asarray(xy, dtype=np.float64)
        index = geometry.world_to_map_idx(world)
        anchor = geometry.map_idx_to_world(index)
        records.append(
            {
                "name": name,
                "world_xy": world.tolist(),
                "map_ij": index.tolist(),
                "index_anchor_xy": anchor.tolist(),
                "quantization_error_xy": (world - anchor).tolist(),
                "roundtrip_index": geometry.world_to_map_idx(anchor).tolist(),
            }
        )
    return records
