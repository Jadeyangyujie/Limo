"""Local collision-debug visualizations for open-loop evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_collision_debug_visualization(
    output_dir: Path,
    *,
    dataset: str,
    method_id: str,
    metric: Any,
    store: Any,
    sample: Any,
    path: np.ndarray,
    result: Any,
) -> Path:
    """Save the exact elevation/traversability state used for one collision."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    gridmap = metric.gridmap_for_sample(store, sample)
    metric.set_goal(gridmap, sample.goal)
    elevation = gridmap.elevation.detach().cpu().numpy()
    traversability = metric.objective._trav.detach().cpu().numpy()
    fatal = np.isfinite(traversability) & (
        traversability >= float(metric.mppi_cfg.fatal_value)
    )
    unknown = np.isnan(traversability)

    resolution = float(gridmap.resolution)
    origin_x, origin_y = gridmap.origin_xy.detach().cpu().numpy().tolist()
    height, width = elevation.shape
    extent = (
        origin_x,
        origin_x + height * resolution,
        origin_y,
        origin_y + width * resolution,
    )
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), constrained_layout=True)
    _draw_elevation(axes[0], elevation, extent)
    _draw_safety(axes[1], traversability, fatal, unknown, extent)
    contacts = _fatal_footprint_contacts(
        path=np.asarray(path, dtype=np.float32),
        footprint_cells=metric.objective._robot_footprint.detach().cpu().numpy(),
        traversability=traversability,
        fatal_value=float(metric.mppi_cfg.fatal_value),
        resolution=resolution,
        origin_xy=(origin_x, origin_y),
    )
    _overlay_path_and_footprints(
        axes,
        np.asarray(path, dtype=np.float32),
        np.asarray(sample.goal, dtype=np.float32),
        metric.objective._robot_footprint.detach().cpu().numpy(),
        contacts,
    )
    contact_steps = [str(contact["waypoint_index"]) for contact in contacts]
    contact_text = ", ".join(contact_steps[:10]) or "none"
    if len(contact_steps) > 10:
        contact_text += ", ..."
    fig.suptitle(
        f"{dataset} / {method_id} | mission={sample.mission}, "
        f"sample={sample.index}, image={sample.image_id}\n"
        f"collision={int(result.collision)}, footprint_any_fatal="
        f"{int(result.footprint_any_fatal)}, max_fatal_fraction="
        f"{result.max_fatal_footprint_fraction:.3f}, out_of_map="
        f"{int(result.out_of_map)}, unknown={int(result.unknown_contact)}\n"
        f"fatal-contact waypoint(s): {contact_text} "
        f"({sum(len(contact['cell_centers']) for contact in contacts)} cell(s))",
        fontsize=10,
    )
    output_path = output_dir / (
        f"{dataset}__{method_id}__{sample.mission}__image_"
        f"{sample.image_id:06d}__sample_{sample.index:06d}.png"
    )
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _draw_elevation(ax: Any, elevation: np.ndarray, extent: tuple[float, ...]) -> None:
    finite = elevation[np.isfinite(elevation)]
    vmin, vmax = (0.0, 1.0) if not finite.size else np.percentile(finite, (2, 98))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1e-6
    image = ax.imshow(
        elevation.T, origin="lower", extent=extent, cmap="terrain", vmin=vmin,
        vmax=vmax, interpolation="nearest", aspect="equal",
    )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="elevation")
    ax.set_title("Elevation map")


def _draw_safety(
    ax: Any,
    traversability: np.ndarray,
    fatal: np.ndarray,
    unknown: np.ndarray,
    extent: tuple[float, ...],
) -> None:
    normal = np.where(np.isfinite(traversability), np.minimum(traversability, 2.0), np.nan)
    image = ax.imshow(
        normal.T, origin="lower", extent=extent, cmap="viridis", vmin=0.0,
        vmax=2.0, interpolation="nearest", aspect="equal",
    )
    ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="traversability")
    ax.imshow(
        np.ma.masked_where(~fatal.T, fatal.T), origin="lower", extent=extent,
        cmap="Reds", vmin=0.0, vmax=1.0, interpolation="nearest", alpha=0.85,
    )
    ax.imshow(
        np.ma.masked_where(~unknown.T, unknown.T), origin="lower", extent=extent,
        cmap="Greys", vmin=0.0, vmax=1.0, interpolation="nearest", alpha=0.45,
    )
    ax.set_title("Traversability (red=fatal, gray=unknown)")


def _overlay_path_and_footprints(
    axes: Any,
    path: np.ndarray,
    goal: np.ndarray,
    footprint_cells: np.ndarray,
    contacts: list[dict[str, Any]],
) -> None:
    ids = np.unique(np.linspace(0, len(path) - 1, min(len(path), 10), dtype=int))
    for index in ids:
        state = path[index]
        cosine, sine = np.cos(state[2]), np.sin(state[2])
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        footprint = footprint_cells @ rotation.T + state[:2]
        for ax in axes:
            ax.scatter(footprint[:, 0], footprint[:, 1], s=2, c="#ff00ff", alpha=0.18, linewidths=0, zorder=3)
    for ax in axes:
        ax.plot(path[:, 0], path[:, 1], c="#00d5ff", lw=2, label="evaluated path", zorder=4)
        ax.scatter([0.0], [0.0], c="white", edgecolor="black", s=45, label="start", zorder=5)
        ax.scatter([goal[0]], [goal[1]], c="#ffe600", edgecolor="black", marker="*", s=110, label="goal", zorder=5)
        for contact_index, contact in enumerate(contacts):
            waypoint_index = contact["waypoint_index"]
            waypoint = path[waypoint_index, :2]
            ax.scatter(
                [waypoint[0]], [waypoint[1]], marker="X", s=75, c="#ff0000",
                edgecolor="white", linewidths=0.7, zorder=7,
                label="fatal-contact waypoint" if contact_index == 0 else None,
            )
            if contact_index < 10:
                ax.annotate(
                    f"t={waypoint_index}", xy=waypoint, xytext=(4, 4),
                    textcoords="offset points", color="#cc0000", fontsize=8,
                    fontweight="bold", zorder=8,
                )
            centers = contact["cell_centers"]
            ax.scatter(
                centers[:, 0], centers[:, 1], marker="s", s=32, c="#ff0000",
                edgecolor="white", linewidths=0.5, zorder=8,
                label="fatal footprint cell" if contact_index == 0 else None,
            )
        ax.set_xlabel("base-frame x [m]")
        ax.set_ylabel("base-frame y [m]")
        ax.legend(loc="upper right", fontsize=8)


def _fatal_footprint_contacts(
    *,
    path: np.ndarray,
    footprint_cells: np.ndarray,
    traversability: np.ndarray,
    fatal_value: float,
    resolution: float,
    origin_xy: tuple[float, float],
) -> list[dict[str, Any]]:
    """Reproduce metrics.py's footprint-to-grid fatal lookup for visualization."""
    height, width = traversability.shape
    origin = np.asarray(origin_xy, dtype=np.float32)
    contacts: list[dict[str, Any]] = []
    for waypoint_index, state in enumerate(path):
        cosine, sine = np.cos(state[2]), np.sin(state[2])
        rotation = np.asarray([[cosine, -sine], [sine, cosine]], dtype=np.float32)
        footprint_world = footprint_cells @ rotation.T + state[:2]
        ij = np.floor((footprint_world - origin) / resolution).astype(np.int64)
        valid = (
            (ij[:, 0] >= 0) & (ij[:, 0] < height) &
            (ij[:, 1] >= 0) & (ij[:, 1] < width)
        )
        is_fatal = np.zeros(len(ij), dtype=bool)
        valid_ij = ij[valid]
        values = traversability[valid_ij[:, 0], valid_ij[:, 1]]
        is_fatal[valid] = np.isfinite(values) & (values >= fatal_value)
        if not is_fatal.any():
            continue
        fatal_ij = np.unique(ij[is_fatal], axis=0)
        centers = origin + (fatal_ij.astype(np.float32) + 0.5) * resolution
        contacts.append({"waypoint_index": waypoint_index, "cell_centers": centers})
    return contacts
