from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/limo_teacher_a_matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon

from dataset_builder.reachability.path_calibration import (
    DensePath,
    PathEvaluation,
    first_true_index,
)
from dataset_builder.reachability.teacher_a import ReachabilityState, TeacherAResult
from dataset_builder.reachability.traversability import TraversabilityResult
from dataset_builder.reachability.visualization import (
    STATE_CMAP,
    STATE_NORM,
    imshow_bev,
    local_traversability_display,
    overlay_root,
)


PATH_STATE_COLORS = {
    -1: "#000000",
    int(ReachabilityState.OUTSIDE_DOMAIN): "#000000",
    int(ReachabilityState.UNKNOWN_CENTER): "#9e9e9e",
    int(ReachabilityState.LOCALLY_BLOCKED): "#ff2222",
    int(ReachabilityState.CLEARANCE_BLOCKED): "#760000",
    int(ReachabilityState.UNKNOWN_FOOTPRINT): "#666666",
    int(ReachabilityState.TRAVERSABLE_BUT_DISCONNECTED): "#ff9800",
    int(ReachabilityState.REACHABLE): "#00e5ff",
}


def _plot_path_by_state(ax, dense: DensePath, evaluation: PathEvaluation, linewidth=2.2):
    xy = dense.xytheta[:, :2]
    ax.plot(xy[:, 1], xy[:, 0], color="white", linewidth=linewidth + 1.4, alpha=0.8)
    for code, color in PATH_STATE_COLORS.items():
        mask = evaluation.display_state == code
        if mask.any():
            ax.scatter(xy[mask, 1], xy[mask, 0], s=7, c=color, linewidths=0, zorder=15)


def _rectangle_corners(x, y, theta, length, width):
    local = np.array(
        [
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
            [length / 2, width / 2],
            [length / 2, -width / 2],
        ]
    )
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    return local @ rotation.T + np.array([x, y])


def _draw_rectangles(
    ax,
    dense: DensePath,
    evaluation: PathEvaluation,
    length: float,
    width: float,
    spacing_m: float = 0.3,
):
    desired = np.arange(0.0, dense.total_length_m + spacing_m * 0.5, spacing_m)
    indices = sorted(
        set(int(np.argmin(np.abs(dense.arc_length_m - value))) for value in desired)
    )
    for index in indices:
        x, y, theta = dense.xytheta[index]
        corners = _rectangle_corners(x, y, theta, length, width)
        if evaluation.overlaps_blocked[index]:
            color = "#ff2222"
        elif evaluation.overlaps_unknown[index]:
            color = "#888888"
        elif evaluation.crosses_domain[index]:
            color = "black"
        else:
            color = "#00e676"
        ax.add_patch(
            Polygon(
                np.column_stack([corners[:, 1], corners[:, 0]]),
                closed=True,
                fill=False,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.85,
                zorder=17,
            )
        )


def _choose_focus(evaluations: dict[str, PathEvaluation], dense: DensePath):
    candidates = []
    for name in ("rectangle", "circle_0p61", "circle_0p26", "center"):
        first = first_true_index(evaluations[name].forbidden)
        if first is not None:
            candidates.append((float(dense.arc_length_m[first]), name, first))
    return min(candidates, default=(None, None, None), key=lambda item: item[0] or 0.0)


def save_path_calibration_figure(
    *,
    rgb: np.ndarray,
    elevation: np.ndarray,
    trav: TraversabilityResult,
    results: dict[str, TeacherAResult],
    dense: DensePath,
    evaluations: dict[str, PathEvaluation],
    rectangle_length_m: float,
    rectangle_width_m: float,
    title: str,
    output_path: Path,
) -> None:
    geometry = results["center"].geometry
    fig, axes = plt.subplots(2, 4, figsize=(22, 11))
    ax = axes.ravel()
    ax[0].imshow(rgb)
    ax[0].set_title("Front RGB", fontsize=9)
    ax[0].axis("off")

    finite = elevation[np.isfinite(elevation)]
    vmin, vmax = np.percentile(finite, [2, 98]) if len(finite) else (-1, 1)
    elev_cmap = plt.get_cmap("terrain").copy()
    elev_cmap.set_bad("#888888")
    imshow_bev(
        ax[1], elevation, geometry, title="Elevation + expert centerline",
        cmap=elev_cmap, vmin=float(vmin), vmax=float(vmax)
    )
    xy = dense.xytheta[:, :2]
    ax[1].plot(xy[:, 1], xy[:, 0], color="#00e5ff", linewidth=2)
    overlay_root(ax[1], results["center"])

    local_cmap = ListedColormap(["black", "#999999", "#e53935", "#2ca25f"])
    imshow_bev(
        ax[2], local_traversability_display(results["center"]), geometry,
        title="Local traversability + point-center state", cmap=local_cmap, vmin=0, vmax=3
    )
    _plot_path_by_state(ax[2], dense, evaluations["center"])
    overlay_root(ax[2], results["center"])

    for axis, key, label in (
        (ax[3], "circle_0p26", "r=0.26 m Teacher-A state + path"),
        (ax[4], "circle_0p61", "r=0.61 m Teacher-A state + path"),
    ):
        imshow_bev(
            axis, results[key].state, geometry, title=label, cmap=STATE_CMAP, norm=STATE_NORM
        )
        _plot_path_by_state(axis, dense, evaluations[key])
        overlay_root(axis, results[key])

    imshow_bev(
        ax[5], results["center"].state, geometry,
        title="Yaw-aware rectangular footprint", cmap=STATE_CMAP, norm=STATE_NORM
    )
    _plot_path_by_state(ax[5], dense, evaluations["rectangle"])
    _draw_rectangles(
        ax[5], dense, evaluations["rectangle"], rectangle_length_m, rectangle_width_m
    )
    overlay_root(ax[5], results["center"])

    focus_s, focus_model, focus_index = _choose_focus(evaluations, dense)
    focus_background = results["circle_0p26"]
    imshow_bev(
        ax[6], focus_background.state, geometry,
        title="First violation local view", cmap=STATE_CMAP, norm=STATE_NORM
    )
    _plot_path_by_state(ax[6], dense, evaluations[focus_model] if focus_model else evaluations["center"])
    if focus_index is not None:
        x, y, theta = dense.xytheta[focus_index]
        ax[6].plot(y, x, marker="*", color="#ff00ff", markersize=14, zorder=25)
        corners = _rectangle_corners(x, y, theta, rectangle_length_m, rectangle_width_m)
        ax[6].add_patch(
            Polygon(
                np.column_stack([corners[:, 1], corners[:, 0]]), closed=True,
                fill=False, edgecolor="#ff00ff", linewidth=1.5, zorder=24
            )
        )
        ax[6].set_xlim(y + 0.8, y - 0.8)
        ax[6].set_ylim(x - 0.8, x + 0.8)
        ax[6].set_title(f"First violation: {focus_model} at s={focus_s:.2f} m", fontsize=9)

    strip_names = ["center", "circle_0p26", "circle_0p61", "rectangle"]
    strip = np.stack([evaluations[name].display_state for name in strip_names])
    strip_colors = [PATH_STATE_COLORS[k] for k in (-1, 0, 1, 2, 3, 4, 5, 6)]
    strip_index = np.clip(strip + 1, 0, 7)
    ax[7].imshow(
        strip_index,
        aspect="auto",
        interpolation="nearest",
        origin="upper",
        extent=(0, max(dense.total_length_m, 1e-6), 3.5, -0.5),
        cmap=ListedColormap(strip_colors),
        vmin=0,
        vmax=7,
    )
    ax[7].set_yticks(range(4), ["center", "circle 0.26", "circle 0.61", "rectangle"])
    ax[7].set_xlabel("path arc length [m]")
    ax[7].set_title("Path diagnostic state along arc length", fontsize=9)
    ax[7].grid(axis="x", alpha=0.25)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
