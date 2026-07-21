from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from dataset_builder.reachability.coordinates import (
    MapGeometry,
    coordinate_validation_records,
)
from dataset_builder.reachability.teacher_a import (
    DiagnosticTarget,
    ReachabilityState,
    TeacherAResult,
)
from dataset_builder.reachability.traversability import TraversabilityResult


STATE_COLORS = [
    "#000000",  # outside
    "#a8a8a8",  # unknown centre
    "#e53935",  # locally blocked
    "#7f0000",  # clearance blocked
    "#666666",  # unknown footprint
    "#f59e0b",  # disconnected
    "#2ca25f",  # reachable
]
STATE_CMAP = ListedColormap(STATE_COLORS)
STATE_NORM = BoundaryNorm(np.arange(-0.5, 7.5, 1.0), STATE_CMAP.N)

TARGET_COLORS = {
    "reachable_near": "#00ffff",
    "reachable_mid": "#ff00ff",
    "reachable_far": "#ffff00",
    "traversable_but_disconnected": "#ff9800",
    "clearance_blocked": "#ff1744",
    "unknown": "#ffffff",
}


def _format_bev_axis(ax, geometry: MapGeometry, title: str) -> None:
    extent = geometry.imshow_extent_yx
    # Array rows are x and columns are y.  No rot90/flip is applied to data.
    # Explicitly reverse the displayed y axis so physical robot-left is page-left.
    ax.set_xlim(extent[1], extent[0])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xlabel("y [m] (left +)", fontsize=7)
    ax.set_ylabel("x [m] (forward +)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.set_title(title, fontsize=8)
    ax.axhline(0.0, color="white", alpha=0.2, linewidth=0.5)
    ax.axvline(0.0, color="white", alpha=0.2, linewidth=0.5)


def imshow_bev(
    ax,
    data: np.ndarray,
    geometry: MapGeometry,
    *,
    title: str,
    cmap="viridis",
    vmin=None,
    vmax=None,
    norm=None,
):
    image = ax.imshow(
        data,
        origin="lower",
        extent=geometry.imshow_extent_yx,
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        norm=norm,
        aspect="equal",
    )
    _format_bev_axis(ax, geometry, title)
    return image


def overlay_root(ax, result: TeacherAResult) -> None:
    ri, rj = result.root.index
    if 0 <= ri < result.geometry.height and 0 <= rj < result.geometry.width:
        xy = result.geometry.map_idx_to_world((ri, rj))
        ax.plot(
            xy[1],
            xy[0],
            marker="o",
            markersize=7,
            markerfacecolor="#0066ff",
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=20,
        )


def overlay_invalid_root_notice(ax, result: TeacherAResult) -> None:
    if result.root.configuration_valid:
        return
    ax.text(
        0.5,
        0.5,
        f"INVALID ROOT\n{result.root.failure_reason}",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.72, "edgecolor": "white"},
        zorder=30,
    )


def overlay_targets_and_paths(
    ax,
    result: TeacherAResult,
    targets: Iterable[DiagnosticTarget],
) -> None:
    for target in targets:
        color = TARGET_COLORS.get(target.category, "white")
        target_xy = result.geometry.map_idx_to_world(target.index)
        ax.plot(
            target_xy[1],
            target_xy[0],
            marker="D",
            color=color,
            markeredgecolor="black",
            markersize=5,
            zorder=18,
        )
        if target.path is not None:
            xy = result.geometry.map_idx_to_world(target.path)
            ax.plot(xy[:, 1], xy[:, 0], color=color, linewidth=1.5, zorder=17)
    overlay_root(ax, result)


def state_legend_handles():
    handles = []
    labels = [
        "outside",
        "unknown center",
        "local blocked",
        "clearance blocked",
        "unknown footprint",
        "disconnected",
        "reachable",
    ]
    for color, label in zip(STATE_COLORS, labels):
        handles.append(plt.Line2D([0], [0], marker="s", color=color, linestyle="", label=label))
    return handles


def known_display(result: TeacherAResult) -> np.ndarray:
    display = np.zeros(result.state.shape, dtype=np.uint8)
    display[result.planning_domain & ~result.known_trav] = 1
    display[result.planning_domain & result.known_trav] = 2
    return display


def local_traversability_display(result: TeacherAResult) -> np.ndarray:
    display = np.zeros(result.state.shape, dtype=np.uint8)
    display[result.unknown_center] = 1
    display[result.local_blocked] = 2
    display[result.local_traversable] = 3
    return display


def configuration_display(result: TeacherAResult) -> np.ndarray:
    display = np.zeros(result.state.shape, dtype=np.uint8)
    display[result.unknown_center] = 1
    display[result.local_blocked] = 2
    display[result.clearance_blocked] = 3
    display[result.unknown_footprint] = 4
    display[result.configuration_free] = 5
    return display


def difference_display(result: TeacherAResult) -> np.ndarray:
    """Decompose why local traversability does not become reachability."""
    display = np.zeros(result.state.shape, dtype=np.uint8)
    display[result.local_blocked | result.unknown_center] = 1
    display[result.clearance_blocked] = 2
    display[result.unknown_footprint] = 3
    display[result.traversable_but_disconnected] = 4
    display[result.reachable] = 5
    return display


def save_coordinate_validation(
    geometry: MapGeometry, output_png: Path, output_json: Path
) -> list[dict]:
    records = coordinate_validation_records(geometry)
    fig, ax = plt.subplots(figsize=(7, 7))
    background = np.zeros((geometry.height, geometry.width), dtype=np.float32)
    imshow_bev(
        ax,
        background,
        geometry,
        title="Coordinate validation — raw data, no rot90/flip; y display explicitly reversed",
        cmap=ListedColormap(["#eeeeee"]),
        vmin=0,
        vmax=1,
    )
    colors = {"root": "blue", "front": "green", "left": "orange", "right": "purple"}
    for record in records:
        x, y = record["world_xy"]
        name = record["name"]
        ax.plot(y, x, "o", color=colors[name], markersize=9, zorder=10)
        ax.annotate(
            f"{name}\nxy={record['world_xy']}\nij={record['map_ij']}",
            (y, x),
            xytext=(7, 7),
            textcoords="offset points",
            fontsize=8,
            color=colors[name],
            zorder=11,
        )
    ax.arrow(0, 0, 0, 0.75, width=0.01, color="green", length_includes_head=True)
    ax.text(0, 0.82, "+x forward", color="green", ha="center", fontsize=9)
    ax.arrow(0, 0, 0.75, 0, width=0.01, color="orange", length_includes_head=True)
    ax.text(0.82, 0, "+y left", color="orange", va="center", fontsize=9)
    ax.set_xlim(1.5, -1.5)
    ax.set_ylim(-1.5, 1.5)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=180)
    plt.close(fig)
    output_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records


def save_frame_overview(
    *,
    rgb: np.ndarray,
    elevation: np.ndarray,
    trav: TraversabilityResult,
    result: TeacherAResult,
    targets: list[DiagnosticTarget],
    title: str,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.ravel()
    axes[0].imshow(rgb)
    axes[0].set_title("Front RGB", fontsize=8)
    axes[0].axis("off")

    finite_elev = elevation[np.isfinite(elevation)]
    elev_min, elev_max = (
        np.percentile(finite_elev, [2, 98]) if len(finite_elev) else (-1.0, 1.0)
    )
    elev_cmap = plt.get_cmap("terrain").copy()
    elev_cmap.set_bad("#888888")
    imshow_bev(
        axes[1], elevation, result.geometry, title="Raw elevation", cmap=elev_cmap,
        vmin=float(elev_min), vmax=float(elev_max)
    )

    known_cmap = ListedColormap(["black", "#999999", "#f7f7f7"])
    imshow_bev(
        axes[2], known_display(result), result.geometry,
        title="Known / unknown / outside", cmap=known_cmap, vmin=0, vmax=2
    )

    risk_cmap = plt.get_cmap("magma").copy()
    risk_cmap.set_bad("#888888")
    risk_img = imshow_bev(
        axes[3], trav.risk, result.geometry,
        title="Continuous risk = 1 - score", cmap=risk_cmap, vmin=0, vmax=1
    )
    fig.colorbar(risk_img, ax=axes[3], fraction=0.045, pad=0.02)

    local_cmap = ListedColormap(["black", "#999999", "#e53935", "#2ca25f"])
    imshow_bev(
        axes[4], local_traversability_display(result), result.geometry,
        title="Local traversability", cmap=local_cmap, vmin=0, vmax=3
    )

    config_cmap = ListedColormap(
        ["black", "#aaaaaa", "#e53935", "#7f0000", "#555555", "#2ca25f"]
    )
    imshow_bev(
        axes[5], configuration_display(result), result.geometry,
        title=f"Configuration-free (r={result.inflation_radius_m:.2f} m; grid={result.effective_radius_m:.2f} m)",
        cmap=config_cmap, vmin=0, vmax=5
    )
    overlay_root(axes[5], result)

    imshow_bev(
        axes[6], result.state, result.geometry,
        title="Full-domain connectivity state", cmap=STATE_CMAP, norm=STATE_NORM
    )
    overlay_root(axes[6], result)
    overlay_invalid_root_notice(axes[6], result)
    axes[6].legend(handles=state_legend_handles(), fontsize=5, loc="lower left", framealpha=0.7)

    geo_cmap = plt.get_cmap("viridis").copy()
    geo_cmap.set_bad("#222222")
    geo_img = imshow_bev(
        axes[7], result.geodesic_m, result.geometry,
        title="Root geometric geodesic [m]", cmap=geo_cmap
    )
    fig.colorbar(geo_img, ax=axes[7], fraction=0.045, pad=0.02)
    overlay_root(axes[7], result)
    overlay_invalid_root_notice(axes[7], result)

    imshow_bev(
        axes[8], result.state, result.geometry,
        title="Diagnostic targets and predecessor paths", cmap=STATE_CMAP, norm=STATE_NORM
    )
    overlay_targets_and_paths(axes[8], result, targets)
    overlay_invalid_root_notice(axes[8], result)

    difference_cmap = ListedColormap(
        ["black", "#444444", "#b2182b", "#999999", "#f59e0b", "#2ca25f"]
    )
    imshow_bev(
        axes[9], difference_display(result), result.geometry,
        title="Difference: base / clearance / unknown support / disconnected / reachable",
        cmap=difference_cmap, vmin=0, vmax=5
    )
    overlay_root(axes[9], result)

    status = (
        "valid" if result.root.configuration_valid else f"INVALID: {result.root.failure_reason}"
    )
    fig.suptitle(f"{title} — canonical full-domain connectivity — root {status}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_radius_comparison(
    *,
    results: list[TeacherAResult],
    targets_by_radius: list[list[DiagnosticTarget]],
    title: str,
    output_path: Path,
) -> None:
    columns = len(results)
    fig, axes = plt.subplots(4, columns, figsize=(6.2 * columns, 20), squeeze=False)
    finite_values = np.concatenate(
        [result.geodesic_m[np.isfinite(result.geodesic_m)] for result in results]
    )
    geo_max = float(np.percentile(finite_values, 98)) if len(finite_values) else 1.0
    config_cmap = ListedColormap(
        ["black", "#aaaaaa", "#e53935", "#7f0000", "#555555", "#2ca25f"]
    )
    difference_cmap = ListedColormap(
        ["black", "#444444", "#b2182b", "#999999", "#f59e0b", "#2ca25f"]
    )
    geo_cmap = plt.get_cmap("viridis").copy()
    geo_cmap.set_bad("#222222")

    for col, (result, targets) in enumerate(zip(results, targets_by_radius)):
        radius_title = f"requested r={result.inflation_radius_m:.2f} m (grid r={result.effective_radius_m:.2f} m)"
        imshow_bev(
            axes[0, col], configuration_display(result), result.geometry,
            title=f"{radius_title}\nConfiguration-free", cmap=config_cmap, vmin=0, vmax=5
        )
        overlay_root(axes[0, col], result)
        imshow_bev(
            axes[1, col], result.state, result.geometry,
            title="Full-domain reachability state", cmap=STATE_CMAP, norm=STATE_NORM
        )
        overlay_targets_and_paths(axes[1, col], result, targets)
        overlay_invalid_root_notice(axes[1, col], result)
        imshow_bev(
            axes[2, col], result.geodesic_m, result.geometry,
            title="Root geometric geodesic [m]", cmap=geo_cmap, vmin=0, vmax=geo_max
        )
        overlay_targets_and_paths(axes[2, col], result, targets)
        overlay_invalid_root_notice(axes[2, col], result)
        imshow_bev(
            axes[3, col], difference_display(result), result.geometry,
            title="Difference sources", cmap=difference_cmap, vmin=0, vmax=5
        )
        overlay_root(axes[3, col], result)
        status = "valid" if result.root.configuration_valid else result.root.failure_reason
        axes[0, col].text(
            0.02, 0.02, f"root: {status}", transform=axes[0, col].transAxes,
            fontsize=7, color="white", bbox={"facecolor": "black", "alpha": 0.6}
        )

    fig.suptitle(
        f"{title}\nThree circular footprint approximations; canonical full-domain connectivity",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=145)
    plt.close(fig)
