from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import zarr


LABEL_COLORS = ["#9e9e9e", "#e53935", "#f39c12", "#2e7d32", "#000000"]
LABEL_NAMES = ["unknown", "blocked", "disconnected_free", "reachable", "ignore"]
TARGET_X0 = 0.0
TARGET_Y0 = -3.0
TARGET_RESOLUTION = 0.1
TARGET_SHAPE = (40, 60)


def choose_rows(group, num_samples: int, seed: int, rows: str | None, topology_only: bool):
    if rows:
        values = [int(x.strip()) for x in rows.split(",") if x.strip()]
        return np.asarray(values, dtype=np.int64)
    valid = np.asarray(group["label_valid"][:], dtype=bool)
    if topology_only and "state" in group:
        state = np.asarray(group["state"][:])
        valid &= np.any(state == 2, axis=(1, 2))
    positions = np.flatnonzero(valid)
    if not len(positions):
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(positions, size=min(num_samples, len(positions)), replace=False))


def add_paths(ax, mission: Path, image_id: int):
    found = False
    for source, color in (("geometric_paths", "#1565c0"), ("teleop_paths", "#8e24aa")):
        path_dir = mission / "data" / source
        if not path_dir.exists():
            continue
        group = zarr.open_group(str(path_dir), mode="r")
        ids = np.asarray(group["image_id"][:], dtype=np.int64)
        for row in np.flatnonzero(ids == image_id):
            path = np.asarray(group["path"][int(row)], dtype=np.float32)
            x = np.floor((path[:, 0] - TARGET_X0) / TARGET_RESOLUTION).astype(int)
            y = np.floor((path[:, 1] - TARGET_Y0) / TARGET_RESOLUTION).astype(int)
            ok = (x >= 0) & (x < TARGET_SHAPE[0]) & (y >= 0) & (y < TARGET_SHAPE[1])
            if ok.any():
                ax.plot((TARGET_SHAPE[1] - 1) - y[ok], x[ok], color=color, linewidth=1.0, alpha=0.75, label=source)
                found = True
    return found


def visualize(mission: Path, output_dir: Path, num_samples: int, seed: int, rows: str | None, topology_only: bool):
    labels_dir = mission / "reachability_labels_minimal"
    group = zarr.open_group(str(labels_dir), mode="r")
    selected = choose_rows(group, num_samples, seed, rows, topology_only)
    output_dir.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(LABEL_COLORS)
    for row in selected:
        row = int(row)
        image_id = int(group["image_id"][row])
        label = np.asarray(group["state"][row], dtype=np.uint8)
        valid = bool(group["label_valid"][row])
        root_valid = bool(group["root_valid"][row])
        reason = str(group["root_invalid_reason"][row])
        fig, axes = plt.subplots(1, 2, figsize=(14, 7), constrained_layout=True)
        image_path = mission / "images" / "hdr_front" / f"{image_id:06d}.jpeg"
        if image_path.exists():
            axes[0].imshow(plt.imread(image_path))
        else:
            axes[0].text(0.5, 0.5, "front RGB unavailable", ha="center", va="center")
        axes[0].set_title(f"front RGB | image_id={image_id}")
        axes[0].axis("off")
        display = np.full(label.shape, 4, dtype=np.uint8)
        display[label == 0] = 0
        display[label == 1] = 1
        display[label == 2] = 2
        display[label == 3] = 3
        axes[1].imshow(np.fliplr(display), origin="lower", cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[1].set_title(f"four-state [40,60] | label_valid={valid} | root_valid={root_valid}")
        axes[1].set_xlabel("left = robot-left; columns reversed for display")
        axes[1].set_ylabel("x-forward rows")
        axes[1].set_xticks([0, 20, 40, 59], labels=["3", "1", "-1", "-3"])
        axes[1].set_yticks([0, 20, 39], labels=["0", "2", "4"])
        has_path = add_paths(axes[1], mission, image_id)
        if has_path:
            axes[1].legend(fontsize=7)
        handles = [plt.Line2D([], [], color=c, linewidth=8, label=n) for c, n in zip(LABEL_COLORS, LABEL_NAMES)]
        axes[1].legend(handles=handles, loc="upper right", fontsize=7)
        fig.suptitle(f"{mission.name} | row={row} | image_id={image_id} | reason={reason}", fontsize=13)
        fig.savefig(output_dir / f"reachability_row_{row:06d}_image_{image_id:06d}.png", dpi=180)
        plt.close(fig)
    print(f"generated {len(selected)} visualizations in {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows", type=str, default=None, help="sidecar row indices, not image_id")
    parser.add_argument("--topology-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or (args.mission / "reachability_labels_minimal" / "visualizations_check")
    visualize(args.mission, output, args.num_samples, args.seed, args.rows, args.topology_only)


if __name__ == "__main__":
    main()

