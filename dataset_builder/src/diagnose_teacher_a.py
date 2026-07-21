"""Small-sample Teacher-A diagnostics; never writes formal dataset labels.

Usage:
  conda run -n limo python -m dataset_builder.src.diagnose_teacher_a
  conda run -n limo python -m dataset_builder.src.diagnose_teacher_a \
      --config dataset_builder/configs/teacher_a_diagnostic.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

# Keep diagnostic rendering self-contained on machines with a read-only home.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/limo_teacher_a_matplotlib")

import numpy as np
from omegaconf import OmegaConf
from scipy import ndimage

from dataset_builder.mppi_planner.traversability_filter import get_filter_torch
from dataset_builder.reachability.coordinates import MapGeometry
from dataset_builder.reachability.teacher_a import (
    TeacherAResult,
    build_teacher_a,
    select_diagnostic_targets,
    validate_reconstructed_path,
)
from dataset_builder.reachability.traversability import compute_limo_traversability
from dataset_builder.reachability.visualization import (
    save_coordinate_validation,
    save_frame_overview,
    save_radius_comparison,
)
from dataset_builder.src.mission_data_source import GrandTourZarrSource

log = logging.getLogger(__name__)


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialise {type(value)}")


def _candidate_indices(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    return sorted(set(np.linspace(0, length - 1, count, dtype=np.int64).tolist()))


def _disconnected_boundary_causes(result: TeacherAResult) -> dict:
    disconnected = result.traversable_but_disconnected
    if not disconnected.any():
        return {"obstacle": 0, "unknown": 0, "domain_edge": 0, "dominant": "none"}
    boundary = ndimage.binary_dilation(disconnected, structure=np.ones((3, 3), bool)) & ~disconnected
    contacts = {
        "obstacle": int(np.count_nonzero(boundary & (result.local_blocked | result.clearance_blocked))),
        "unknown": int(np.count_nonzero(boundary & (result.unknown_center | result.unknown_footprint))),
        "domain_edge": int(np.count_nonzero(boundary & result.outside_domain)),
    }
    contacts["dominant"] = max(contacts, key=contacts.get) if any(contacts.values()) else "unresolved"
    return contacts


def _frame_metrics(
    mission: str,
    frame_idx: int,
    known_raw: np.ndarray,
    result: TeacherAResult,
) -> dict:
    summary = result.summary()
    domain_n = max(int(result.planning_domain.sum()), 1)
    output = {
        "mission": mission,
        "frame_idx": frame_idx,
        "inflation_radius_m": result.inflation_radius_m,
        "effective_radius_m": result.effective_radius_m,
        "known_raw_fraction": float(known_raw.mean()),
        "known_trav_fraction_of_domain": float((result.known_trav & result.planning_domain).sum())
        / domain_n,
        "local_blocked_fraction_of_domain": float(result.local_blocked.sum()) / domain_n,
        "clearance_blocked_fraction_of_domain": float(result.clearance_blocked.sum()) / domain_n,
        "unknown_center_fraction_of_domain": float(result.unknown_center.sum()) / domain_n,
        "unknown_footprint_fraction_of_domain": float(result.unknown_footprint.sum()) / domain_n,
        "configuration_free_fraction_of_domain": summary["configuration_free_fraction_of_domain"],
        "reachable_fraction_of_domain": summary["reachable_fraction_of_domain"],
        "disconnected_fraction_of_domain": summary["disconnected_fraction_of_domain"],
        "root_center_valid": result.root.center_valid,
        "root_configuration_valid": result.root.configuration_valid,
        "root_failure_reason": result.root.failure_reason or "",
        "max_geodesic_m": summary["max_geodesic_m"],
    }
    causes = _disconnected_boundary_causes(result)
    output.update({f"disconnected_contact_{key}": value for key, value in causes.items()})
    return output


def _process_frame(source, frame_idx, geometry, filter_model, mppi_cfg, cfg):
    elevation = source.get_elevation(frame_idx)
    trav = compute_limo_traversability(elevation, filter_model, mppi_cfg, cfg.device)
    results = [
        build_teacher_a(
            geometry=geometry,
            known_trav=trav.known_trav,
            risk=trav.risk,
            fatal_threshold=float(mppi_cfg.fatal_th),
            inflation_radius_m=float(radius),
            root_xy=tuple(float(v) for v in cfg.root_xy),
        )
        for radius in cfg.inflation_radii_m
    ]
    return elevation, trav, results


def _pick_representative_frames(
    per_frame: dict[int, list[dict]], selected_count: int
) -> list[dict]:
    frame_ids = sorted(per_frame)
    middle_radius_index = 1 if len(per_frame[frame_ids[0]]) > 1 else 0

    def metric(frame_idx, key, radius_index=middle_radius_index):
        value = per_frame[frame_idx][radius_index].get(key)
        if value is None:
            return -np.inf
        return float(value)

    chosen: list[dict] = []
    used: set[int] = set()

    def add_best(label: str, candidates: list[int], key_fn, reverse=True):
        candidates = [idx for idx in candidates if idx not in used]
        if not candidates or len(chosen) >= selected_count:
            return
        best = sorted(candidates, key=key_fn, reverse=reverse)[0]
        chosen.append({"frame_idx": best, "selection_reason": label})
        used.add(best)

    valid_middle = [
        idx for idx in frame_ids if per_frame[idx][middle_radius_index]["root_configuration_valid"]
    ]
    invalid_middle = [idx for idx in frame_ids if idx not in valid_middle]
    add_best(
        "open/high_reachable",
        valid_middle,
        lambda idx: metric(idx, "reachable_fraction_of_domain"),
    )
    add_best(
        "obstacle_rich",
        valid_middle,
        lambda idx: metric(idx, "local_blocked_fraction_of_domain"),
    )
    add_best(
        "unknown_rich",
        frame_ids,
        lambda idx: metric(idx, "unknown_center_fraction_of_domain"),
    )
    add_best(
        "max_disconnected",
        valid_middle,
        lambda idx: max(
            [
                float(row["disconnected_fraction_of_domain"])
                for row in per_frame[idx]
                if row["root_configuration_valid"]
            ]
            or [-np.inf]
        ),
    )
    add_best(
        "footprint_sensitive",
        frame_ids,
        lambda idx: float(per_frame[idx][0]["reachable_fraction_of_domain"])
        - float(per_frame[idx][-1]["reachable_fraction_of_domain"]),
    )
    add_best(
        "invalid_root",
        invalid_middle,
        lambda idx: metric(idx, "unknown_center_fraction_of_domain"),
    )
    for idx in frame_ids:
        if len(chosen) >= selected_count:
            break
        if idx not in used:
            chosen.append({"frame_idx": idx, "selection_reason": "coverage_fill"})
            used.add(idx)
    return chosen


def _write_metrics_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _monotonicity_report(results: list[TeacherAResult]) -> dict:
    configuration_violations = []
    reachable_violations = []
    for smaller, larger in zip(results[:-1], results[1:]):
        configuration_violations.append(
            int(np.count_nonzero(larger.configuration_free & ~smaller.configuration_free))
        )
        reachable_violations.append(int(np.count_nonzero(larger.reachable & ~smaller.reachable)))
    return {
        "configuration_free_violation_pixels": configuration_violations,
        "reachable_violation_pixels": reachable_violations,
        "passed": not any(configuration_violations) and not any(reachable_violations),
    }


def run(cfg) -> Path:
    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    mppi_path = Path(cfg.mppi_config)
    if not mppi_path.is_absolute():
        mppi_path = Path.cwd() / mppi_path
    mppi_cfg = OmegaConf.load(mppi_path).mppi
    dataset_root = Path(cfg.dataset_root)
    filter_model = get_filter_torch(str(cfg.device))

    cells = int(round(2 * float(cfg.map_size) / float(cfg.map_resolution)))
    geometry = MapGeometry(
        height=cells,
        width=cells,
        resolution=float(cfg.map_resolution),
        origin_xy=(-float(cfg.map_size), -float(cfg.map_size)),
    )
    coordinate_records = save_coordinate_validation(
        geometry,
        output_root / "coordinate_validation.png",
        output_root / "coordinate_validation.json",
    )

    all_metric_rows: list[dict] = []
    selection_manifest: dict[str, list[dict]] = {}
    rendered_records: list[dict] = []

    for mission in cfg.missions:
        mission = str(mission)
        source = GrandTourZarrSource(
            dataset_root / mission,
            map_size=float(cfg.map_size),
            map_resolution=float(cfg.map_resolution),
        )
        candidate_ids = _candidate_indices(len(source), int(cfg.candidate_frames_per_mission))
        per_frame: dict[int, list[dict]] = {}
        log.info("[%s] evaluating %d candidate frames", mission, len(candidate_ids))
        for frame_idx in candidate_ids:
            elevation, trav, results = _process_frame(
                source, frame_idx, geometry, filter_model, mppi_cfg, cfg
            )
            rows = [
                _frame_metrics(mission, frame_idx, trav.known_raw, result)
                for result in results
            ]
            per_frame[frame_idx] = rows
            all_metric_rows.extend(rows)

        selected = _pick_representative_frames(
            per_frame, int(cfg.selected_frames_per_mission)
        )
        selection_manifest[mission] = selected
        mission_output = output_root / mission
        mission_output.mkdir(parents=True, exist_ok=True)

        for selected_item in selected:
            frame_idx = int(selected_item["frame_idx"])
            elevation, trav, results = _process_frame(
                source, frame_idx, geometry, filter_model, mppi_cfg, cfg
            )
            rgb = source.get_image(frame_idx)
            targets_by_radius = [select_diagnostic_targets(result) for result in results]
            frame_output = mission_output / f"frame_{frame_idx:06d}"
            frame_output.mkdir(parents=True, exist_ok=True)
            title = f"{mission} frame {frame_idx} ({selected_item['selection_reason']})"

            # The middle radius (0.26 m in the default config) gets the complete
            # ten-panel diagnostic; all three radii are rendered side by side.
            canonical_index = 1 if len(results) > 1 else 0
            save_frame_overview(
                rgb=rgb,
                elevation=elevation,
                trav=trav,
                result=results[canonical_index],
                targets=targets_by_radius[canonical_index],
                title=title,
                output_path=frame_output / "overview_radius_0p26.png",
            )
            save_radius_comparison(
                results=results,
                targets_by_radius=targets_by_radius,
                title=title,
                output_path=frame_output / "radius_comparison.png",
            )

            target_records = []
            all_paths_valid = True
            for result, targets in zip(results, targets_by_radius):
                for target in targets:
                    record = {
                        "radius_m": result.inflation_radius_m,
                        "category": target.category,
                        "index": list(target.index),
                        "result": target.expected_result,
                        "path_length_cells": len(target.path) if target.path is not None else None,
                    }
                    if target.path is not None:
                        valid, errors = validate_reconstructed_path(
                            result, target.index, target.path
                        )
                        record["path_valid"] = valid
                        record["path_errors"] = errors
                        all_paths_valid = all_paths_valid and valid
                    target_records.append(record)

            frame_record = {
                "mission": mission,
                "frame_idx": frame_idx,
                "selection_reason": selected_item["selection_reason"],
                "timestamp": source.get_timestamp(frame_idx),
                "radii": [result.summary() for result in results],
                "monotonicity": _monotonicity_report(results),
                "all_reconstructed_paths_valid": all_paths_valid,
                "targets": target_records,
                "overview": str(frame_output / "overview_radius_0p26.png"),
                "radius_comparison": str(frame_output / "radius_comparison.png"),
            }
            (frame_output / "diagnostics.json").write_text(
                json.dumps(frame_record, indent=2, default=_json_default), encoding="utf-8"
            )
            rendered_records.append(frame_record)

    _write_metrics_csv(output_root / "candidate_metrics.csv", all_metric_rows)
    (output_root / "selection_manifest.json").write_text(
        json.dumps(selection_manifest, indent=2), encoding="utf-8"
    )
    summary = {
        "teacher_definition": "2-D root geometric reachability under circular footprint constraints",
        "connectivity": "canonical full-domain 8-neighbour Dijkstra; no diagonal corner cutting",
        "coordinate_contract": coordinate_records,
        "inflation_radii_m": [float(v) for v in cfg.inflation_radii_m],
        "selection": selection_manifest,
        "rendered_frames": rendered_records,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, default=_json_default), encoding="utf-8"
    )
    log.info("Diagnostics written to %s", output_root)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="dataset_builder/configs/teacher_a_diagnostic.yaml",
        help="diagnostic YAML configuration",
    )
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = OmegaConf.load(args.config)
    if args.output_dir is not None:
        cfg.output_dir = args.output_dir
    run(cfg)


if __name__ == "__main__":
    main()
