"""Протоколы одного пролёта и полного витка для N=2,3,10.

Запуск: .venv/bin/python scripts/check_grid.py
Файлы: docs/grid_checks/grid_n{N}.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from square_demo.core import observability, visibility_summary
from square_demo.providers import SCENARIOS, calculate, load_scenario


def threshold_check(result, angle_deg):
    metadata = result.metadata
    original_tolerances = metadata["grid_tolerances"]
    original_observation = metadata["observability"]
    try:
        metadata["grid_tolerances"] = {**original_tolerances,
                                       "min_neighbor_angle_deg":angle_deg}
        metadata["observability"] = observability(result)
        summary = visibility_summary(result)
        return {"accepted_nodes":summary["accepted_nodes"],
                "interval_around_epoch_s":summary["zenith_interval_s"],
                "duration_s":summary["zenith_contiguous_duration_s"],
                "longest_interval_s":summary["longest_interval_s"],
                "has_positive_duration_interval":summary["has_positive_duration_interval"]}
    finally:
        metadata["grid_tolerances"] = original_tolerances
        metadata["observability"] = original_observation


def run(size):
    scenario = load_scenario(SCENARIOS["independent_j2"])
    scenario["grid_size"] = size
    result = calculate(scenario)
    metadata = result.metadata
    center = int(np.argmin(abs(result.t_s)))
    audit = metadata["full_orbit_audit"]
    closest = min(audit["pair_minima"].values(),key=lambda x:x["distance_km"])
    return {
        "scenario_id":scenario["scenario_id"],
        "model":"independent osculating orbits, nonlinear J2, zero thrust",
        "epoch_utc":scenario["epoch_utc"],
        "grid_size":size,"satellite_count":size*size,
        "index_map_rows_bottom_to_top":metadata["grid_indices"],
        "corner_indices":metadata["corner_indices"],
        "angular_side_deg":scenario["angular_side_deg"],
        "orbit_constants":scenario["orbit"],
        "observer":scenario["observer"],
        "time":scenario["time"],
        "integration":scenario["integration"],
        "visibility":scenario["visibility"],
        "grid_tolerances":scenario["grid_tolerances"],
        "elements_at_epoch":metadata["orbital_elements"],
        "display":{
            "array_shape":list(result.r_eci_km.shape),
            "max_initial_projection_error_gnomonic":float(np.max(abs(
                result.image_gnomonic[center]-result.target_image_gnomonic))),
            "quality":metadata["quality"],
            "zero_applied_thrust":bool(np.all(result.accel_oso_m_s2==0)),
            "max_grid_point_error_steps":metadata["quality"]["max_point_error_steps"],
            "max_neighbor_step_error":metadata["quality"]["max_neighbor_step_error"],
            "max_cell_side_error":metadata["quality"]["max_cell_side_error"],
            "max_cell_diagonal_error":metadata["quality"]["max_cell_diagonal_error"],
            "max_cell_angle_error_deg":metadata["quality"]["max_cell_angle_error_deg"],
            "min_true_neighbor_angle_deg":metadata["quality"]["min_true_neighbor_angle_deg"],
            "first_failure_reason":next((text for ok,text in zip(
                metadata["observability"]["eligible"],metadata["observability"]["reasons"])
                if not ok),None),
        },
        "angle_threshold_examples_deg":{
            "1":threshold_check(result,1.0),"2":threshold_check(result,2.0)},
        "full_orbit":{
            "status":metadata["audit_status"],
            "nominal_period_s":metadata["reference_period_s"],
            "min_altitude_km":metadata["full_orbit_min_altitude_km"],
            "closest_pair":closest,
            "audit":audit,
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes",nargs="+",type=int,default=[2,3,10])
    parser.add_argument("--output-dir",type=Path,default=Path("docs/grid_checks"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for size in args.sizes:
        data = run(size)
        path = args.output_dir/f"grid_n{size}.json"
        if size == 10:
            audit = data["full_orbit"]["audit"]
            minima = audit.pop("pair_minima")
            high = audit.pop("high_latitude_sample_minima")
            csv_path = args.output_dir/f"grid_n{size}_pair_minima.csv"
            with csv_path.open("w",encoding="utf-8",newline="") as f:
                writer = csv.writer(f, lineterminator="\n")
                writer.writerow(("pair","t_s","utc","distance_km","latitude_deg",
                    "longitude_deg","method","high_lat_t_s","high_lat_distance_km",
                    "high_lat_latitude_deg","high_lat_longitude_deg"))
                for pair,event in minima.items():
                    polar = high[pair]
                    writer.writerow((pair,event["t_s"],event["utc"],event["distance_km"],
                        event["latitude_deg"],event["longitude_deg"],event["method"],
                        polar["t_s"],polar["distance_km"],polar["latitude_deg"],
                        polar["longitude_deg"]))
            audit["all_pair_minima_csv"] = csv_path.name
        path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
        quality = data["display"]["quality"]
        closest = data["full_orbit"]["closest_pair"]
        print(f"N={size}: {size*size} КА, {data['full_orbit']['audit']['pair_count']} пар, "
              f"допустимо {quality['zenith_contiguous_duration_s']:.0f} с, "
              f"min {closest['distance_km']:.6f} км ({closest['pair']}), {path}")
