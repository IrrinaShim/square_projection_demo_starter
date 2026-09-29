"""Повторить N=10/20°: .venv/bin/python -m orbit_study.drag_demo.run."""
from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np

from orbit_study import drag_model
from orbit_study.drag_demo import audit, search
from orbit_study.evaluate import geometry, intervals
from orbit_study.model import make_candidate, reconstruct
from square_demo import core, grid, orbit_j2
from square_demo.drag_provider import calculate_drag_case

HERE=Path(__file__).resolve().parent


def model_fingerprint():
    functions=(drag_model.gmst_approx_deg,drag_model.density_kg_m3,
        drag_model.drag_acceleration_km_s2,drag_model.acceleration,
        drag_model._rk4,drag_model.integrate,search.initial_candidate,
        search.preview,search.checked_geometry,search.search,audit.audit_one_pass,
        make_candidate,grid.initial_grid_from_corners,
        geometry,intervals,orbit_j2.centroid_relative_frame,
        core.observer_eci,core.project,core.square_geometry,core.observability,
        core.assemble_absolute,grid.grid_metrics,grid.grid_acceptance,
        orbit_j2.elements_to_state,calculate_drag_case)
    digest=hashlib.sha256()
    for function in functions:
        digest.update(inspect.getsource(function).encode())
    digest.update((HERE.parent.parent/"scenarios"/"independent_j2_4sat.json").read_bytes())
    return digest.hexdigest()


def _minimum_pair(states):
    positions=states[:,:,0]
    best=np.inf
    for i in range(positions.shape[1]-1):
        best=min(best,float(np.min(np.linalg.norm(
            positions[:,i+1:]-positions[:,i:i+1],axis=2))))
    return best


def run(config_path=HERE/"config.json",output=HERE/"results",size=10,side_deg=20.,
        budget=None,candidate_override=None,rows_override=None):
    config=json.loads(Path(config_path).read_text(encoding="utf-8"))
    if candidate_override is None:
        cfg,candidate,preview,rows=search.search(config,size,side_deg,budget)
    else:
        cfg=search.prepare_config(config)
        cfg["angular_side_deg"]=float(side_deg)
        candidate=candidate_override
        preview=search.preview(candidate,cfg)
        rows=rows_override if rows_override is not None else []
    result=calculate_drag_case(cfg,candidate)
    times=np.arange(cfg["display"]["start_s"],cfg["display"]["end_s"]+0.1,
                    cfg["search"]["audit_step_s"])
    states=drag_model.integrate(candidate["states"],times,cfg,
                                min(1.,cfg["display"]["integration_step_s"]))
    pair_audit=audit.audit_one_pass(states,times,cfg)
    orbital_limits,orbital_violations=search.orbit_limits(candidate,cfg)
    chief_a=candidate["elements_at_epoch"][0]["a_km"]
    period=2*np.pi*np.sqrt(chief_a**3/cfg["orbit"]["mu_km3_s2"])
    full_times=np.r_[np.arange(0.,period,5.),period]
    full_states=drag_model.integrate(candidate["states"],full_times,cfg,
                                     cfg["display"]["integration_step_s"])
    full_audit=audit.audit_one_pass(full_states,full_times,cfg)
    full_heights=np.linalg.norm(full_states[:,:,0],axis=2)-cfg["orbit"]["earth_radius_km"]
    if np.min(full_heights)<cfg["search"]["minimum_altitude_km"] or np.max(full_heights)>cfg["search"]["maximum_altitude_km"]:
        orbital_violations.append("высота на полном витке вне диапазона модели")
    if full_audit["closest_pair"]["distance_km"]<full_audit["threshold_km"]:
        orbital_violations.append("опасное сближение на полном витке")
    elements=candidate["elements_at_epoch"]
    inverse_states=candidate.get("inverse_states",candidate["states"])
    roundtrip=float(np.max(np.abs(reconstruct(elements,cfg["orbit"]["mu_km3_s2"])
                                  -inverse_states)))
    if roundtrip>1e-7: raise AssertionError("Оскулирующие элементы не воспроизводят эпоху")
    comparison=drag_model.integrate(candidate["states"],times,cfg,1.,with_drag=False)
    no_drag_geometry=search.checked_geometry(comparison,times,cfg,size)
    drag_geometry=search.checked_geometry(states,times,cfg,size)
    accepted=drag_geometry["accepted"]["0"]
    accepted_metrics=({"outer_side_range_deg":[float(np.min(drag_geometry["outer_side_deg"][accepted])),
        float(np.max(drag_geometry["outer_side_deg"][accepted]))],
        "minimum_elevation_deg":float(np.min(drag_geometry["min_elevation_deg"][accepted])),
        "minimum_true_neighbor_deg":float(np.min(drag_geometry["min_true_neighbor_deg"][accepted])),
        "maximum_shape_errors":[float(x) for x in np.max(drag_geometry["shape_errors"][accepted],axis=0)],
        "maximum_point_error_steps":float(np.max(drag_geometry["point_error_steps"][accepted])),
        "maximum_neighbor_step_error":float(np.max(drag_geometry["neighbor_step_error"][accepted]))}
        if np.any(accepted) else None)
    sensitivity={"same_initial_elements":True,
        "max_position_difference_over_pass_km":float(np.max(np.linalg.norm(
            states[:,:,0]-comparison[:,:,0],axis=2))),
        "end_position_difference_max_km":float(np.max(np.linalg.norm(
            states[-1,:,0]-comparison[-1,:,0],axis=1))),
        "accepted_duration_square_with_drag_s":max((v["duration_s"] for v in
            intervals(drag_geometry,0.)),default=0.),
        "accepted_duration_2deg_with_drag_s":max((v["duration_s"] for v in
            intervals(drag_geometry,2.)),default=0.),
        "accepted_duration_2deg_without_drag_s":max((v["duration_s"] for v in
            intervals(no_drag_geometry,2.)),default=0.),
        "minimum_pair_with_drag_sample_km":_minimum_pair(states),
        "minimum_pair_without_drag_sample_km":_minimum_pair(comparison)}
    heights=np.linalg.norm(states[:,:,0],axis=2)-cfg["orbit"]["earth_radius_km"]
    full_orbit_ok=bool(not orbital_violations and not pair_audit["dangerous_events"])
    square_duration=result.metadata["quality"]["longest_contiguous_duration_s"]
    resolved_duration=max((v["duration_s"] for v in intervals(drag_geometry,2.)),default=0.)
    resolved_display_duration=preview["best_duration_2deg_s"]
    projection_violations=[] if square_duration>0 else ["нет связного интервала наблюдаемого квадрата"]
    instrument_violations=[] if resolved_display_duration>0 else ["при исследовательском пороге разрешения 2° нет интервала"]
    verified=bool(full_orbit_ok and square_duration>0 and resolved_display_duration>0)
    result_data={"schema_version":3,"model_id":cfg["model_id"],
        "model_fingerprint_sha256":model_fingerprint(),"config":cfg,
        "N":size,"satellite_count":size*size,"grid_indices":candidate["grid_indices"],
        "parameters":candidate["parameters"],"elements_at_epoch":elements,
        "initial_state_roundtrip_max_error_km_or_km_s":roundtrip,
        "orbit_closeness":orbital_limits,"verified_acceptable":verified,
        "full_orbit_ok":full_orbit_ok,"full_orbit_violations":orbital_violations,
        "projection_violations":projection_violations,
        "instrument_violations":instrument_violations,
        "violations":orbital_violations+projection_violations+instrument_violations,
        "included_forces":["центральное притяжение Земли","J₂","атмосферное сопротивление"],
        "excluded_forces":["гармоники выше J₂","Солнце","Луна","световое давление","тяга"],
        "verification_scope":"один пролёт и полный виток всех пар; многосуточная устойчивость не проверена",
        "one_pass":{"time_s":[float(times[0]),float(times[-1])],
            "accepted_intervals_square":intervals(drag_geometry,0.),
            "accepted_intervals_1deg":intervals(drag_geometry,1.),
            "accepted_intervals_2deg":intervals(drag_geometry,2.),
            "longest_contiguous_duration_square_s":square_duration,
            "longest_contiguous_duration_2deg_display_s":resolved_display_duration,
            "longest_contiguous_duration_2deg_s":resolved_duration,
            "longest_interval_square_s":result.metadata["quality"]["longest_interval_s"],
            "accepted_square_metrics_on_1s_grid":accepted_metrics,
            "min_elevation_deg_over_window":float(np.min(result.elevation_deg)),
            "min_elevation_each_satellite_deg":np.min(result.elevation_deg,axis=0).tolist(),
            "min_true_neighbor_deg_at_epoch":float(result.metadata["grid_metrics"]["min_true_neighbor_angle_deg"][90]),
            "min_true_neighbor_deg_over_window":float(np.nanmin(drag_geometry["min_true_neighbor_deg"])),
            "max_shape_errors_over_window":preview["maximum_shape_errors"],
            "minimum_altitude_km":float(np.min(heights)),
            "maximum_altitude_km":float(np.max(heights)),
            "minimum_altitude_each_satellite_km":np.min(heights,axis=0).tolist(),
            "pair_count":pair_audit["pair_count"],
            "closest_pair":pair_audit["closest_pair"],
            "dangerous_event_count":len(pair_audit["dangerous_events"]),
            "pair_sample_step_s":pair_audit["sample_step_max_s"],
            "pair_refinement_step_s":pair_audit["refinement_step_max_s"],
            "refined_event_count":pair_audit["refined_event_count"],
            "pair_minima_file":"pair_minima.csv"},
        "full_orbit":{"time_s":[0.,float(period)],"pair_count":full_audit["pair_count"],
            "closest_pair":full_audit["closest_pair"],
            "dangerous_event_count":len(full_audit["dangerous_events"]),
            "minimum_altitude_km":float(np.min(full_heights)),
            "maximum_altitude_km":float(np.max(full_heights)),
            "sample_step_max_s":full_audit["sample_step_max_s"],
            "pair_refinement_step_s":full_audit["refinement_step_max_s"],
            "pair_minima_file":"full_orbit_pair_minima.csv"},
        "sensitivity":sensitivity,"survey_count":len(rows),
        "limitations":["фиксированная экспоненциальная плотность без космической погоды",
            "UT1≈UTC и ECI средней даты без полного преобразования ITRF/ICRF",
            "масса/площадь/CD и ориентация относятся к допущениям, не к сертифицированной платформе",
            "один пролёт не доказывает повторяемость и безопасность на много суток"]}
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    (out/"case.json").write_text(json.dumps(result_data,ensure_ascii=False,
        indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (out/"survey.json").write_text(json.dumps(rows,ensure_ascii=False,
        indent=2,allow_nan=False)+"\n",encoding="utf-8")
    with (out/"pair_minima.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(("pair","t_s","utc","distance_km","latitude_deg",
                         "longitude_deg","method"))
        for event in pair_audit["pair_minima"].values():
            writer.writerow(tuple(event[key] for key in ("pair","t_s","utc","distance_km",
                        "latitude_deg","longitude_deg","method")))
    with (out/"full_orbit_pair_minima.csv").open("w",encoding="utf-8",newline="") as f:
        writer=csv.writer(f,lineterminator="\n")
        writer.writerow(("pair","t_s","utc","distance_km","latitude_deg","longitude_deg","method"))
        for event in full_audit["pair_minima"].values():
            writer.writerow(tuple(event[key] for key in ("pair","t_s","utc","distance_km",
                        "latitude_deg","longitude_deg","method")))
    return result_data


if __name__=="__main__":
    outcome=run()
    quality=outcome["one_pass"]
    print(f"N={outcome['N']}, θ={outcome['parameters']['side_deg']:g}°: "
          f"квадрат {quality['longest_contiguous_duration_square_s']:g} с; "
          f"min пар за пролёт {quality['closest_pair']['distance_km']:.6f} км, "
          f"за полный виток {outcome['full_orbit']['closest_pair']['distance_km']:.6f} км; "
          f"приемлем={outcome['verified_acceptable']}; "
          f"данные {HERE/'results'}")
