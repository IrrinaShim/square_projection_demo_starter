"""Загрузка проверенного пролёта либо новый ограниченный поиск для изменённых полей."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from orbit_study.drag_demo.audit import audit_one_pass
from orbit_study.drag_demo.run import model_fingerprint
from orbit_study.drag_demo.search import initial_candidate, prepare_config, search, orbit_limits
from orbit_study.drag_model import integrate
from orbit_study.model import reconstruct

ROOT=Path(__file__).resolve().parent.parent/"orbit_study"/"drag_demo"


def load_or_search(size,side_deg,neighbor_deg,latitude_deg=25.3026,
                   longitude_deg=51.5132,min_elevation_deg=15.,
                   altitude_km=700.,inclination_deg=97.8,budget=12,cancel_event=None):
    base=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    cfg=deepcopy(base)
    cfg["observer"].update(latitude_deg=float(latitude_deg),
                           longitude_deg=float(longitude_deg))
    cfg["visibility"]["min_elevation_deg"]=float(min_elevation_deg)
    cfg["orbit"].update(altitude_km=float(altitude_km),
                        inclination_deg=float(inclination_deg))
    cfg["grid_tolerances"]["min_neighbor_angle_deg"]=float(neighbor_deg)
    cfg=prepare_config(cfg)
    cfg["angular_side_deg"]=float(side_deg)
    case_path=ROOT/"results"/(f"n{size}/case.json" if size in (2,3) else "case.json")
    saved=json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else None
    same=(saved is not None and size==saved["N"] and np.isclose(side_deg,saved["parameters"]["side_deg"])
          and cfg==saved["config"] and
          saved["model_fingerprint_sha256"]==model_fingerprint())
    if same:
        parameters=saved["parameters"]
        candidate=initial_candidate(cfg,size,side_deg,
            parameters["depth_factor"],parameters["range_rate_factor"],
            parameters["warp_factor"])
        states=reconstruct(saved["elements_at_epoch"],cfg["orbit"]["mu_km3_s2"])
        if np.max(np.abs(states-candidate["states"]))>1e-7:
            raise ValueError("Сохранённые элементы не воспроизводят начальные состояния")
        candidate["states"]=states
        candidate["elements_at_epoch"]=saved["elements_at_epoch"]
        full=saved["full_orbit"]
        audit={"closest_pair":full["closest_pair"],
            "pair_count":full["pair_count"],
            "threshold_km":cfg["spacecraft"]["collision_radius_sum_km"]+
                            cfg["spacecraft"]["additional_safety_margin_km"],
            "horizon_s":full["time_s"],
            "one_pass_horizon_s":saved["one_pass"]["time_s"],
            "one_pass_closest_pair":saved["one_pass"]["closest_pair"],
            "dangerous_events_count":full["dangerous_event_count"],
            "saved_duration_s":saved["one_pass"]["longest_contiguous_duration_2deg_s"],
            "minimum_altitude_km":full["minimum_altitude_km"],
            "maximum_altitude_km":full["maximum_altitude_km"],
            "verified_acceptable":saved["verified_acceptable"],
            "violations":saved["violations"],"orbit_closeness":saved["orbit_closeness"]}
        cfg["angular_side_deg"]=float(side_deg)
        return cfg,candidate,audit,("сохранённый кандидат: проверены один пролёт и полный виток" if saved["verified_acceptable"]
                                    else "сохранённая исследовательская попытка; подходящий пассивный вариант не найден")
    cfg,candidate,preview,_=search(cfg,size,side_deg,budget,cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise ValueError("Расчёт отменён после изменения параметров")
    period=2*np.pi*np.sqrt(candidate["elements_at_epoch"][0]["a_km"]**3/
                           cfg["orbit"]["mu_km3_s2"])
    step=5.
    times=np.r_[np.arange(0.,period,step),period]
    states=integrate(candidate["states"],times,cfg,cfg["display"]["integration_step_s"])
    checked=audit_one_pass(states,times,cfg,cancel_event=cancel_event)
    audit={key:checked[key] for key in ("closest_pair","pair_count","threshold_km","horizon_s")}
    audit["one_pass_horizon_s"]=[float(cfg["display"]["start_s"]),float(cfg["display"]["end_s"])]
    audit["one_pass_closest_pair"]={"distance_km":preview["minimum_pair_sample_km"],
                                      "method":"выборка показанного пролёта"}
    audit["dangerous_events_count"]=len(checked["dangerous_events"])
    heights=np.linalg.norm(states[:,:,0],axis=2)-cfg["orbit"]["earth_radius_km"]
    audit["minimum_altitude_km"]=float(np.min(heights))
    audit["maximum_altitude_km"]=float(np.max(heights))
    audit["orbit_closeness"],audit["violations"]=orbit_limits(candidate,cfg)
    if audit["minimum_altitude_km"]<cfg["search"]["minimum_altitude_km"] or audit["maximum_altitude_km"]>cfg["search"]["maximum_altitude_km"]:
        audit["violations"].append("высота на полном витке вне диапазона модели")
    if checked["dangerous_events"]:
        audit["violations"].append("опасное сближение на полном витке")
    if preview["best_duration_2deg_s"]<=0:
        audit["violations"].append("нет связного интервала приемлемой проекции")
    audit["verified_acceptable"]=not audit["violations"]
    cfg["angular_side_deg"]=float(side_deg)
    return cfg,candidate,audit,"новый подбор; проверены пролёт и полный виток"
