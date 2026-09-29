"""Загрузка проверенного пролёта либо новый ограниченный поиск для изменённых полей."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from orbit_study.drag_demo.audit import audit_one_pass
from orbit_study.drag_demo.run import model_fingerprint
from orbit_study.drag_demo.search import initial_candidate, prepare_config, search
from orbit_study.drag_model import integrate
from orbit_study.model import reconstruct

ROOT=Path(__file__).resolve().parent.parent/"orbit_study"/"drag_demo"


def load_or_search(size,side_deg,neighbor_deg,latitude_deg=25.3026,
                   longitude_deg=51.5132,min_elevation_deg=15.,
                   altitude_km=700.,inclination_deg=97.8,budget=12):
    base=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    cfg=deepcopy(base)
    cfg["observer"].update(latitude_deg=float(latitude_deg),
                           longitude_deg=float(longitude_deg))
    cfg["visibility"]["min_elevation_deg"]=float(min_elevation_deg)
    cfg["orbit"].update(altitude_km=float(altitude_km),
                        inclination_deg=float(inclination_deg))
    cfg["grid_tolerances"]["min_neighbor_angle_deg"]=float(neighbor_deg)
    cfg=prepare_config(cfg)
    saved=json.loads((ROOT/"results"/"case.json").read_text(encoding="utf-8"))
    same=(size==saved["N"] and np.isclose(side_deg,saved["parameters"]["side_deg"])
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
        audit={"closest_pair":saved["one_pass"]["closest_pair"],
            "pair_count":saved["one_pass"]["pair_count"],
            "threshold_km":cfg["spacecraft"]["collision_radius_sum_km"]+
                            cfg["spacecraft"]["additional_safety_margin_km"],
            "horizon_s":saved["one_pass"]["time_s"],
            "dangerous_events_count":saved["one_pass"]["dangerous_event_count"],
            "saved_duration_s":saved["one_pass"]["longest_contiguous_duration_2deg_s"],
            "minimum_altitude_km":saved["one_pass"]["minimum_altitude_km"],
            "maximum_altitude_km":saved["one_pass"]["maximum_altitude_km"]}
        cfg["angular_side_deg"]=float(side_deg)
        return cfg,candidate,audit,"сохранённый проверенный кандидат"
    cfg,candidate,preview,_=search(cfg,size,side_deg,budget)
    step=cfg["search"]["audit_step_s"]
    times=np.arange(cfg["display"]["start_s"],cfg["display"]["end_s"]+step*.25,step)
    states=integrate(candidate["states"],times,cfg,min(1.,cfg["display"]["integration_step_s"]))
    checked=audit_one_pass(states,times,cfg)
    audit={key:checked[key] for key in ("closest_pair","pair_count","threshold_km","horizon_s")}
    audit["dangerous_events_count"]=len(checked["dangerous_events"])
    audit["minimum_altitude_km"]=preview["minimum_altitude_km"]
    audit["maximum_altitude_km"]=preview["maximum_altitude_km"]
    cfg["angular_side_deg"]=float(side_deg)
    return cfg,candidate,audit,"новый ограниченный подбор; проверен один пролёт"
