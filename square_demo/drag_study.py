"""Загрузка проверенного пролёта либо новый ограниченный поиск для изменённых полей."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from orbit_study.drag_demo.audit import audit_one_pass
from orbit_study.drag_demo.search import initial_candidate, prepare_config, search, orbit_limits
from orbit_study.drag_model import integrate
from orbit_study.model import reconstruct
from orbit_study.orbit_direction import audit_orbit_normals

ROOT=Path(__file__).resolve().parent.parent/"orbit_study"/"drag_demo"


def _direction_audit(candidate,cfg):
    initial=audit_orbit_normals(candidate["states"][None],np.array([0.]))
    if not initial["ok"]:raise ValueError("Сохранённые орбиты разнонаправленны: "+initial["reason"])
    display=cfg["display"]
    times=np.arange(display["start_s"],display["end_s"]+display["step_s"]*.25,display["step_s"])
    shown=audit_orbit_normals(integrate(candidate["states"],times,cfg,
        display["integration_step_s"]),times)
    period=2*np.pi*np.sqrt(candidate["elements_at_epoch"][0]["a_km"]**3/
                           cfg["orbit"]["mu_km3_s2"])
    full_times=np.r_[np.arange(0.,period,5.),period]
    full=audit_orbit_normals(integrate(candidate["states"],full_times,cfg,
        display["integration_step_s"]),full_times)
    for name,item in (("пролёт",shown),("полный виток",full)):
        if not item["ok"]:raise ValueError(f"Сохранённые орбиты, {name}: "+item["reason"])
    return {"epoch":initial,"one_pass":shown,"full_orbit":full,"fixed_limit_deg":5.}


def _checked_audit(cfg,checked):
    full=checked["full_orbit"];one=checked["one_pass"]
    violations=list(checked["physical_violations"])
    projection=[] if one["best_square_duration_display_s"]>0 else ["нет связного интервала наблюдаемого квадрата"]
    instrument=[] if one["best_resolution_2deg_display_s"]>0 else ["при исследовательском пороге разрешения 2° нет интервала"]
    return {"closest_pair":full["closest_pair"],"pair_count":full["pair_count"],
        "threshold_km":full["threshold_km"],"horizon_s":[0.,full["period_s"]],
        "one_pass_horizon_s":[cfg["display"]["start_s"],cfg["display"]["end_s"]],
        "one_pass_closest_pair":{"distance_km":one["minimum_pair_sample_km"],"method":"узлы 1 с"},
        "dangerous_events_count":full["dangerous_event_count"],
        "saved_duration_s":one["best_square_duration_display_s"],
        "instrument_duration_2deg_s":one["best_resolution_2deg_display_s"],
        "minimum_altitude_km":full["altitude_range_km"][0],
        "maximum_altitude_km":full["altitude_range_km"][1],
        "orbit_closeness":checked["orbit_limits"],"full_orbit_ok":not violations,
        "orbital_direction":checked["orbital_direction"],
        "full_orbit_violations":violations,"projection_violations":projection,
        "instrument_violations":instrument,
        "violations":violations+projection+instrument,
        "verified_acceptable":not (violations or projection or instrument)}


def load_or_search(size,side_deg,neighbor_deg,latitude_deg=25.3026,
                   longitude_deg=51.5132,min_elevation_deg=15.,
                   altitude_km=700.,inclination_deg=97.8,budget=12,cancel_event=None,
                   epoch_utc=None,allow_search=True):
    base=json.loads((ROOT/"config.json").read_text(encoding="utf-8"))
    cfg=deepcopy(base)
    if epoch_utc is not None:cfg["epoch_utc"]=str(epoch_utc)
    cfg["observer"].update(latitude_deg=float(latitude_deg),
                           longitude_deg=float(longitude_deg))
    cfg["visibility"]["min_elevation_deg"]=float(min_elevation_deg)
    cfg["orbit"].update(altitude_km=float(altitude_km),
                        inclination_deg=float(inclination_deg))
    cfg["grid_tolerances"]["min_neighbor_angle_deg"]=float(neighbor_deg)
    cfg=prepare_config(cfg)
    cfg["angular_side_deg"]=float(side_deg)
    from orbit_study.passive_optimizer import load_cached
    cached=load_cached(cfg,size)
    if cached is not None:
        cfg,candidate,data,path=cached
        audit=_checked_audit(cfg,data["verification"])
        return cfg,candidate,audit,f"подбор из {path}: один пролёт и полный виток проверены"
    case_path=ROOT/"results"/(f"n{size}/case.json" if size in (2,3) else "case.json")
    saved=json.loads(case_path.read_text(encoding="utf-8")) if case_path.exists() else None
    same=(saved is not None and saved.get("schema_version")==3
          and size==saved["N"] and np.isclose(side_deg,saved["parameters"]["side_deg"])
          and cfg==saved["config"])
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
        from orbit_study.passive_optimizer import verify
        checked=verify(candidate,cfg,cancel_event)
        if not all(checked["orbital_direction"][scope]["ok"]
                   for scope in ("epoch","one_pass","full_orbit")):
            raise ValueError("Ранее сохранённый набор нарушает сонаправленность орбит")
        audit=_checked_audit(cfg,checked)
        cfg["angular_side_deg"]=float(side_deg)
        return cfg,candidate,audit,("архивные элементы повторно проверены в текущей модели" if audit["verified_acceptable"]
                                    else "архивные элементы повторно проверены; критерии выполнены частично")
    if not allow_search:
        raise FileNotFoundError("Для выбранных параметров сохранённого расчёта нет. Нажмите «Подобрать орбиты». Старые траектории не используются.")
    cfg,candidate,preview,_=search(cfg,size,side_deg,budget,cancel_event)
    if cancel_event is not None and cancel_event.is_set():
        raise ValueError("Расчёт отменён после изменения параметров")
    period=2*np.pi*np.sqrt(candidate["elements_at_epoch"][0]["a_km"]**3/
                           cfg["orbit"]["mu_km3_s2"])
    step=5.
    times=np.r_[np.arange(0.,period,step),period]
    states=integrate(candidate["states"],times,cfg,cfg["display"]["integration_step_s"])
    checked=audit_one_pass(states,times,cfg,cancel_event=cancel_event)
    direction=_direction_audit(candidate,cfg)
    audit={key:checked[key] for key in ("closest_pair","pair_count","threshold_km","horizon_s")}
    audit["one_pass_horizon_s"]=[float(cfg["display"]["start_s"]),float(cfg["display"]["end_s"])]
    audit["one_pass_closest_pair"]={"distance_km":preview["minimum_pair_sample_km"],
                                      "method":"выборка показанного пролёта"}
    audit["dangerous_events_count"]=len(checked["dangerous_events"])
    heights=np.linalg.norm(states[:,:,0],axis=2)-cfg["orbit"]["earth_radius_km"]
    audit["minimum_altitude_km"]=float(np.min(heights))
    audit["maximum_altitude_km"]=float(np.max(heights))
    audit["orbit_closeness"],audit["violations"]=orbit_limits(candidate,cfg)
    audit["orbital_direction"]=direction
    if audit["minimum_altitude_km"]<cfg["search"]["minimum_altitude_km"] or audit["maximum_altitude_km"]>cfg["search"]["maximum_altitude_km"]:
        audit["violations"].append("высота на полном витке вне диапазона модели")
    if checked["dangerous_events"]:
        audit["violations"].append("опасное сближение на полном витке")
    audit["full_orbit_violations"]=audit["violations"].copy()
    audit["full_orbit_ok"]=not audit["full_orbit_violations"]
    audit["projection_violations"]=([] if preview["best_duration_square_s"]>0 else
                                    ["нет связного интервала наблюдаемого квадрата"])
    audit["instrument_duration_2deg_s"]=preview["best_duration_2deg_s"]
    audit["instrument_violations"]=([] if preview["best_duration_2deg_s"]>0 else
                                    ["при исследовательском пороге разрешения 2° нет интервала"])
    audit["violations"]+=audit["projection_violations"]+audit["instrument_violations"]
    audit["verified_acceptable"]=not audit["violations"]
    cfg["angular_side_deg"]=float(side_deg)
    return cfg,candidate,audit,"новый подбор; проверены пролёт и полный виток"
