"""Загрузка проверенных данных orbit_study для существующего провайдера J₂."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from orbit_study.provenance import (FRAME, MODEL_VERSION, physics_fingerprint,
                                    solutions_fingerprint)
from orbit_study.model import reconstruct
from .grid import grid_indices
from .orbit_j2 import state_to_elements
from .providers import ROOT, SCENARIOS, load_scenario

STUDY_ROOT = ROOT / "orbit_study"
SOLUTIONS = STUDY_ROOT / "results" / "solutions.json"


def _same_mapping(saved, current, label):
    if set(saved) != set(current):
        raise ValueError(f"Исследование несовместимо: изменён набор параметров «{label}»")
    for key in saved:
        a,b=saved[key],current[key]
        if isinstance(a,(int,float)) and isinstance(b,(int,float)):
            equal=bool(np.isclose(a,b,rtol=0,atol=1e-12))
        else:
            equal=a==b
        if not equal:
            raise ValueError(f"Исследование несовместимо: {label}.{key}: сохранено {a}, сейчас {b}")


def load_study_case(size, neighbor_angle_deg=2.0, solutions_path=SOLUTIONS):
    """Возвращает сценарий, запись и общий протокол; без пересчёта орбит.

    При изменённой физике, эпохе или допусках сохранённые выводы не выдаются.
    """
    grid_indices(size)
    if neighbor_angle_deg not in (1.0,2.0):
        raise ValueError("Исследованы только пороги различимости 1° и 2°")
    path=Path(solutions_path)
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("solutions_fingerprint_sha256")!=solutions_fingerprint(data):
        raise ValueError("Исследование несовместимо: содержимое сохранённых решений изменилось")
    if data.get("schema_version")!=1 or data.get("model_version")!=MODEL_VERSION:
        raise ValueError("Исследование несовместимо: версия модели отличается")
    if data.get("physics_fingerprint_sha256")!=physics_fingerprint():
        raise ValueError("Исследование несовместимо: численные определения изменились")
    if (data.get("frame")!=FRAME or data.get("dynamics")!=
            "independent RK4 nonlinear central gravity + J2; no thrust"):
        raise ValueError("Исследование несовместимо: система координат или динамика изменены")
    saved_config=data.get("search_config")
    current_config=json.loads((STUDY_ROOT/"search.json").read_text(encoding="utf-8"))
    if saved_config!=current_config:
        raise ValueError("Исследование несовместимо: настройки поиска или безопасности изменены")
    base=load_scenario(SCENARIOS["independent_j2"])
    if data.get("epoch_utc")!=base["epoch_utc"]:
        raise ValueError("Исследование несовместимо: эпоха изменилась")
    _same_mapping(data["orbit_constants"],base["orbit"],"орбита")
    _same_mapping(data["observer"],base["observer"],"наблюдатель")
    _same_mapping(data["visibility"],base["visibility"],"видимость")
    _same_mapping(data["grid_tolerances"],base["grid_tolerances"],"сетка")
    record=data["solutions"].get(str(size))
    if record is None:
        raise ValueError(f"Для N={size} в исследовании нет сохранённого варианта")
    if (int(record["N"])!=size or int(record["satellite_count"])!=size*size
            or record["grid_indices"]!=grid_indices(size).tolist()):
        raise ValueError("Исследование несовместимо: индексы или число КА изменились")
    elements=record["elements_at_epoch"]
    if len(elements)!=size*size:
        raise ValueError("Исследование несовместимо: недостающие элементы КА")
    orbit=dict(data["orbit_constants"])
    orbit["altitude_km"]=float(record["parameters"]["height_km"])
    orbit["inclination_deg"]=float(record["parameters"]["inclination_deg"])
    initial=reconstruct(elements,orbit["mu_km3_s2"])
    if initial.shape!=(size*size,2,3) or not np.isfinite(initial).all():
        raise ValueError("Некорректные начальные состояния исследования")
    chief=state_to_elements(*initial[0],orbit["mu_km3_s2"])
    if (abs(np.linalg.norm(initial[0,0])-orbit["earth_radius_km"]-orbit["altitude_km"])>1e-5
            or abs(chief["inclination_deg"]-orbit["inclination_deg"])>1e-5):
        raise ValueError("Исследование несовместимо: элементы опорной орбиты и параметры расходятся")
    scenario=base
    scenario["grid_size"]=size
    scenario["orbit"]=orbit
    scenario["angular_side_deg"]=float(record["parameters"]["side_deg"])
    scenario["elements_at_epoch"]=elements
    scenario["initialization"]="full_elements"
    scenario["grid_tolerances"]["min_neighbor_angle_deg"]=neighbor_angle_deg
    scenario["safety"]["illustrative_min_separation_km"]=(
        float(current_config["collision_radius_sum_km"])
        +float(current_config["additional_safety_margin_km"]))
    scenario["study_provenance"]={"model_version":MODEL_VERSION,
        "physics_fingerprint_sha256":data["physics_fingerprint_sha256"],
        "solutions_path":str(path.resolve()),"selected_N":size,
        "resolution_threshold_deg":neighbor_angle_deg}
    return scenario,record,data


def study_status(record, threshold_deg=2.0, config=None):
    """Три честных статуса по выполненному горизонту и выбранному порогу."""
    if config is None:
        config=json.loads((STUDY_ROOT/"search.json").read_text(encoding="utf-8"))
    audit=record["verification"]
    key=str(int(threshold_deg))
    passes=audit["repeat"][key]["qualifying_passes"]
    minimum=audit["closest_pair"]["distance_km"]
    collision=float(config["collision_radius_sum_km"])
    required=collision+float(config["additional_safety_margin_km"])
    high_enough=audit["minimum_altitude_km"]>=config["minimum_model_altitude_km"]
    complete=audit["horizon_days"]>=config["detailed_horizon_base_days"]
    if passes<1 or minimum<collision:
        return "нет проверенного решения"
    if complete and passes>=2 and minimum>=required and high_enough:
        return "требования выполнены"
    return "выполнены частично"
