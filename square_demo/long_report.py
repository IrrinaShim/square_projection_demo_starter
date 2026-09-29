"""Отдельный отчёт по запросу долгосрочной проверки новой модели.

Численные результаты старой J₂-модели сюда не переносятся. Пока нет
проверенного многосуточного расчёта с сопротивлением, отчёт фиксирует
невыполненную проверку и исходные данные, необходимые для её повторения.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4


def create_long_report(config, candidate, audit, result, horizon_days, output_parent):
    if not 0.05 <= float(horizon_days) <= 30:
        raise ValueError("Горизонт запроса должен быть в пределах 0,05…30 сут")
    if output_parent is None:
        raise ValueError("Нужна папка для отдельного отчёта")
    elements=candidate["elements_at_epoch"]
    radius=float(config["orbit"]["earth_radius_km"])
    perigee=min(float(e["a_km"])*(1-float(e["e"]))-radius for e in elements)
    apogee=max(float(e["a_km"])*(1+float(e["e"]))-radius for e in elements)
    domain=config["atmosphere"]["documented_range_km"]
    outside=perigee<domain[0] or apogee>domain[1]
    reason=(f"Оценки перигея и апогея по элементам эпохи {perigee:.1f}…{apogee:.1f} км "
            f"выходят за область принятой плотности {domain[0]:g}…{domain[1]:g} км."
            if outside else
            "Для выбранного набора пока не выполнены многосуточное распространение и проверка всех пар в модели с сопротивлением.")
    now=datetime.now(timezone.utc)
    folder=Path(output_parent)/(
        f"N{candidate['N']}_{candidate['parameters']['side_deg']:g}deg_"
        f"{now:%Y%m%dT%H%M%SZ}_{uuid4().hex[:6]}")
    folder.mkdir(parents=True,exist_ok=False)
    data={"schema_version":1,"status":"not_computed",
          "created_utc":now.isoformat().replace("+00:00","Z"),
          "requested_horizon_days":float(horizon_days),
          "reason":reason,
          "verification_scope":"one displayed pass and one full-orbit pair audit; no multi-day propagation",
          "model_id":config["model_id"],"included_forces":result.metadata["included_forces"],
          "N":candidate["N"],"satellite_count":candidate["N"]**2,
          "target_outer_side_deg":candidate["parameters"]["side_deg"],
          "selection_parameters":candidate["parameters"],
          "grid_indices":candidate["grid_indices"],
          "epoch_utc":config["epoch_utc"],"config":config,
          "elements_at_epoch":elements,
          "osculating_perigee_estimate_min_km":perigee,
          "osculating_apogee_estimate_max_km":apogee,
          "one_pass":{"time_s":audit["one_pass_horizon_s"],
                      "longest_accepted_interval_s":result.metadata["quality"]["longest_interval_s"],
                      "longest_accepted_duration_s":result.metadata["quality"]["longest_contiguous_duration_s"],
                      "closest_pair":audit["one_pass_closest_pair"],"pair_count":audit["pair_count"]},
          "full_orbit_pair_audit":{"time_s":audit["horizon_s"],
              "closest_pair":audit["closest_pair"],"pair_count":audit["pair_count"]},
          "long_term":{"qualifying_passes":None,"closest_pair":None,
                       "collision_assessment":"not_computed"}}
    (folder/"long_term_assessment.json").write_text(
        json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    (folder/"README.md").write_text(
        f"# Запрос долгосрочной проверки N={candidate['N']}\n\n"
        f"Запрошенный горизонт: {horizon_days:g} сут. Статус: **не вычислено**.\n\n"
        f"Причина: {reason}\n\n"
        "Файл `long_term_assessment.json` хранит модель, конфигурацию, "
        "элементы всех КА, итоги показанного пролёта и аудит пар полного витка. "
        "Число подходящих последующих пролётов и минимальное расстояние "
        "на запрошенном горизонте неизвестны.\n",encoding="utf-8")
    return {"folder":str(folder.resolve()),"status":"not_computed","reason":reason}
