"""Проверка пригодности исторического управляемого CSV; не меняет его динамику."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from square_demo.providers import SCENARIOS, calculate, load_scenario

ROOT=Path(__file__).resolve().parents[2]


def run(output=Path(__file__).resolve().parent/"results"/"controlled_check.json"):
    result=calculate(load_scenario(SCENARIOS["legacy_replay"]))
    legacy=json.loads((ROOT/"legacy"/"example_controlled"/"config.json").read_text(encoding="utf-8"))
    controls=legacy["control"]
    maximum=float(np.max(np.linalg.norm(result.accel_oso_m_s2,axis=2)))
    quality=result.metadata["quality"]
    data={"N":2,"source":"historical 4-satellite CSV, not a J2+drag trajectory",
          "frame":"conditional Greenwich=0 at assigned zenith; UTC epoch absent",
          "applied_acceleration_max_m_s2":maximum,
          "declared_mass_kg":controls["mass_kg"],
          "declared_thrust_limit_N":controls["F_max_N"],
          "thrust_implied_max_N":maximum*controls["mass_kg"],
          "max_delta_v_m_s":float(np.max(result.dv_cumulative_m_s[-1])),
          "declared_engine_hardware_verified":False,
          "target_outer_side_deg":result.metadata["angular_side_deg"],
          "reconstructed_projection_interval_s":quality["longest_interval_s"],
          "reconstructed_projection_duration_s":quality["longest_contiguous_duration_s"],
          "compatible_with_passive_drag_study":False,
          "reasons":["нет UTC эпохи и согласованной земной ориентации",
                     "исторический CSV не распространён с моделью J2+сопротивление",
                     "Fmax, масса и характеристики двигателя из внутреннего config.json не подтверждены для реальной платформы 16U",
                     "внешняя сторона исторического случая не равна 20°"]}
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return data


if __name__=="__main__":
    print(json.dumps(run(),ensure_ascii=False,indent=2))
