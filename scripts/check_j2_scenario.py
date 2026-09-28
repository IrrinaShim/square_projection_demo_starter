"""Воспроизводимый протокол одного пролёта и полного витка J₂.

Запуск: .venv/bin/python scripts/check_j2_scenario.py > docs/j2_pass_results.json
"""
from __future__ import annotations

import json

import numpy as np

from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))

from square_demo.orbit_j2 import (elements_to_state, integrate, separation_audit,
                                  specific_energy_j2)
from square_demo.providers import SCENARIOS, calculate, load_scenario


def report():
    scenario = load_scenario(SCENARIOS["independent_j2"])
    result = calculate(scenario)
    metadata = result.metadata
    audit = metadata["full_orbit_audit"]
    quality = metadata["quality"]
    zero = int(np.argmin(abs(result.t_s)))
    orbit = scenario["orbit"]
    mu,re,j2 = (float(orbit[k]) for k in
                ("mu_km3_s2","earth_radius_km","j2"))
    initial = np.stack([elements_to_state(item,mu) for item in scenario["elements_at_epoch"]])
    period = metadata["reference_period_s"]
    full_t = np.linspace(0,period,int(np.ceil(period/2.5))+1)
    fine = integrate(initial,full_t,mu,re,j2,max_step_s=1.0)
    fine_audit = separation_audit(fine,full_t,mu,re,j2,scenario["observer"],
                                  scenario["epoch_utc"],
                                  scenario["safety"]["illustrative_min_separation_km"],0.05)
    coarse_end = integrate(initial,np.array([0.,period]),mu,re,j2,max_step_s=2.0)[-1]
    fine_end = integrate(initial,np.array([0.,period]),mu,re,j2,max_step_s=1.0)[-1]
    energy = specific_energy_j2(fine,mu,re,j2)
    max_relative_energy_drift = float(np.max(abs((energy-energy[0])/energy[0])))
    return {
        "scenario_id":scenario["scenario_id"],
        "epoch_utc":scenario["epoch_utc"],
        "model":"four independent osculating orbits; nonlinear Earth gravity including J2; zero applied thrust",
        "units":{"length":"km","time":"s","velocity":"km/s","angle":"deg"},
        "elements_at_epoch":scenario["elements_at_epoch"],
        "reference_nominal_period_s":period,
        "display_window_s":[float(result.t_s[0]),float(result.t_s[-1])],
        "display_nodes":len(result.t_s),
        "projection_at_epoch_max_abs_error_gnomonic":float(np.max(abs(
            result.image_gnomonic[zero]-result.target_image_gnomonic))),
        "accepted_nodes":quality["accepted_nodes"],
        "accepted_interval_around_epoch_s":quality["zenith_interval_s"],
        "accepted_duration_s":quality["zenith_contiguous_duration_s"],
        "max_side_error":quality["max_side_error"],
        "max_diagonal_error":quality["max_diagonal_error"],
        "max_angle_error_deg":quality["max_angle_error_deg"],
        "min_elevation_over_display_deg":quality["min_elevation_deg"],
        "min_pair_distance_over_display_km":quality["min_pair_distance_km"],
        "full_orbit_min_altitude_km":metadata["full_orbit_min_altitude_km"],
        "full_orbit_audit":audit,
        "convergence_check":{
            "fine_max_integrator_step_s":1.0,
            "fine_audit_sample_step_max_s":float(np.max(np.diff(full_t))),
            "fine_refinement_step_max_s":0.05,
            "max_end_position_difference_km":float(np.max(np.linalg.norm(
                coarse_end[:,0]-fine_end[:,0],axis=1))),
            "max_pair_minimum_distance_difference_km":float(max(abs(
                audit["pair_minima"][key]["distance_km"]-
                fine_audit["pair_minima"][key]["distance_km"])
                for key in audit["pair_minima"])),
            "max_relative_specific_energy_drift":max_relative_energy_drift,
        },
    }


if __name__ == "__main__":
    print(json.dumps(report(),ensure_ascii=False,indent=2))
