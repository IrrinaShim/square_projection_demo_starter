"""Готовый TrajectoryResult для независимых пассивных орбит J₂+сопротивление."""
from __future__ import annotations

import numpy as np

from orbit_study.drag_model import drag_acceleration_km_s2, integrate
from .core import assemble_absolute, spatial_size_metrics, visibility_summary
from .orbit_j2 import centroid_relative_frame
from .providers import _metadata


def calculate_drag_case(config,candidate):
    size=candidate["N"]
    orbit=config["orbit"]
    scenario=dict(config,mode="independent_drag",scenario_id="drag_study",
                  angular_side_deg=candidate["parameters"]["side_deg"],
                  grid_size=size,safety={"illustrative_min_separation_km":
                    config["spacecraft"]["collision_radius_sum_km"]
                    +config["spacecraft"]["additional_safety_margin_km"]})
    display=config["display"]
    t=np.arange(display["start_s"],display["end_s"]+display["step_s"]*.25,
                display["step_s"])
    states=integrate(candidate["states"],t,config,display["integration_step_s"])
    period=2*np.pi*np.sqrt(candidate["elements_at_epoch"][0]["a_km"]**3
                          /orbit["mu_km3_s2"])
    line_t=np.linspace(0,period,361)
    orbit_line=integrate(candidate["states"][:1],line_t,config,
                         display["integration_step_s"])[:,0,0]
    extra=drag_acceleration_km_s2(states[:,0,0],states[:,0,1],
        orbit["earth_radius_km"],config["observer"]["earth_rotation_rad_s"],
        config["spacecraft"],config["atmosphere"])
    center,center_velocity,basis,rho,local_velocity=centroid_relative_frame(
        states,orbit["mu_km3_s2"],orbit["earth_radius_km"],orbit["j2"],extra)
    from .core import observer_eci
    robs=observer_eci(t,0,orbit["earth_radius_km"],config["observer"])
    metadata=_metadata(scenario,"Независимые орбиты: центральное поле + J₂ + атмосферное сопротивление",
        orbit,"Плотность приближённая; исследовательский расчёт, не эксплуатационный прогноз.",0.)
    metadata.update({"angular_side_deg":candidate["parameters"]["side_deg"],
        "variant":"passive_drag","epoch_utc":config["epoch_utc"],
        "orbital_elements":candidate["elements_at_epoch"],"grid_size":size,
        "grid_indices":candidate["grid_indices"],"grid_tolerances":config["grid_tolerances"],
        "initial_epoch_s":0.,"initial_choice":"собственные элементы всех КА на общей эпохе",
        "local_frame_origin":"centroid","center_eci_km":center,
        "center_velocity_eci_km_s":center_velocity,
        "local_basis_eci":basis,
        "local_velocity_oso_km_s":local_velocity,
        "local_frame_orientation":"e_x=r0/|r0|; e_z=(r0×v0)/|r0×v0|; e_y=e_z×e_x",
        "included_forces":["центральное притяжение Земли","J₂","атмосферное сопротивление"],
        "research_only":True,"frame":config["frame"],
        "safety_horizon":"показанный пролёт; уточнение всех пар из протокола"})
    accel=np.zeros((len(t)-1,size*size,3))
    result=assemble_absolute(t,states[:,:,0],basis,robs,accel,candidate["target"],
                             orbit,metadata,orbit_line,period)
    result.metadata["spatial_size_metrics"]=spatial_size_metrics(result.r_eci_km,size)
    observation=result.metadata["observability"]
    limit=float(config["grid_tolerances"]["max_outer_side_relative_error"])*100
    deviation=observation["size_deviation_percent"]
    size_ok=np.isfinite(deviation)&(np.abs(deviation)<=limit)
    observation["eligible"] &= size_ok
    for k in np.flatnonzero(~size_ok):
        observation["reasons"][k]+=("; " if observation["reasons"][k] else "")+(
            f"внешний размер: отклонение {deviation[k]:+.2f}% > {limit:g}%")
    result.metadata["quality"]=visibility_summary(result)
    if not np.allclose(result.rho_oso_km,rho,atol=1e-9):
        raise ValueError("Центрированная система не согласована с абсолютными состояниями")
    return result
