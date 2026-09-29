"""Три фиксированных случая и центрированная постановка: численный протокол.

Запуск: .venv/bin/python -m orbit_study.drag_demo.observability_regression
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from orbit_study.drag_demo.audit import audit_one_pass
from orbit_study.drag_demo.centered_projection import centered_initial_candidate, centered_metrics
from orbit_study.drag_demo.search import checked_geometry, initial_candidate, orbit_limits, prepare_config
from orbit_study.drag_model import integrate
from orbit_study.evaluate import intervals
from square_demo.core import zenith_orbit
from square_demo.core import project
from square_demo.drag_provider import calculate_drag_case
from square_demo.drag_study import load_or_search
from square_demo.grid import expected_neighbor_angle_deg
from square_demo.orbit_j2 import state_to_elements
from square_demo.providers import SCENARIOS, free_hcw_two_body, load_scenario

HERE=Path(__file__).resolve().parent


def windows(t,mask):
    differences=np.diff(np.r_[False,np.asarray(mask,dtype=bool),False].astype(int))
    starts,ends=np.flatnonzero(differences==1),np.flatnonzero(differences==-1)-1
    return [[float(t[a]),float(t[b])] for a,b in zip(starts,ends) if b>a]


def hcw_elements(result,scenario):
    orbit=scenario["orbit"]
    chief,basis,_,n=zenith_orbit(np.array([0.]),0.,orbit,scenario["observer"])
    b=basis[0]
    rho=np.vstack((np.zeros(3),result.metadata["initial_rho_oso_km"]))
    rel_v=np.vstack((np.zeros(3),result.metadata["initial_vel_oso_m_s"]/1000))
    r0=chief[0]
    v0=n*(orbit["earth_radius_km"]+orbit["altitude_km"])*b[:,1]
    initial=[]
    for x,u in zip(rho,rel_v):
        r=r0+b@x
        v=v0+b@(u+np.array([-n*x[1],n*x[0],0.]))
        initial.append(state_to_elements(r,v,orbit["mu_km3_s2"]))
    return initial


def hcw_record(variant):
    scenario=load_scenario(SCENARIOS["free_hcw_two_body"])
    scenario["variant"]=variant
    result=free_hcw_two_body(scenario)
    obs=result.metadata["observability"]
    shape=(np.all(result.image_valid,axis=1)&obs["convex"]
           &np.all(np.isfinite(result.shape_errors),axis=1)
           &(result.shape_errors[:,0]<=scenario["visibility"]["max_side_error"])
           &(result.shape_errors[:,1]<=scenario["visibility"]["max_diagonal_error"])
           &(np.rad2deg(result.shape_errors[:,2])<=scenario["visibility"]["max_angle_error_deg"]))
    visible=np.min(result.elevation_deg,axis=1)>=scenario["visibility"]["min_elevation_deg"]
    k0=int(np.argmin(abs(result.t_s)))
    best=int(np.argmin(result.min_pair_distance_km))
    return {"model":"linear_HCW_two_body","variant":variant,"epoch":"t=0, conditional Greenwich 0 deg",
            "angular_side_target_deg":scenario["angular_side_deg"],
            "form_intervals_s":windows(result.t_s,shape),
            "all_visible_intervals_s":windows(result.t_s,visible),
            "observed_square_intervals_s":windows(result.t_s,shape&visible),
            "outer_size_at_epoch_deg":float(np.rad2deg(np.arctan(np.mean(result.square_sides_gnomonic[k0])))),
            "minimum_pair_on_displayed_pass_km":float(result.min_pair_distance_km[best]),
            "minimum_pair_time_s":float(result.t_s[best]),
            "osculating_elements_if_interpreted_as_ECI_at_epoch":hcw_elements(result,scenario),
            "qualification":"Оскулирующие элементы лишь пересчёт мгновенных HCW состояний; сама HCW траектория не интегрирует эти орбиты"}


def drag_record(config,candidate,label,full_audit=None):
    result=calculate_drag_case(config,candidate)
    t=np.arange(config["display"]["start_s"],config["display"]["end_s"]+1,1.)
    propagated=integrate(candidate["states"],t,config,1.)
    geom=checked_geometry(propagated,t,config,candidate["N"])
    no_visibility=deepcopy(config)
    no_visibility["visibility"]["min_elevation_deg"]=-90.
    form=checked_geometry(propagated,t,no_visibility,candidate["N"])["accepted"]["0"]
    visible=geom["min_elevation_deg"]>=config["visibility"]["min_elevation_deg"]
    chief_a=candidate["elements_at_epoch"][0]["a_km"]
    period=2*np.pi*np.sqrt(chief_a**3/config["orbit"]["mu_km3_s2"])
    if full_audit is None:
        full_t=np.r_[np.arange(0.,period,5.),period]
        full_states=integrate(candidate["states"],full_t,config,2.)
        full_audit=audit_one_pass(full_states,full_t,config)
        full_heights=np.linalg.norm(full_states[:,:,0],axis=2)-config["orbit"]["earth_radius_km"]
        full_min_alt=float(np.min(full_heights));full_max_alt=float(np.max(full_heights))
    else:
        full_min_alt=full_audit["minimum_altitude_km"]
        full_max_alt=full_audit["maximum_altitude_km"]
    limits,failures=orbit_limits(candidate,config)
    center=candidate["states"][:,0].mean(axis=0)
    center_radius=np.linalg.norm(candidate["states"][:,0]-center,axis=1)
    return {"model":"independent_central_J2_drag","label":label,
        "epoch_utc":config["epoch_utc"],"N":candidate["N"],
        "parameters":candidate["parameters"],
        "form_intervals_s":windows(t,form),
        "all_visible_intervals_s":windows(t,visible),
        "observed_square_intervals_s":windows(t,geom["accepted"]["0"]),
        "instrument_resolved_square_intervals_2deg_s":windows(t,geom["accepted"]["2"]),
        "outer_size_at_epoch_deg":float(geom["outer_side_deg"][np.argmin(abs(t))]),
        "true_minimum_neighbor_at_epoch_deg":float(geom["min_true_neighbor_deg"][np.argmin(abs(t))]),
        "minimum_pair_on_displayed_pass_km":float(np.min(result.min_pair_distance_km)),
        "minimum_pair_full_orbit":full_audit["closest_pair"],
        "full_orbit_altitude_range_km":[full_min_alt,full_max_alt],
        "orbit_closeness":limits,"orbit_closeness_violations":failures,
        "center_radius_epoch_range_km":[float(np.min(center_radius)),float(np.max(center_radius))],
        "elements_at_epoch":candidate["elements_at_epoch"]}


def centered_comparison(config,size,side):
    candidate=centered_initial_candidate(config,size,side)
    t=np.arange(-450.,451.,5.)
    states=integrate(candidate["states"],t,config,2.)
    metrics=centered_metrics(states,t,config,candidate)
    limits,failures=orbit_limits(candidate,config)
    return {"N":size,"target_error_at_epoch":float(np.max(abs(metrics["image"][90]-candidate["target"]))),
            "centroid_image_mean_at_epoch":np.mean(metrics["image"][90],axis=0).tolist(),
            "form_intervals_s":windows(t,metrics["form"]),
            "observed_square_intervals_s":windows(t,metrics["square"]),
            "instrument_resolved_intervals_2deg_s":windows(t,metrics["resolved_square"]),
            "true_minimum_neighbor_at_epoch_deg":float(metrics["min_true_neighbor_deg"][90]),
            "orbit_closeness":limits,"orbit_closeness_violations":failures,
            "qualification":"Равнодальные симметричные лучи и вращательная начальная скорость; исследовательское начальное приближение, не выбранная орбита"}


def physics_validation(config,candidate):
    t=np.arange(-450.,451.,5.)
    propagated={step:integrate(candidate["states"],t,config,step) for step in (2.,1.,.5)}
    position_delta={str(step):float(np.max(np.linalg.norm(
        propagated[step][:,:,0]-propagated[.5][:,:,0],axis=2))) for step in (2.,1.)}
    pass_windows={str(step):windows(t,checked_geometry(states,t,config,candidate["N"])["accepted"]["0"])
                  for step,states in propagated.items()}
    result=calculate_drag_case(config,candidate)
    basis=result.metadata["local_basis_eci"]
    absolute=result.metadata["center_eci_km"][:,None,:]+np.einsum(
        "tij,tsj->tsi",basis,result.rho_oso_km)
    image,valid,_=project(absolute,result.observer_eci_km,basis,
                          config["orbit"]["earth_radius_km"])
    epoch=int(np.argmin(abs(t)))
    chief_displacement=float(np.linalg.norm(result.r_eci_km[epoch+24,0]-
                                            result.r_eci_km[epoch,0]))
    return {"N":candidate["N"],"integrator_step_s":[2.,1.,.5],
        "max_position_difference_vs_half_second_km":position_delta,
        "square_intervals_5s_grid":pass_windows,
        "local_to_absolute_max_error_km":float(np.max(abs(absolute-result.r_eci_km))),
        "projection_after_local_roundtrip_max_error":float(np.max(abs(image[valid]-
                                                   result.image_gnomonic[valid]))),
        "centroid_local_mean_max_km":float(np.max(abs(result.rho_oso_km.mean(axis=1)))),
        "chief_displacement_120s_in_fixed_ECI_camera_km":chief_displacement,
        "following_camera_center_displacement_120s_km":chief_displacement,
        "note":"Камеры меняют только пределы осей; обе используют один массив абсолютных рассчитанных положений"}


def main():
    hcwt=hcw_record("trial")
    hcwf=hcw_record("fitted")
    config=json.loads((HERE/"config.json").read_text(encoding="utf-8"))
    config["orbit"]["altitude_km"]=630.
    config=prepare_config(config)
    config["angular_side_deg"]=1.8183029644518331
    current_small=initial_candidate(config,2,config["angular_side_deg"],0.,.03,0.)
    independent_small=drag_record(config,current_small,"same_angle_and_altitude_as_HCW")
    cfg20,candidate20,audit20,_=load_or_search(2,20.,2.)
    independent20=drag_record(cfg20,candidate20,"saved_n2_20deg_700km",
                               dict(audit20,minimum_altitude_km=audit20["minimum_altitude_km"],
                                    maximum_altitude_km=audit20["maximum_altitude_km"]))
    output={"fixed_cases":{"HCW_trial":hcwt,"HCW_fitted_context":hcwf,
                           "independent_same_parameters":independent_small,
                           "independent_saved_20deg":independent20},
            "ideal_n10_20deg_noncentered_min_true_step_deg":expected_neighbor_angle_deg(10,20.),
            "centered_initial_comparison":[centered_comparison(cfg20,n,20.) for n in (2,10)],
            "physics_validation":[physics_validation(cfg20,candidate20)]}
    path=HERE/"results"/"observability_regression.json"
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(path)
    for key,item in output["fixed_cases"].items():
        print(key,"форма",item["form_intervals_s"],"видимость",item["all_visible_intervals_s"],
              "квадрат",item["observed_square_intervals_s"],"размер",item["outer_size_at_epoch_deg"])
    for item in output["centered_initial_comparison"]:
        print("центрирование N",item["N"],"квадрат",item["observed_square_intervals_s"],
              "нарушения орбит",item["orbit_closeness_violations"])


if __name__=="__main__":
    main()
