"""Повторяемый ограниченный подбор начальных состояний пассивной сетки."""
from __future__ import annotations

from copy import deepcopy
from itertools import product

import numpy as np

from orbit_study.drag_model import gmst_approx_deg, integrate
from orbit_study.evaluate import geometry, intervals
from orbit_study.model import make_candidate
from square_demo.core import image_frame, observer_eci
from square_demo.orbit_j2 import orbital_basis, state_to_elements
from square_demo.providers import SCENARIOS, load_scenario


def prepare_config(config):
    cfg=deepcopy(config)
    cfg["observer"]["greenwich_angle_at_zenith_deg"]=gmst_approx_deg(cfg["epoch_utc"])
    for section,fields in (("spacecraft",("mass_kg","effective_area_m2","drag_coefficient")),
                           ("atmosphere",("rho_reference_kg_m3","scale_height_km"))):
        if any(float(cfg[section][key])<=0 for key in fields):
            raise ValueError(f"Некорректные параметры {section}")
    return cfg


def initial_candidate(config,size,side_deg,depth_factor,range_rate_factor,warp_factor):
    """Положение — лучи изображения; скорость подбирается для замедления сдвига.

    Небольшое квадратичное смещение внутренних узлов сохраняет четыре крайних
    угла и внешний размер, поднимая истинный минимальный шаг над 2°.
    """
    base=load_scenario(SCENARIOS["independent_j2"])
    base["epoch_utc"]=config["epoch_utc"]
    base["observer"]=deepcopy(config["observer"])
    base["visibility"]=deepcopy(config["visibility"])
    base["grid_tolerances"]=deepcopy(config["grid_tolerances"])
    base["orbit"].update(config["orbit"])
    raw=make_candidate(base,size,config["orbit"]["altitude_km"],side_deg,
                       velocity_factor=0,depth_factor=depth_factor,
                       inclination_deg=config["orbit"]["inclination_deg"])
    states=raw["states"].copy()
    times=np.array([-5.,0.,5.])
    chief=integrate(states[:1],times,config,1.)
    robs=observer_eci(times,0,config["orbit"]["earth_radius_km"],config["observer"])
    _,sight,exi,eeta=image_frame(chief[:,0,0],robs,orbital_basis(chief[:,0]))
    alpha=np.tan(np.deg2rad(side_deg))
    normalized=raw["target"]/alpha
    warped=alpha*(normalized+warp_factor*normalized*(normalized-1))
    lam=(states[:,0]-robs[1])@sight[1]
    ray=sight[1]+warped[:,0,None]*exi[1]+warped[:,1,None]*eeta[1]
    def desired(k):
        return robs[k]+lam[:,None]*(sight[k]+warped[:,0,None]*exi[k]
                                  +warped[:,1,None]*eeta[k])
    states[:,0]=desired(1)
    base_velocity=(desired(2)-desired(0))/10
    r0,v0=chief[1,0]
    omega=np.cross(r0,v0)/np.dot(r0,r0)
    approximate_circular=np.cross(omega,states[:,0])
    range_rate=np.sum(ray*(approximate_circular-base_velocity),axis=1)/np.sum(ray*ray,axis=1)
    states[:,1]=base_velocity+range_rate_factor*range_rate[:,None]*ray
    states[0]=chief[1,0]
    mu=config["orbit"]["mu_km3_s2"]
    elements=[state_to_elements(*state,mu) for state in states]
    return {"states":states,"elements_at_epoch":elements,"target":raw["target"],
            "grid_indices":raw["grid_indices"],"N":size,
            "parameters":{"side_deg":float(side_deg),"depth_factor":float(depth_factor),
                "range_rate_factor":float(range_rate_factor),"warp_factor":float(warp_factor),
                "height_km":float(config["orbit"]["altitude_km"]),
                "inclination_deg":float(config["orbit"]["inclination_deg"])}}


def preview(candidate,config,step_s=None):
    display=config["display"]
    step=float(step_s or display["step_s"])
    times=np.arange(display["start_s"],display["end_s"]+step*.25,step)
    states=integrate(candidate["states"],times,config,min(step,display["integration_step_s"]))
    metrics=geometry(states,times,config,candidate["N"])
    accepted={key:intervals(metrics,float(key)) for key in ("1","2")}
    min_sep=np.inf
    r=states[:,:,0]
    for i in range(r.shape[1]-1):
        min_sep=min(min_sep,float(np.min(np.linalg.norm(r[:,i+1:]-r[:,i:i+1],axis=2))))
    heights=np.linalg.norm(r,axis=2)-config["orbit"]["earth_radius_km"]
    finite=metrics["shape_errors"][np.isfinite(metrics["shape_errors"]).all(axis=1)]
    return {"times_s":times,"states":states,"metrics":metrics,"intervals":accepted,
        "best_duration_1deg_s":max((x["duration_s"] for x in accepted["1"]),default=0.),
        "best_duration_2deg_s":max((x["duration_s"] for x in accepted["2"]),default=0.),
        "minimum_pair_sample_km":min_sep,"minimum_altitude_km":float(np.min(heights)),
        "maximum_altitude_km":float(np.max(heights)),
        "minimum_elevation_deg":float(np.min(metrics["min_elevation_deg"])),
        "minimum_true_neighbor_deg":float(np.nanmin(metrics["min_true_neighbor_deg"])),
        "maximum_shape_errors":np.max(finite,axis=0).tolist() if len(finite) else [None]*3,
        "maximum_point_error_steps":float(np.nanmax(metrics["point_error_steps"]))}


def search(config,size,side_deg,budget=None):
    cfg=prepare_config(config)
    if not 2<=size<=10 or not 1<=side_deg<=30:
        raise ValueError("N=2…10, внешняя сторона 1…30°")
    choices=list(product(cfg["search"]["depth_factors"],
                         cfg["search"]["range_rate_factors"],
                         cfg["search"]["warp_factors"]))
    if budget is not None: choices=choices[:int(budget)]
    if not choices: raise ValueError("Нулевой бюджет поиска")
    rows=[]
    best=None
    for depth,rate,warp in choices:
        try:
            candidate=initial_candidate(cfg,size,side_deg,depth,rate,warp)
            result=preview(candidate,cfg)
        except (ValueError,FloatingPointError) as exc:
            rows.append({"parameters":{"depth_factor":depth,"range_rate_factor":rate,
                                        "warp_factor":warp},"error":str(exc)})
            continue
        valid_height=result["minimum_altitude_km"]>=cfg["search"]["minimum_altitude_km"]
        valid_density=result["maximum_altitude_km"]<=cfg["atmosphere"]["documented_range_km"][1]
        safe=result["minimum_pair_sample_km"]>=(cfg["spacecraft"]["collision_radius_sum_km"]
                    +cfg["spacecraft"]["additional_safety_margin_km"])
        score=(bool(valid_height and valid_density and safe),
               result["best_duration_2deg_s"],result["best_duration_1deg_s"],
               result["minimum_altitude_km"],-result["maximum_point_error_steps"])
        rows.append({"parameters":candidate["parameters"],"score":score,
            "best_duration_2deg_s":result["best_duration_2deg_s"],
            "minimum_altitude_km":result["minimum_altitude_km"],
            "minimum_pair_sample_km":result["minimum_pair_sample_km"]})
        if best is None or score>best[0]: best=(score,candidate,result)
    if best is None: raise ValueError("В бюджете поиска нет допустимых начальных элементов")
    return cfg,best[1],best[2],rows
