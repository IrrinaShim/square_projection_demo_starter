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


def checked_geometry(states,times,config,size):
    """Квадрат (0) и отдельные сценарии разрешения прибором (1, 2 градуса)."""
    metrics=geometry(states,times,config,size,thresholds=(0.,1.,2.))
    target=np.tan(np.deg2rad(config["angular_side_deg"]))
    actual=np.tan(np.deg2rad(metrics["outer_side_deg"]))
    within=np.isfinite(actual)&(np.abs(actual/target-1)<=
        config["grid_tolerances"]["max_outer_side_relative_error"])
    for accepted in metrics["accepted"].values(): accepted &= within
    metrics["outer_size_ok"]=within
    return metrics
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
    """Точная картинка в эпоху; λ и v уточняются по нескольким моментам пролёта.

    Четыре переменные каждого ведомого -- дальность по своему лучу и три
    инерциальные компоненты скорости. Никакой тяги после эпохи нет.
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
    times=np.array([-180.,-120.,-60.,0.,60.,120.,180.])
    chief=integrate(states[:1],times,config,config["display"]["integration_step_s"])
    robs=observer_eci(times,0,config["orbit"]["earth_radius_km"],config["observer"])
    _,sight,exi,eeta=image_frame(chief[:,0,0],robs,orbital_basis(chief[:,0]))
    alpha=np.tan(np.deg2rad(side_deg))
    target=raw["target"]
    mid=3
    lam0=float(np.dot(chief[mid,0,0]-robs[mid],sight[mid]))
    ray=sight[mid]+target[:,0,None]*exi[mid]+target[:,1,None]*eeta[mid]
    lam=lam0*(1+depth_factor*(target[:,0]+target[:,1])/max(alpha,1e-12))
    states[:,0]=robs[mid]+lam[:,None]*ray
    r0,v0=chief[mid,0]
    omega=np.cross(r0,v0)/np.dot(r0,r0)
    states[:,1]=v0+np.cross(omega,states[:,0]-r0)
    states[0]=chief[mid,0]
    # H_i(t) r_i(t)=H_i(t) r_obs(t), H rows e_xi-xi*s, e_eta-eta*s.
    # Finite differences of independently propagated nonlinear trajectories
    # supply the P,V sensitivities; regularized least squares limits the
    # otherwise poorly determined range and velocity.
    h=np.stack((exi[:,None,:]-target[None,:,0,None]*sight[:,None,:],
                eeta[:,None,:]-target[None,:,1,None]*sight[:,None,:]),axis=2)
    scales=np.array([80.,.12,.12,.12])
    perturb=np.array([1.,.001,.001,.001])
    for _ in range(2):
        base_states=integrate(states,times,config,config["display"]["integration_step_s"])
        variants=np.repeat(states[None],4,axis=0)
        variants[0,:,0]+=ray*perturb[0]
        for axis in range(3): variants[axis+1,:,1,axis]+=perturb[axis+1]
        propagated=integrate(variants.reshape(-1,2,3),times,config,
                             config["display"]["integration_step_s"])
        perturbed=propagated.reshape(len(times),4,len(states),2,3)[:,:,:,0]
        jac=np.einsum('tsad,tpsd->tsap',h,
            ((perturbed-base_states[:,None,:,0])/perturb[None,:,None,None]).transpose(0,1,2,3))
        # einsum above orders perturbed axes as time, parameter, satellite, xyz.
        rhs=np.einsum('tsad,tsd->tsa',h,robs[:,None,:]-base_states[:,:,0])
        for j in range(1,len(states)):
            matrix=jac[:,j].reshape(-1,4)*scales
            vector=rhs[:,j].reshape(-1)
            augmented=np.vstack((matrix,np.diag([35.,35.,35.,35.])))
            solution=np.linalg.lstsq(augmented,np.r_[vector,np.zeros(4)],rcond=None)[0]*scales
            states[j,0]+=np.clip(solution[0]*range_rate_factor,-40.,40.)*ray[j]
            dv=solution[1:]*range_rate_factor
            norm=np.linalg.norm(dv)
            states[j,1]+=dv*min(1.,.12/max(norm,1e-12))
        states[0]=chief[mid,0]
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
    metrics=checked_geometry(states,times,config,candidate["N"])
    accepted={key:intervals(metrics,float(key)) for key in ("0","1","2")}
    min_sep=np.inf
    r=states[:,:,0]
    for i in range(r.shape[1]-1):
        min_sep=min(min_sep,float(np.min(np.linalg.norm(r[:,i+1:]-r[:,i:i+1],axis=2))))
    heights=np.linalg.norm(r,axis=2)-config["orbit"]["earth_radius_km"]
    finite=metrics["shape_errors"][np.isfinite(metrics["shape_errors"]).all(axis=1)]
    return {"times_s":times,"states":states,"metrics":metrics,"intervals":accepted,
        "best_duration_square_s":max((x["duration_s"] for x in accepted["0"]),default=0.),
        "best_duration_1deg_s":max((x["duration_s"] for x in accepted["1"]),default=0.),
        "best_duration_2deg_s":max((x["duration_s"] for x in accepted["2"]),default=0.),
        "minimum_pair_sample_km":min_sep,"minimum_altitude_km":float(np.min(heights)),
        "maximum_altitude_km":float(np.max(heights)),
        "minimum_elevation_deg":float(np.min(metrics["min_elevation_deg"])),
        "minimum_true_neighbor_deg":float(np.nanmin(metrics["min_true_neighbor_deg"])),
        "maximum_shape_errors":np.max(finite,axis=0).tolist() if len(finite) else [None]*3,
        "maximum_point_error_steps":float(np.nanmax(metrics["point_error_steps"]))}


def orbit_limits(candidate,config):
    """Эпохальные оскулирующие элементы; ω перигея здесь не показатель близости."""
    elements=candidate["elements_at_epoch"]
    mu=config["orbit"]["mu_km3_s2"]
    re=config["orbit"]["earth_radius_km"]
    a=np.array([entry["a_km"] for entry in elements])
    e=np.array([entry["e"] for entry in elements])
    inc=np.array([entry["inclination_deg"] for entry in elements])
    period=2*np.pi*np.sqrt(a**3/mu)
    states=candidate["states"]
    speed=np.linalg.norm(states[:,1]-states[0,1],axis=1)
    values={"a_range_km":[float(a.min()),float(a.max())],
        "e_range":[float(e.min()),float(e.max())],
        "i_range_deg":[float(inc.min()),float(inc.max())],
        "perigee_altitude_range_km":[float(np.min(a*(1-e)-re)),float(np.max(a*(1-e)-re))],
        "apogee_altitude_range_km":[float(np.min(a*(1+e)-re)),float(np.max(a*(1+e)-re))],
        "period_spread_s":float(np.ptp(period)),
        "max_epoch_relative_speed_km_s":float(speed.max())}
    limits=config["search"]
    failures=[]
    tests=(("перигей",values["perigee_altitude_range_km"][0],limits["minimum_altitude_km"],">="),
        ("апогей",values["apogee_altitude_range_km"][1],limits["maximum_altitude_km"],"<="),
        ("разброс a",float(np.ptp(a)),limits["max_semimajor_axis_spread_km"],"<="),
        ("макс. e",float(e.max()),limits["max_eccentricity"],"<="),
        ("разброс i",float(np.ptp(inc)),limits["max_inclination_spread_deg"],"<="),
        ("разброс периодов",values["period_spread_s"],limits["max_period_spread_s"],"<="),
        ("относительная скорость",values["max_epoch_relative_speed_km_s"],
         limits["max_epoch_relative_speed_km_s"],"<="))
    for name,value,bound,sense in tests:
        if (value<bound if sense==">=" else value>bound):
            failures.append(f"{name}: {value:.3f} {sense} {bound:g} не выполнено")
    return values,failures


def full_orbit_sample(candidate,config):
    """Недорогой отсев; окончательное уточнение всех пар выполняет audit."""
    a=candidate["elements_at_epoch"][0]["a_km"]
    period=2*np.pi*np.sqrt(a**3/config["orbit"]["mu_km3_s2"])
    times=np.arange(0.,period,10.)
    times=np.r_[times,period]
    positions=integrate(candidate["states"],times,config,
                        config["display"]["integration_step_s"])[:,:,0]
    separation=np.inf
    for j in range(positions.shape[1]-1):
        separation=min(separation,float(np.min(np.linalg.norm(
            positions[:,j+1:]-positions[:,j:j+1],axis=2))))
    heights=np.linalg.norm(positions,axis=2)-config["orbit"]["earth_radius_km"]
    return {"sample_minimum_pair_km":separation,
        "minimum_altitude_km":float(np.min(heights)),
        "maximum_altitude_km":float(np.max(heights)),"period_s":float(period)}


def search(config,size,side_deg,budget=None,cancel_event=None):
    cfg=prepare_config(config)
    cfg["angular_side_deg"]=float(side_deg)
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
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Подбор отменён после изменения параметров")
        try:
            candidate=initial_candidate(cfg,size,side_deg,depth,rate,warp)
            result=preview(candidate,cfg)
        except (ValueError,FloatingPointError) as exc:
            rows.append({"parameters":{"depth_factor":depth,"range_rate_factor":rate,
                                        "warp_factor":warp},"error":str(exc)})
            continue
        limits,failures=orbit_limits(candidate,cfg)
        orbit_sample=full_orbit_sample(candidate,cfg) if not failures else None
        if orbit_sample is not None:
            if orbit_sample["minimum_altitude_km"]<cfg["search"]["minimum_altitude_km"]:
                failures.append("высота на полном витке ниже границы модели")
            if orbit_sample["maximum_altitude_km"]>cfg["search"]["maximum_altitude_km"]:
                failures.append("высота на полном витке выше границы модели")
        valid_height=result["minimum_altitude_km"]>=cfg["search"]["minimum_altitude_km"]
        valid_density=result["maximum_altitude_km"]<=cfg["search"]["maximum_altitude_km"]
        safe=result["minimum_pair_sample_km"]>=(cfg["spacecraft"]["collision_radius_sum_km"]
                    +cfg["spacecraft"]["additional_safety_margin_km"])
        safe_orbit=(orbit_sample is not None and orbit_sample["sample_minimum_pair_km"]
                    >=cfg["spacecraft"]["collision_radius_sum_km"]
                      +cfg["spacecraft"]["additional_safety_margin_km"])
        score=(bool(valid_height and valid_density and safe and not failures and safe_orbit),
               result["best_duration_square_s"],result["best_duration_2deg_s"],
               orbit_sample["sample_minimum_pair_km"] if orbit_sample else -1.,
               -result["maximum_point_error_steps"])
        rows.append({"parameters":candidate["parameters"],"score":score,
            "best_duration_2deg_s":result["best_duration_2deg_s"],
            "minimum_altitude_km":result["minimum_altitude_km"],
            "minimum_pair_sample_km":result["minimum_pair_sample_km"],
            "orbit_limits":limits,"full_orbit_sample":orbit_sample,
            "violations":failures+([] if safe_orbit else ["сближение на полном витке: выборка или ограничения орбиты"])})
        if best is None or score>best[0]: best=(score,candidate,result)
    if best is None: raise ValueError("В бюджете поиска нет допустимых начальных элементов")
    return cfg,best[1],best[2],rows
