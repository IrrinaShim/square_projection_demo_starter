"""Пассивный обратный подбор наблюдаемой сетки; км, с, км/с.

ХКУ используется только для начальных приближений. Итоговая длительность и
сближения получаются независимым RK4 для каждого КА в J2+сопротивлении.
Число прямых расчётов задаёт --budget; --resume продолжает от сохранённого
лучшего состояния, не выдавая ограниченный поиск за глобальный оптимум.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from orbit_study.drag_demo.audit import audit_one_pass
from orbit_study.drag_demo.run import model_fingerprint
from orbit_study.drag_demo.search import checked_geometry, orbit_limits, prepare_config
from orbit_study.drag_model import integrate
from orbit_study.evaluate import intervals
from orbit_study.model import reconstruct
from square_demo.core import image_frame, observer_eci, project, spatial_size_metrics, zenith_orbit
from square_demo.grid import corner_indices, grid_indices
from square_demo.orbit_j2 import orbital_basis, state_to_elements
from square_demo.providers import _fit_initial_states

ROOT=Path(__file__).resolve().parent
BASE=ROOT/"drag_demo"/"config.json"
RESULTS=ROOT/"optimizer_results"
VERSION=2


def configuration(size,side_deg,latitude_deg=25.3026,longitude_deg=51.5132,
                  min_elevation_deg=15.,epoch_utc="2026-09-28T00:00:00Z",
                  altitude_km=700.,inclination_deg=97.8):
    if not 2<=int(size)<=10 or int(size)!=size or not 1<=float(side_deg)<=30:
        raise ValueError("N должно быть целым 2…10, внешняя сторона 1…30°")
    cfg=json.loads(BASE.read_text(encoding="utf-8"))
    cfg["epoch_utc"]=epoch_utc
    cfg["observer"].update(latitude_deg=float(latitude_deg),longitude_deg=float(longitude_deg))
    cfg["visibility"]["min_elevation_deg"]=float(min_elevation_deg)
    cfg["orbit"].update(altitude_km=float(altitude_km),inclination_deg=float(inclination_deg))
    cfg["angular_side_deg"]=float(side_deg)
    return prepare_config(cfg)


def signature(config,size):
    payload=json.dumps({"version":VERSION,"config":config,"N":int(size),
                        "physics":model_fingerprint()},sort_keys=True,ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def geometry_at_epoch(config,size):
    """Точные регулярные лучи относительно направления на угловой КА 0."""
    orbit=config["orbit"]
    chief_r,basis,_,_=zenith_orbit(np.array([0.]),0.,orbit,config["observer"])
    b=basis[0]; r0=chief_r[0]
    v0=b[:,1]*np.sqrt(orbit["mu_km3_s2"]/np.linalg.norm(r0))
    robs=observer_eci(np.array([0.]),0.,orbit["earth_radius_km"],config["observer"])[0]
    _,sight,exi,eeta=image_frame(r0[None],robs[None],b[None])
    sight,exi,eeta=sight[0],exi[0],eeta[0]
    ids=grid_indices(size)
    alpha=np.tan(np.deg2rad(config["angular_side_deg"]))
    target=np.empty((size*size,2))
    for row in range(size):
        for col in range(size):
            target[ids[row,col]]=alpha*np.array([col/(size-1),row/(size-1)])
    rays=sight+target[:,0,None]*exi+target[:,1,None]*eeta
    lam0=float(np.dot(r0-robs,sight))
    return np.stack((r0,v0)),b,robs,rays,target,lam0


def _candidate(states,target,size,config,source):
    elements=[state_to_elements(*state,config["orbit"]["mu_km3_s2"]) for state in states]
    return {"N":int(size),"states":np.asarray(states),"target":target,
            "grid_indices":grid_indices(size).tolist(),"elements_at_epoch":elements,
            "parameters":{"side_deg":float(config["angular_side_deg"]),
                          "height_km":float(config["orbit"]["altitude_km"]),
                          "inclination_deg":float(config["orbit"]["inclination_deg"]),
                          "source":source}}


def corner_seed_hcw(config,size,regularization,half_window_s):
    """H_i(P rho+V v)=b для трёх углов; не считается итоговой орбитой."""
    chief,b,robs,rays,target,lam0=geometry_at_epoch(config,size)
    corners=corner_indices(size)
    t=np.arange(-450.,451.,5.)
    rho,vel,_=_fit_initial_states(t,0.,target[corners],config["orbit"],
        config["observer"],regularization,21,half_window_s)
    states=np.empty((4,2,3));states[0]=chief
    omega=np.cross(chief[0],chief[1])/np.dot(chief[0],chief[0])
    for j in range(1,4):
        relative=b@rho[j-1]
        states[j,0]=chief[0]+relative
        states[j,1]=chief[1]+np.cross(omega,relative)+b@vel[j-1]
    # Билинейная интерполяция только начальной дальности и скорости.
    # Каждое начальное положение остаётся на своём точном целевом луче.
    ids=grid_indices(size)
    ranges=(states[:,0]-robs)@rays[0]
    all_states=np.empty((size*size,2,3))
    for row in range(size):
        for col in range(size):
            u,v=col/(size-1),row/(size-1)
            weights=np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v])
            index=ids[row,col]
            all_states[index,0]=robs+(weights@ranges)*rays[index]
            all_states[index,1]=weights@states[:,1]
    all_states[corners]=states
    return _candidate(all_states,target,size,config,
                      f"HCW H(Pρ+Vv), reg={regularization:g}, window={half_window_s:g}s")


def rotation_seed(config,size):
    chief,b,robs,rays,target,lam0=geometry_at_epoch(config,size)
    states=np.empty((size*size,2,3))
    states[:,0]=robs+lam0*rays
    omega=np.cross(chief[0],chief[1])/np.dot(chief[0],chief[0])
    states[:,1]=chief[1]+np.cross(omega,states[:,0]-chief[0])
    states[0]=chief
    return _candidate(states,target,size,config,"равнодальный вращательный seed")


def archived_seed(config,size):
    folder=ROOT/"drag_demo"/"results"/(f"n{size}" if size in (2,3) else "")
    path=folder/"case.json"
    if not path.exists():return None
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("N")!=size or data.get("config")!=config or data.get("model_fingerprint_sha256")!=model_fingerprint():
        return None
    states=reconstruct(data["elements_at_epoch"],config["orbit"]["mu_km3_s2"])
    _,_,_,_,target,_=geometry_at_epoch(config,size)
    candidate=_candidate(states,target,size,config,"ранее проверенный кандидат")
    candidate["archive_audit"]=data
    return candidate


def _coefficients(size):
    """12 угловых + 4 гладких поля, при N>2 ещё точечные направления."""
    ids=grid_indices(size)
    u,v=np.meshgrid(np.linspace(0,1,size),np.linspace(0,1,size))
    weights=np.stack(((1-u)*(1-v),u*(1-v),u*v,(1-u)*v),axis=-1)
    field=np.empty((size*size,4))
    for row in range(size):
        for col in range(size):field[ids[row,col]]=weights[row,col]
    bubble=16*u*(1-u)*v*(1-v)
    smooth=np.empty(size*size)
    for row in range(size):
        for col in range(size):smooth[ids[row,col]]=bubble[row,col]
    return field,smooth


def perturb(seed,config,coeff,individual=None):
    states=seed["states"].copy();size=seed["N"]
    chief,b,robs,rays,target,_=geometry_at_epoch(config,size)
    field,smooth=_coefficients(size)
    directions=field[:,1:]@coeff[:3,:]+smooth[:,None]*coeff[3,:]
    if individual:
        for index,delta in individual.items():directions[index]+=delta
    # Коэффициенты заданы в км и км/с в локальном базисе эпохи.
    states[:,0]+=directions[:,0,None]*rays
    states[:,1]+=directions[:,1:]@b.T
    states[0]=chief
    return _candidate(states,target,size,config,seed["parameters"]["source"]+" + совместные поправки")


def _near_miss(metrics,config):
    vis=config["visibility"];tol=config["grid_tolerances"]
    e=metrics["shape_errors"]
    terms=[e[:,0]/vis["max_side_error"],e[:,1]/vis["max_diagonal_error"],
           np.rad2deg(e[:,2])/vis["max_angle_error_deg"],
           (vis["min_elevation_deg"]-metrics["min_elevation_deg"])/10+1,
           abs(np.tan(np.deg2rad(metrics["outer_side_deg"]))
               /np.tan(np.deg2rad(config["angular_side_deg"]))-1)
              /tol["max_outer_side_relative_error"]]
    if len(metrics["shape_errors"]) and config.get("N",2)>2:
        terms.extend((metrics["point_error_steps"]/tol["max_point_error_steps"],
                      metrics["neighbor_step_error"]/tol["max_neighbor_step_error"]))
    worst=np.nanmax(np.nan_to_num(np.stack(terms),nan=100.,posinf=100.),axis=0)
    return float(np.min(worst))


def survey(candidate,config,step_s=None):
    step_s=float(step_s or config["search"]["survey_step_s"])
    t=np.arange(config["display"]["start_s"],config["display"]["end_s"]+step_s*.25,step_s)
    states=integrate(candidate["states"],t,config,config["display"]["integration_step_s"])
    metrics=checked_geometry(states,t,config,candidate["N"])
    wins=intervals(metrics,0.);wins2=intervals(metrics,2.)
    duration=max((x["duration_s"] for x in wins),default=0.)
    resolution=max((x["duration_s"] for x in wins2),default=0.)
    limits,failures=orbit_limits(candidate,config)
    heights=np.linalg.norm(states[:,:,0],axis=2)-config["orbit"]["earth_radius_km"]
    if float(np.min(heights))<config["search"]["minimum_altitude_km"]:
        failures.append("высота на пролёте ниже границы")
    if float(np.max(heights))>config["search"]["maximum_altitude_km"]:
        failures.append("высота на пролёте выше границы")
    pair_min=np.inf
    for j in range(candidate["N"]**2-1):
        pair_min=min(pair_min,float(np.min(np.linalg.norm(
            states[:,j+1:,0]-states[:,j:j+1,0],axis=2))))
    threshold=config["spacecraft"]["collision_radius_sum_km"]+config["spacecraft"]["additional_safety_margin_km"]
    if pair_min<threshold:failures.append(f"сближение на пролёте {pair_min:.4f} < {threshold:g} км")
    near=_near_miss(metrics,dict(config,N=candidate["N"]))
    # Продолжительность является главной целью среди физически допустимых seed.
    score=(not failures,duration,-near)
    return {"score":score,"duration_square_s":duration,"duration_resolution_2deg_s":resolution,
            "intervals_square":wins,"near_miss":near,"orbit_limits":limits,
            "minimum_pair_sample_km":pair_min,"violations":failures}


def _record(candidate,metrics,iteration,stage):
    return {"iteration":iteration,"stage":stage,"source":candidate["parameters"]["source"],
            "initial_states_eci_km_km_s":candidate["states"].tolist(),
            "elements_at_epoch":candidate["elements_at_epoch"],
            "score":list(metrics["score"]),"duration_square_s":metrics["duration_square_s"],
            "duration_resolution_2deg_s":metrics["duration_resolution_2deg_s"],
            "near_miss":metrics["near_miss"],"orbit_limits":metrics["orbit_limits"],
            "minimum_pair_sample_km":metrics["minimum_pair_sample_km"],
            "rejection_reasons":metrics["violations"] or (["нулевой связный интервал"] if not metrics["duration_square_s"] else [])}


def verify(candidate,config,cancel_event=None):
    """Окончательный 1-с пролёт, 5-с виток, межузловые минимумы всех пар."""
    size=candidate["N"]
    if cancel_event is not None and cancel_event.is_set():raise InterruptedError("Аудит отменён")
    t=np.arange(config["display"]["start_s"],config["display"]["end_s"]+1,1.)
    states=integrate(candidate["states"],t,config,1.)
    geom=checked_geometry(states,t,config,size)
    wins=intervals(geom,0.);wins2=intervals(geom,2.)
    display_geom=checked_geometry(states[::5],t[::5],config,size)
    display_square=max((x["duration_s"] for x in intervals(display_geom,0.)),default=0.)
    display_resolved=max((x["duration_s"] for x in intervals(display_geom,2.)),default=0.)
    best_window=max(wins,key=lambda x:x["duration_s"],default=None)
    observer_epoch=observer_eci(np.array([0.]),0.,config["orbit"]["earth_radius_km"],config["observer"])
    epoch_index=int(np.argmin(abs(t)))
    initial_image,_,_=project(states[epoch_index:epoch_index+1,:,0],observer_epoch,
        orbital_basis(states[epoch_index:epoch_index+1,0]),config["orbit"]["earth_radius_km"])
    initial_target_error=float(np.max(abs(initial_image[0]-candidate["target"])))
    centroid_direction=states[epoch_index,:,0].mean(axis=0)-observer_epoch[0]
    chief_direction=states[epoch_index,0,0]-observer_epoch[0]
    center_chief_angle=float(np.rad2deg(np.arccos(np.clip(np.dot(centroid_direction,chief_direction)/
        (np.linalg.norm(centroid_direction)*np.linalg.norm(chief_direction)),-1,1))))
    if best_window:
        left=int(round(best_window["start_s"]-t[0]));right=int(round(best_window["end_s"]-t[0]))+1
        spatial=spatial_size_metrics(states[:,:,0],size)
        window_metrics={"outer_side_range_deg":[float(np.min(geom["outer_side_deg"][left:right])),
                                                float(np.max(geom["outer_side_deg"][left:right]))],
            "minimum_elevation_deg":float(np.min(geom["min_elevation_deg"][left:right])),
            "minimum_true_neighbor_deg":float(np.min(geom["min_true_neighbor_deg"][left:right])),
            "max_shape_errors":np.max(geom["shape_errors"][left:right],axis=0).tolist(),
            "max_point_error_steps":float(np.max(geom["point_error_steps"][left:right])),
            "spatial_neighbor_side_range_km":[float(np.min(spatial["min_neighbor_km"][left:right])),
                                              float(np.max(spatial["max_neighbor_km"][left:right]))]}
    else:window_metrics=None
    chief_a=candidate["elements_at_epoch"][0]["a_km"]
    period=2*np.pi*np.sqrt(chief_a**3/config["orbit"]["mu_km3_s2"])
    full_t=np.r_[np.arange(0.,period,5.),period]
    full_states=integrate(candidate["states"],full_t,config,config["display"]["integration_step_s"])
    if cancel_event is not None and cancel_event.is_set():raise InterruptedError("Аудит отменён")
    audit=audit_one_pass(full_states,full_t,config,cancel_event=cancel_event)
    pass_min=np.inf
    for j in range(size*size-1):
        pass_min=min(pass_min,float(np.min(np.linalg.norm(
            states[:,j+1:,0]-states[:,j:j+1,0],axis=2))))
    heights=np.linalg.norm(full_states[:,:,0],axis=2)-config["orbit"]["earth_radius_km"]
    chief_lat=np.rad2deg(np.arcsin(full_states[:,0,0,2]/np.linalg.norm(full_states[:,0,0],axis=1)))
    limits,failures=orbit_limits(candidate,config)
    if heights.min()<config["search"]["minimum_altitude_km"] or heights.max()>config["search"]["maximum_altitude_km"]:
        failures.append("высота на полном витке вне области модели")
    if audit["dangerous_events"]:failures.append("опасное сближение на полном витке")
    finer=integrate(candidate["states"],t,config,.5)
    drift=float(np.max(np.linalg.norm(finer[:,:,0]-states[:,:,0],axis=2)))
    roundtrip=float(np.max(abs(reconstruct(candidate["elements_at_epoch"],
                        config["orbit"]["mu_km3_s2"])-candidate["states"])))
    if roundtrip>1e-7:failures.append("элементы не восстанавливают состояния")
    return {"one_pass":{"sample_step_s":1.,"square_intervals":wins,
                "resolved_intervals_2deg":wins2,
                "best_square_duration_s":max((x["duration_s"] for x in wins),default=0.),
                "best_resolution_2deg_s":max((x["duration_s"] for x in wins2),default=0.),
                "best_square_duration_display_s":display_square,
                "best_resolution_2deg_display_s":display_resolved,
                "minimum_pair_sample_km":pass_min,
                "outer_size_at_epoch_deg":float(geom["outer_side_deg"][epoch_index]),
                "initial_target_max_error_gnomonic":initial_target_error,
                "centroid_to_chief_direction_angle_deg":center_chief_angle,
                "minimum_true_neighbor_at_epoch_deg":float(geom["min_true_neighbor_deg"][epoch_index]),
                "best_interval_metrics":window_metrics},
            "full_orbit":{"period_s":period,"pair_count":audit["pair_count"],
                "closest_pair":audit["closest_pair"],"dangerous_event_count":len(audit["dangerous_events"]),
                "all_pair_minima":audit["pair_minima"],
                "altitude_range_km":[float(heights.min()),float(heights.max())],
                "max_abs_chief_latitude_deg":float(np.max(abs(chief_lat))),
                "threshold_km":audit["threshold_km"]},
            "orbit_limits":limits,"physical_violations":failures,
            "step_1_vs_half_max_position_km":drift,"elements_roundtrip_max_error":roundtrip}


def optimize(config,size,budget=16,output_root=RESULTS,resume=True,cancel_event=None,progress=None):
    cfg=prepare_config(config);cfg["angular_side_deg"]=float(config["angular_side_deg"])
    key=signature(cfg,size);folder=Path(output_root)/key;folder.mkdir(parents=True,exist_ok=True)
    history_file=folder/"candidates.jsonl";best_file=folder/"best.json"
    history=[]
    if resume and history_file.exists():
        history=[json.loads(line) for line in history_file.read_text(encoding="utf-8").splitlines() if line]
    seeds=[]
    if resume and best_file.exists():
        old=json.loads(best_file.read_text(encoding="utf-8"))
        if old.get("signature")==key:
            old_candidate=old["candidate"]
            states=np.asarray(old_candidate["initial_states_eci_km_km_s"],dtype=float)
            _,_,_,_,target,_=geometry_at_epoch(cfg,size)
            seeds.append(_candidate(states,target,size,cfg,"продолжение сохранённого поиска"))
    archived=archived_seed(cfg,size)
    if archived is not None:seeds.append(archived)
    seeds.append(rotation_seed(cfg,size))
    for reg,window in ((.1,120.),(1.,150.),(10.,240.)):
        try:seeds.append(corner_seed_hcw(cfg,size,reg,window))
        except (ValueError,np.linalg.LinAlgError) as exc:
            record={"iteration":len(history)+1,"stage":"начальное ХКУ-приближение",
                "source":f"reg={reg:g}, window={window:g}s","score":[False,0.,-1e6],
                "duration_square_s":0.,"duration_resolution_2deg_s":0.,
                "near_miss":1e6,"orbit_limits":{},"minimum_pair_sample_km":0.,
                "rejection_reasons":[str(exc)]}
            history.append(record)
            with history_file.open("a",encoding="utf-8") as stream:
                stream.write(json.dumps(record,ensure_ascii=False)+"\n")
    best=None;best_index=None;count=0
    def evaluate(candidate,stage):
        nonlocal best,best_index,count
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Поиск отменён; история уже рассчитанных попыток сохранена")
        try:
            metrics=survey(candidate,cfg)
        except (ValueError,FloatingPointError) as exc:
            metrics={"score":(False,0.,-1e6),"duration_square_s":0.,
                "duration_resolution_2deg_s":0.,"near_miss":1e6,
                "orbit_limits":{},"minimum_pair_sample_km":0.,"violations":[str(exc)]}
        count+=1
        improved=best is None or metrics["score"]>best[1]["score"]
        record=_record(candidate,metrics,len(history)+1,stage)
        if not improved and not record["rejection_reasons"]:
            record["rejection_reasons"]=["не увеличил проверяемую длительность или запас критериев относительно текущего лидера"]
        history.append(record)
        with history_file.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+"\n")
        if improved:
            best=(candidate,metrics)
            best_index=len(history)-1
        if progress:progress(f"{count}/{budget}: {stage}; квадрат {metrics['duration_square_s']:g} с; лучший {best[1]['duration_square_s']:g} с")
    for seed in seeds:
        if count>=budget:break
        evaluate(seed,"углы ХКУ / ранее проверенный seed")
    if best is None:raise ValueError("Не удалось построить ни одного кандидата")
    rng=np.random.default_rng(20260929+len(history))
    coeff=np.zeros((4,4));step=np.array([20.,.02,.02,.02])
    corners=set(corner_indices(size).tolist())
    interior=[int(index) for index in grid_indices(size).ravel() if index not in corners]
    while count<budget:
        selected=best[0]
        axis=int(rng.integers(0,16));row,col=divmod(axis,4)
        direction=1 if count%2==0 else -1
        if interior and count%5==0:
            index=interior[(count//5)%len(interior)]
            correction=np.zeros(4)
            correction[col]=direction*step[col]*.5
            candidate=perturb(selected,cfg,coeff,{index:correction})
            stage=f"индивидуальная поправка КА {index}"
        else:
            trial_coeff=coeff.copy();trial_coeff[row,col]+=direction*step[col]
            candidate=perturb(selected,cfg,trial_coeff)
            stage="совместные углы и гладкое поле"
        old_score=best[1]["score"]
        evaluate(candidate,stage)
        if best[1]["score"]>old_score:
            coeff[:]=0
        elif count%16==0:step*=.6
    selected=best[0]
    checked=verify(selected,cfg,cancel_event)
    # Непроверенный по витку выигрыш не вытесняет проверенную базу.
    if archived is not None:
        saved_duration=max((row["duration_s"] for row in archived["archive_audit"]["one_pass"][
            "accepted_intervals_square"]),default=0.)
        if checked["physical_violations"] or checked["one_pass"]["best_square_duration_s"]<=saved_duration:
            archived_check=verify(archived,cfg,cancel_event)
            if (not archived_check["physical_violations"] and
                (checked["physical_violations"] or archived_check["one_pass"]["best_square_duration_s"]
                  >=checked["one_pass"]["best_square_duration_s"])):
                selected,checked=archived,archived_check
    payload={"schema_version":VERSION,"signature":key,"model_fingerprint_sha256":model_fingerprint(),
        "config":cfg,"N":size,"budget_this_run":budget,"survey_count_total":len(history),
        "candidate":{"source":selected["parameters"]["source"],
                     "initial_states_eci_km_km_s":selected["states"].tolist(),
                     "elements_at_epoch":selected["elements_at_epoch"],
                     "target_gnomonic":selected["target"].tolist(),
                     "grid_indices":selected["grid_indices"]},
        "verification":checked,"included_forces":["центральное поле","J₂","атмосферное сопротивление"],
        "motor_acceleration_applied":False,
        "status":("проверен один пролёт и полный виток" if not checked["physical_violations"]
                  else "кандидат не прошёл физическую проверку"),
        "limitations":["ограниченный поиск без гарантии глобального оптимума",
                       "приближённая плотность, без прогноза космической погоды",
                       "многосуточная повторяемость не проверена"]}
    selected_index=best_index
    if selected["parameters"]["source"]=="ранее проверенный кандидат":
        selected_index=next((i for i,row in enumerate(history)
                             if row["source"]=="ранее проверенный кандидат"),best_index)
    for i,row in enumerate(history):
        row["outcome"]="selected" if i==selected_index else "rejected"
        if i!=selected_index and not row["rejection_reasons"]:
            row["rejection_reasons"]=["не выбран после точной проверки пролёта и полного витка"]
    history_file.write_text("".join(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n"
                                    for row in history),encoding="utf-8")
    best_file.write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return payload,best_file


def load_cached(config,size,root=RESULTS):
    cfg=prepare_config(config);key=signature(cfg,size)
    path=Path(root)/key/"best.json"
    if not path.exists():return None
    data=json.loads(path.read_text(encoding="utf-8"))
    if data.get("signature")!=key or data.get("config")!=cfg or data.get("model_fingerprint_sha256")!=model_fingerprint():
        raise ValueError("Сохранённый подбор несовместим с текущей моделью")
    raw=data["candidate"];states=np.asarray(raw["initial_states_eci_km_km_s"])
    _,_,_,_,target,_=geometry_at_epoch(cfg,size)
    candidate=_candidate(states,target,size,cfg,raw["source"])
    if np.max(abs(reconstruct(raw["elements_at_epoch"],cfg["orbit"]["mu_km3_s2"])-states))>1e-7:
        raise ValueError("Сохранённые элементы не воспроизводят состояния")
    return cfg,candidate,data,path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n",type=int,default=2)
    parser.add_argument("--side-deg",type=float,default=20.)
    parser.add_argument("--altitude-km",type=float,default=700.)
    parser.add_argument("--inclination-deg",type=float,default=97.8)
    parser.add_argument("--observer-lat",type=float,default=25.3026)
    parser.add_argument("--observer-lon",type=float,default=51.5132)
    parser.add_argument("--min-elevation-deg",type=float,default=15.)
    parser.add_argument("--epoch-utc",default="2026-09-28T00:00:00Z")
    parser.add_argument("--budget",type=int,default=16)
    parser.add_argument("--output-root",type=Path,default=RESULTS)
    parser.add_argument("--no-resume",action="store_true")
    args=parser.parse_args()
    cfg=configuration(args.n,args.side_deg,args.observer_lat,args.observer_lon,
        args.min_elevation_deg,args.epoch_utc,args.altitude_km,args.inclination_deg)
    result,path=optimize(cfg,args.n,args.budget,args.output_root,not args.no_resume,progress=print)
    v=result["verification"]
    print(f"{path}: квадрат {v['one_pass']['best_square_duration_s']:g} с, "
          f"при 2° {v['one_pass']['best_resolution_2deg_s']:g} с; "
          f"полный виток min={v['full_orbit']['closest_pair']['distance_km']:.6f} км; "
          f"нарушения: {v['physical_violations']}")


if __name__=="__main__":main()
