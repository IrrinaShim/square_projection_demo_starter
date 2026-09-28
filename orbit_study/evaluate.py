"""Критерии одного пролёта и потоковая проверка долгого горизонта."""
from __future__ import annotations

from datetime import datetime, timedelta
from itertools import combinations

import numpy as np

from square_demo.core import observer_eci, project, square_geometry
from square_demo.grid import corner_indices, grid_acceptance, grid_metrics
from square_demo.orbit_j2 import (orbital_basis, integrate, acceleration_j2,
    _separation_audit_many, _advance, _pair_event)


def geometry(states, times, scenario, size, thresholds=(1.,2.)):
    """Строго те же определения формы, что и у GUI; размер отдельно."""
    t = np.asarray(times,dtype=float)
    r = states[:,:,0]
    orbit, observer = scenario["orbit"],scenario["observer"]
    robs = observer_eci(t,0,orbit["earth_radius_km"],observer)
    image,valid,elev = project(r,robs,orbital_basis(states[:,0]),orbit["earth_radius_km"])
    corners = corner_indices(size)
    sides,_,_,err = square_geometry(image[:,corners],valid[:,corners])
    outer = image[:,corners]
    edges = np.roll(outer,-1,axis=1)-outer
    crosses = edges[:,:,0]*np.roll(edges[:,:,1],-1,axis=1)-edges[:,:,1]*np.roll(edges[:,:,0],-1,axis=1)
    convex = np.all(crosses>0,axis=1)
    g = grid_metrics(image,valid,r,robs,size)
    vis = scenario["visibility"]
    base = (np.all(valid,axis=1)&convex&np.all(np.isfinite(err),axis=1)
        &(np.min(elev,axis=1)>=vis["min_elevation_deg"])
        &(err[:,0]<=vis["max_side_error"])
        &(err[:,1]<=vis["max_diagonal_error"])
        &(err[:,2]<=np.deg2rad(vis["max_angle_error_deg"])))
    accepted = {}
    for threshold in thresholds:
        tol = dict(scenario["grid_tolerances"],min_neighbor_angle_deg=float(threshold))
        grid_ok,_ = grid_acceptance(g,size,tol)
        accepted[str(int(threshold))] = base&grid_ok
    finite=np.isfinite(sides)
    counts=np.sum(finite,axis=1)
    outer_side=np.rad2deg(np.arctan(np.divide(np.nansum(sides,axis=1),counts,
        out=np.full(len(sides),np.nan),where=counts>0)))
    return {"times_s":t,"accepted":accepted,"outer_side_deg":outer_side,
            "min_true_neighbor_deg":g["min_true_neighbor_angle_deg"],
            "min_elevation_deg":np.min(elev,axis=1),"shape_errors":err,
            "point_error_steps":g["point_error_steps"],
            "neighbor_step_error":g["neighbor_step_error"],
            "cell_angle_error_deg":g["max_cell_angle_error_deg"],
            "all_valid":np.all(valid,axis=1),
            "ordering_ok":g["ordering_ok"],"cells_convex":g["cells_convex"]}


def intervals(metrics, threshold):
    ok = metrics["accepted"][str(int(threshold))]
    delta = np.diff(np.r_[False,ok,False].astype(int))
    starts,ends = np.flatnonzero(delta==1),np.flatnonzero(delta==-1)-1
    t = metrics["times_s"]
    out = []
    for lo,hi in zip(starts,ends):
        # Одиночный принятый узел не считается положительным интервалом.
        if hi <= lo:
            continue
        side = metrics["outer_side_deg"][lo:hi+1]
        step = metrics["min_true_neighbor_deg"][lo:hi+1]
        out.append({"start_s":float(t[lo]),"end_s":float(t[hi]),
                    "duration_s":float(t[hi]-t[lo]),
                    "min_outer_side_deg":float(np.min(side)),
                    "max_outer_side_deg":float(np.max(side)),
                    "min_true_neighbor_deg":float(np.min(step)),
                    "max_shape_errors":[float(x) for x in np.max(metrics["shape_errors"][lo:hi+1],axis=0)],
                    "max_point_error_steps":float(np.max(metrics["point_error_steps"][lo:hi+1])),
                    "max_step_error":float(np.max(metrics["neighbor_step_error"][lo:hi+1]))})
    return out


def survey(candidate, scenario, sample_s=10.):
    orbit = scenario["orbit"]
    t = np.arange(-450.,450.+sample_s/2,sample_s)
    states = integrate(candidate["states"],t,orbit["mu_km3_s2"],
        orbit["earth_radius_km"],orbit["j2"],min(sample_s,10.))
    m = geometry(states,t,scenario,candidate["size"])
    wins = {key:intervals(m,float(key)) for key in ("1","2")}
    best = {key:max(wins[key],key=lambda x:(x["duration_s"],x["min_outer_side_deg"]),
                    default=None) for key in wins}
    return {"candidate":candidate,"best":best,
            "maximum_point_error_steps":float(np.nanmax(m["point_error_steps"])),
            "minimum_neighbor_angle_deg":float(np.nanmin(m["min_true_neighbor_deg"])),
            "minimum_elevation_deg":float(np.min(m["min_elevation_deg"]))}


def _utc(epoch,t):
    dt=datetime.fromisoformat(epoch.replace("Z","+00:00"))+timedelta(seconds=float(t))
    return dt.isoformat().replace("+00:00","Z")


def _finite_extreme(values, kind="max"):
    finite=np.asarray(values,dtype=float)
    finite=finite[np.isfinite(finite)]
    if not len(finite):
        return None
    return float(np.max(finite) if kind=="max" else np.min(finite))


def detailed(candidate, scenario, horizon_days, sample_s=60., integration_step_s=10.,
             pass_step_s=5., chunk_hours=6., collision_km=0.02, margin_km=1.0):
    """Все пары на всём горизонте; короткие окна пролётов с шагом pass_step_s.

    Аудит каждого непрерывного блока использует Эрмитовы экстремумы между
    узлами; возле порога движение пары заново интегрируется в J₂.
    """
    orbit,observer=scenario["orbit"],scenario["observer"]
    mu,re,j2=[float(orbit[k]) for k in ("mu_km3_s2","earth_radius_km","j2")]
    total=float(horizon_days)*86400
    if total<=0 or sample_s<=0 or integration_step_s<=0 or pass_step_s<=0:
        raise ValueError("все интервалы и шаги должны быть положительными")
    if sample_s>60 or integration_step_s>10 or pass_step_s>5:
        raise ValueError("для итоговой проверки допустимы шаги ≤60/10/5 с")
    steps=int(np.ceil(total/sample_s))
    if steps>2_000_000:
        raise ValueError("превышен бюджет узлов")
    pairs=len(candidate["states"])*(len(candidate["states"])-1)//2
    pair_best={}
    dangerous=[]
    visible=[]
    min_alt=np.inf
    max_lat=0.
    exact_refinements=0
    restart=None
    # Включаем всю первую видимость вокруг общей эпохи, в том числе t<0.
    t0=-450.
    state=integrate(candidate["states"],np.array([t0]),mu,re,j2,integration_step_s)[0]
    chunk_nodes=max(2,int(chunk_hours*3600/sample_s))
    while t0 < total-1e-9:
        end=min(total,t0+chunk_nodes*sample_s)
        tt=np.arange(t0,end+sample_s*.49,sample_s)
        if tt[-1]<end-1e-8:
            tt=np.r_[tt,end]
        tt[-1]=end
        local=tt-t0
        block=integrate(state,local,mu,re,j2,integration_step_s)
        min_alt=min(min_alt,float(np.min(np.linalg.norm(block[:,:,0],axis=2)-re)))
        # Параметры вращения и UTC сдвигаются вместе с началом блока.
        shifted=dict(observer,greenwich_angle_at_zenith_deg=float(observer[
            "greenwich_angle_at_zenith_deg"])+np.rad2deg(observer["earth_rotation_rad_s"]*t0))
        audit=_separation_audit_many(block,local,mu,re,j2,shifted,_utc(scenario["epoch_utc"],t0),
                                     collision_km+margin_km,min(0.1,integration_step_s))
        exact_refinements+=audit["near_threshold_j2_refinements"]
        max_lat=max(max_lat,audit["max_abs_chief_latitude_deg"])
        for key,event in audit["pair_minima"].items():
            if key not in pair_best or event["distance_km"]<pair_best[key]["distance_km"]:
                pair_best[key]=dict(event,t_s=float(event["t_s"]+t0))
            if restart is None or event["distance_km"]<restart["predicted_km"]:
                a,b=map(int,key.split("-"))
                k=int(np.clip(np.searchsorted(local,event["t_s"],side="right")-1,
                              0,len(local)-2))
                restart={"predicted_km":float(event["distance_km"]),
                         "pair":key,"states":block[k,[a,b]].copy(),
                         "left_s":float(tt[k]),"right_s":float(tt[k+1]),
                         "time_s":float(event["t_s"]+t0)}
        dangerous.extend(dict(e,t_s=float(e["t_s"]+t0)) for e in audit["dangerous_events"]
                         if e["distance_km"]<collision_km+margin_km)
        robs=observer_eci(tt,0,re,observer)
        los=block[:,0,0]-robs
        elev=np.rad2deg(np.arcsin(np.clip(np.sum(los*robs,axis=1)/
            (np.linalg.norm(los,axis=1)*re),-1,1)))
        for k in np.flatnonzero(elev>0):
            # Узел на границе блока попадёт дважды; позже удалим дубликат.
            visible.append((float(tt[k]),block[k].copy()))
        state=block[-1]
        t0=end
    visible.sort(key=lambda item:item[0])
    visible=[entry for k,entry in enumerate(visible)
             if k==0 or entry[0]>visible[k-1][0]+1e-7]
    runs=[]
    for item in visible:
        if not runs or item[0]-runs[-1][-1][0]>sample_s+1e-6:
            runs.append([])
        runs[-1].append(item)
    passes=[]
    for run in runs:
        if len(run)<2:
            continue
        first,last=run[0][0],run[-1][0]
        times=np.arange(first,last+pass_step_s*.49,pass_step_s)
        if times[-1]<last-1e-8:
            times=np.r_[times,last]
        states=integrate(run[0][1],times-first,mu,re,j2,min(integration_step_s,pass_step_s))
        geom=geometry(states,times,scenario,candidate["size"])
        wins={key:intervals(geom,float(key)) for key in ("1","2")}
        passes.append({"chief_visible_start_s":first,"chief_visible_end_s":last,
            "intervals":wins,"best":{key:max(wins[key],key=lambda x:x["duration_s"],
                                       default=None) for key in wins},
            "max_shape_errors":[_finite_extreme(geom["shape_errors"][:,j]) for j in range(3)],
            "min_elevation_deg":float(np.min(geom["min_elevation_deg"])),
            "max_point_error_steps":_finite_extreme(geom["point_error_steps"]),
            "min_true_neighbor_deg":_finite_extreme(geom["min_true_neighbor_deg"],"min")})
    repeat={}
    for key in ("1","2"):
        hits=[p for p in passes if p["best"][key] is not None]
        centers=[(p["best"][key]["start_s"]+p["best"][key]["end_s"])/2 for p in hits]
        repeat[key]={"qualifying_passes":len(hits),
                     "interval_between_passes_s":list(map(float,np.diff(centers))),
                     "repeat_span_days":float((centers[-1]-centers[0])/86400) if len(centers)>1 else 0.,
                     "best_pass":max((p["best"][key] for p in hits),
                                      key=lambda x:(x["duration_s"],x["min_outer_side_deg"]),default=None)}
    # Даже если кубический Эрмит дал далёкий от порога глобальный минимум,
    # окончательное число получаем из независимого J₂-интегрирования пары.
    a,b=map(int,restart["pair"].split("-"))
    event_time=restart["time_s"]
    for _ in range(3):
        at=_advance(restart["states"],event_time-restart["left_s"],mu,re,j2,0.1)
        dr=at[0,0]-at[1,0]
        dv=at[0,1]-at[1,1]
        da=acceleration_j2(at[0,0],mu,re,j2)-acceleration_j2(at[1,0],mu,re,j2)
        curvature=float(np.dot(dv,dv)+np.dot(dr,da))
        if curvature<=0:
            break
        event_time=float(np.clip(event_time-np.dot(dr,dv)/curvature,
                                 restart["left_s"],restart["right_s"]))
    at=_advance(restart["states"],event_time-restart["left_s"],mu,re,j2,0.1)
    closest=_pair_event(at[0,0],at[1,0],event_time,a,b,observer,
                        scenario["epoch_utc"],"J2 refined global minimum")
    pair_best[restart["pair"]]=closest
    if closest["distance_km"]<collision_km+margin_km:
        dangerous.append(closest)
    return {"horizon_days":float(horizon_days),"horizon_s":[-450.0,total],"sample_s":sample_s,
            "integration_step_s":integration_step_s,"pass_step_s":pass_step_s,
            "pair_count":pairs,"pair_minima":pair_best,"closest_pair":closest,
            "near_threshold_events":sorted(dangerous,key=lambda x:x["t_s"]),
            "collision_events":[event for event in dangerous
                                if event["distance_km"]<collision_km],
            "collision_threshold_km":float(collision_km),
            "safety_margin_km":float(margin_km),
            "absolute_minimum_requirement_km":float(collision_km+margin_km),
            "near_threshold_j2_refinements":exact_refinements,
            "minimum_altitude_km":min_alt,"maximum_chief_latitude_deg":max_lat,
            "visible_pass_count":len(passes),"passes":passes,"repeat":repeat}
