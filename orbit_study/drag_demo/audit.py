"""Все пары на одном пролёте: кубический Эрмит и точная проверка близких минимумов."""
from __future__ import annotations

from datetime import datetime, timedelta
from itertools import combinations

import numpy as np

from orbit_study.drag_model import acceleration, integrate
from square_demo.orbit_j2 import _hermite_relative, earth_location


def _event(position_i,position_j,t_s,pair,config,method):
    center=(position_i+position_j)/2
    obs=config["observer"]
    lat,lon=earth_location(center,t_s,obs["earth_rotation_rad_s"],
                           obs["greenwich_angle_at_zenith_deg"])
    utc=datetime.fromisoformat(config["epoch_utc"].replace("Z","+00:00"))
    return {"pair":pair,"t_s":float(t_s),
        "utc":(utc+timedelta(seconds=float(t_s))).isoformat().replace("+00:00","Z"),
        "distance_km":float(np.linalg.norm(position_i-position_j)),
        "latitude_deg":lat,"longitude_deg":lon,"method":method}


def _exact_pair(states,left_s,right_s,guess_s,config):
    """Уточнение корня Δr·Δv=0 повторным интегрированием двух КА."""
    event_time=float(guess_s)
    for _ in range(4):
        pair_state=integrate(states,np.array([event_time-left_s]),config,.05)[0]
        dr=pair_state[0,0]-pair_state[1,0]
        dv=pair_state[0,1]-pair_state[1,1]
        a=acceleration(pair_state,config)
        curvature=np.dot(dv,dv)+np.dot(dr,a[0]-a[1])
        if curvature<=0: break
        event_time=float(np.clip(event_time-np.dot(dr,dv)/curvature,left_s,right_s))
    pair_state=integrate(states,np.array([event_time-left_s]),config,.05)[0]
    return event_time,pair_state[:,0]


def audit_one_pass(states,times,config,threshold_km=None,cancel_event=None):
    count=states.shape[1]
    pairs=np.array(list(combinations(range(count),2)),dtype=int)
    threshold=float(threshold_km if threshold_km is not None else
        config["spacecraft"]["collision_radius_sum_km"]
        +config["spacecraft"]["additional_safety_margin_km"])
    events={}
    refine_candidates=[]
    for offset in range(0,len(pairs),192):
        if cancel_event is not None and cancel_event.is_set():
            raise ValueError("Аудит пар отменён после изменения параметров")
        block=pairs[offset:offset+192]
        i,j=block.T
        dr=states[:,i,0]-states[:,j,0]
        dv=states[:,i,1]-states[:,j,1]
        distance2=np.einsum("tbi,tbi->tb",dr,dr)
        rates=np.einsum("tbi,tbi->tb",dr,dv)
        sample_min=np.argmin(distance2,axis=0)
        best=[_event(states[k,a,0],states[k,b,0],times[k],f"{a}-{b}",config,"узел")
              for a,b,k in zip(i,j,sample_min)]
        rows,cols=np.nonzero((rates[:-1]<=0)&(rates[1:]>0))
        if len(rows):
            dt=times[rows+1]-times[rows]
            low,high=np.zeros(len(rows)),np.ones(len(rows))
            for _ in range(28):
                mid=(low+high)/2
                rp,dp=_hermite_relative(dr[rows,cols],dr[rows+1,cols],
                    dv[rows,cols],dv[rows+1,cols],dt,mid)
                negative=np.einsum("bi,bi->b",rp,dp)<0
                low=np.where(negative,mid,low)
                high=np.where(negative,high,mid)
            fraction=(low+high)/2
            for q,(row,col) in enumerate(zip(rows,cols)):
                a,b=map(int,block[col]);pair=f"{a}-{b}"
                instant=float(times[row]+fraction[q]*dt[q])
                pos,_=_hermite_relative(states[row,[a,b],0],states[row+1,[a,b],0],
                    states[row,[a,b],1],states[row+1,[a,b],1],
                    np.full(2,dt[q]),np.full(2,fraction[q]))
                event=_event(pos[0],pos[1],instant,pair,config,"Эрмит между узлами")
                if event["distance_km"]<best[col]["distance_km"]:
                    best[col]=event
                    refine_candidates.append((pair,row,instant))
        for col,(a,b) in enumerate(block): events[f"{a}-{b}"]=best[col]
    closest_key=min(events,key=lambda key:events[key]["distance_km"])
    exact_count=0
    for pair,row,guess in refine_candidates:
        if pair!=closest_key and events[pair]["distance_km"]>threshold*1.1:
            continue
        a,b=map(int,pair.split("-"))
        instant,positions=_exact_pair(states[row,[a,b]],times[row],times[row+1],
                                      guess,config)
        event=_event(positions[0],positions[1],instant,pair,config,
                     "повторное интегрирование J₂+сопротивление")
        if event["distance_km"]<events[pair]["distance_km"]+1e-5:
            events[pair]=event
        exact_count+=1
    closest=min(events.values(),key=lambda item:item["distance_km"])
    return {"pair_count":len(pairs),"pair_minima":events,"closest_pair":closest,
            "threshold_km":threshold,"refined_event_count":exact_count,
            "horizon_s":[float(times[0]),float(times[-1])],
            "sample_step_max_s":float(np.max(np.diff(times))),
            "refinement_step_max_s":.05,
            "dangerous_events":[item for item in events.values()
                                if item["distance_km"]<threshold]}
