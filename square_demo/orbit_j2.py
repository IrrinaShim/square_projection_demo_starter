"""Абсолютное движение КА в осесимметричном поле Земли с J₂.

Оскулирующие классические элементы заданы в одной условной геоцентрической
инерциальной системе на общей эпохе. Длины: км; время: с; углы: градусы.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from itertools import combinations

import numpy as np


def elements_to_state(elements, mu):
    """(a,e,i,Ω,ω,ν) -> (r,v) в ECI; поддерживает малый, но ненулевой e."""
    a = float(elements["a_km"])
    e = float(elements["e"])
    inc, node, peri, anomaly = np.deg2rad([float(elements[k]) for k in
        ("inclination_deg", "raan_deg", "arg_perigee_deg", "true_anomaly_deg")])
    if not (np.isfinite([a,e,inc,node,peri,anomaly]).all() and a > 0 and 0 <= e < 1):
        raise ValueError("Некорректные эллиптические элементы орбиты")
    p = a*(1-e*e)
    radius = p/(1+e*np.cos(anomaly))
    rp = radius*np.array([np.cos(anomaly),np.sin(anomaly),0.0])
    vp = np.sqrt(mu/p)*np.array([-np.sin(anomaly),e+np.cos(anomaly),0.0])
    co,so = np.cos(node),np.sin(node)
    ci,si = np.cos(inc),np.sin(inc)
    cw,sw = np.cos(peri),np.sin(peri)
    rot = np.array([
        [co*cw-so*sw*ci,-co*sw-so*cw*ci,so*si],
        [so*cw+co*sw*ci,-so*sw+co*cw*ci,-co*si],
        [sw*si,cw*si,ci],
    ])
    return np.stack((rot@rp,rot@vp))


def state_to_elements(position, velocity, mu):
    """ECI -> оскулирующие элементы; применяется для подготовки сценария."""
    r,v = np.asarray(position,dtype=float),np.asarray(velocity,dtype=float)
    radius = np.linalg.norm(r)
    h = np.cross(r,v)
    hn = np.linalg.norm(h)
    node = np.cross([0.,0.,1.],h)
    nn = np.linalg.norm(node)
    eccentric = np.cross(v,h)/mu-r/radius
    e = np.linalg.norm(eccentric)
    if hn <= 0 or nn < 1e-10 or e >= 1:
        raise ValueError("Для классических элементов нужен невырожденный узел")
    a = 1/(2/radius-np.dot(v,v)/mu)
    inc = np.arccos(np.clip(h[2]/hn,-1,1))
    raan = np.arctan2(node[1],node[0])
    if e < 1e-10:
        # Для круговой орбиты перигей не определён; фиксируем ω=0, ν=u.
        arg = 0.0
        nu = np.arctan2(np.dot(np.cross(node,r),h)/hn,np.dot(node,r))
    else:
        arg = np.arctan2(np.dot(np.cross(node,eccentric),h)/hn,np.dot(node,eccentric))
        nu = np.arctan2(np.dot(np.cross(eccentric,r),h)/hn,np.dot(eccentric,r))
    return {"a_km":float(a),"e":float(e),"inclination_deg":float(np.rad2deg(inc)),
            "raan_deg":float(np.rad2deg(raan)%360),
            "arg_perigee_deg":float(np.rad2deg(arg)%360),
            "true_anomaly_deg":float(np.rad2deg(nu)%360)}


def acceleration_j2(position, mu, earth_radius, j2):
    r = np.asarray(position)
    r2 = np.sum(r*r,axis=-1)
    rn = np.sqrt(r2)
    z2r2 = r[...,2]**2/r2
    factor = 1.5*j2*mu*earth_radius**2/rn**5
    shape = np.stack((r[...,0]*(5*z2r2-1),r[...,1]*(5*z2r2-1),
                      r[...,2]*(5*z2r2-3)),axis=-1)
    return -mu*r/rn[...,None]**3+factor[...,None]*shape


def specific_energy_j2(states, mu, earth_radius, j2):
    """Удельная сохраняемая энергия в стационарном осесимметричном поле."""
    position = np.asarray(states)[...,0,:]
    velocity = np.asarray(states)[...,1,:]
    radius = np.linalg.norm(position,axis=-1)
    latitude_sine = position[...,2]/radius
    return (np.sum(velocity*velocity,axis=-1)/2-mu/radius
            +mu*j2*earth_radius**2/(2*radius**3)*(3*latitude_sine**2-1))


def _derivative(states, mu, earth_radius, j2):
    out = np.empty_like(states)
    out[:,0] = states[:,1]
    out[:,1] = acceleration_j2(states[:,0],mu,earth_radius,j2)
    return out


def rk4(states, dt, mu, earth_radius, j2):
    k1 = _derivative(states,mu,earth_radius,j2)
    k2 = _derivative(states+dt*k1/2,mu,earth_radius,j2)
    k3 = _derivative(states+dt*k2/2,mu,earth_radius,j2)
    k4 = _derivative(states+dt*k3,mu,earth_radius,j2)
    return states+dt*(k1+2*k2+2*k3+k4)/6


def _advance(state, duration, mu, earth_radius, j2, max_step_s):
    steps = max(1,int(np.ceil(abs(duration)/max_step_s)))
    step = duration/steps
    for _ in range(steps):
        state = rk4(state,step,mu,earth_radius,j2)
    return state


def integrate(initial, times, mu, earth_radius, j2, max_step_s=2.0):
    """Интегрирование от общей эпохи t=0 к произвольной возрастающей сетке."""
    times = np.asarray(times,dtype=float)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("Время интегрирования должно строго возрастать")
    if not np.isfinite(max_step_s) or max_step_s <= 0:
        raise ValueError("Шаг интегрирования должен быть положительным")
    result = np.empty((len(times),*initial.shape))
    for indices in (np.flatnonzero(times<0)[::-1],np.flatnonzero(times>=0)):
        state = np.asarray(initial,dtype=float).copy()
        current = 0.0
        for k in indices:
            state = _advance(state,float(times[k]-current),mu,earth_radius,j2,max_step_s)
            result[k] = state
            current = float(times[k])
    return result


def orbital_basis(chief_states):
    r,v = chief_states[:,0],chief_states[:,1]
    ex = r/np.linalg.norm(r,axis=1)[:,None]
    h = np.cross(r,v)
    ez = h/np.linalg.norm(h,axis=1)[:,None]
    ey = np.cross(ez,ex)
    return np.stack((ex,ey,ez),axis=-1)


def centroid_relative_frame(states, mu, earth_radius, j2):
    """Центр масс равновесных точек и вращающаяся ОСО КА 0.

    B=[e_r,e_t,e_h] ориентирована по мгновенным r0,v0. Скорость в
    координатах B равна B^T(v_j-v_c)+Bdot^T(r_j-r_c).
    """
    positions=states[:,:,0]
    velocities=states[:,:,1]
    center=positions.mean(axis=1)
    center_velocity=velocities.mean(axis=1)
    chief=states[:,0]
    r,v=chief[:,0],chief[:,1]
    acceleration=acceleration_j2(r,mu,earth_radius,j2)
    radius=np.linalg.norm(r,axis=1)
    ex=r/radius[:,None]
    exdot=(v-ex*np.sum(ex*v,axis=1)[:,None])/radius[:,None]
    h=np.cross(r,v)
    hdot=np.cross(r,acceleration)
    hnorm=np.linalg.norm(h,axis=1)
    ez=h/hnorm[:,None]
    ezdot=(hdot-ez*np.sum(ez*hdot,axis=1)[:,None])/hnorm[:,None]
    ey=np.cross(ez,ex)
    eydot=np.cross(ezdot,ex)+np.cross(ez,exdot)
    basis=np.stack((ex,ey,ez),axis=-1)
    basis_dot=np.stack((exdot,eydot,ezdot),axis=-1)
    dr=positions-center[:,None,:]
    dv=velocities-center_velocity[:,None,:]
    local_position=np.einsum("nji,nsj->nsi",basis,dr)
    local_velocity=(np.einsum("nji,nsj->nsi",basis,dv)
                    +np.einsum("nji,nsj->nsi",basis_dot,dr))
    return center,center_velocity,basis,local_position,local_velocity


def earth_location(position, t_s, earth_rotation_rad_s, greenwich_angle_deg):
    """Геоцентрические широта и долгота подспутниковой точки, градусы."""
    x,y,z = np.asarray(position,dtype=float)
    lat = np.rad2deg(np.arcsin(np.clip(z/np.linalg.norm(position),-1,1)))
    lon = np.rad2deg(np.arctan2(y,x)-np.deg2rad(greenwich_angle_deg)
                     -earth_rotation_rad_s*t_s)
    return float(lat),float((lon+180)%360-180)


def _distance_rate(states, i, j):
    dr = states[i,0]-states[j,0]
    dv = states[i,1]-states[j,1]
    return float(np.dot(dr,dv))


def _event(states, t_s, i, j, observer, epoch):
    dr = states[i,0]-states[j,0]
    midpoint = (states[i,0]+states[j,0])/2
    lat,lon = earth_location(midpoint,t_s,float(observer["earth_rotation_rad_s"]),
                             float(observer["greenwich_angle_at_zenith_deg"]))
    utc = datetime.fromisoformat(epoch.replace("Z","+00:00"))+timedelta(seconds=float(t_s))
    return {"pair":f"{i}-{j}","t_s":float(t_s),"utc":utc.isoformat().replace("+00:00","Z"),
            "distance_km":float(np.linalg.norm(dr)),"latitude_deg":lat,"longitude_deg":lon}


def separation_audit(states, times, mu, earth_radius, j2, observer, epoch,
                     threshold_km, refinement_step_s=0.1):
    """Все пары: корни d|Δr|²/dt между узлами и опасные локальные минимумы."""
    if times[0] != 0 or not np.all(np.diff(times)>0):
        raise ValueError("Аудит начинается на общей эпохе t=0")
    if states.shape[1] > 4:
        return _separation_audit_many(states,times,mu,earth_radius,j2,observer,
                                      epoch,threshold_km,refinement_step_s)
    pair_minima = {}
    events = []
    high_latitude_minima = {}
    chief_lat = np.rad2deg(np.arcsin(states[:,0,0,2]/np.linalg.norm(states[:,0,0],axis=1)))
    high = np.flatnonzero(abs(chief_lat)>=60)
    for i,j in combinations(range(4),2):
        pair = f"{i}-{j}"
        rates = np.array([_distance_rate(state,i,j) for state in states])
        candidates = [_event(states[0],times[0],i,j,observer,epoch),
                      _event(states[-1],times[-1],i,j,observer,epoch)]
        for k in np.flatnonzero((rates[:-1]<=0)&(rates[1:]>0)):
            left,right = float(times[k]),float(times[k+1])
            base = states[k]
            # Бисекция корня скалярного произведения Δr·Δv; каждый пробный
            # момент заново получен интегрированием J₂ от левого узла.
            for _ in range(28):
                middle = (left+right)/2
                at_mid = _advance(base,middle-times[k],mu,earth_radius,j2,refinement_step_s)
                if _distance_rate(at_mid,i,j)<0:
                    left = middle
                else:
                    right = middle
            t_min = (left+right)/2
            at_min = _advance(base,t_min-times[k],mu,earth_radius,j2,refinement_step_s)
            candidates.append(_event(at_min,t_min,i,j,observer,epoch))
        pair_minima[pair] = min(candidates,key=lambda item:item["distance_km"])
        events.extend(item for item in candidates if item["distance_km"]<threshold_km)
        if len(high):
            k = high[np.argmin(np.linalg.norm(states[high,i,0]-states[high,j,0],axis=1))]
            high_latitude_minima[pair] = _event(states[k],times[k],i,j,observer,epoch)
    return {"horizon_s":[float(times[0]),float(times[-1])],
            "status":"complete","pair_count":6,
            "sample_step_max_s":float(np.max(np.diff(times))),
            "refinement_step_max_s":float(refinement_step_s),
            "threshold_km":float(threshold_km),
            "max_abs_chief_latitude_deg":float(np.max(abs(chief_lat))),
            "high_latitude_definition_deg":60.0,
            "pair_minima":pair_minima,"high_latitude_sample_minima":high_latitude_minima,
            "dangerous_events":sorted(events,key=lambda item:item["t_s"])}


def _hermite_relative(r0,r1,v0,v1,dt,u):
    u = np.asarray(u)[:,None]
    d = dt[:,None]
    h00 = 2*u**3-3*u**2+1
    h10 = u**3-2*u**2+u
    h01 = -2*u**3+3*u**2
    h11 = u**3-u**2
    position = h00*r0+h10*d*v0+h01*r1+h11*d*v1
    derivative = ((6*u*u-6*u)*r0+(3*u*u-4*u+1)*d*v0
                  +(-6*u*u+6*u)*r1+(3*u*u-2*u)*d*v1)
    return position,derivative


def _pair_event(position_i, position_j, t_s, i, j, observer, epoch, method):
    midpoint = (position_i+position_j)/2
    lat,lon = earth_location(midpoint,t_s,float(observer["earth_rotation_rad_s"]),
                             float(observer["greenwich_angle_at_zenith_deg"]))
    utc = datetime.fromisoformat(epoch.replace("Z","+00:00"))+timedelta(seconds=float(t_s))
    return {"pair":f"{i}-{j}","t_s":float(t_s),"utc":utc.isoformat().replace("+00:00","Z"),
            "distance_km":float(np.linalg.norm(position_i-position_j)),
            "latitude_deg":lat,"longitude_deg":lon,"method":method}


def _separation_audit_many(states, times, mu, earth_radius, j2, observer,
                           epoch, threshold_km, refinement_step_s):
    """Блочный аудит O(T S²); J₂-переинтегрирование только у порога опасности.

    Кубический Эрмит уточняет все внутренние минимумы между узлами. Если
    оценка находится вблизи порога, пара дополнительно интегрируется в
    нелинейном поле J₂ от левого узла с шагом refinement_step_s.
    """
    count = states.shape[1]
    pairs = np.array(list(combinations(range(count),2)),dtype=int)
    chief_lat = np.rad2deg(np.arcsin(states[:,0,0,2]/np.linalg.norm(states[:,0,0],axis=1)))
    high = np.flatnonzero(abs(chief_lat)>=60)
    pair_minima = {}
    high_minima = {}
    dangerous = []
    verified_count = 0
    margin = max(0.1,0.1*threshold_km)
    for offset in range(0,len(pairs),192):
        block = pairs[offset:offset+192]
        i,j = block.T
        dr = states[:,i,0]-states[:,j,0]
        dv = states[:,i,1]-states[:,j,1]
        distance2 = np.einsum("tbi,tbi->tb",dr,dr)
        rates = np.einsum("tbi,tbi->tb",dr,dv)
        sample_min = np.argmin(distance2,axis=0)
        best = [_pair_event(states[k,a,0],states[k,b,0],times[k],a,b,
                            observer,epoch,"узел")
                for a,b,k in zip(i,j,sample_min)]
        crossings = (rates[:-1]<=0)&(rates[1:]>0)
        rows,cols = np.nonzero(crossings)
        if len(rows):
            r0,r1 = dr[rows,cols],dr[rows+1,cols]
            v0,v1 = dv[rows,cols],dv[rows+1,cols]
            dt = times[rows+1]-times[rows]
            low,high_u = np.zeros(len(rows)),np.ones(len(rows))
            for _ in range(28):
                mid = (low+high_u)/2
                rp,dp = _hermite_relative(r0,r1,v0,v1,dt,mid)
                negative = np.einsum("bi,bi->b",rp,dp)<0
                low = np.where(negative,mid,low)
                high_u = np.where(negative,high_u,mid)
            fraction = (low+high_u)/2
            predicted,_ = _hermite_relative(r0,r1,v0,v1,dt,fraction)
            predicted_dist = np.linalg.norm(predicted,axis=1)
            for q,(row,col) in enumerate(zip(rows,cols)):
                a,b = map(int,block[col])
                event_time = float(times[row]+fraction[q]*dt[q])
                near = predicted_dist[q] <= threshold_km+margin
                if near:
                    pair_start = states[row,[a,b]]
                    for _ in range(2):
                        pair_state = _advance(pair_start,event_time-times[row],
                                              mu,earth_radius,j2,refinement_step_s)
                        dr_exact = pair_state[0,0]-pair_state[1,0]
                        dv_exact = pair_state[0,1]-pair_state[1,1]
                        da_exact = (acceleration_j2(pair_state[0,0],mu,earth_radius,j2)
                                    -acceleration_j2(pair_state[1,0],mu,earth_radius,j2))
                        curvature = np.dot(dv_exact,dv_exact)+np.dot(dr_exact,da_exact)
                        if curvature <= 0:
                            break
                        correction = -np.dot(dr_exact,dv_exact)/curvature
                        event_time = float(np.clip(event_time+correction,times[row],times[row+1]))
                    pair_state = _advance(pair_start,event_time-times[row],
                                          mu,earth_radius,j2,refinement_step_s)
                    p_i,p_j = pair_state[:,0]
                    method = "J2 refined"
                    verified_count += 1
                else:
                    # Положение каждого КА интерполируется только для отчёта
                    # о минимуме; динамика траекторий всегда интегрировалась.
                    pos0 = states[row,[a,b],0]
                    pos1 = states[row+1,[a,b],0]
                    vel0 = states[row,[a,b],1]
                    vel1 = states[row+1,[a,b],1]
                    pos,_ = _hermite_relative(pos0,pos1,vel0,vel1,
                        np.full(2,dt[q]),np.full(2,fraction[q]))
                    p_i,p_j = pos
                    method = "Hermite between J2 nodes"
                event = _pair_event(p_i,p_j,event_time,a,b,observer,epoch,method)
                if event["distance_km"] < best[col]["distance_km"]:
                    best[col] = event
                if event["distance_km"] < threshold_km:
                    dangerous.append(event)
        for col,(a,b) in enumerate(block):
            pair = f"{a}-{b}"
            pair_minima[pair] = best[col]
            for endpoint in (0,len(times)-1):
                if np.sqrt(distance2[endpoint,col]) < threshold_km:
                    dangerous.append(_pair_event(states[endpoint,a,0],states[endpoint,b,0],
                        times[endpoint],a,b,observer,epoch,"граница интервала"))
            if len(high):
                k = int(high[np.argmin(distance2[high,col])])
                high_minima[pair] = _pair_event(states[k,a,0],states[k,b,0],
                    times[k],a,b,observer,epoch,"узел высокой широты")
    return {"status":"complete","pair_count":int(len(pairs)),
            "horizon_s":[float(times[0]),float(times[-1])],
            "sample_step_max_s":float(np.max(np.diff(times))),
            "refinement_step_max_s":float(refinement_step_s),
            "near_threshold_j2_refinements":verified_count,
            "minimum_method":"cubic Hermite root; near threshold independently re-integrated with J2",
            "threshold_km":float(threshold_km),
            "max_abs_chief_latitude_deg":float(np.max(abs(chief_lat))),
            "high_latitude_definition_deg":60.0,
            "pair_minima":pair_minima,"high_latitude_sample_minima":high_minima,
            "dangerous_events":sorted(dangerous,key=lambda item:item["t_s"])}
