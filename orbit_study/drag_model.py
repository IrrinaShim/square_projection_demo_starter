"""Исследовательская ECI-модель: центральное поле, J₂, вращающаяся атмосфера."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from square_demo.orbit_j2 import acceleration_j2


def gmst_approx_deg(epoch_utc):
    """GMST среднего равноденствия даты; UT1≈UTC, прецессия не преобразована."""
    instant=datetime.fromisoformat(epoch_utc.replace("Z","+00:00"))
    if instant.tzinfo is None or instant.utcoffset().total_seconds()!=0:
        raise ValueError("Эпоха должна быть UTC")
    jd=instant.timestamp()/86400+2440587.5
    d=jd-2451545.0
    century=d/36525.0
    return float((280.46061837+360.98564736629*d+0.000387933*century**2
                  -century**3/38710000.0)%360)


def density_kg_m3(altitude_km, atmosphere):
    altitude=np.asarray(altitude_km,dtype=float)
    return (float(atmosphere["rho_reference_kg_m3"])
            *np.exp(-(altitude-float(atmosphere["reference_altitude_km"]))
                    /float(atmosphere["scale_height_km"])))


def drag_acceleration_km_s2(position_km, velocity_km_s, earth_radius_km,
                            rotation_rad_s, spacecraft, atmosphere):
    """-ρ C_D A/(2m)|v_rel|v_rel; площади в м², выход в км/с²."""
    r=np.asarray(position_km,dtype=float)
    v=np.asarray(velocity_km_s,dtype=float)
    omega=np.array([0.,0.,float(rotation_rad_s)])
    relative=v-np.cross(omega,r)
    rho=density_kg_m3(np.linalg.norm(r,axis=-1)-earth_radius_km,atmosphere)
    coefficient=(.5*rho*float(spacecraft["drag_coefficient"])
                 *float(spacecraft["effective_area_m2"])
                 /float(spacecraft["mass_kg"])*1000.)
    return -coefficient[...,None]*np.linalg.norm(relative,axis=-1)[...,None]*relative


def acceleration(states,config,with_drag=True):
    orbit=config["orbit"]
    out=acceleration_j2(states[...,0,:],orbit["mu_km3_s2"],
                        orbit["earth_radius_km"],orbit["j2"])
    if with_drag:
        out+=drag_acceleration_km_s2(states[...,0,:],states[...,1,:],
            orbit["earth_radius_km"],config["observer"]["earth_rotation_rad_s"],
            config["spacecraft"],config["atmosphere"])
    return out


def _rk4(states,dt,config,with_drag):
    def derivative(x):
        return np.stack((x[...,1,:],acceleration(x,config,with_drag)),axis=-2)
    k1=derivative(states)
    k2=derivative(states+dt*k1/2)
    k3=derivative(states+dt*k2/2)
    k4=derivative(states+dt*k3)
    return states+dt*(k1+2*k2+2*k3+k4)/6


def integrate(initial,times,config,max_step_s=2.,with_drag=True):
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or len(times)<1 or not np.isfinite(times).all() or np.any(np.diff(times)<=0):
        raise ValueError("Нужна строго возрастающая конечная сетка времени")
    if max_step_s<=0: raise ValueError("Шаг интегрирования должен быть положительным")
    result=np.empty((len(times),*initial.shape))
    for indices in (np.flatnonzero(times<0)[::-1],np.flatnonzero(times>=0)):
        state=np.asarray(initial,dtype=float).copy()
        current=0.
        for k in indices:
            duration=float(times[k]-current)
            count=max(1,int(np.ceil(abs(duration)/max_step_s)))
            for _ in range(count): state=_rk4(state,duration/count,config,with_drag)
            result[k]=state
            current=float(times[k])
    return result
