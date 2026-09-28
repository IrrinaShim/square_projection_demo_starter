"""Независимая проверка пробной формации на точных двухтельных орбитах.

Запуск: .venv/bin/python scripts/two_body_check.py
Единицы: км, с, км/с. SciPy не требуется.
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from square_demo.core import zenith_orbit
from square_demo.providers import SCENARIOS, load_scenario


def initial_states(scenario):
    orbit, observer = scenario["orbit"], scenario["observer"]
    r0, basis, _, n = zenith_orbit(np.array([0.0]),0.0,orbit,observer)
    a = float(orbit["earth_radius_km"])+float(orbit["altitude_km"])
    length = float(orbit["altitude_km"])*np.tan(np.deg2rad(scenario["angular_side_deg"]))
    rho = np.array([[0,0,0],[0,length,0],[0,length,length],[0,0,length]],dtype=float)
    # v_I = v_chief + B [v_rel + (n e_z) x rho], здесь v_rel=0.
    rotating_term = np.column_stack((-n*rho[:,1],n*rho[:,0],np.zeros(4)))
    position = r0[0]+(basis[0]@rho.T).T
    velocity = a*n*basis[0,:,1]+(basis[0]@rotating_term.T).T
    return np.stack((position,velocity),axis=1),n


def derivative(states,mu):
    out = np.empty_like(states)
    position = states[:,0,:]
    out[:,0,:] = states[:,1,:]
    out[:,1,:] = -mu*position/np.linalg.norm(position,axis=1)[:,None]**3
    return out


def rk4(states,dt,mu):
    k1 = derivative(states,mu)
    k2 = derivative(states+dt*k1/2,mu)
    k3 = derivative(states+dt*k2/2,mu)
    k4 = derivative(states+dt*k3,mu)
    return states+dt*(k1+2*k2+2*k3+k4)/6


def propagate(initial,mu,period,divisions=8192,orbits=3):
    dt = period/divisions
    samples = divisions*orbits
    states = np.empty((samples+1,4,2,3))
    states[0] = initial
    for k in range(samples):
        states[k+1] = rk4(states[k],dt,mu)
    return states,dt


def specific_energy(states,mu):
    r = np.linalg.norm(states[...,0,:],axis=-1)
    v2 = np.sum(states[...,1,:]**2,axis=-1)
    return v2/2-mu/r


def refined_minimum(states,dt,mu,pair,start,end,subdivisions=200):
    """Перебирает все узловые локальные минимумы и уточняет каждый."""
    i,j = pair
    distance = np.linalg.norm(states[:end+1,i,0]-states[:end+1,j,0],axis=1)
    local = np.flatnonzero((distance[1:-1] <= distance[:-2])
                           & (distance[1:-1] <= distance[2:]))+1
    candidates = np.r_[start,end,local[(local>=start)&(local<=end)]]
    best = (float("inf"),None)
    for k in candidates:
        left = max(start,int(k)-1)
        right = min(end,int(k)+1)
        step = dt/subdivisions
        state = states[left].copy()
        for q in range((right-left)*subdivisions+1):
            d = float(np.linalg.norm(state[i,0]-state[j,0]))
            if d < best[0]:
                best = (d,(left*subdivisions+q)*step)
            if q < (right-left)*subdivisions:
                state = rk4(state,step,mu)
    return {"km":best[0],"t_s":best[1]}


def report(divisions=8192):
    scenario = load_scenario(SCENARIOS["free_hcw_two_body"])
    orbit = scenario["orbit"]
    mu = float(orbit["mu_km3_s2"])
    initial,n = initial_states(scenario)
    period = 2*np.pi/n
    states,dt = propagate(initial,mu,period,divisions)
    energy = specific_energy(states,mu)
    initial_energy = energy[0]
    semimajor = -mu/(2*initial_energy)
    satellite_periods = 2*np.pi*np.sqrt(semimajor**3/mu)
    pairs = list(combinations(range(4),2))
    minima = {}
    for horizon in (1,3):
        end = divisions*horizon
        minima[str(horizon)] = {f"{i}-{j}":refined_minimum(states,dt,mu,(i,j),0,end)
                                for i,j in pairs}
    checkpoints = {}
    for name,index in (("T/4",divisions//4),("3T/4",3*divisions//4),
                       ("T",divisions),("3T",3*divisions)):
        checkpoints[name] = {f"{i}-{j}":float(np.linalg.norm(
            states[index,i,0]-states[index,j,0])) for i,j in pairs}
    relative_energy_error = np.max(np.abs((energy-initial_energy)/initial_energy),axis=0)
    return {
        "method":"fixed-step classical RK4; all pairwise local minima refined with RK4 step/200",
        "units":{"distance":"km","time":"s","velocity":"km/s","energy":"km²/s²"},
        "T_reference_s":period,"coarse_step_s":dt,"refined_step_s":dt/200,
        "initial_r_eci_km":initial[:,0,:].tolist(),
        "initial_v_eci_km_s":initial[:,1,:].tolist(),
        "semimajor_axes_km":semimajor.tolist(),
        "orbital_periods_s":satellite_periods.tolist(),
        "max_relative_energy_drift":relative_energy_error.tolist(),
        "minima":minima,"checkpoints_km":checkpoints,
    }


if __name__ == "__main__":
    print(json.dumps(report(),ensure_ascii=False,indent=2))
