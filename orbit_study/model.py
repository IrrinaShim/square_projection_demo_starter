"""Кандидаты: оскулирующие элементы на общей эпохе, затем только J₂.

Единицы: км, с, рад внутри формул; углы элементов и отчёта в градусах.
Никакого управления после эпохи не прикладывается.
"""
from __future__ import annotations

import numpy as np

from square_demo.core import image_frame, observer_eci, zenith_orbit
from square_demo.grid import grid_indices, initial_grid_from_corners
from square_demo.orbit_j2 import elements_to_state, orbital_basis, state_to_elements


def make_candidate(base, size, height_km, side_deg, velocity_factor=1.0,
                   depth_factor=1.0, phase_s=0.0, inclination_deg=97.8):
    """Интерполируются только начальные состояния внутренних КА.

    Скорости углов наследуют относительную скорость исходного четырёхточечного
    кандидата, масштабированную по поперечному размеру; затем все N² объектов
    распространяются независимо. Параметры являются поисковыми, не импульсами.
    """
    orbit = dict(base["orbit"], altitude_km=float(height_km),
                 inclination_deg=float(inclination_deg))
    observer = base["observer"]
    mu, re = float(orbit["mu_km3_s2"]), float(orbit["earth_radius_km"])
    old_corner = np.stack([elements_to_state(e,mu) for e in base["elements_at_epoch"]])
    old_basis = orbital_basis(old_corner[:1])[0]
    old_rel = old_corner[:,0]-old_corner[0,0]
    old_local = (old_basis.T@old_rel.T).T
    old_omega = np.cross(old_corner[0,0],old_corner[0,1])/np.dot(old_corner[0,0],old_corner[0,0])
    old_vlocal = (old_basis.T@(old_corner[:,1]-old_corner[0,1]
                  -np.cross(old_omega,old_rel)).T).T

    chief_r,bases,_,_ = zenith_orbit(np.array([float(phase_s)]),0.0,orbit,observer)
    b = bases[0]
    r0 = chief_r[0]
    v0 = b[:,1]*np.sqrt(mu/np.linalg.norm(r0))
    chief = np.stack((r0,v0))
    obs = observer_eci(np.array([0.0]),0.0,re,observer)[0]
    _,sight,exi,eeta = image_frame(r0[None,:],obs[None,:],b[None,:,:])
    lam0 = float(np.dot(r0-obs,sight[0]))
    old_obs = observer_eci(np.array([0.0]),0.0,re,observer)[0]
    old_sight = (old_corner[0,0]-old_obs)/np.linalg.norm(old_corner[0,0]-old_obs)
    old_ranges = (old_corner[:,0]-old_obs)@old_sight
    scale = height_km/float(base["orbit"]["altitude_km"])*(
        np.tan(np.deg2rad(side_deg))/np.tan(np.deg2rad(base["angular_side_deg"])))
    omega = np.cross(r0,v0)/np.dot(r0,r0)
    alpha = np.tan(np.deg2rad(side_deg))
    xy = np.array([[0,0],[alpha,0],[alpha,alpha],[0,alpha]])
    corners = np.empty((4,2,3))
    corners[0] = chief
    for k in range(1,4):
        lam = lam0+depth_factor*(old_ranges[k]-old_ranges[0])*scale
        corners[k,0] = obs+lam*(sight[0]+xy[k,0]*exi[0]+xy[k,1]*eeta[0])
        rel = corners[k,0]-r0
        corners[k,1] = v0+np.cross(omega,rel)+b@(old_vlocal[k]*scale*velocity_factor)
    states,target,elements = initial_grid_from_corners(corners,size,obs,sight[0],
                                                        exi[0],eeta[0],side_deg,mu)
    if np.any(np.linalg.norm(states[:,0],axis=1)<re+100):
        raise ValueError("начальная высота ниже 100 км")
    if any(e["a_km"]*(1-e["e"])<re+100 for e in elements):
        raise ValueError("оскул. перигей ниже 100 км")
    return {"states":states,"elements":elements,"target":target,
            "grid_indices":grid_indices(size).tolist(),"size":size,
            "parameters":{"height_km":float(height_km),"side_deg":float(side_deg),
                          "velocity_factor":float(velocity_factor),
                          "depth_factor":float(depth_factor),"phase_s":float(phase_s),
                          "inclination_deg":float(inclination_deg)}}


def reconstruct(elements, mu):
    return np.stack([elements_to_state(e,mu) for e in elements])
