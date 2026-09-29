"""Исследовательская целевая сетка вокруг луча на геометрический центр группы.

КА 0 остаётся на заданной опорной орбите. Начальные положения всех КА
вычисляются заново по симметричным лучам с общей дальностью; это не сдвиг
уже готовой картинки и не удержание конфигурации после эпохи.
"""
from __future__ import annotations

import numpy as np

from square_demo.core import observer_eci, square_geometry, zenith_orbit
from square_demo.grid import corner_indices, grid_acceptance, grid_indices, grid_metrics
from square_demo.orbit_j2 import orbital_basis, state_to_elements


def _axes(sight, transverse):
    transverse=transverse-np.dot(transverse,sight)*sight
    exi=transverse/np.linalg.norm(transverse)
    return exi,np.cross(sight,exi)


def centered_initial_candidate(config,size,side_deg):
    """Симметричные лучи, КА 0 фиксирован, затем независимые состояния.

    Один и тот же множитель дальности для всех лучей обеспечивает
    r_c(0)-R(0) || s_c. Положение КА 0 совпадает с опорным состоянием.
    """
    ids=grid_indices(size)
    orbit=config["orbit"]
    chief,basis,_,_=zenith_orbit(np.array([0.]),0.,orbit,config["observer"])
    r0=chief[0]
    v0=basis[0,:,1]*np.sqrt(orbit["mu_km3_s2"]/np.linalg.norm(r0))
    robs=observer_eci(np.array([0.]),0.,orbit["earth_radius_km"],config["observer"])[0]
    s0=(r0-robs)/np.linalg.norm(r0-robs)
    e0,f0=_axes(s0,basis[0,:,1])
    alpha=np.tan(np.deg2rad(side_deg))
    target=np.empty((size*size,2))
    for row in range(size):
        for col in range(size):
            target[ids[row,col]]=alpha*np.array([col/(size-1)-.5,row/(size-1)-.5])

    # Решаем условие: луч на КА 0 имеет в центрированной картинной
    # плоскости координаты (-alpha/2,-alpha/2).
    def frame(offset):
        sight=s0+offset[0]*e0+offset[1]*f0
        sight/=np.linalg.norm(sight)
        exi,eeta=_axes(sight,basis[0,:,1])
        return sight,exi,eeta
    def residual(offset):
        sight,exi,eeta=frame(offset)
        d=r0-robs
        return np.array([np.dot(d,exi),np.dot(d,eeta)])/np.dot(d,sight)-target[0]
    offset=np.array([alpha/2,alpha/2])
    for _ in range(8):
        error=residual(offset)
        if np.max(np.abs(error))<2e-14: break
        perturb=1e-6
        jac=np.column_stack([(residual(offset+np.eye(2)[j]*perturb)-error)/perturb
                             for j in range(2)])
        offset-=np.linalg.solve(jac,error)
    sight,exi,eeta=frame(offset)
    rays=sight+target[:,0,None]*exi+target[:,1,None]*eeta
    lam=np.dot(r0-robs,rays[0])/np.dot(rays[0],rays[0])
    states=np.empty((size*size,2,3))
    states[:,0]=robs+lam*rays
    states[0,0]=r0
    omega=np.cross(r0,v0)/np.dot(r0,r0)
    states[:,1]=v0+np.cross(omega,states[:,0]-r0)
    states[0,1]=v0
    center=np.mean(states[:,0],axis=0)
    if np.linalg.norm(np.cross(center-robs,sight))>1e-8:
        raise ValueError("Начальная геометрия не центрирована по группе")
    return {"N":size,"states":states,"target":target,"grid_indices":ids.tolist(),
            "elements_at_epoch":[state_to_elements(*state,orbit["mu_km3_s2"])
                                 for state in states],
            "parameters":{"side_deg":float(side_deg),"projection_origin":"centroid",
                          "initialization":"equal-range orbital-rotation baseline",
                          "chief_height_km":float(orbit["altitude_km"]),
                          "inclination_deg":float(orbit["inclination_deg"]),
                          "range_multiplier_km":float(lam)}}


def project_centered(states,times,config):
    """Абсолютные положения → луч на r_c и гномонические координаты."""
    positions=states[:,:,0]
    re=config["orbit"]["earth_radius_km"]
    observer=observer_eci(np.asarray(times),0.,re,config["observer"])
    center=np.mean(positions,axis=1)
    sight=center-observer
    sight/=np.linalg.norm(sight,axis=1)[:,None]
    basis=orbital_basis(states[:,0])
    transverse=basis[:,:,1]
    exi=transverse-np.sum(transverse*sight,axis=1)[:,None]*sight
    exi/=np.linalg.norm(exi,axis=1)[:,None]
    eeta=np.cross(sight,exi)
    los=positions-observer[:,None,:]
    denom=np.einsum('tsd,td->ts',los,sight)
    valid=np.isfinite(denom)&(denom>1e-9)
    image=np.stack((np.einsum('tsd,td->ts',los,exi)/denom,
                    np.einsum('tsd,td->ts',los,eeta)/denom),axis=-1)
    image[~valid]=np.nan
    direction=los/np.linalg.norm(los,axis=2)[:,:,None]
    elevation=np.rad2deg(np.arcsin(np.clip(np.einsum('tsd,td->ts',direction,
                                                    observer/re),-1,1)))
    return image,valid,elevation,observer


def centered_metrics(states,times,config,candidate):
    """Те же допуски формы и размера, видимость и разрешение раздельно."""
    size=candidate["N"]
    image,valid,elevation,observer=project_centered(states,times,config)
    corners=corner_indices(size)
    sides,_,_,errors=square_geometry(image[:,corners],valid[:,corners])
    outer=image[:,corners]
    edges=np.roll(outer,-1,axis=1)-outer
    cross=edges[:,:,0]*np.roll(edges[:,:,1],-1,axis=1)-edges[:,:,1]*np.roll(edges[:,:,0],-1,axis=1)
    convex=np.all(cross>0,axis=1)
    grid=grid_metrics(image,valid,states[:,:,0],observer,size)
    shape_limits=dict(config["grid_tolerances"],min_neighbor_angle_deg=0.)
    grid_ok,_=grid_acceptance(grid,size,shape_limits)
    mean_side=np.nanmean(sides,axis=1)
    alpha=np.tan(np.deg2rad(config["angular_side_deg"]))
    size_ok=np.isfinite(mean_side)&(np.abs(mean_side/alpha-1)<=
        config["grid_tolerances"]["max_outer_side_relative_error"])
    vis=config["visibility"]
    form=(np.all(valid,axis=1)&convex&grid_ok&np.all(np.isfinite(errors),axis=1)
          &(errors[:,0]<=vis["max_side_error"])
          &(errors[:,1]<=vis["max_diagonal_error"])
          &(errors[:,2]<=np.deg2rad(vis["max_angle_error_deg"]))&size_ok)
    visible=np.min(elevation,axis=1)>=vis["min_elevation_deg"]
    resolved=grid["min_true_neighbor_angle_deg"]>=config["grid_tolerances"]["min_neighbor_angle_deg"]
    return {"image":image,"valid":valid,"elevation_deg":elevation,
            "shape_errors":errors,"outer_side_deg":np.rad2deg(np.arctan(mean_side)),
            "min_true_neighbor_deg":grid["min_true_neighbor_angle_deg"],
            "form":form,"visible":visible,"square":form&visible,
            "resolved":resolved,"resolved_square":form&visible&resolved,
            "point_error_steps":grid["point_error_steps"],
            "neighbor_step_error":grid["neighbor_step_error"]}
