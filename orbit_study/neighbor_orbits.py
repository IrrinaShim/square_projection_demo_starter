"""Exact sight-line positions and nearby, co-directed osculating orbits.

The gnomonic rays are fixed at the epoch. Range and velocity are the free
variables; the circular family is a physical starting point, not a forced
trajectory after the epoch. Units are km and km/s.
"""
from __future__ import annotations

import numpy as np


def circular_neighbor_states(chief, observer, rays, mu):
    """Intersect each sight line with the chief's radius and align orbit normals.

    Of the two sphere intersections choose the one nearest the chief range.
    Each velocity is tangential to its own position, with normal equal to the
    projection of the chief's *directed* angular-momentum unit vector onto
    that position's tangent plane. This gives nearby prograde or retrograde
    orbits according to the chief, without guessing from inclination alone.
    """
    chief=np.asarray(chief,dtype=float)
    observer=np.asarray(observer,dtype=float)
    rays=np.asarray(rays,dtype=float)
    radius=np.linalg.norm(chief[0])
    a=np.einsum("ij,ij->i",rays,rays)
    b=2*rays@observer
    c=np.dot(observer,observer)-radius**2
    discriminant=b*b-4*a*c
    if np.any(discriminant<0):
        raise ValueError("Целевой луч не пересекает сферу опорной орбиты")
    roots=np.stack(((-b-np.sqrt(discriminant))/(2*a),
                    (-b+np.sqrt(discriminant))/(2*a)),axis=1)
    chief_range=np.linalg.norm(chief[0]-observer)
    roots[roots<=0]=np.inf
    distances=roots[np.arange(len(rays)),np.argmin(abs(roots-chief_range),axis=1)]
    if not np.all(np.isfinite(distances)):
        raise ValueError("Нет положительной дальности до целевого луча")
    positions=observer+distances[:,None]*rays
    normal=np.cross(*chief)
    normal/=np.linalg.norm(normal)
    rhat=positions/np.linalg.norm(positions,axis=1)[:,None]
    normals=normal[None,:]-np.einsum("ij,j->i",rhat,normal)[:,None]*rhat
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    velocities=np.cross(normals,rhat)*np.sqrt(mu/radius)
    states=np.stack((positions,velocities),axis=1)
    states[0]=chief
    return states


def blend_on_rays(base, proposal, permitted, iterations=24):
    """Keep a proposal inside the configured orbital neighborhood.

    Both inputs must lie on the same fixed image rays. This is a construction
    constraint for all search proposals, separate from final time audits.
    """
    if not permitted(base):
        raise ValueError("Базовая круговая группа не удовлетворяет ограничениям")
    if permitted(proposal):
        return proposal.copy(),1.
    low,high=0.,1.
    for _ in range(iterations):
        mid=(low+high)/2
        trial=base+mid*(proposal-base)
        if permitted(trial):low=mid
        else:high=mid
    # Keep a small numerical margin to every active inequality.
    fraction=max(0.,low-1e-5)
    return base+fraction*(proposal-base),fraction
