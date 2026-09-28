"""Индексация и критерии наблюдаемой сетки N×N; N=2 совпадает с 0–1–2–3."""
from __future__ import annotations

import numpy as np

from .orbit_j2 import state_to_elements


def grid_indices(size):
    if not isinstance(size,(int,np.integer)) or not 2 <= size <= 10:
        raise ValueError("N должно быть целым от 2 до 10")
    return np.array([[row*size+(col if row%2 == 0 else size-1-col)
                      for col in range(size)] for row in range(size)],dtype=int)


def corner_indices(size):
    ids = grid_indices(size)
    return np.array([ids[0,0],ids[0,-1],ids[-1,-1],ids[-1,0]],dtype=int)


def initial_grid_from_corners(corner_states, size, observer_position, sight, exi, eeta,
                              angular_side_deg, mu):
    """Интерполяция только начальных данных, затем отдельные элементы каждого КА."""
    ids = grid_indices(size)
    alpha = np.tan(np.deg2rad(angular_side_deg))
    sight_ranges = (corner_states[:,0]-observer_position)@sight
    states = np.empty((size*size,2,3))
    targets = np.empty((size*size,2))
    for row in range(size):
        v = row/(size-1)
        for col in range(size):
            u = col/(size-1)
            k = ids[row,col]
            w = np.array([(1-u)*(1-v),u*(1-v),u*v,(1-u)*v])
            xi,eta = alpha*u,alpha*v
            targets[k] = (xi,eta)
            lam = w@sight_ranges
            states[k,0] = observer_position+lam*(sight+xi*exi+eta*eeta)
            states[k,1] = w@corner_states[:,1]
    states[corner_indices(size)] = corner_states
    elements = [state_to_elements(*state,mu) for state in states]
    return states,targets,elements


def grid_metrics(image, valid, positions, observer_positions, size):
    """Показатели всех точек, соседей и ячеек в каждом узле пролёта."""
    ids = grid_indices(size)
    t = len(image)
    points = image[:,ids]
    good = np.all(valid,axis=1)
    result = {key:np.full(t,np.nan) for key in (
        "point_error_steps","neighbor_step_error","max_cell_side_error",
        "max_cell_diagonal_error","max_cell_angle_error_deg",
        "min_true_neighbor_angle_deg")}
    result["ordering_ok"] = np.zeros(t,dtype=bool)
    result["cells_convex"] = np.zeros(t,dtype=bool)
    los = positions-observer_positions[:,None,:]
    directions = los/np.linalg.norm(los,axis=2)[:,:,None]
    for k in np.flatnonzero(good):
        p = points[k]
        u,v = np.meshgrid(np.linspace(0,1,size),np.linspace(0,1,size))
        q = (u+1j*v).ravel()
        z = (p[...,0]+1j*p[...,1]).ravel()
        q0,z0 = q-q.mean(),z-z.mean()
        scale = np.vdot(q0,z0)/np.vdot(q0,q0)
        step = abs(scale)/(size-1)
        if step <= 1e-15:
            continue
        result["point_error_steps"][k] = np.max(abs(z-z.mean()-scale*q0))/step
        horizontal = p[:,1:]-p[:,:-1]
        vertical = p[1:]-p[:-1]
        lengths = np.r_[np.linalg.norm(horizontal,axis=2).ravel(),
                        np.linalg.norm(vertical,axis=2).ravel()]
        mean_length = np.mean(lengths)
        if mean_length <= 1e-15:
            continue
        result["neighbor_step_error"][k] = np.max(abs(lengths-mean_length))/mean_length
        hx = p[0,-1]-p[0,0]
        hy = p[-1,0]-p[0,0]
        result["ordering_ok"][k] = bool(
            np.all(np.einsum("...i,i->...",horizontal,hx)>0)
            and np.all(np.einsum("...i,i->...",vertical,hy)>0))
        a,b,c,d = p[:-1,:-1],p[:-1,1:],p[1:,1:],p[1:,:-1]
        corners = np.stack((a,b,c,d),axis=2)
        edges = np.roll(corners,-1,axis=2)-corners
        sides = np.linalg.norm(edges,axis=3)
        cell_mean = np.mean(sides,axis=2)
        diag1 = np.linalg.norm(c-a,axis=2)
        diag2 = np.linalg.norm(d-b,axis=2)
        incoming = np.roll(corners,1,axis=2)-corners
        cosines = np.sum(incoming*edges,axis=3)/(
            np.linalg.norm(incoming,axis=3)*sides)
        angles = np.arccos(np.clip(cosines,-1,1))
        result["max_cell_side_error"][k] = np.max(abs(sides-cell_mean[:,:,None])/cell_mean[:,:,None])
        result["max_cell_diagonal_error"][k] = np.max(2*abs(diag1-diag2)/(diag1+diag2))
        result["max_cell_angle_error_deg"][k] = np.rad2deg(np.max(abs(angles-np.pi/2)))
        cross = edges[...,0]*np.roll(edges[...,1],-1,axis=2)-edges[...,1]*np.roll(edges[...,0],-1,axis=2)
        result["cells_convex"][k] = bool(np.all(cross>0))
        unit = directions[k,ids]
        neighbor_cosines = np.r_[np.sum(unit[:,:-1]*unit[:,1:],axis=2).ravel(),
                                 np.sum(unit[:-1]*unit[1:],axis=2).ravel()]
        result["min_true_neighbor_angle_deg"][k] = float(np.rad2deg(
            np.min(np.arccos(np.clip(neighbor_cosines,-1,1)))))
    result["indices"] = ids
    return result


def grid_acceptance(metrics, size, tolerances):
    """Форма внутренних ячеек и истинный угловой шаг, отдельно от внешнего квадрата."""
    count = len(metrics["ordering_ok"])
    accepted = np.ones(count,dtype=bool)
    reasons = [[] for _ in range(count)]
    tests = [("min_true_neighbor_angle_deg","min_neighbor_angle_deg",lambda value,limit:value<limit,
              "истинный угловой шаг", "°")]
    if size > 2:
        tests += [
            ("point_error_steps","max_point_error_steps",lambda value,limit:value>limit,
             "отклонение точки", " шага"),
            ("neighbor_step_error","max_neighbor_step_error",lambda value,limit:value>limit,
             "неравномерность соседних шагов", ""),
            ("max_cell_side_error","max_cell_side_error",lambda value,limit:value>limit,
             "ошибка сторон ячейки", ""),
            ("max_cell_diagonal_error","max_cell_diagonal_error",lambda value,limit:value>limit,
             "ошибка диагоналей ячейки", ""),
            ("max_cell_angle_error_deg","max_cell_angle_error_deg",lambda value,limit:value>limit,
             "ошибка угла ячейки", "°"),
        ]
    for field,threshold,failed,title,unit in tests:
        values = metrics[field]
        limit = float(tolerances[threshold])
        bad = ~np.isfinite(values) | failed(values,limit)
        accepted &= ~bad
        for k in np.flatnonzero(bad):
            value = "—" if not np.isfinite(values[k]) else f"{values[k]:.3f}"
            sign = "<" if field == "min_true_neighbor_angle_deg" else ">"
            reasons[k].append(f"{title} {value}{unit} {sign} {limit:g}{unit}")
    if size > 2:
        for field,title in (("ordering_ok","нарушен порядок строк/столбцов"),
                            ("cells_convex","ячейка невыпуклая или перевёрнута")):
            bad = ~metrics[field]
            accepted &= ~bad
            for k in np.flatnonzero(bad):
                reasons[k].append(title)
    return accepted,["; ".join(items) for items in reasons]
