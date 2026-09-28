"""Геометрия, динамика и проверяемый контракт расчёта (км, с, м/с²)."""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from .grid import corner_indices, grid_acceptance, grid_metrics


@dataclass
class TrajectoryResult:
    t_s: np.ndarray
    r_eci_km: np.ndarray
    rho_oso_km: np.ndarray
    observer_eci_km: np.ndarray
    image_gnomonic: np.ndarray
    image_valid: np.ndarray
    elevation_deg: np.ndarray
    accel_oso_m_s2: np.ndarray
    dv_cumulative_m_s: np.ndarray
    shape_errors: np.ndarray
    min_pair_distance_km: np.ndarray
    target_image_gnomonic: np.ndarray
    reference_orbit_eci_km: np.ndarray
    square_sides_gnomonic: np.ndarray
    square_diagonals_gnomonic: np.ndarray
    square_angles_rad: np.ndarray
    target_yoz_km: np.ndarray
    metadata: dict = field(default_factory=dict)

    def validate(self) -> None:
        expected_units = {"time": "s", "position": "km", "acceleration": "m/s²", "delta_v": "m/s"}
        if self.metadata.get("units") != expected_units:
            raise ValueError("metadata.units: несогласованные единицы")
        t = self.t_s
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or not (np.diff(t) > 0).all():
            raise ValueError("t_s: требуется не менее двух строго возрастающих конечных отсчётов")
        n, s = self.r_eci_km.shape[:2]
        if n != len(t) or s < 4:
            raise ValueError("r_eci_km: неверное число отсчётов или КА")
        shapes = {
            "r_eci_km": (n, s, 3), "rho_oso_km": (n, s, 3),
            "observer_eci_km": (n, 3), "image_gnomonic": (n, s, 2),
            "image_valid": (n, s), "elevation_deg": (n, s),
            "accel_oso_m_s2": (n-1, s, 3), "dv_cumulative_m_s": (n, s),
            "shape_errors": (n, 3), "min_pair_distance_km": (n,),
            "target_image_gnomonic": (s, 2),
            "square_sides_gnomonic": (n,4), "square_diagonals_gnomonic": (n,2),
            "square_angles_rad": (n,4),
            "target_yoz_km": (s,2),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name}: ожидалась размерность {shape}")
        if self.reference_orbit_eci_km.ndim != 2 or self.reference_orbit_eci_km.shape[1] != 3:
            raise ValueError("reference_orbit_eci_km: ожидалась размерность (M,3)")
        for name in shapes.keys() - {"image_gnomonic", "image_valid", "shape_errors",
                                     "square_sides_gnomonic", "square_diagonals_gnomonic", "square_angles_rad"}:
            if not np.isfinite(getattr(self, name)).all():
                raise ValueError(f"{name}: встречено NaN или бесконечность")
        if not np.isfinite(self.reference_orbit_eci_km).all():
            raise ValueError("reference_orbit_eci_km: нечисловые значения")
        if not np.allclose(self.rho_oso_km[:, 0], 0, atol=1e-10):
            raise ValueError("Индекс 0 должен быть опорным КА")
        if not np.isfinite(self.image_gnomonic[self.image_valid]).all():
            raise ValueError("В действительной проекции есть NaN")
        if not np.isnan(self.image_gnomonic[~self.image_valid]).all():
            raise ValueError("Недействительная проекция должна содержать NaN")
        if not self.image_valid[:,0].all() or not np.allclose(self.image_gnomonic[:,0],0,atol=1e-12):
            raise ValueError("Проекция опорного КА должна быть (0,0)")
        if not np.isfinite(self.shape_errors[np.all(self.image_valid[:, :4], axis=1)]).all():
            raise ValueError("Некорректные показатели формы")
        for name in ("square_sides_gnomonic", "square_diagonals_gnomonic", "square_angles_rad"):
            if not np.isfinite(getattr(self,name)[np.all(self.image_valid[:,:4],axis=1)]).all():
                raise ValueError(f"{name}: неверная геометрия")
        if np.any(self.dv_cumulative_m_s < -1e-12) or not np.allclose(self.dv_cumulative_m_s[0], 0):
            raise ValueError("Delta v должно начинаться с нуля и быть неотрицательным")
        if np.any(np.diff(self.dv_cumulative_m_s, axis=0) < -1e-10):
            raise ValueError("Delta v должно возрастать")
        if np.max(np.abs(self.dv_cumulative_m_s[1:] - self.dv_cumulative_m_s[:-1]
                         - np.linalg.norm(self.accel_oso_m_s2, axis=2) * np.diff(t)[:, None])) > 1e-7:
            raise ValueError("Delta v не согласовано с приложенным ускорением")


def zenith_orbit(t_s, zenith_s, orbit, observer):
    """Опорная круговая орбита и вращающийся наблюдатель из TeX."""
    t = np.asarray(t_s, dtype=float)
    re = float(orbit["earth_radius_km"])
    a = re + float(orbit["altitude_km"])
    mu = float(orbit["mu_km3_s2"])
    inc = np.deg2rad(float(orbit["inclination_deg"]))
    lat = np.deg2rad(float(observer["latitude_deg"]))
    lon = np.deg2rad(float(observer["longitude_deg"]))
    omega_e = float(observer["earth_rotation_rad_s"])
    theta_0 = np.deg2rad(float(observer["greenwich_angle_at_zenith_deg"]))
    if re <= 0 or a <= re or mu <= 0 or abs(np.sin(lat)) > abs(np.sin(inc)):
        raise ValueError("Невозможная высота, гравитационный параметр или геометрия зенита")
    n = np.sqrt(mu / a**3)
    u0 = np.arcsin(np.sin(lat) / np.sin(inc))  # восходящая ветвь
    node = lon + theta_0 - np.arctan2(np.cos(inc)*np.sin(u0), np.cos(u0))
    cn, sn = np.cos(node), np.sin(node)
    ci, si = np.cos(inc), np.sin(inc)
    ex0 = np.array([cn, sn, 0.0])
    ey0 = np.array([-sn*ci, cn*ci, si])
    ez0 = np.cross(ex0, ey0)
    u = u0 + n*(t-zenith_s)
    ex = np.cos(u)[:, None]*ex0 + np.sin(u)[:, None]*ey0
    ey = -np.sin(u)[:, None]*ex0 + np.cos(u)[:, None]*ey0
    ez = np.broadcast_to(ez0, ex.shape)
    basis = np.stack((ex, ey, ez), axis=-1)
    r0 = a*ex
    robs = observer_eci(t,zenith_s,re,observer)
    return r0, basis, robs, n


def observer_eci(t_s, epoch_s, earth_radius_km, observer):
    """Условная ECI: Гринвичский угол задан в epoch_s, Земля сферическая."""
    t = np.asarray(t_s,dtype=float)
    lat = np.deg2rad(float(observer["latitude_deg"]))
    lon = np.deg2rad(float(observer["longitude_deg"]))
    theta = np.deg2rad(float(observer["greenwich_angle_at_zenith_deg"]))
    lam = lon+theta+float(observer["earth_rotation_rad_s"])*(t-epoch_s)
    return float(earth_radius_km)*np.stack((np.cos(lat)*np.cos(lam),
        np.cos(lat)*np.sin(lam),np.full_like(lam,np.sin(lat))),axis=-1)


def hcw_free(tau_s, rho0_km, vel0_km_s, n):
    """Аналитическое решение HCW для произвольного числа ведомых."""
    tau = np.asarray(tau_s, dtype=float)[:, None]
    x0, y0, z0 = np.asarray(rho0_km, dtype=float).T
    vx0, vy0, vz0 = np.asarray(vel0_km_s, dtype=float).T
    c = vy0 + 2*n*x0
    xc = 2*c/n
    cosine, sine = np.cos(n*tau), np.sin(n*tau)
    x = xc + (x0-xc)*cosine + vx0*sine/n
    y = y0 + (c-2*n*xc)*tau - 2*(x0-xc)*sine + 2*vx0*(cosine-1)/n
    z = z0*cosine + vz0*sine/n
    vx = -(x0-xc)*n*sine + vx0*cosine
    vy = c - 2*n*x
    vz = -z0*n*sine + vz0*cosine
    return np.stack((x,y,z), axis=-1), np.stack((vx,vy,vz), axis=-1)


def hcw_transition(tau_s, n):
    """Матрицы P,V уравнений (position_transition, velocity_transition) из TeX."""
    tau = np.asarray(tau_s, dtype=float)
    c, s = np.cos(n*tau), np.sin(n*tau)
    p = np.zeros((len(tau),3,3))
    v = np.zeros_like(p)
    p[:,0,0] = 4-3*c
    p[:,1,0] = 6*(s-n*tau)
    p[:,1,1] = 1
    p[:,2,2] = c
    v[:,0,0] = s/n
    v[:,0,1] = 2*(1-c)/n
    v[:,1,0] = 2*(c-1)/n
    v[:,1,1] = (4*s-3*n*tau)/n
    v[:,2,2] = s/n
    return p,v


def image_frame(r0_eci_km, observer_eci_km, basis):
    chief = r0_eci_km - observer_eci_km
    distance = np.linalg.norm(chief, axis=1)
    sight = chief / distance[:, None]
    ey = basis[:, :, 1]
    w = ey - np.sum(ey*sight, axis=1)[:, None]*sight
    wn = np.linalg.norm(w, axis=1)
    if np.any(wn < 1e-12):
        raise ValueError("Вырожден базис картинной плоскости")
    exi = w/wn[:, None]
    eeta = np.cross(sight, exi)
    return distance, sight, exi, eeta


def project(r_eci_km, observer_eci_km, basis, earth_radius_km):
    _, sight, exi, eeta = image_frame(r_eci_km[:,0], observer_eci_km, basis)
    los = r_eci_km - observer_eci_km[:, None, :]
    denom = np.sum(los*sight[:, None, :], axis=2)
    valid = np.isfinite(denom) & (denom > 1e-9)
    image = np.full((*denom.shape, 2), np.nan)
    for j, axis in enumerate((exi, eeta)):
        numerator = np.sum(los*axis[:, None, :], axis=2)
        image[:, :, j] = np.divide(numerator, denom, out=np.full_like(denom, np.nan), where=valid)
    image[~valid] = np.nan
    direction = los / np.linalg.norm(los, axis=2)[:, :, None]
    vertical = observer_eci_km / earth_radius_km
    elev = np.rad2deg(np.arcsin(np.clip(np.sum(direction*vertical[:, None, :], axis=2), -1, 1)))
    return image, valid, elev


def square_geometry(image, valid):
    sides_all = np.full((len(image),4),np.nan)
    diagonals_all = np.full((len(image),2),np.nan)
    angles_all = np.full((len(image),4),np.nan)
    result = np.full((len(image), 3), np.nan)
    for k, points in enumerate(image[:, :4]):
        if not valid[k, :4].all():
            continue
        edges = np.roll(points, -1, axis=0)-points
        sides = np.linalg.norm(edges, axis=1)
        d1 = np.linalg.norm(points[2]-points[0])
        d2 = np.linalg.norm(points[3]-points[1])
        if min(sides) <= 1e-15 or d1+d2 <= 1e-15:
            continue
        mean = np.mean(sides)
        incoming = np.roll(points, 1, axis=0)-points
        outgoing = np.roll(points, -1, axis=0)-points
        cosg = np.sum(incoming*outgoing, axis=1)/(np.linalg.norm(incoming,axis=1)*sides)
        angles = np.arccos(np.clip(cosg, -1, 1))
        sides_all[k] = sides
        diagonals_all[k] = (d1,d2)
        angles_all[k] = angles
        result[k] = (np.max(np.abs(sides-mean))/mean, 2*abs(d1-d2)/(d1+d2),
                     np.max(np.abs(angles-np.pi/2)))
    return sides_all,diagonals_all,angles_all,result


def shape_metrics(image, valid):
    return square_geometry(image,valid)[3]


def observability(result):
    """Критерии TeX, отклонение размера и причины отказа во всех узлах."""
    vis = result.metadata["visibility"]
    errors = result.shape_errors
    corners = result.metadata.get("corner_indices",[0,1,2,3])
    min_elevation = np.min(result.elevation_deg,axis=1)
    mean_side = np.mean(result.square_sides_gnomonic,axis=1)
    desired_side = np.linalg.norm(result.target_image_gnomonic[corners[1]]
                                  -result.target_image_gnomonic[corners[0]])
    size_deviation_percent = 100*(mean_side/desired_side-1)
    outer = result.image_gnomonic[:,corners]
    edges = np.roll(outer,-1,axis=1)-outer
    next_edges = np.roll(edges,-1,axis=1)
    crosses = edges[:,:,0]*next_edges[:,:,1]-edges[:,:,1]*next_edges[:,:,0]
    convex = np.all(crosses > 0,axis=1)
    eligible = (np.all(result.image_valid,axis=1)
                & convex
                & np.all(np.isfinite(errors),axis=1)
                & (min_elevation >= vis["min_elevation_deg"])
                & (errors[:,0] <= vis["max_side_error"])
                & (errors[:,1] <= vis["max_diagonal_error"])
                & (errors[:,2] <= np.deg2rad(vis["max_angle_error_deg"])))
    reasons = []
    for k in range(len(result.t_s)):
        problems = []
        if not np.all(result.image_valid[k]):
            problems.append("гномоническая проекция не определена")
        if np.all(result.image_valid[k]) and not convex[k]:
            problems.append("контур не является выпуклым")
        if not np.isfinite(errors[k]).all():
            problems.append("показатели формы не определены")
        if min_elevation[k] < vis["min_elevation_deg"]:
            problems.append(f"угол места {min_elevation[k]:.2f}° ниже {vis['min_elevation_deg']:g}°")
        if np.isfinite(errors[k,0]) and errors[k,0] > vis["max_side_error"]:
            problems.append(f"εb={errors[k,0]:.4f} > {vis['max_side_error']:g}")
        if np.isfinite(errors[k,1]) and errors[k,1] > vis["max_diagonal_error"]:
            problems.append(f"εd={errors[k,1]:.4f} > {vis['max_diagonal_error']:g}")
        angle_deg = np.rad2deg(errors[k,2])
        if np.isfinite(angle_deg) and angle_deg > vis["max_angle_error_deg"]:
            problems.append(f"εγ={angle_deg:.2f}° > {vis['max_angle_error_deg']:g}°")
        reasons.append(problems)
    if "grid_metrics" in result.metadata:
        grid_ok,grid_reasons = grid_acceptance(result.metadata["grid_metrics"],
            result.metadata["grid_size"],result.metadata["grid_tolerances"])
        eligible &= grid_ok
        for k,reason in enumerate(grid_reasons):
            if reason:
                reasons[k].append(reason)
    reasons = ["; ".join(items) if items else "все условия формы и видимости выполнены"
               for items in reasons]
    threshold = result.metadata["illustrative_min_separation_km"]
    return {"eligible":eligible, "reasons":reasons, "min_elevation_deg":min_elevation,
            "mean_side_gnomonic":mean_side, "size_deviation_percent":size_deviation_percent,
            "convex":convex, "safety_illustrative_ok":result.min_pair_distance_km >= threshold}


def visibility_summary(result):
    """Критерии TeX во всех узлах и связный интервал вокруг зенита."""
    eligible = result.metadata["observability"]["eligible"]
    errors = result.shape_errors
    center = int(np.argmin(abs(result.t_s-result.metadata["zenith_s"])))
    if eligible[center]:
        left = right = center
        while left > 0 and eligible[left-1]:
            left -= 1
        while right+1 < len(eligible) and eligible[right+1]:
            right += 1
        duration = float(result.t_s[right]-result.t_s[left])
        interval = (float(result.t_s[left]),float(result.t_s[right]))
    else:
        duration = 0.0
        interval = None
    transitions = np.diff(np.r_[False,eligible,False].astype(int))
    starts = np.flatnonzero(transitions==1)
    ends = np.flatnonzero(transitions==-1)-1
    if len(starts):
        lengths = result.t_s[ends]-result.t_s[starts]
        longest = int(np.argmax(lengths))
        longest_duration = float(lengths[longest])
        longest_interval = (float(result.t_s[starts[longest]]),
                            float(result.t_s[ends[longest]]))
    else:
        longest_duration = 0.0
        longest_interval = None
    finite = np.isfinite(errors).all(axis=1)
    maxima = np.max(errors[finite],axis=0) if finite.any() else np.full(3,np.nan)
    summary = {"zenith_contiguous_duration_s": duration,
            "zenith_interval_s": interval,
            "longest_contiguous_duration_s":longest_duration,
            "longest_interval_s":longest_interval,
            "has_positive_duration_interval":bool(longest_duration>0),
            "accepted_nodes": int(eligible.sum()),
            "max_side_error": float(maxima[0]), "max_diagonal_error": float(maxima[1]),
            "max_angle_error_deg": float(np.rad2deg(maxima[2])),
            "side_range_gnomonic": (float(np.nanmin(result.square_sides_gnomonic)),
                                      float(np.nanmax(result.square_sides_gnomonic))),
            "diagonal_range_gnomonic": (float(np.nanmin(result.square_diagonals_gnomonic)),
                                          float(np.nanmax(result.square_diagonals_gnomonic))),
            "angle_range_deg": (float(np.rad2deg(np.nanmin(result.square_angles_rad))),
                                float(np.rad2deg(np.nanmax(result.square_angles_rad)))),
            "min_elevation_deg": float(np.min(result.elevation_deg)),
            "min_pair_distance_km": float(np.min(result.min_pair_distance_km)),
            "all_image_valid": bool(np.all(result.image_valid))}
    if "grid_metrics" in result.metadata:
        metrics = result.metadata["grid_metrics"]
        summary.update({
            "max_point_error_steps":float(np.nanmax(metrics["point_error_steps"])),
            "max_neighbor_step_error":float(np.nanmax(metrics["neighbor_step_error"])),
            "max_cell_side_error":float(np.nanmax(metrics["max_cell_side_error"])),
            "max_cell_diagonal_error":float(np.nanmax(metrics["max_cell_diagonal_error"])),
            "max_cell_angle_error_deg":float(np.nanmax(metrics["max_cell_angle_error_deg"])),
            "min_true_neighbor_angle_deg":float(np.nanmin(metrics["min_true_neighbor_angle_deg"])),
        })
    return summary


def assemble(t, rho, accel, target, orbit, observer, zenith_s, metadata):
    r0, basis, robs, n = zenith_orbit(t, zenith_s, orbit, observer)
    r = r0[:, None, :] + np.einsum("nij,nsj->nsi", basis, rho)
    orbit_t = zenith_s + np.linspace(0, 2*np.pi/n, 361)
    orbit_line = zenith_orbit(orbit_t, zenith_s, orbit, observer)[0]
    return assemble_absolute(t,r,basis,robs,accel,target,orbit,metadata,orbit_line,
                             reference_period_s=2*np.pi/n)


def assemble_absolute(t, r, basis, robs, accel, target, orbit, metadata,
                      orbit_line, reference_period_s):
    """Единая геометрия GUI для HCW, CSV и абсолютных орбит с J₂."""
    rho = np.einsum("nji,nsj->nsi",basis,r-r[:,0:1])
    rho[:,0] = 0
    earth_radius = float(orbit["earth_radius_km"])
    min_altitude = float(np.min(np.linalg.norm(r,axis=2)-earth_radius))
    if min_altitude <= 0:
        raise ValueError("Траектория пересекает поверхность сферической Земли")
    image, valid, elev = project(r, robs, basis, float(orbit["earth_radius_km"]))
    dt = np.diff(t)
    dv = np.zeros((len(t), rho.shape[1]))
    dv[1:] = np.cumsum(np.linalg.norm(accel, axis=2)*dt[:, None], axis=0)
    minpair = np.full(len(t),np.inf)
    for i in range(rho.shape[1]-1):
        distances = np.linalg.norm(r[:,i+1:]-r[:,i:i+1],axis=2)
        minpair = np.minimum(minpair,np.min(distances,axis=1))
    corners = corner_indices(metadata["grid_size"]) if "grid_size" in metadata else np.arange(4)
    sides,diagonals,angles,errors = square_geometry(image[:,corners],valid[:,corners])
    if "grid_size" in metadata:
        metadata["corner_indices"] = corners.tolist()
        metadata["grid_metrics"] = grid_metrics(image,valid,r,robs,metadata["grid_size"])
    metadata = dict(metadata, orbit_radius_km=float(orbit["earth_radius_km"]),
                    reference_altitude_km=float(orbit["altitude_km"]),
                    reference_period_s=float(reference_period_s),
                    min_altitude_km=min_altitude,
                    min_pair_distance_over_window_km=float(np.min(minpair)),
                    separation_meets_illustrative_threshold=bool(
                        np.min(minpair) >= metadata["illustrative_min_separation_km"]),
                    max_rho_over_a=float(np.max(np.linalg.norm(rho,axis=2)) /
                                         (float(orbit["earth_radius_km"])+float(orbit["altitude_km"]))))
    target_yoz = float(orbit["altitude_km"])*target
    result = TrajectoryResult(t, r, rho, robs, image, valid, elev, accel, dv,
                              errors, minpair, target, orbit_line, sides,diagonals,angles,target_yoz,metadata)
    result.validate()
    result.metadata["observability"] = observability(result)
    observation = result.metadata["observability"]
    observation["outer_side_deg"] = np.rad2deg(np.arctan(
        observation["mean_side_gnomonic"]))
    visible = (np.all(result.image_valid,axis=1)
               & (observation["min_elevation_deg"] >=
                  float(metadata["visibility"]["min_elevation_deg"])))
    changes = np.diff(np.r_[False,visible,False].astype(int))
    starts = np.flatnonzero(changes==1)
    ends = np.flatnonzero(changes==-1)-1
    observation["visibility_intervals_s"] = [
        (float(t[a]),float(t[b])) for a,b in zip(starts,ends)]
    result.metadata["quality"] = visibility_summary(result)
    return result
