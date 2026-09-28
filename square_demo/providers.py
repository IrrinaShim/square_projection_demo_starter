"""Провайдеры свободной модели и воспроизведения исторических CSV."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core import (assemble, assemble_absolute, hcw_free, hcw_transition,
                   image_frame, observer_eci, zenith_orbit)
from .orbit_j2 import (elements_to_state, integrate, orbital_basis,
                       separation_audit)
from .grid import grid_indices, initial_grid_from_corners


ROOT = Path(__file__).resolve().parent.parent
SCENARIOS = {
    "free_hcw_two_body": ROOT / "scenarios/free_zenith.json",
    "independent_j2": ROOT / "scenarios/independent_j2_4sat.json",
    "legacy_replay": ROOT / "scenarios/legacy_replay.json",
}


def load_scenario(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as f:
        scenario = json.load(f)
    if scenario.get("schema_version") != 1 or scenario.get("mode") not in SCENARIOS:
        raise ValueError("Неподдерживаемая версия или режим сценария")
    scenario["_path"] = str(path)
    return scenario


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name}: требуется положительное конечное число")
    return value


def _observer(data):
    lat, lon = float(data["latitude_deg"]), float(data["longitude_deg"])
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("Широта должна быть от −90 до 90°, долгота — от −180 до 180°")
    _positive(data["earth_rotation_rad_s"], "Угловая скорость Земли")
    return data


def _visibility(data):
    for key in ("min_elevation_deg", "max_side_error", "max_diagonal_error", "max_angle_error_deg"):
        if not np.isfinite(float(data[key])):
            raise ValueError(f"Некорректный параметр видимости: {key}")
    return data


def _target(alpha):
    return np.array([[0.,0.], [alpha,0.], [alpha,alpha], [0.,alpha]])


def _metadata(scenario, source, orbit, warning, zenith_s):
    return {
        "mode": scenario["mode"], "scenario_id": scenario["scenario_id"],
        "source": source, "warning": warning,
        "zenith_s": float(zenith_s), "visibility": _visibility(scenario["visibility"]),
        "illustrative_min_separation_km": _positive(scenario["safety"]["illustrative_min_separation_km"], "Порог сближения"),
        "safety_horizon": "показываемый интервал",
        "angle_convention": "theta=atan(xi) или atan(eta), градусы; xi,eta безразмерны",
        "units": {"time": "s", "position": "km", "acceleration": "m/s²", "delta_v": "m/s"},
        "inclination_deg": float(orbit["inclination_deg"]),
        "observer_latitude_deg": float(scenario["observer"]["latitude_deg"]),
        "observer_longitude_deg": float(scenario["observer"]["longitude_deg"]),
    }


def _fit_initial_states(t, t0, target, orbit, observer, regularization, sample_count, half_window_s):
    """Точная начальная проекция и регуляризованное H(Pρ0+Vv0)=b."""
    r0, basis, robs, n = zenith_orbit(t,t0,orbit,observer)
    distance, sight, exi, eeta = image_frame(r0,robs,basis)
    p,v = hcw_transition(t-t0,n)
    center = int(np.argmin(abs(t-t0)))
    if abs(t[center]-t0) > 1e-8:
        raise ValueError("Сетка подбора должна содержать зенит")
    window = np.flatnonzero(abs(t-t0) <= half_window_s+1e-9)
    if len(window) < sample_count:
        raise ValueError("Интервал подбора слишком короток для sample_count")
    sample = np.unique(np.r_[np.linspace(window[0],window[-1],sample_count,dtype=int),center])
    span = max(float(half_window_s),1)
    length = max(float(orbit["altitude_km"])*target[1,0],1)
    scales = np.array([length,length/span,length/span,length/span])
    rho0 = np.zeros((3,3))
    vel0 = np.zeros((3,3))
    diagnostics = []
    for j in range(3):
        xi,eta = target[j+1]
        h = -distance[center]*(basis[center].T@sight[center])
        g = basis[center].T@(sight[center]+xi*exi[center]+eta*eeta[center])
        base = h + distance[center]*g  # нулевой начальный радиальный сдвиг
        h_rows = np.stack((np.einsum("ni,nij->nj",exi-xi*sight,basis),
                           np.einsum("ni,nij->nj",eeta-eta*sight,basis)),axis=1)
        a = np.empty((len(sample)*2,4))
        rhs = np.empty(len(sample)*2)
        for m,k in enumerate(sample):
            hp = h_rows[k]@p[k]
            a[2*m:2*m+2,0] = hp@g
            a[2*m:2*m+2,1:] = h_rows[k]@v[k]
            rhs[2*m:2*m+2] = distance[k]*target[j+1] - hp@base
        # Безразмерные неизвестные: Δλ/L и v/(L/T). Нормировка по числу узлов.
        design = a*scales[None,:]/(length*np.sqrt(len(sample)))
        values = rhs/(length*np.sqrt(len(sample)))
        augmented = np.vstack((design,np.sqrt(regularization)*np.eye(4)))
        y = np.r_[values,np.zeros(4)]
        q,_,rank,singular = np.linalg.lstsq(augmented,y,rcond=None)
        delta_lambda, *velocity = q*scales
        if distance[center]+delta_lambda <= 0:
            raise ValueError("Подбор дал неположительную дальность одного из КА")
        rho0[j] = base + delta_lambda*g
        vel0[j] = velocity
        diagnostics.append({"lambda_km":float(distance[center]+delta_lambda),
                            "radial_offset_km":float(rho0[j,0]),
                            "initial_speed_m_s":float(np.linalg.norm(vel0[j])*1000),
                            "fit_equation_rms_km":float(np.sqrt(np.mean((a@(q*scales)-rhs)**2))),
                            "rank":int(rank), "condition":float(singular[0]/singular[-1])})
    return rho0,vel0,diagnostics


def free_hcw_two_body(scenario):
    if scenario["mode"] != "free_hcw_two_body":
        raise ValueError("Неверный режим для свободного провайдера")
    theta = float(scenario["angular_side_deg"])
    lo, hi = map(float, scenario["angular_side_range_deg"])
    if not np.isfinite(theta) or not 1 <= lo <= theta <= hi <= 20:
        raise ValueError("Угловая сторона должна быть в пределах 1–20°")
    orbit = scenario["orbit"]
    _positive(orbit["earth_radius_km"], "Радиус Земли")
    altitude = _positive(orbit["altitude_km"], "Высота")
    _positive(orbit["mu_km3_s2"], "Гравитационный параметр")
    inc = float(orbit["inclination_deg"])
    if not 0 < inc < 180:
        raise ValueError("Наклонение должно быть между 0 и 180°")
    _observer(scenario["observer"])
    time = scenario["time"]
    t0, start, end, step = map(float, (time["zenith_s"], time["start_s"], time["end_s"], time["step_s"]))
    if not (np.isfinite([t0,start,end,step]).all() and start < t0 < end and step > 0 and (end-start)/step <= 20000):
        raise ValueError("Неверный временной интервал")
    t = np.arange(start, end + step*0.5, step)
    if not np.any(np.isclose(t,t0,atol=1e-8)):
        raise ValueError("Сетка времени должна содержать момент зенита")
    t[np.argmin(abs(t-t0))] = t0
    if t[-1] > end+1e-8:
        t = t[:-1]
    alpha = np.tan(np.deg2rad(theta))
    target = _target(alpha)
    offsets = np.asarray(scenario["initial_family"]["radial_offsets_km"],dtype=float)
    velocities = np.asarray(scenario["initial_family"]["relative_velocities_m_s"],dtype=float)
    if offsets.shape != (3,) or velocities.shape != (3,3) or not np.isfinite(offsets).all() or not np.isfinite(velocities).all():
        raise ValueError("Начальные данные: ожидались три радиальные координаты и три вектора скорости")
    if not np.allclose(offsets,0) or not np.allclose(velocities,0):
        raise ValueError("Этот демонстрационный провайдер поддерживает только x(0)=0 и v_rel(0)=0")
    rho0 = np.column_stack((offsets, altitude*target[1:,0], altitude*target[1:,1]))
    n = np.sqrt(float(orbit["mu_km3_s2"])/(float(orbit["earth_radius_km"])+altitude)**3)
    variant = scenario.get("variant","trial")
    fit_details = None
    if variant == "trial":
        relative, _ = hcw_free(t-t0, rho0, velocities/1000, n)
        initial_rho, initial_vel = rho0, velocities
        source = "Пробное свободное HCW, c=1"
        warning = "Пробное x(0)=0, v_rel(0)=0; линейная модель не валидирована."
    elif variant == "fitted":
        settings = scenario.get("fit",{})
        regularization = float(settings.get("regularization",1.0))
        sample_count = int(settings.get("sample_count",21))
        half_window_s = float(settings.get("half_window_s",150))
        if (not np.isfinite(regularization) or regularization <= 0 or not 5 <= sample_count <= len(t)
                or not np.isfinite(half_window_s) or half_window_s <= 0):
            raise ValueError("Подбор: неверные regularization или sample_count")
        fitted_rho, fitted_vel, fit_details = _fit_initial_states(
            t,t0,target,orbit,scenario["observer"],regularization,sample_count,half_window_s)
        relative, _ = hcw_free(t-t0,fitted_rho,fitted_vel,n)
        initial_rho, initial_vel = fitted_rho, fitted_vel*1000
        source = "Подобранное свободное HCW, регуляризованный МНК"
        warning = "Подбор H(Pρ+Vv)=b на нескольких узлах; начальная проекция точна, тяга не применялась. Линейная модель не валидирована."
    elif variant == "ideal":
        relative = np.broadcast_to(rho0,(len(t),3,3)).copy()
        initial_rho, initial_vel = rho0, np.zeros((3,3))
        source = "Идеально удерживаемый квадрат ОСО (геометрический ориентир)"
        warning = "Идеализированное непрерывное удержание пространственного квадрата в модели HCW; это не свободный полёт и не проект реального управления."
    else:
        raise ValueError(f"Неизвестный вариант свободного сценария: {variant}")
    rho = np.zeros((len(t),4,3))
    rho[:,1:] = relative
    accel = np.zeros((len(t)-1,4,3))
    if variant == "ideal":
        # Для неподвижных в ОСО точек HCW требует a_x=-3n²x, a_z=n²z.
        accel[:,:,0] = -3*n*n*rho[0,:,0][None,:]*1000
        accel[:,:,2] = n*n*rho[0,:,2][None,:]*1000
    metadata = _metadata(scenario,source,orbit,warning,t0)
    metadata["angular_side_deg"] = theta
    metadata["variant"] = variant
    metadata["fit_details"] = fit_details
    metadata["initial_rho_oso_km"] = initial_rho.copy()
    metadata["initial_vel_oso_m_s"] = initial_vel.copy()
    metadata["initial_lambda_km"] = altitude+initial_rho[:,0]
    metadata["initial_epoch_s"] = t0
    metadata["initial_choice"] = {"trial":"пробные: x(0)=0, v_rel(0)=0",
                                  "fitted":"подобраны по проекции (регуляризованный МНК)",
                                  "ideal":"геометрический ориентир: удерживаемый квадрат ОСО"}[variant]
    if variant == "fitted":
        metadata["fit_half_window_s"] = half_window_s
    result = assemble(t,rho,accel,target,orbit,scenario["observer"],t0,metadata)
    return result


def calculate_free_variants(scenario):
    if scenario["mode"] != "free_hcw_two_body":
        raise ValueError("Сравнение доступно только для свободного сценария")
    return {name:free_hcw_two_body({**scenario,"variant":name})
            for name in ("ideal","trial","fitted")}


def audit_independent_grid(scenario, elements):
    """Полный номинальный виток; вызывается и из фонового потока GUI."""
    orbit = scenario["orbit"]
    mu,re,j2 = (float(orbit[key]) for key in
                ("mu_km3_s2","earth_radius_km","j2"))
    settings = scenario["integration"]
    initial = np.stack([elements_to_state(item,mu) for item in elements])
    period = 2*np.pi*np.sqrt(float(elements[0]["a_km"])**3/mu)
    audit_step = float(settings["full_orbit_sample_step_s"])
    full_t = np.linspace(0,period,int(np.ceil(period/audit_step))+1)
    full = integrate(initial,full_t,mu,re,j2,float(settings["max_step_s"]))
    min_altitude = float(np.min(np.linalg.norm(full[:,:,0],axis=2)-re))
    if min_altitude <= 0:
        raise ValueError("Одна из орбит пересекла сферическую Землю на полном витке")
    threshold = float(scenario["safety"]["illustrative_min_separation_km"])
    audit = separation_audit(full,full_t,mu,re,j2,scenario["observer"],
                             scenario["epoch_utc"],threshold,
                             float(settings["minimum_refinement_step_s"]))
    minimum = min(item["distance_km"] for item in audit["pair_minima"].values())
    return {"full_orbit_audit":audit,"full_orbit_min_altitude_km":min_altitude,
            "full_orbit_meets_illustrative_threshold":bool(minimum>=threshold),
            "audit_status":"complete"}


def independent_j2(scenario, include_audit=True):
    """N² собственных оскулирующих орбит и независимое движение с J₂."""
    if scenario["mode"] != "independent_j2":
        raise ValueError("Неверный режим для провайдера J₂")
    from datetime import datetime

    epoch = scenario["epoch_utc"]
    parsed = datetime.fromisoformat(epoch.replace("Z","+00:00"))
    if (parsed.tzinfo is None or parsed.utcoffset().total_seconds() != 0
            or float(scenario["epoch_s"]) != 0):
        raise ValueError("Нужна общая UTC-эпоха с t=0 с")
    orbit = dict(scenario["orbit"])
    re = _positive(orbit["earth_radius_km"],"Радиус Земли")
    mu = _positive(orbit["mu_km3_s2"],"Гравитационный параметр")
    j2 = float(orbit["j2"])
    if not np.isfinite(j2) or not 0 < j2 < 0.01:
        raise ValueError("Некорректный J₂")
    _observer(scenario["observer"])
    theta = float(scenario["angular_side_deg"])
    if not np.isfinite(theta) or not 1 <= theta <= 20:
        raise ValueError("Угловая сторона должна быть в пределах 1–20°")
    size = scenario.get("grid_size",2)
    grid_indices(size)
    for key in ("min_neighbor_angle_deg","max_point_error_steps",
                "max_neighbor_step_error","max_cell_side_error",
                "max_cell_diagonal_error","max_cell_angle_error_deg"):
        _positive(scenario["grid_tolerances"][key],key)
    entries = scenario["elements_at_epoch"]
    if len(entries) != 4:
        raise ValueError("Нужны элементы четырёх угловых КА в порядке 0–1–2–3")
    for k,entry in enumerate(entries):
        if float(entry["a_km"])*(1-float(entry["e"])) <= re:
            raise ValueError(f"Перигей КА {k} внутри сферической Земли")
    corner_initial = np.stack([elements_to_state(entry,mu) for entry in entries])
    basis_at_epoch = orbital_basis(corner_initial[0:1])
    observer_at_epoch = observer_eci(np.array([0.]),0,re,scenario["observer"])
    _,sight,exi,eeta = image_frame(corner_initial[0:1,0],observer_at_epoch,basis_at_epoch)
    initial,target,all_elements = initial_grid_from_corners(
        corner_initial,size,observer_at_epoch[0],sight[0],exi[0],eeta[0],theta,mu)
    if size == 2:
        all_elements = list(entries)
    for k,entry in enumerate(all_elements):
        if float(entry["a_km"])*(1-float(entry["e"])) <= re:
            raise ValueError(f"Перигей КА {k} внутри сферической Земли")
    chief_h = np.cross(initial[0,0],initial[0,1])
    orbit["altitude_km"] = float(np.linalg.norm(initial[0,0])-re)
    orbit["inclination_deg"] = float(np.rad2deg(np.arccos(
        np.clip(chief_h[2]/np.linalg.norm(chief_h),-1,1))))
    if (abs(orbit["altitude_km"]-float(scenario["orbit"]["altitude_km"])) > 1e-6
            or abs(orbit["inclination_deg"]-float(scenario["orbit"]["inclination_deg"])) > 1e-6):
        raise ValueError("Высота и наклонение в orbit должны совпадать с элементами КА 0")
    settings = scenario["integration"]
    max_step = _positive(settings["max_step_s"],"Шаг интегрирования")
    audit_step = _positive(settings["full_orbit_sample_step_s"],"Шаг аудита")
    refine_step = _positive(settings["minimum_refinement_step_s"],"Шаг уточнения")
    if max_step > 10 or audit_step > 30 or refine_step > 1:
        raise ValueError("Шаги интегрирования/аудита слишком крупны")
    time = scenario["time"]
    start,end,step = map(float,(time["start_s"],time["end_s"],time["step_s"]))
    if not (np.isfinite([start,end,step]).all() and start < 0 < end and step > 0
            and (end-start)/step <= 20000):
        raise ValueError("Неверное окно показываемого пролёта")
    t = np.arange(start,end+step*0.5,step)
    t = t[t<=end+1e-8]
    if not np.any(np.isclose(t,0,atol=1e-8)):
        raise ValueError("Временная сетка должна содержать общую эпоху t=0")
    t[np.argmin(abs(t))] = 0
    display = integrate(initial,t,mu,re,j2,max_step)
    period = 2*np.pi*np.sqrt(float(entries[0]["a_km"])**3/mu)
    line_t = np.linspace(0,period,361)
    orbit_line = integrate(initial[0:1],line_t,mu,re,j2,max_step)[:,0,0]
    threshold = _positive(scenario["safety"]["illustrative_min_separation_km"],"Порог сближения")
    basis = orbital_basis(display[:,0])
    robs = observer_eci(t,0,re,scenario["observer"])
    accel = np.zeros((len(t)-1,size*size,3))  # гравитационное J₂ не является тягой
    warning = (f"{size*size} независимых абсолютных орбит с J₂ без тяги. "
               "Порог расстояния иллюстративный; долгосрочная безопасность не установлена.")
    metadata = _metadata(scenario,"Нелинейная абсолютная модель J₂",orbit,warning,0)
    metadata.update({
        "angular_side_deg":theta,"variant":"independent_j2",
        "epoch_utc":epoch,"orbital_elements":all_elements,
        "grid_size":size,"grid_indices":grid_indices(size).tolist(),
        "grid_tolerances":scenario["grid_tolerances"],
        "initial_epoch_s":0.0,"initial_choice":"собственные оскулирующие элементы на общей UTC-эпохе",
        "safety_horizon":"один номинальный виток опорного КА от общей эпохи",
        "audit_status":"pending",
    })
    b0 = orbital_basis(initial[0:1])
    relative = initial[:,0]-initial[0,0]
    relative_oso = (b0[0].T@relative.T).T
    h = np.cross(initial[0,0],initial[0,1])
    angular_rate = h/np.dot(initial[0,0],initial[0,0])
    relative_velocity_eci = initial[:,1]-initial[0,1]-np.cross(angular_rate,relative)
    velocity_oso = (b0[0].T@relative_velocity_eci.T).T
    obs0 = observer_eci(np.array([0.]),0,re,scenario["observer"])[0]
    metadata["initial_rho_oso_km"] = relative_oso[1:]
    metadata["initial_vel_oso_m_s"] = velocity_oso[1:]*1000
    metadata["initial_lambda_km"] = np.linalg.norm(initial[1:,0]-obs0,axis=1)
    result = assemble_absolute(t,display[:,:,0],basis,robs,accel,target,orbit,
                               metadata,orbit_line,period)
    if include_audit:
        result.metadata.update(audit_independent_grid(scenario,all_elements))
    return result


def legacy_replay(scenario):
    if scenario["mode"] != "legacy_replay":
        raise ValueError("Неверный режим для исторического провайдера")
    _observer(scenario["observer"])
    data_dir = (Path(scenario["_path"]).parent / scenario["data_directory"]).resolve()
    with (data_dir / "config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    orbit_old = config["orbit"]
    orbit = {"earth_radius_km": orbit_old["Re_km"], "mu_km3_s2": orbit_old["mu_km3_s2"],
             "altitude_km": orbit_old["h_km"], "inclination_deg": orbit_old["i_deg"]}
    states = np.genfromtxt(data_dir / "trajectories.csv", delimiter=",", names=True)
    controls = np.genfromtxt(data_dir / "controls.csv", delimiter=",", names=True)
    if len(states) != 241 or len(controls) != 240:
        raise ValueError("Исторические CSV должны содержать 241 узел и 240 интервалов")
    t, ct = np.asarray(states["t"]), np.asarray(controls["t"])
    if not np.all(np.diff(t)>0) or not np.allclose(ct,t[:-1]) or not np.allclose(t,config["time"]["time_grid"]):
        raise ValueError("Время управления или состояний не соответствует интервалам")
    # Старые [0,1,2,3] -> новые [0,1,3,2], одновременно для состояний и тяги.
    rho = np.zeros((len(t),4,3))
    accel = np.zeros((len(t)-1,4,3))
    initial_vel = np.zeros((3,3))
    for new, old in ((1,1),(2,3),(3,2)):
        rho[:,new] = np.column_stack([states[f"{axis}{old}"] for axis in "xyz"])/1000
        initial_vel[new-1] = [states[f"v{axis}{old}"][0] for axis in "xyz"]
        f = controls[f"f{old}"]
        psi, theta = states[f"psi{old}"][:-1], states[f"theta{old}"][:-1]
        accel[:,new] = np.column_stack((f*np.cos(theta)*np.cos(psi),
                                        f*np.cos(theta)*np.sin(psi), f*np.sin(theta)))
    zenith = float(scenario["legacy_zenith_time_s"])
    if not t[0] <= zenith <= t[-1]:
        raise ValueError("Назначенный зенит вне данных")
    alpha = float(config["formation"]["d_m"])/1000/float(orbit["altitude_km"])
    target = _target(alpha)
    warning = (f"Зенит t={zenith:g} с назначен для реконструкции изображения; "
               "это не опубликованный интервал наблюдения. Старые координаты заданы в метрах, управление — в м/с².")
    metadata = _metadata(scenario, "Согласованные исторические CSV (241/240)", orbit, warning, zenith)
    metadata["legacy_spatial_side_km"] = float(config["formation"]["d_m"])/1000
    metadata["angular_side_deg"] = float(np.rad2deg(np.arctan(alpha)))
    metadata["initial_rho_oso_km"] = rho[0,1:].copy()
    metadata["initial_vel_oso_m_s"] = initial_vel
    metadata["initial_lambda_km"] = None
    metadata["initial_epoch_s"] = float(t[0])
    metadata["initial_choice"] = "исторические состояния CSV; дальности не подбирались"
    return assemble(t,rho,accel,target,orbit,scenario["observer"],zenith,metadata)


PROVIDERS = {"free_hcw_two_body": free_hcw_two_body,
             "independent_j2": independent_j2, "legacy_replay": legacy_replay}


def calculate(scenario, include_audit=True):
    if scenario["mode"] == "independent_j2":
        return independent_j2(scenario,include_audit=include_audit)
    return PROVIDERS[scenario["mode"]](scenario)
