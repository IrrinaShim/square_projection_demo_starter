from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np


@dataclass
class OrbitConfig:
    """
    Параметры опорной орбиты и коэффициенты модели Швайгарта--Седвика.
    Длины для орбитальных констант задаются в км, как в исходных формулах.
    Относительные координаты в задаче управления задаются в м.
    """

    Re_km: float = 6371.0
    mu_km3_s2: float = 398600.4
    J2: float = 1.08263e-3

    h_km: float = 630.0
    i_deg: float = 97.8

    def __post_init__(self):
        self.r0_km = self.Re_km + self.h_km
        self.i_rad = np.deg2rad(self.i_deg)

        self.n = np.sqrt(self.mu_km3_s2 / self.r0_km**3)

        self.s = (
            3.0
            * self.J2
            * self.Re_km**2
            / (8.0 * self.r0_km**2)
            * (1.0 + 3.0 * np.cos(2.0 * self.i_rad))
        )

        self.c = np.sqrt(1.0 + self.s)
        self.q = self.n * np.sqrt(3.0 * self.c**2 - 2.0)

        self.T_orb = 2.0 * np.pi / self.n


@dataclass
class FormationConfig:
    """
    Геометрия требуемой квадратной конфигурации.
    Опорный аппарат расположен в точке rho_0 = (0, 0, 0).
    Три ведомых аппарата должны находиться в вершинах:
        rho_1 = (0, d, 0),
        rho_2 = (0, 0, d),
        rho_3 = (0, d, d).
    """

    d_m: float = 20_000.0
    rel_error_fraction: float = 0.10
    l_free: float = 0.0

    def __post_init__(self):
        self.rho_tol_m = self.rel_error_fraction * self.d_m

        self.targets_m = np.array(
            [
                [0.0, self.d_m, 0.0],
                [0.0, 0.0, self.d_m],
                [0.0, self.d_m, self.d_m],
            ],
            dtype=float,
        )


@dataclass
class ControlConfig:
    """
    Ограничения на управляющие воздействия.
    f_i -- модуль управляющего ускорения.
    psi_i, theta_i -- углы ориентации тяги.
    omega_psi_i, omega_theta_i -- скорости изменения углов.
    """

    mass_kg: float = 22.0
    F_max_N: float = 1.0

    propulsion_mass_kg: float = 7.9
    propulsion_volume_U: float = 8.0
    total_impulse_Ns: float = 3500.0
    specific_impulse_s: float = 248.0
    g0_m_s2: float = 9.80665
    thruster_count: int = 4

    omega_psi_max_rad_s: float = np.pi / 180.0
    omega_theta_max_rad_s: float = np.pi / 180.0

    psi_min: float = -np.pi
    psi_max: float = np.pi

    theta_min: float = -np.pi / 2.0
    theta_max: float = np.pi / 2.0

    def __post_init__(self):
        self.f_max_m_s2 = self.F_max_N / self.mass_kg


@dataclass
class TimeConfig:
    """
    Временной интервал моделирования.
    Первые 300 секунд считаются участком подготовки к наблюдению.
    Интервал [t_obs, T] используется для оценки удержания квадрата.
    """

    T_s: float = 1200.0
    t_obs_s: float = 300.0

    N: int = 240

    def __post_init__(self):
        self.dt_s = self.T_s / self.N
        self.time_grid = np.linspace(0.0, self.T_s, self.N + 1)


@dataclass
class CostConfig:
    """
    Параметры функционала качества.
    Основной штраф задаётся по ошибкам положения и скорости.
    На подготовительном участке можно использовать малый вес.
    """

    w_prep: float = 0.0
    w_obs: float = 1.0

    eps_thrust: float = 1.0e-4
    eps_angle_rate: float = 1.0e-5

    def compute_velocity_tolerance(self, orbit: OrbitConfig, formation: FormationConfig):
        self.v_tol_m_s = orbit.q * formation.rho_tol_m


@dataclass
class SolverConfig:
    """
    Настройки NLP-решателя.
    """

    ipopt_max_iter: int = 3000
    ipopt_tol: float = 1.0e-6
    ipopt_acceptable_tol: float = 1.0e-5
    print_level: int = 5


@dataclass
class ProjectConfig:
    """
    Общая конфигурация проекта.
    """

    results_dir: Path = Path("results_16u_cubedrive")

    def __post_init__(self):
        self.figures_dir = self.results_dir / "figures"


def build_initial_state(
    orbit: OrbitConfig,
    formation: FormationConfig,
    time: TimeConfig,
):
    """
    Формирует начальный вектор состояния для трёх ведомых аппаратов.

    Для одного аппарата используем порядок:
        X_i = [x_i, vx_i, y_i, vy_i, z_i, vz_i, psi_i, theta_i].

    Полный вектор состояния:
        X = [X_1, X_2, X_3].
    """

    d = formation.d_m
    q = orbit.q
    t_obs = time.t_obs_s

    z0 = d * np.cos(q * t_obs)
    vz0 = q * d * np.sin(q * t_obs)

    psi0 = 0.0
    theta0 = np.pi / 2.0

    X1_0 = np.array(
        [
            0.0,     # x_1
            0.0,     # vx_1
            d,       # y_1
            0.0,     # vy_1
            0.0,     # z_1
            0.0,     # vz_1
            psi0,    # psi_1
            theta0,  # theta_1
        ],
        dtype=float,
    )

    X2_0 = np.array(
        [
            0.0,     # x_2
            0.0,     # vx_2
            0.0,     # y_2
            0.0,     # vy_2
            z0,      # z_2
            vz0,     # vz_2
            psi0,    # psi_2
            theta0,  # theta_2
        ],
        dtype=float,
    )

    X3_0 = np.array(
        [
            0.0,     # x_3
            0.0,     # vx_3
            d,       # y_3
            0.0,     # vy_3
            z0,      # z_3
            vz0,     # vz_3
            psi0,    # psi_3
            theta0,  # theta_3
        ],
        dtype=float,
    )

    X0 = np.concatenate([X1_0, X2_0, X3_0])

    return X0, z0, vz0


def make_result_dirs(project: ProjectConfig):
    project.results_dir.mkdir(parents=True, exist_ok=True)
    project.figures_dir.mkdir(parents=True, exist_ok=True)


def to_jsonable(obj):
    """
    Переводит dataclass, numpy-массивы и Path в формат,
    который можно сохранить в JSON.
    """

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)

    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)

    if hasattr(obj, "__dict__"):
        return {
            key: to_jsonable(value)
            for key, value in obj.__dict__.items()
        }

    return obj


def build_config():
    orbit = OrbitConfig()
    formation = FormationConfig()
    control = ControlConfig()
    time = TimeConfig()
    cost = CostConfig()
    solver = SolverConfig()
    project = ProjectConfig()

    cost.compute_velocity_tolerance(orbit, formation)

    X0, z0, vz0 = build_initial_state(orbit, formation, time)

    config = {
        "orbit": to_jsonable(orbit),
        "formation": to_jsonable(formation),
        "control": to_jsonable(control),
        "time": to_jsonable(time),
        "cost": to_jsonable(cost),
        "solver": to_jsonable(solver),
        "project": to_jsonable(project),
        "initial_state": {
            "X0": X0.tolist(),
            "z0_m": float(z0),
            "vz0_m_s": float(vz0),
            "state_order_one_satellite": [
                "x",
                "vx",
                "y",
                "vy",
                "z",
                "vz",
                "psi",
                "theta",
            ],
        },
    }

    return config


def save_config(config, filename):
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    config = build_config()

    project = ProjectConfig()
    make_result_dirs(project)

    save_path = project.results_dir / "config.json"
    save_config(config, save_path)

    print("Конфигурация задачи сохранена:")
    print(save_path)

    print()
    print("Основные параметры:")
    print(f"n  = {config['orbit']['n']:.8e} рад/с")
    print(f"s  = {config['orbit']['s']:.8e}")
    print(f"c  = {config['orbit']['c']:.8e}")
    print(f"q  = {config['orbit']['q']:.8e} рад/с")

    print()
    print("Начальное состояние:")
    print(f"z0  = {config['initial_state']['z0_m']:.6f} м")
    print(f"vz0 = {config['initial_state']['vz0_m_s']:.6f} м/с")

    print()
    print("Ограничения управления:")
    print(f"f_max     = {config['control']['f_max_m_s2']:.8e} м/с^2")
    print(f"omega_max = {config['control']['omega_psi_max_rad_s']:.8e} рад/с")