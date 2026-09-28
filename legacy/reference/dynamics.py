import numpy as np
import casadi as ca


N_SAT = 3
NX_ONE = 8
NU_ONE = 3

NX = N_SAT * NX_ONE
NU = N_SAT * NU_ONE


def extract_dynamics_params(config):
    """
    Извлекает из общего словаря config параметры, необходимые для правой части системы.
    """

    orbit = config["orbit"]
    formation = config["formation"]
    time = config["time"]

    params = {
        "n": float(orbit["n"]),
        "c": float(orbit["c"]),
        "q": float(orbit["q"]),
        "dt": float(time["dt_s"]),
        "l_values": [float(formation["l_free"])] * N_SAT,
        "chi_values": [0.0] * N_SAT,
    }

    return params


def single_sat_dynamics_ca(xi, ui, t, n, c, q, l_i=0.0, chi_i=0.0):
    """
    Правая часть системы Швайгарта--Седвика для одного ведомого аппарата.

    Порядок состояния:
        xi = [x, vx, y, vy, z, vz, psi, theta]

    Порядок управления:
        ui = [f, omega_psi, omega_theta]

    Здесь ui -- вектор управляющих воздействий двигателя.
    Компоненты управляющего ускорения a_x, a_y, a_z вычисляются через
    f, psi, theta.
    """

    x = xi[0]
    vx = xi[1]
    y = xi[2]
    vy = xi[3]
    z = xi[4]
    vz = xi[5]
    psi = xi[6]
    theta = xi[7]

    f = ui[0]
    omega_psi = ui[1]
    omega_theta = ui[2]

    ax = f * ca.cos(theta) * ca.cos(psi)
    ay = f * ca.cos(theta) * ca.sin(psi)
    az = f * ca.sin(theta)

    x_dot = vx
    vx_dot = 2.0 * n * c * vy + (5.0 * c**2 - 2.0) * n**2 * x + ax

    y_dot = vy
    vy_dot = -2.0 * n * c * vx + ay

    z_dot = vz
    vz_dot = -q**2 * z + 2.0 * l_i * q * ca.cos(q * t + chi_i) + az

    psi_dot = omega_psi
    theta_dot = omega_theta

    return ca.vertcat(
        x_dot,
        vx_dot,
        y_dot,
        vy_dot,
        z_dot,
        vz_dot,
        psi_dot,
        theta_dot,
    )


def group_dynamics_ca(X, U, t, params):
    """
    Правая часть системы для трёх ведомых аппаратов.

    Полный вектор состояния:
        X = [X1, X2, X3]

    Полный вектор управления:
        U = [U1, U2, U3]
    """

    n = params["n"]
    c = params["c"]
    q = params["q"]

    l_values = params["l_values"]
    chi_values = params["chi_values"]

    rhs_list = []

    for i in range(N_SAT):
        x_start = i * NX_ONE
        u_start = i * NU_ONE

        xi = X[x_start:x_start + NX_ONE]
        ui = U[u_start:u_start + NU_ONE]

        rhs_i = single_sat_dynamics_ca(
            xi=xi,
            ui=ui,
            t=t,
            n=n,
            c=c,
            q=q,
            l_i=l_values[i],
            chi_i=chi_values[i],
        )

        rhs_list.append(rhs_i)

    return ca.vertcat(*rhs_list)


def rk4_step_ca(X, U, t, dt, params):
    """
    Один шаг метода Рунге--Кутты 4-го порядка для полной системы.
    Управление U считается постоянным на интервале [t, t + dt].
    """

    k1 = group_dynamics_ca(X, U, t, params)
    k2 = group_dynamics_ca(X + 0.5 * dt * k1, U, t + 0.5 * dt, params)
    k3 = group_dynamics_ca(X + 0.5 * dt * k2, U, t + 0.5 * dt, params)
    k4 = group_dynamics_ca(X + dt * k3, U, t + dt, params)

    X_next = X + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    return X_next


def make_casadi_functions(config):
    """
    Создаёт CasADi-функции правой части и одного шага интегрирования.

    rhs_fun(X, U, t) возвращает X_dot.
    rk4_fun(X, U, t) возвращает X_next.
    """

    params = extract_dynamics_params(config)
    dt = params["dt"]

    X = ca.MX.sym("X", NX)
    U = ca.MX.sym("U", NU)
    t = ca.MX.sym("t")

    X_dot = group_dynamics_ca(X, U, t, params)
    X_next = rk4_step_ca(X, U, t, dt, params)

    rhs_fun = ca.Function(
        "rhs_fun",
        [X, U, t],
        [X_dot],
        ["X", "U", "t"],
        ["X_dot"],
    )

    rk4_fun = ca.Function(
        "rk4_fun",
        [X, U, t],
        [X_next],
        ["X", "U", "t"],
        ["X_next"],
    )

    return rhs_fun, rk4_fun


def single_sat_dynamics_np(xi, ui, t, n, c, q, l_i=0.0, chi_i=0.0):
    """
    NumPy-версия правой части для одного аппарата.
    Используется только для проверок и постобработки.
    """

    x = xi[0]
    vx = xi[1]
    y = xi[2]
    vy = xi[3]
    z = xi[4]
    vz = xi[5]
    psi = xi[6]
    theta = xi[7]

    f = ui[0]
    omega_psi = ui[1]
    omega_theta = ui[2]

    ax = f * np.cos(theta) * np.cos(psi)
    ay = f * np.cos(theta) * np.sin(psi)
    az = f * np.sin(theta)

    return np.array(
        [
            vx,
            2.0 * n * c * vy + (5.0 * c**2 - 2.0) * n**2 * x + ax,
            vy,
            -2.0 * n * c * vx + ay,
            vz,
            -q**2 * z + 2.0 * l_i * q * np.cos(q * t + chi_i) + az,
            omega_psi,
            omega_theta,
        ],
        dtype=float,
    )


def group_dynamics_np(X, U, t, params):
    """
    NumPy-версия правой части для всей группы.
    """

    n = params["n"]
    c = params["c"]
    q = params["q"]

    l_values = params["l_values"]
    chi_values = params["chi_values"]

    rhs = np.zeros(NX)

    for i in range(N_SAT):
        x_start = i * NX_ONE
        u_start = i * NU_ONE

        xi = X[x_start:x_start + NX_ONE]
        ui = U[u_start:u_start + NU_ONE]

        rhs[x_start:x_start + NX_ONE] = single_sat_dynamics_np(
            xi=xi,
            ui=ui,
            t=t,
            n=n,
            c=c,
            q=q,
            l_i=l_values[i],
            chi_i=chi_values[i],
        )

    return rhs


def rk4_step_np(X, U, t, dt, params):
    """
    NumPy-версия одного шага RK4.
    """

    k1 = group_dynamics_np(X, U, t, params)
    k2 = group_dynamics_np(X + 0.5 * dt * k1, U, t + 0.5 * dt, params)
    k3 = group_dynamics_np(X + 0.5 * dt * k2, U, t + 0.5 * dt, params)
    k4 = group_dynamics_np(X + dt * k3, U, t + dt, params)

    return X + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def propagate_open_loop_np(X0, U_grid, time_grid, params):
    """
    Интегрирует систему при заданной сетке управления.
    U_grid имеет размерность (N, NU).
    time_grid имеет размерность (N + 1,).
    """

    N = len(time_grid) - 1
    X_grid = np.zeros((N + 1, NX))
    X_grid[0, :] = X0

    for k in range(N):
        dt = time_grid[k + 1] - time_grid[k]
        X_grid[k + 1, :] = rk4_step_np(
            X=X_grid[k, :],
            U=U_grid[k, :],
            t=time_grid[k],
            dt=dt,
            params=params,
        )

    return X_grid


def check_initial_free_motion(config):
    """
    Проверяет выбранное начальное состояние при нулевом управлении.
    Для аппаратов 2 и 3 должно получиться:
        z(t_obs) примерно равно d,
        vz(t_obs) примерно равно 0.
    """

    params = extract_dynamics_params(config)

    X0 = np.array(config["initial_state"]["X0"], dtype=float)
    time_grid = np.array(config["time"]["time_grid"], dtype=float)

    N = len(time_grid) - 1
    U_grid = np.zeros((N, NU))

    X_grid = propagate_open_loop_np(
        X0=X0,
        U_grid=U_grid,
        time_grid=time_grid,
        params=params,
    )

    t_obs = config["time"]["t_obs_s"]
    k_obs = int(np.argmin(np.abs(time_grid - t_obs)))

    d = config["formation"]["d_m"]

    z2 = X_grid[k_obs, 1 * NX_ONE + 4]
    vz2 = X_grid[k_obs, 1 * NX_ONE + 5]

    z3 = X_grid[k_obs, 2 * NX_ONE - 4]
    vz3 = X_grid[k_obs, 2 * NX_ONE - 3]

    print("Проверка свободного поперечного движения:")
    print(f"t_obs = {time_grid[k_obs]:.3f} с")
    print()
    print("Аппарат 2:")
    print(f"z2(t_obs)  = {z2:.6f} м")
    print(f"vz2(t_obs) = {vz2:.6e} м/с")
    print(f"ошибка z2  = {z2 - d:.6e} м")
    print()
    print("Аппарат 3:")
    print(f"z3(t_obs)  = {z3:.6f} м")
    print(f"vz3(t_obs) = {vz3:.6e} м/с")
    print(f"ошибка z3  = {z3 - d:.6e} м")


if __name__ == "__main__":
    from config import build_config

    config = build_config()

    rhs_fun, rk4_fun = make_casadi_functions(config)

    print("CasADi-функции успешно созданы:")
    print(rhs_fun)
    print(rk4_fun)

    print()
    check_initial_free_motion(config)