from scipy.integrate import solve_ivp


def WK3(t, y, p_old, p_current, p_ao, p_dia, R_ao, R_circ, C_circ):
    # Defining WK3 function based on scipy.integrate.solve_ivp
    # The main equations are, with p_{ao} and its derivatives are unkowns:
    # 1. Q = \frac{p_{lv} - p_{ao}}{R_{ao}}
    # 2. Q_R = \frac{p_{ao}}{R_{circ}}
    # 3. Q_C = C_{circ} \cdot \frac{dp_{ao}}{dt}
    # 4. Q = Q_R + Q_C
    # 5. \frac{dp_{ao}}{dt} = y[1]
    # 6. \frac{d^2p_{ao}}{dt^2} = \frac{Q - Q_R - Q_C}{C_{circ}}
    p_ao = y[0]
    dp_ao_dt = y[1]

    # Calculating flows
    p_lv_interpolated = p_old + (p_current - p_old) * t
    Q = (p_lv_interpolated - p_ao) / R_ao
    Q_R = (p_ao - p_dia) / R_circ
    Q_C = C_circ * dp_ao_dt

    # Conservation of flow
    dQ_C_dt = (Q - Q_R - Q_C) / C_circ
    d2p_ao_dt2 = dQ_C_dt

    return [dp_ao_dt, d2p_ao_dt2]


def dV_WK3(p_current, tau, R_ao, circ_p_ao, circ_dp_ao, p_ao, R_circ, C_circ):
    p_current_backup = p_current
    circ_solution = solve_ivp(
        WK3,
        [0, tau],
        [circ_p_ao, circ_dp_ao],
        t_eval=[0, tau],
        args=(p_current, p_current, p_ao, p_ao, R_ao, R_circ, C_circ),
    )
    if p_current > p_ao:
        circ_p_ao_1 = circ_solution.y[0][1]
        Q1 = (p_current - circ_p_ao_1) / R_ao
    else:
        Q1 = 0
    p_current = p_current * 1.01
    circ_solution = solve_ivp(
        WK3,
        [0, tau],
        [circ_p_ao, circ_dp_ao],
        t_eval=[0, tau],
        args=(p_current, p_current, p_ao, p_ao, R_ao, R_circ, C_circ),
    )
    if p_current > p_ao:
        circ_p_ao_2 = circ_solution.y[0][1]
        Q2 = (p_current - circ_p_ao_2) / R_ao
    else:
        Q2 = 0
    p_current = p_current_backup
    return (Q2 - Q1) / (p_current * 0.01) * tau
