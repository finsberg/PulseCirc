import logging
import numpy as np

from typing import Protocol
from scipy.integrate import solve_ivp
from . import wk3
from .datacollector import DataCollector

logger = logging.getLogger(__name__)

global p_current, p_old


class Model(Protocol):
    @property
    def volume(self) -> float:
        ...

    @property
    def pressure(self) -> float:
        ...

    @pressure.setter
    def pressure(self, value: float) -> None:
        ...

    @property
    def activation(self) -> float:
        ...

    @activation.setter
    def activation(self, value: float) -> None:
        ...

    def save(self, t: float) -> None:
        ...

    def dVdp(self) -> float:
        ...


def circulation(
    model: Model,
    activation: np.ndarray,
    p_ao: float,
    tau: float,
    R_ao: float,
    p_dia: float,
    R_circ: float,
    C_circ: float,
    collector: DataCollector | None = None,
) -> DataCollector:
    if collector is None:
        collector = DataCollector()

    # Saving the initial pressure and volume
    collector.collect(0.0, 0.0, model.volume, model.pressure, 0, p_ao)

    # Initialization to the atrium pressure of 0.2 kPa
    model.pressure = 0.2
    collector.collect(0.5, 0.0, model.volume, model.pressure, 0, p_ao)

    model.save(0.0)

    for t, target_activation in enumerate(activation, start=1):
        model.activation = target_activation

        circ_iter = 0

        # initial guess for new pressure
        if t == 1:
            p_current = collector.pressures[-1] * 1.01
            circ_p_ao = p_ao
            circ_dp_ao = 0
        else:
            p_current = collector.pressures[-1] + (
                collector.pressures[-1] - collector.pressures[-2]
            )

        p_old = collector.pressures[-1]
        v_old = collector.volumes[-1]
        R = []
        tol = 1e-4 * v_old

        while len(R) == 0 or (np.abs(R[-1]) > tol and circ_iter < 20):
            model.pressure = p_current

            v_current = model.volume
            circ_solution = solve_ivp(
                wk3.WK3,
                [0, tau],
                [circ_p_ao, circ_dp_ao],
                t_eval=[0, tau],
                args=(p_old, p_current, p_ao, p_dia, R_ao, R_circ, C_circ),
            )
            # check the current p_ao vs previous p_ao to open the ao valve
            if circ_solution.y[0][1] > p_ao:
                circ_p_ao_current = circ_solution.y[0][1]
                circ_dp_ao_current = circ_solution.y[1][1]
                Q = (p_current - circ_p_ao_current) / R_ao
            else:
                circ_p_ao_current = circ_p_ao
                circ_dp_ao_current = circ_dp_ao
                Q = 0
            v_fe = v_current
            v_circ = v_old - Q * tau
            R.append(v_fe - v_circ)
            if np.abs(R[-1]) > tol:
                dVFE_dP = model.dVdp()
                dVCirc_dP = wk3.dV_WK3(
                    p_current,
                    tau,
                    R_ao,
                    circ_p_ao_current,
                    circ_dp_ao_current,
                    p_ao=p_ao,
                    R_circ=R_circ,
                    C_circ=C_circ,
                )
                J = dVFE_dP + dVCirc_dP
                p_current = p_current - R[-1] / J
                circ_iter += 1
        p_current = model.pressure
        v_current = model.volume
        if circ_solution.y[0][1] > p_ao:
            circ_p_ao = circ_solution.y[0][1]
            circ_dp_ao = circ_solution.y[1][1]
            p_ao = circ_p_ao

        collector.collect(t, target_activation, v_current, p_current, Q * tau, p_ao)
        model.save(t)

        if p_current < 0.01:
            break

    return collector
