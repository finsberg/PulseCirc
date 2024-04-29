import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from circ.solver import circulation_solver
from circ.circulation_model import CirculationModel
from circ.datacollector import DataCollector
from heart_model_pulse import HeartModelPulse
import activation_model


logging.getLogger("pulse").setLevel(logging.WARNING)

t_res = 1000
t_span = (0.0, 1.0)
t_eval = np.linspace(*t_span, t_res)
normal_activation_params = activation_model.default_parameters()
normal_activation_params["t_sys"] = 0.03

normal_activation = (
    activation_model.activation_function(
        t_span=t_span,
        t_eval=t_eval,
        parameters=normal_activation_params,
    )
    / 1000.0
)
outdir = Path("results")
fe_model = HeartModelPulse(geo_folder=Path(outdir / "mesh"))
collector = DataCollector(outdir=outdir, problem=fe_model)
circ_model = CirculationModel()


# Increase to atrial pressure
for pressure in [0.0, 0.01]:
    volume = fe_model.compute_volume(activation_value=0, pressure_value=pressure)
    collector.collect(
        time=0,
        pressure=pressure,
        volume=volume,
        activation=0.0,
        flow=circ_model.flow,
        p_ao=circ_model.aortic_pressure,
    )

t_end = 600
#  we use only the first 700ms, as the relaxation is not yet implemented
circulation_solver(
    heart_model=fe_model,
    circulation_model=circ_model,
    activation=normal_activation[:t_end],
    time=t_eval[:t_end] * 1000,
    collector=collector,
    start_time=2,
)
