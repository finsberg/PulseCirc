# %%
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from circ.solver import circulation_solver
from circ.circulation_model import CirculationModel
from circ.datacollector import DataCollector
from heart_model_thin_sphere import ThinSphere
import activation_model

# %% Output directory

results_name = "results.xdmf"
outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)
outname = Path(outdir) / results_name
if outname.is_file():
    outname.unlink()
    outname.with_suffix(".h5").unlink()


t_res = 1000
t_span = (0.0, 1.0)
t_eval = np.linspace(*t_span, t_res)
normal_activation_params = activation_model.default_parameters()

normal_activation = (
    activation_model.activation_function(
        t_span=t_span,
        t_eval=t_eval,
        parameters=normal_activation_params,
    )
    / 1000.0
)

# %%

fe_model = ThinSphere(geo_folder=Path(outdir / "mesh"))
fe_model.compute_volume(0, 0)
fe_model.save(0, outname)

fe_model.compute_volume(0, 0.01)
fe_model.save(1, outname)

circ_model = CirculationModel()
t_end = 600
#  we use only the first 700ms, as the relaxation is not yet implemented
collector = DataCollector(outdir=Path("results"), problem=fe_model)
circulation_solver(
    heart_model=fe_model,
    circulation_model=circ_model,
    activation=normal_activation[:t_end],
    time=t_eval[:t_end] * 1000,
    collector=collector,
    start_time=2,
)

# %% Saving the results
fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # Create a figure and two subplots
axs[0, 0].plot(t_eval[:t_end] * 1000, normal_activation[:t_end])
axs[0, 0].set_ylabel("Activation (kPa)")
axs[0, 0].set_xlabel("Time (ms)")
axs[0, 1].plot(np.array(volumes), presures)
axs[0, 1].set_ylabel("Pressure (kPa)")
axs[0, 1].set_xlabel("Volume (ml)")
axs[1, 0].plot(t_eval[: len(presures)] * 1000, outflows)
axs[1, 0].set_ylabel("Outflow (ml/s)")
axs[1, 0].set_xlabel("Time (ms)")
axs[1, 1].plot(t_eval[: len(presures)] * 1000, presures, label="LV Pressure")
axs[1, 1].plot(
    t_eval[: len(presures)] * 1000, aortic_pressures, label="Aortic Pressure"
)
axs[1, 1].legend()
axs[1, 1].set_ylabel("Pressure (kPa)")
axs[1, 1].set_xlabel("Time (ms)")
plt.tight_layout()
name = "results.png"
plt.savefig(Path(outdir) / name)
# %%
