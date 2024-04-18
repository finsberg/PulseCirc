#%%
import numpy as np 
import matplotlib.pyplot as plt
from pathlib import Path

from solver import *
from circulation_model import *
from heart_model_pulse import *
import activation_model

#%% Output directory

results_name='results.xdmf'
outdir = Path("results_refactored")
outdir.mkdir(exist_ok=True, parents=True)
outname = Path(outdir) / results_name
if outname.is_file():
    outname.unlink()
    outname.with_suffix(".h5").unlink()
    

t_res=1000
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

#%%

fe_model=HeartModelPulse(geo_folder=Path(outdir / 'mesh'))
fe_model.compute_volume(0,0)
fe_model.save(0,outname)

fe_model.compute_volume(0,0.02)
fe_model.save(1,outname)

circ_model=CirculationModel()
#%%
presures, volumes, outflows, aortic_pressures = circulation_solver(fe_model, circ_model, normal_activation[:400], t_eval[:400]*1000, outdir, start_time=2)
# %%
