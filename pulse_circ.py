#%%
from pathlib import Path
import logging
import cardiac_geometries
import pulse
import dolfin
import ufl_legacy as ufl
from pulse.solver import NonlinearSolver
from pulse.solver import NonlinearProblem
import copy

import numpy as np
import matplotlib.pyplot as plt
import activation_model
from pulse.utils import getLogger
logger = getLogger(__name__)


#%%Parameters

t_res=1000
t_span = (0.0, 1.0)
# Aortic Pressure: the pressure from which the ejection start
P_ao=5

#%%
def get_ellipsoid_geometry(folder=Path("lv")):
    if not folder.is_dir():
        # Create geometry
        cardiac_geometries.create_lv_ellipsoid(
            folder,
            create_fibers=True,
        )

    geo = cardiac_geometries.geometry.Geometry.from_folder(folder)
    marker_functions = pulse.MarkerFunctions(cfun=geo.cfun, ffun=geo.ffun)
    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)
    return pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        marker_functions=marker_functions,
        microstructure=microstructure,
    )

geometry = get_ellipsoid_geometry()

#%%
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
systole_ind=np.where(normal_activation == 0)[0][-1]+1
normal_activation_systole=normal_activation[systole_ind:]
t_eval_systole=t_eval[systole_ind:]
# %% Defining activation as dolfin.constant

activation = dolfin.Constant(0.0, name='gamma')

#%% Material Properties
matparams = pulse.HolzapfelOgden.default_parameters()
material = pulse.HolzapfelOgden(
    activation=activation,
    active_model="active_stress",
    parameters=matparams,
    f0=geometry.f0,
    s0=geometry.s0,
    n0=geometry.n0,
)
#%% Boundary Conditions
# Add spring term at the epicardium of stiffness 1.0 kPa/cm^2 to represent pericardium
# Fix the basal plane in the longitudinal direction
# 0 in V.sub(0) refers to x-direction, which is the longitudinal direction
def fix_basal_plane(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc = dolfin.DirichletBC(
        V.sub(0),
        dolfin.Constant(0.0),
        geometry.ffun,
        geometry.markers["BASE"][0],
    )
    return bc

dirichlet_bc = (fix_basal_plane,)

# LV Pressure
lvp = dolfin.Constant(0.0, name='LV Pressure')
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]

# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc,
    neumann=neumann_bc,
    # robin=robin_bc,
)
#%%
problem = pulse.MechanicsProblem(geometry, material, bcs)

outdir = Path("results_pulse_circ")
outdir.mkdir(exist_ok=True, parents=True)
outname = Path(outdir) / "results.xdmf"
if outname.is_file():
    outname.unlink()
    outname.with_suffix(".h5").unlink()
    
#%%
vols=[]
pres=[]
# Saving the initial pressure and volume
v_current=geometry.cavity_volume()
p_current=lvp.values()[0]
vols.append(v_current)
pres.append(p_current)
# %% Initialization to the atrium pressure of 0.2 kPa
pulse.iterate.iterate(problem, lvp, 0.2, initial_number_of_steps=15)
v_current=geometry.cavity_volume(u=problem.state.sub(0))
p_current=lvp.values()[0]
vols.append(v_current)
pres.append(p_current)
# %%
tau=t_eval[1]
p_ao=1

#%%
def WK2(tau,p_ao,p_old,p_current,R,C):
    if p_current>p_ao:
        dp=(p_current-p_old)/tau
        Q=p_current/R+dp*C
    else:
        Q=0
    return Q
def dV_FE(problem,p):
    #u_0=problem.state.split(deepcopy=True)[0]
    p_old=p.values()[0]
    v_old=geometry.cavity_volume(u=problem.state.sub(0))
    p_new=p_old*1.001
    pulse.iterate.iterate(problem, lvp, p_new, initial_number_of_steps=5)
    v_new=geometry.cavity_volume(u=problem.state.sub(0))
    #u_1=problem.state.split(deepcopy=True)[0]
    pulse.iterate.iterate(problem, lvp, p_old, initial_number_of_steps=5)
    #u_2=problem.state.split(deepcopy=True)[0]
    dVdp=(v_new-v_old)/(p_new-p_old)
    return dVdp
    
def dV_WK2(fun,tau,p_old,p_current,R,C):
    eval1=fun(tau,p_ao,p_old,p_current,R,C)
    eval2=fun(tau,p_ao,p_old,p_current*1.01,R,C)
    return (eval2-eval1)/(p_current*.01)

def copy_problem(problem,lvp_name=None):
        # FIXME Add it to the class
        # Copying the problem as new simple Mechanics problem, and defining a new Coefficient for Neumann BC. Note that the Coefficient in the new problem is a different dolfin.coefficient as we do not want to change the original Pendo
        lvp=problem.bcs.neumann[0].traction
        lvp_value=lvp.values()[0]
        if lvp_name==None:
            lvp_name=lvp.name()+'_new'
        lvp_new=dolfin.Constant(lvp_value,name=lvp_name)
        lv_pressure = pulse.NeumannBC(traction=lvp_new, marker=problem.geometry.markers["ENDO"][0], name="lv")
        new_bcs_neumann = [lv_pressure]
        new_bcs_dirichlet = copy.deepcopy(problem.bcs.dirichlet)
        new_bcs_robin = copy.deepcopy(problem.bcs.robin)
        new_bcs = pulse.BoundaryConditions(
            dirichlet=new_bcs_dirichlet,
            neumann=new_bcs_neumann,
            robin=new_bcs_robin,
        )
        new_problem = pulse.MechanicsProblem(problem.geometry, problem.material, new_bcs)
        dolfin.assign(new_problem.state.sub(0), problem.state.sub(0))
        dolfin.assign(new_problem.state.sub(1), problem.state.sub(1))
        return new_problem

#%%
for t in range(len(normal_activation_systole)):
    target_activation=normal_activation_systole[t]
    pulse.iterate.iterate(problem, activation, target_activation, initial_number_of_steps=5)
    #### Circulation
    R=[]
    circ_iter=0
    tol=0.1
    # initial guess for new pressure
    if t==0:
        p_current=p_current*1.01
    else:
        p_current=p_current+(p_current-pres[-2])
    while len(R)==0 or (np.abs(R[-1])>tol and circ_iter<10):
        pulse.iterate.iterate(problem, lvp, p_current, initial_number_of_steps=15)
        v_current=geometry.cavity_volume(u=problem.state.sub(0))
        p_old=pres[-1]
        v_old=vols[-1]
        Q=WK2(tau,p_ao,p_old,p_current,0.01,1)
        v_fe=v_current
        v_circ=v_old-Q
        R.append(v_fe-v_circ)
        dVFE_dP=dV_FE(problem,lvp)
        dQCirc_dP=dV_WK2(WK2,tau,p_old,p_current,0.01,1)
        J=dVFE_dP+dQCirc_dP
        p_current=p_current-R[-1]/J
        circ_iter+=1
    pulse.iterate.iterate(problem, lvp, p_current, initial_number_of_steps=5)
    v_current=geometry.cavity_volume(u=problem.state.sub(0))
    vols.append(v_current)
    pres.append(p_current)
    if t>10:
        break
    
# %%
