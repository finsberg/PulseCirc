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


#%% Parameters
R_circ=10
C_circ=0.1

t_res=1000
t_span = (0.0, 1.0)

# # Aortic Pressure: the pressure from which the ejection start
p_ao=12
# p_ao=2

# # Assuming 50 mm for LVEDd (ED diameter), and 7.5mm for wall thickness.  
# r_short_endo = 30
# r_short_epi = 37.5
# r_long_endo = 55
# r_long_epi = 60
# mesh_size=10
r_short_endo = 7
r_short_epi = 10
r_long_endo = 17
r_long_epi = 20
mesh_size=3
# # Sigma_0 for activation parameter
sigma_0=50e3

results_name='results_R10_C01_P12_Sigma50.xdmf'
#%%
def get_ellipsoid_geometry(folder=Path("lv"),r_short_endo = 7,r_short_epi = 10,r_long_endo = 17,r_long_epi = 20, mesh_size=3):
    geo = cardiac_geometries.mesh.create_lv_ellipsoid(
        outdir= folder,
        r_short_endo = r_short_endo,
        r_short_epi = r_short_epi,
        r_long_endo = r_long_endo,
        r_long_epi = r_long_epi,
        psize_ref = mesh_size,
        mu_apex_endo = -np.pi,
        mu_base_endo = -np.arccos(r_short_epi / r_long_endo/2),
        mu_apex_epi = -np.pi,
        mu_base_epi = -np.arccos(r_short_epi / r_long_epi/2),
        create_fibers = True,
        fiber_angle_endo = -60,
        fiber_angle_epi = +60,
        fiber_space = "P_1",
        aha = True,
    )
    marker_functions = pulse.MarkerFunctions(cfun=geo.cfun, ffun=geo.ffun, efun=geo.efun)
    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)
    geometry=pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        marker_functions=marker_functions,
        microstructure=microstructure,
    )
    return geometry

geometry = get_ellipsoid_geometry(folder=Path("lv"),r_short_endo = r_short_endo, r_short_epi = r_short_epi, r_long_endo = r_long_endo, r_long_epi = r_long_epi, mesh_size=mesh_size)
geometry.mesh
print(geometry.cavity_volume())
#%%
t_eval = np.linspace(*t_span, t_res)
normal_activation_params = activation_model.default_parameters()
normal_activation_params['sigma_0']=sigma_0
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
# ------------------- Fix base and/or endoring  -------------------------
# Finidng the endo ring radius
pnts=[]
radii=[]
for fc in dolfin.facets(geometry.mesh):
    if geometry.ffun[fc]==geometry.markers['BASE'][0]:
        for vertex in dolfin.vertices(fc):
            pnts.append(vertex.point().array())
pnts=np.array(pnts)            
EndoRing_radius=np.sqrt(np.min((pnts[:,1]**2+pnts[:,2]**2)))
print(f'Endoring radius is {EndoRing_radius}')
# EndoRing_subDomain = dolfin.CompiledSubDomain('near(x[0], 1, 0.001) && near(pow(x[1],2)+pow(x[2],2), radius, 0.001)', radius=EndoRing_radius)

def AllBCs(W):
    V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
    bc_fixed_based = dolfin.DirichletBC(
        V.sub(0),
        dolfin.Constant(0.0),
        geometry.ffun,
        geometry.markers["BASE"][0],
    )
    class EndoRing_subDomain(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return dolfin.near(x[0], 5, .01) and dolfin.near(pow(pow(x[1],2)+pow(x[2],2),0.5), 6.69, .5)
    endo_ring_fixed=dolfin.DirichletBC(
        V,
        dolfin.Constant((0.0,0.0,0.0)),
        EndoRing_subDomain(),
        method="pointwise",
    )
    return [bc_fixed_based,endo_ring_fixed]
dirichlet_bc = (AllBCs,)


# ------------------- LV pressure on ENDO surface -------------------------
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

outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)
outname = Path(outdir) / results_name
if outname.is_file():
    outname.unlink()
    outname.with_suffix(".h5").unlink()
    
#%%
vols=[]
pres=[]
flows=[]
# Saving the initial pressure and volume
v_current=geometry.cavity_volume()
p_current=lvp.values()[0]
vols.append(v_current)
pres.append(p_current)
flows.append(0)
# %% Initialization to the atrium pressure of 0.2 kPa
pulse.iterate.iterate(problem, lvp, 0.02, initial_number_of_steps=15)
v_current=geometry.cavity_volume(u=problem.state.sub(0))
p_current=lvp.values()[0]
vols.append(v_current)
pres.append(p_current)
flows.append(0)
reults_u, p = problem.state.split(deepcopy=True)
reults_u.t=0
with dolfin.XDMFFile(outname.as_posix()) as xdmf:
    xdmf.write_checkpoint(reults_u, "u", float(0), dolfin.XDMFFile.Encoding.HDF5, True)
# %%
tau=t_eval_systole[1]-t_eval_systole[0]
#%%
def WK2(tau,p_ao,p_old,p_current,R,C,AVC_flag):
    # AVC Aortic Valve Closure after ejection phase become True
    if AVC_flag:
        Q=0
    elif p_current>p_ao:
        dp=(p_current-p_old)/tau
        Q=p_current/R+dp*C
    else:
        Q=0
    return Q
def dV_FE(problem):
    """
    Calculating the dV/dP based on FE model. 
    
    :pulse.MechanicsProblem problem:    The mechanics problem containg the infromation on FE model.
    
    """
    #
    #  Backup the problem
    state_backup_dv = problem.state.copy(deepcopy=True)
    lvp_value_backup_dv=get_lvp_from_problem(problem).values()[0]
    #
    #
    lvp=get_lvp_from_problem(problem)
    p_old=lvp.values()[0]
    v_old=get_lvv_from_problem(problem)
    dp0=0.001*p_old
    dp=dp0
    k=0
    flag_solved=False
    while (not flag_solved) and k<20:
        try:
            p_new=p_old+dp
            lvp.assign(p_new)
            problem.solve()
            flag_solved=True
        except pulse.mechanicsproblem.SolverDidNotConverge:
            problem.state.assign(state_backup_dv)
            lvp.assign(lvp_value_backup_dv)
            # problem.solve()
            dp+=dp0
            print(f"Derivation not Converged, increasin the dp to : {dp}")
            k+=1
        
    # pulse.iterate.iterate(dummy_problem, dummy_lvp, p_new, initial_number_of_steps=5)
    v_new=get_lvv_from_problem(problem)
    dVdp=(v_new-v_old)/(p_new-p_old)
    problem.state.assign(state_backup_dv)
    lvp.assign(lvp_value_backup_dv)
    # FIXME: I think we need to solve the problem here too
    # problem.solve()
    return dVdp
    
def dV_WK2(fun,tau,p_old,p_current,R,C,AVC_flag):
    eval1=fun(tau,p_ao,p_old,p_current,R,C,AVC_flag)
    eval2=fun(tau,p_ao,p_old,p_current*1.01,R,C,AVC_flag)
    return (eval2-eval1)/(p_current*.01)



def get_lvp_from_problem(problem):
    # getting the LV pressure which is assinged as Neumann BC from a Pulse.MechanicsProblem
    return problem.bcs.neumann[0].traction
def get_lvv_from_problem(problem):
    # getting the LV volume from a Pulse.MechanicsProblem and its solution
    return problem.geometry.cavity_volume(u=problem.state.sub(0))

#%%
AVC_flag=False
import csv
with open(Path(outdir) / 'data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['P_ao', p_ao, '   R_circ', R_circ,'   C_circ', C_circ])
    writer.writerow(['Time', 'Activation', 'Volume', 'Pressure'])
    for t in range(len(normal_activation_systole)):
        target_activation=normal_activation_systole[t]
        pulse.iterate.iterate(problem, activation, target_activation)
        #### Circulation
        circ_iter=0
        # initial guess for new pressure
        if t==0:
            p_current=p_current*1.01
        else:
            p_current=pres[-1]+(pres[-1]-pres[-2])
            
        # if p_current<p_ao and (pres[-1]-pres[-2])<0:
        #     AVC_flag=True
        # else:
        #     AVC_flag=False
        problem.solve()
        p_old=pres[-1]
        v_old=vols[-1]
        R=[]
        tol=0.00001*v_old
        while len(R)==0 or (np.abs(R[-1])>tol and circ_iter<20):
            pulse.iterate.iterate(problem, lvp, p_current)
            v_current=get_lvv_from_problem(problem)
            Q=WK2(tau,p_ao,p_old,p_current,R_circ,C_circ,AVC_flag)
            v_fe=v_current
            v_circ=v_old-Q
            R.append(v_fe-v_circ)
            if np.abs(R[-1])>tol:
                dVFE_dP=dV_FE(problem)
                dQCirc_dP=dV_WK2(WK2,tau,p_old,p_current,R_circ,C_circ,AVC_flag)
                J=dVFE_dP+dQCirc_dP
                p_current=p_current-R[-1]/J
                circ_iter+=1
        p_current=get_lvp_from_problem(problem).values()[0]
        v_current=get_lvv_from_problem(problem)
        vols.append(v_current)
        pres.append(p_current)
        flows.append(Q)
        reults_u, p = problem.state.split(deepcopy=True)
        reults_u.t=t+1
        with dolfin.XDMFFile(outname.as_posix()) as xdmf:
            xdmf.write_checkpoint(reults_u, "u", float(t+1), dolfin.XDMFFile.Encoding.HDF5, True)
        writer.writerow([t,target_activation, v_current, p_current])
        # if t%20==0:
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # Create a figure and two subplots
        axs[0].scatter(t_eval_systole[t], target_activation)
        axs[0].plot(t_eval_systole, normal_activation_systole)
        axs[0].set_ylabel('Activation (kPa)')
        axs[0].set_xlabel('Cardiac Cycle (-)')
        axs[1].plot(np.array(vols), pres)
        axs[1].set_ylabel('Pressure (kPa)')
        axs[1].set_xlabel('Volume (mm3)')
        axs[1].set_xlim([0, 2700])  
        axs[1].set_ylim([0, 15])  
        axs[2].plot(np.linspace(0,t_eval_systole[t],t+3), flows)
        axs[2].set_ylabel('Outflow (mm2/s)')
        axs[2].set_xlabel('Cardiac Cycle (-)')
        axs[2].set_xlim([0, 1])  
        axs[2].set_ylim([0, 1800]) 
        # Adjust layout
        plt.tight_layout()
        # Save the figure
        name = 'plot_' + str(t) + '.png'
        plt.savefig(Path(outdir) / name)
#%%    