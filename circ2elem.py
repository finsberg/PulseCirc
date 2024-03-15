#%%
from pathlib import Path
import logging
import cardiac_geometries
import pulse
import dolfin
import ufl_legacy as ufl
from pulse.solver import NonlinearSolver
from pulse.solver import NonlinearProblem

import numpy as np
import matplotlib.pyplot as plt
import activation_model
from pulse.utils import getLogger
logger = getLogger(__name__)
#%%Parameters

t_res=200
t_span = (0.0, 1.0)
# Aortic Pressure: the pressure from which the ejection start
P_ao=5

#%%
class MechanicsProblem_modal(pulse.MechanicsProblem):
    """
    Base class for mechanics problem
    """
    def __init__(
        self,
        geometry: pulse.Geometry,
        material: pulse.Material,
        control_mode: str,
        control_value: dolfin.Constant,
        bcs=None,
        bcs_parameters=None,
        solver_parameters=None,
    ):
        logger.debug("Initialize mechanics problem")
        if control_mode not in ["pressure", "volume"]:
            raise ValueError("Invalid control mode. Only 'pressure' and 'volume' are allowed.")
        self.geometry = geometry
        self.material = material
        self.control_mode = control_mode
        self.control_value = control_value
        self._handle_bcs(bcs=bcs, bcs_parameters=bcs_parameters)

        # Make sure that the material has microstructure information
        for attr in ("f0", "s0", "n0"):
            setattr(self.material, attr, getattr(self.geometry, attr))

        self.solver_parameters = NonlinearSolver.default_solver_parameters()
        if solver_parameters is not None:
            self.solver_parameters.update(**solver_parameters)

        self._init_spaces()
        self._init_forms()

    def _init_spaces(self):
        logger.debug("Initialize spaces for mechanics problem -- with lvp")
        mesh = self.geometry.mesh

        P2 = dolfin.VectorElement("Lagrange", mesh.ufl_cell(), 2)
        P1 = dolfin.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
        if self.control_mode=='volume':
            R = dolfin.FiniteElement("Real", mesh.ufl_cell(),0)
            self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P2,P1,R]))
        else:
            self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P2,P1]))

        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):
        logger.debug("Initialize forms mechanics problem")
        # Displacement and hydrostatic_pressure
        u=dolfin.split(self.state)[0]
        p=dolfin.split(self.state)[1]
        v  = dolfin.split(self.state_test)[0]
        q  = dolfin.split(self.state_test)[1]
        if self.control_mode=='volume':
            Pendo=dolfin.split(self.state)[2]
            qendo  = dolfin.split(self.state_test)[2]
            Vendo=self.control_value
        elif self.control_mode=='pressure':
            Vendo=None
            Pendo=self.control_value

        # Some mechanical quantities
        F = dolfin.variable(pulse.kinematics.DeformationGradient(u))
        J = pulse.kinematics.Jacobian(F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            F,
        ) + self.material.compressibility(p, J) 
        volume_constraint=self._inner_volume_constraint(u, Pendo, Vendo, self.geometry.markers["ENDO"][0])
        self._virtual_work = dolfin.derivative(
        internal_energy * dx + volume_constraint,
        self.state,
        self.state_test,
        )

        external_work = self._external_work(u, v)
        if external_work is not None:
            self._virtual_work += external_work

        self._set_dirichlet_bc()
        self._jacobian = dolfin.derivative(
            self._virtual_work,
            self.state,
            dolfin.TrialFunction(self.state_space),
        )
        self._init_solver()
        
    def _inner_volume_constraint(self, u, pendo, V, sigma):
        """
        Adapted from PULSE 2
        
        Compute the form
            (V(u) - V, pendo) * ds(sigma)
        where V(u) is the volume computed from u and
            u = displacement
            V = volume enclosed by sigma
            pendo = Lagrange multiplier
        sigma is the boundary of the volume.
        """

        geo = self.geometry


        # ufl doesn't support any measure for duality
        # between two Real spaces, so we have to divide
        # by the total measure of the domain
        ds_sigma = geo.ds(sigma)
        area = dolfin.assemble(dolfin.Constant(1.0) * ds_sigma)
        # Calculation of Vu which is the current volume based on deformation
        geo.update_xshift()
        xshift_val = geometry.xshift
        xshift = [0.0, 0.0, 0.0]
        xshift[0] = geo.base_mean_position[0]
        xshift = dolfin.Constant(tuple(xshift))
        #xshift = dolfin.Constant(xshift_val)
        x = ufl.SpatialCoordinate(geo.mesh) + u - xshift
        F = ufl.grad(x)
        n = ufl.cofac(F) * ufl.FacetNormal(geo.mesh)        
        V_u = -1 / float(geo.mesh.geometry().dim()) * ufl.inner(x, n)
        
        L = -pendo * V_u * ds_sigma
        if V is not None:
            L += dolfin.Constant(1.0 / area) * pendo * V * ds_sigma
        # L += pendo * V * ds_sigma
        return L
    def change_mode_and_reinit(self,new_control_mode: str) -> None:
        if self.control_mode == new_control_mode:
            return

        # Save the current state
        state_old = self.state.copy(True)
        if self.control_mode=='volume':
            pendo_old=dolfin.split(self.state)[2]
            Vendo_old=self.control_value
            pendo_old_value=pendo_old(self.geometry.mesh.coordinates()[0])
            self.control_value=dolfin.Constant(pendo_old_value,name='pressure')
        elif self.control_mode=='pressure':
            Vendo_old=self.geometry.cavity_volume(u=state_old.sub(0))
            pendo_old=self.control_value
            self.control_value=dolfin.Constant(Vendo_old,name='volume')

        # Reinit problem
        self.control_mode = new_control_mode
        self._init_spaces()
        self._init_forms()

        # Assign old values
        dolfin.assign(self.state.sub(0), state_old.sub(0))
        dolfin.assign(self.state.sub(1), state_old.sub(1))

        # ???
        # ??? What is this for?
        # ???
        # if self.parameters["bc_type"] not in [BCType.fix_base]:
        #     dolfin.assign(self.state.sub(2), state_old.sub(2))
        
    def copy_as_simple_problem(self):
        # Copying the problem as new simple Mechanics problem, and defining Pendo as Neumann BC. Note that the Pendo in the new problem is a different dolfin.coefficient as we do not want to change the original Pendo
        # FIXME:  The code can be much simpler if we just use the neumann BC
        Pendo_value=self.control_value.values()[0]
        Pendo_new=dolfin.Constant(Pendo_value)
        lv_pressure = pulse.NeumannBC(traction=Pendo_new, marker=self.geometry.markers["ENDO"][0], name="lv")
        neumann_bc = [lv_pressure]
        new_bcs = copy.deepcopy(self.bcs)
        new_bcs.neumann=neumann_bc
        new_problem = pulse.MechanicsProblem(self.geometry, self.material, new_bcs)
        dolfin.assign(new_problem.state.sub(0), self.state.sub(0))
        dolfin.assign(new_problem.state.sub(1), self.state.sub(1))
        return new_problem

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
base_spring = 1.0
robin_bc = [
    pulse.RobinBC(
        value=dolfin.Constant(base_spring),
        marker=geometry.markers["EPI"][0],
    ),
]

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
# lvp = dolfin.Constant(0.0)
# lv_marker = geometry.markers["ENDO"][0]
# lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
# neumann_bc = [lv_pressure]

# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc,
    #neumann=neumann_bc,
    # robin=robin_bc,
)
#%%
# V0=geometry.cavity_volume()
# Vendo=dolfin.Constant(V0, name='volume')
# problem = MechanicsProblem_modal(geometry, material, 'volume',Vendo, bcs)
Pendo=dolfin.Constant(0, name='pressure')
problem = MechanicsProblem_modal(geometry, material,'pressure',Pendo, bcs)

#%% Output directory for saving the results

outdir = Path("results")
outdir.mkdir(exist_ok=True, parents=True)
outname = Path(outdir) / "results.xdmf"
if outname.is_file():
    outname.unlink()
    outname.with_suffix(".h5").unlink()
    
outname_mesh = Path(outdir) / "mesh.xdmf"
if outname.is_file():
    outname.unlink()
    outname.with_suffix(".h5").unlink()
#%%
vols=[]
pres=[]
# Saving the initial pressure and volume
v_current=geometry.cavity_volume()
p_current=0
vols.append(v_current)
pres.append(p_current)
point=problem.geometry.mesh.coordinates()[0]
# Initialization to the atrium pressure of 0.2 kPa
pulse.iterate.iterate(problem, Pendo, 0.2, initial_number_of_steps=15)
v_current=geometry.cavity_volume(u=problem.state.sub(0))
p_current=Pendo(point)
vols.append(v_current)
pres.append(p_current)
#%%
problem.change_mode_and_reinit('volume')
for t in range(len(normal_activation_systole)):
    target=normal_activation_systole[t]
    pulse.iterate.iterate(problem, activation, target, initial_number_of_steps=5)
    u = problem.state.split(deepcopy=True)[0]
    v_current=geometry.cavity_volume(u=problem.state.sub(0))
    Pendo=problem.state.split(deepcopy=True)[2]
    p_current=Pendo(point)
    vols.append(v_current)
    pres.append(p_current)
    u.t=t
    with dolfin.XDMFFile(outname.as_posix()) as xdmf:
        xdmf.write_checkpoint(u, "u", float(t), dolfin.XDMFFile.Encoding.HDF5, True)
    if p_current>P_ao:
        break
#%%
def copy_simple_problem(simple_problem):
        # Copying the problem as new simple Mechanics problem, and defining Pendo as Neumann BC. Note that the Pendo in the new problem is a different dolfin.coefficient as we do not want to change the original Pendo
        # FIXME:  The code can be much simpler if we just use the neumann BC
        Pendo_value=simple_problem.bcs.neumann[0].traction.values()[0]
        Pendo_new=dolfin.Constant(Pendo_value)
        lv_pressure = pulse.NeumannBC(traction=Pendo_new, marker=simple_problem.geometry.markers["ENDO"][0], name="lv")
        neumann_bc = [lv_pressure]
        new_bcs_dirichlet = copy.deepcopy(simple_problem.bcs.dirichlet)
        new_bcs = pulse.BoundaryConditions(
            dirichlet=new_bcs_dirichlet,
            neumann=neumann_bc,
            # robin=robin_bc,
        )
        new_simple_problem = pulse.MechanicsProblem(simple_problem.geometry, simple_problem.material, new_bcs)
        dolfin.assign(new_simple_problem.state.sub(0), simple_problem.state.sub(0))
        dolfin.assign(new_simple_problem.state.sub(1), simple_problem.state.sub(1))
        return new_simple_problem
#%%
def circ(problem,v_current,v_old,p_current,p_old,tau):
    """
    Check the v_current and p_current with respect to the circulation model. Then the 
    problem is updated first with the caluclated pressure and then with the activation.
    FIXME for now circulation model is hard coded here but it should be an input.
    

    :pulse.MechanicsProblem problem: The mechanics problem containg the infromation on FE model.
    :param v_current:   The current calculated volume based on FE model.
    :param v_old:       The previous calculated volume based on FE model and circulation.
    :param p_current:   The current calculated pressure based on FE model.
    :param p_old:       The previous calculated pressure based on FE model and circulation.
    :param tau:         The time step
    """
    # Parameters of Circulation model
    new_problem=problem.copy_as_simple_problem()
    tol=0.1
    R_wk2=0.001
    C_wk2=10
    # initialize the windkessel
    R=[]
    Q0=WK2(tau,p_old,p_current,R_wk2,C_wk2)*tau
    V_WK2=v_current-Q0
    V_FE=v_current
    R.append(V_FE-V_WK2)
    dV_WK2=dWK2(WK2,tau,p_old,p_current,R_wk2,C_wk2)
    dV_FE=dFE(new_problem,p_current)
    J=dV_FE+dV_WK2
    p_new=p_current-R[-1]/J
    k=0
    while R[-1]>tol and k<10:
        k+=1
        new_problem=problem.copy_as_simple_problem()
        Pendo_dummy=new_problem.bcs.neumann[0].traction
        Pendo_dummy.rename('Pendo_dummy','dummy Pendo for Newton method')
        pulse.iterate.iterate(new_problem, Pendo_dummy, p_new, initial_number_of_steps=15)
        v_new=geometry.cavity_volume(u=new_problem.state.sub(0))
        Q=WK2(tau,p_current,p_new,R_wk2,C_wk2)*tau
        V_WK2=v_new-Q
        V_FE=v_new
        R.append(V_FE-V_WK2)
        dV_WK2=dWK2(WK2,tau,p_current,p_new,R_wk2,C_wk2)
        dV_FE=dFE(new_problem, p_new)
        J=dV_FE+dV_WK2
        p_new=p_new-R[-1]/J

def WK2(tau,p_old,p_current,R,C):
    dp=(p_current-p_old)/tau
    Q1=p_current/R
    Q2=dp*C
    return Q1+Q2

def dWK2(fun,tau,p_old,p_current,R,C):
    eval1=fun(tau,p_old,p_current,R,C)
    eval2=fun(tau,p_old,p_current*1.001,R,C)
    return (eval2-eval1)/(p_current*.001)

def dFE(problem,Pendo_value):
    dummy_problem = copy_simple_problem(problem)
    P_dummy=dummy_problem.bcs.neumann[0].traction
    P_dummy.rename('dummy_P1','dummy Pendo for dFE')
    pulse.iterate.iterate(dummy_problem, P_dummy, Pendo_value, initial_number_of_steps=15)
    V1=geometry.cavity_volume(u=dummy_problem.state.sub(0))
    dummy_problem_2 = copy_simple_problem(problem)
    P_dummy=dummy_problem_2.bcs.neumann[0].traction
    P_dummy.rename('dummy_P2','dummy Pendo for dFE')
    pulse.iterate.iterate(dummy_problem, P_dummy, Pendo_value*1.01, initial_number_of_steps=15)
    V2=geometry.cavity_volume(u=dummy_problem_2.state.sub(0))
    return (V2-V1)/(0.01*Pendo_value)
    # dummy_problem = pulse.MechanicsProblem_modal(problem.geometry, problem.material, problem.bcs)
    # dolfin.assign(dummy_problem.state.sub(0), problem.state.sub(0))
    # dolfin.assign(dummy_problem.state.sub(1), problem.state.sub(1))
    # pulse.iterate.iterate(dummy_problem, Pendo, p_current, initial_number_of_steps=15)
    # V1=geometry.cavity_volume(u=dummy_problem.state.sub(0))
    # pulse.iterate.iterate(dummy_problem, Pendo, p_current*1.01, initial_number_of_steps=15)
    # V2=geometry.cavity_volume(u=dummy_problem.state.sub(0))
    # return (V2-V1)/(0.01*p_current)

import copy
#FIXME we need normal pulse for this
problem.change_mode_and_reinit('pressure')
t0=copy.copy(t)+1
for t in range(t0,len(normal_activation_systole)):
#t=t0
    target=normal_activation_systole[t]
    pulse.iterate.iterate(problem, activation, target, initial_number_of_steps=5)
    u = problem.state.split(deepcopy=True)[0]
    v_current=geometry.cavity_volume(u=problem.state.sub(0))
    Pendo=problem.control_value
    p_current=Pendo(point)
    p_old,tau,v_old=pres[-1],t_eval[t],vols[-1]
    v0,p0,p_old,tau=v_current,pres[-1],pres[-2],t_eval[t]
    # initialize the windkessel
    R=[]
    Q0=WK2(tau,p_old,p0,R_wk2,C_wk2)
    V_WK2=v0-Q0*tau
    V_FE=v0
    R.append(V_FE-V_WK2)
    dV_WK2=dWK2(WK2,tau,p_old,p0,R_wk2,C_wk2)
    dV_FE=dFE(problem, p0)
    J=dV_FE-dV_WK2
    p=p0+R[-1]/J
    k=0
    while R[-1]>tol and k<100:
        k+=1
        pulse.iterate.iterate(problem, Pendo, p, initial_number_of_steps=15)
        v=geometry.cavity_volume(u=problem.state.sub(0))
        Q=WK2(tau,p_old,p,R_wk2,C_wk2)
        V_WK2=v-Q*tau
        V_FE=v
        R.append(V_FE-V_WK2)
        dV_WK2=dWK2(WK2,tau,p_old,p,R_wk2,C_wk2)
        dV_FE=dFE(problem, p)
        J=dV_FE-dV_WK2
        p=p+R[-1]/J
    vols.append(v)
    pres.append(p)
    u.t=t
    with dolfin.XDMFFile(outname.as_posix()) as xdmf:
        xdmf.write_checkpoint(u, "u", float(t), dolfin.XDMFFile.Encoding.HDF5, True)
    if target==np.max(normal_activation_systole):
        break
    
#%%
problem.change_mode_and_reinit('volume')
t0=copy.copy(t)
for t in range(t0,len(normal_activation_systole)):
    target=normal_activation_systole[t]
    pulse.iterate.iterate(problem, activation, target, initial_number_of_steps=5)
    u = problem.state.split(deepcopy=True)[0]
    v_current=geometry.cavity_volume(u=problem.state.sub(0))
    Pendo=problem.state.split(deepcopy=True)[2]
    p_current=Pendo(point)
    vols.append(v_current)
    pres.append(p_current)
    u.t=t
    with dolfin.XDMFFile(outname.as_posix()) as xdmf:
        xdmf.write_checkpoint(u, "u", float(t), dolfin.XDMFFile.Encoding.HDF5, True)
    if p_current<0.05:
        break
#%%
deformed_mesh= dolfin.Mesh(problem.geometry.mesh)
V = dolfin.VectorFunctionSpace(deformed_mesh, "Lagrange", 2)
U=dolfin.Function(V)
U.vector()[:] = u.vector()
dolfin.ALE.move(deformed_mesh, U)

from fenics_plotly import plot
fig = plot(deformed_mesh, color="red", opacity=0.3, show=False)
fig.add_plot(plot(problem.geometry.mesh, opacity=0.3, show=False))
fig.show()    
    # # Create a dummy function on the deformed mesh
    # V_dummy = dolfin.FunctionSpace(deformed_mesh, 'P', 1)
    # dummy_function = dolfin.Function(V_dummy)
    # dummy_function.vector()[:] = 0  # Arbitrary values