#%%
from pathlib import Path

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

t_res=100
t_span = (0.0, 1.0)
P_ED=.1

#%%
class MechanicsProblem_VConst(pulse.MechanicsProblem):
    """
    Base class for mechanics problem
    """

    def __init__(
        self,
        geometry: pulse.Geometry,
        material: pulse.Material,
        Vendo: dolfin.Constant,
        bcs=None,
        bcs_parameters=None,
        solver_parameters=None,
    ):
        logger.debug("Initialize mechanics problem")
        self.geometry = geometry
        self.material = material
        self.Vendo = Vendo
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
        R = dolfin.FiniteElement("Real", mesh.ufl_cell(),0)

        self.state_space = dolfin.FunctionSpace(mesh, dolfin.MixedElement([P2,P1,R]))

        self.state = dolfin.Function(self.state_space, name="state")
        self.state_test = dolfin.TestFunction(self.state_space)

    def _init_forms(self):
        logger.debug("Initialize forms mechanics problem")
        # Displacement and hydrostatic_pressure
        u, p, pendo = dolfin.split(self.state)
        v, q, qendo  = dolfin.split(self.state_test)

        # Some mechanical quantities
        F = dolfin.variable(pulse.kinematics.DeformationGradient(u))
        J = pulse.kinematics.Jacobian(F)
        dx = self.geometry.dx

        internal_energy = self.material.strain_energy(
            F,
        ) + self.material.compressibility(p, J) 

        volume_constraint=self._inner_volume_constraint(u, pendo, self.Vendo, self.geometry.markers["ENDO"][0])

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
    # def _init_solver(self):
    #     try:
    #         import dolfin_adjoint  # noqa: F401
    #     except ImportError:
    #             has_dolfin_adjoint = False
    #     else:
    #             has_dolfin_adjoint = True
    #     if has_dolfin_adjoint:
    #         from dolfin_adjoint import (
    #             NonlinearVariationalProblem,
    #             NonlinearVariationalSolver,
    #         )

    #         self._problem = NonlinearVariationalProblem(
    #             J=self._jacobian,
    #             F=self._virtual_work,
    #             u=self.state,
    #             bcs=self._dirichlet_bc,
    #         )
    #         self.solver = NonlinearVariationalSolver(self._problem)
    #     else:
    #         self._problem = NonlinearProblem(
    #             J=self._jacobian,
    #             F=self._virtual_work,
    #             bcs=self._dirichlet_bc,
    #         )
    #         self.solver = NonlinearSolver(
    #             self._problem,
    #             self.state,
    #             parameters=self.solver_parameters,
    #         )
        
    def _inner_volume_constraint(self, u, pendo, V, sigma):
        """
        Excerpted from PULSE 2
        
        Compute the form
            (V(u) - V, pendo) * ds(sigma)
        where V(u) is the volume computed from u and
            u = displacement
            V = volume enclosed by sigma
            pendo = Lagrange multiplier
        sigma is the boundary of the volume.
        """

        geo = self.geometry

# ???
# ??? what is the comment about?
# ???
        # ufl doesn't support any measure for duality
        # between two Real spaces, so we have to divide
        # by the total measure of the domain
        ds_sigma = geo.ds(sigma)
        area = dolfin.assemble(dolfin.Constant(1.0) * ds_sigma)
        # Calculation of Vu which is the current volume based on deformation
        xshift_val = geo.base_mean_position
        xshift = dolfin.Constant(tuple(xshift_val))
        x = ufl.SpatialCoordinate(geo.mesh) + u - xshift
        F = ufl.grad(x)
        n = ufl.cofac(F) * ufl.FacetNormal(geo.mesh)        
        V_u = -1 / float(geo.mesh.geometry().dim()) * ufl.inner(x, n)
        
        L = -pendo * V_u * ds_sigma
        L += dolfin.Constant(1.0 / area) * pendo * V * ds_sigma
        # L += pendo * V * ds_sigma
        return L

# ???
# ??? why dolfin.Constant(1.0 / area)
# ???
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
# %%
V = dolfin.FunctionSpace(geometry.mesh, "CG", 1)
target_activation = dolfin.Function(V)
activation = dolfin.Function(V)

# ???
# ??? why not defining actiction as a real field
# ???
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
# LV Pressure
lvp = dolfin.Constant(0.0)
lv_marker = geometry.markers["ENDO"][0]
lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
neumann_bc = [lv_pressure]
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


# Collect boundary conditions
bcs = pulse.BoundaryConditions(
    dirichlet=dirichlet_bc,
    neumann=neumann_bc,
    robin=robin_bc,
)


#%%
V0=geometry.cavity_volume()
Vendo=dolfin.Constant(V0)
problem = MechanicsProblem_VConst(geometry, material, Vendo, bcs)
target_activation.vector()[:] = normal_activation_systole[0]
target_activation.vector()[:] = normal_activation_systole[0]/10
pulse.iterate.iterate(problem, activation, target_activation)
u, p, pendo = problem.state.split(deepcopy=True)
v_current = geometry.cavity_volume(u=u)
p_current=pendo(dolfin.Point(geometry.mesh.coordinates()[0]))

