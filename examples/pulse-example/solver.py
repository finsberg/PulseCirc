import dolfin
import pulse
import logging
from collections import deque

logger = logging.getLogger(__name__)


def enlist(obj):
    try:
        return list(obj)
    except TypeError:
        return [obj]


class NonlinearProblem(dolfin.NonlinearProblem):
    def __init__(
        self,
        J,
        F,
        bcs,
        output_matrix=False,
        output_matrix_path="output",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._J = J
        self._F = F

        self.bcs = enlist(bcs)
        self.output_matrix = output_matrix
        self.output_matrix_path = output_matrix_path
        self.verbose = True
        self.n = 0

    def F(self, b: dolfin.PETScVector, x: dolfin.PETScVector):
        dolfin.assemble(self._F, tensor=b)
        for bc in self.bcs:
            bc.apply(b, x)

    def J(self, A: dolfin.PETScMatrix, x: dolfin.PETScVector):
        dolfin.assemble(self._J, tensor=A)
        for bc in self.bcs:
            bc.apply(A)


class ContinuationProblem(NonlinearProblem):
    def __init__(self, J, F, bcs, **kwargs):
        # self.problem = problem
        super().__init__(J, F, bcs, **kwargs)
        self._J = J
        self._F = F

        self.bcs = [bcs]

        # super(ContinuationProblem, self).__init__()

        self.fres = deque(maxlen=2)

        self.first_call = True
        self.skipF = False

        self._assemble_jacobian = True

    def form(self, A, P, b, x):
        # pb = self.problem
        if self._assemble_jacobian:
            dolfin.assemble_system(self._J, self._F, self.bcs, A_tensor=A, b_tensor=b)
        else:
            dolfin.assemble(self._F, tensor=b)
            if self.bcs:
                for bc in self.bcs:
                    bc.apply(b)
        self._assemble_jacobian = not self._assemble_jacobian

        return
        # Timer("ContinuationSolver: form")
        # pb = self.problem

        # # check if we need to assemble the jacobian
        # if self.first_call:
        #     reset_jacobian = True
        #     self.first_call = False
        #     self.skipF = True
        # else:
        #     reset_jacobian = b.empty() and not A.empty()
        #     self.skipF = reset_jacobian

        #     if len(self.fres) == 2 and reset_jacobian:
        #         if self.fres[1] < 0.1 * self.fres[0]:
        #             debug("REUSE J")
        #             reset_jacobian = False

        # if reset_jacobian:
        #     # requested J, assemble both
        #     debug("ASSEMBLE J")
        #     assemble_system(pb.dG, pb.G, pb.bcs, x0=x, A_tensor=A, b_tensor=b)

    def J(self, A, x):
        pass

    def F(self, b, x):
        return
        # if self.skipF:
        #     return
        # pb = self.problem
        # assemble(pb.G, tensor=b)
        # for bc in pb.bcs:
        #     bc.apply(b)
        # self.fres.append(b.norm("l2"))


class NewtonSolver(dolfin.NewtonSolver):
    def __init__(
        self,
        problem: pulse.NonlinearProblem,
        state: dolfin.Function,
        active,
        update_cb=None,
        parameters=None,
    ):
        self.active = active
        print(f"Initialize NewtonSolver with parameters: {parameters!r}")
        dolfin.PETScOptions.clear()
        self._problem = problem
        self._state = state
        self._update_cb = update_cb
        self._prev_state = dolfin.Vector(state.vector().copy())

        #
        # Initializing Newton solver (parent class)
        self.petsc_solver = dolfin.PETScKrylovSolver()
        super().__init__(
            self._state.function_space().mesh().mpi_comm(),
            self.petsc_solver,
            dolfin.PETScFactory.instance(),
        )
        self._handle_parameters(parameters)

    def _handle_parameters(self, parameters):
        # Setting default parameters
        params = type(self).default_solver_parameters()

        if parameters is not None:
            params.update(parameters)

        for k, v in params.items():
            if self.parameters.has_parameter(k):
                self.parameters[k] = v
            if self.parameters.has_parameter_set(k):
                for subk, subv in params[k].items():
                    self.parameters[k][subk] = subv
        petsc = params.pop("petsc", {})
        for k, v in petsc.items():
            if v is not None:
                dolfin.PETScOptions.set(k, v)
        self.newton_verbose = params.pop("newton_verbose", False)
        self.ksp_verbose = params.pop("ksp_verbose", False)
        self.debug = params.pop("debug", False)
        if self.newton_verbose:
            dolfin.set_log_level(dolfin.LogLevel.INFO)
            self.parameters["report"] = True
        if self.ksp_verbose:
            self.parameters["lu_solver"]["report"] = True
            self.parameters["lu_solver"]["verbose"] = True
            self.parameters["krylov_solver"]["monitor_convergence"] = True
            dolfin.PETScOptions.set("ksp_monitor_true_residual")
        self.linear_solver().set_from_options()
        self._residual_index = 0
        self._residuals = []
        self.parameters["convergence_criterion"] = "incremental"
        # self.parameters["relaxation_parameter"] = 0.8

    @staticmethod
    def default_solver_parameters():
        return {
            "petsc": {
                "ksp_type": "preonly",
                # "ksp_type": "gmres",
                # "pc_type": "lu",
                "pc_type": "cholesky",
                "pc_factor_mat_solver_type": "mumps",
                "mat_mumps_icntl_33": 0,
                "mat_mumps_icntl_7": 6,
            },
            "newton_verbose": False,
            "ksp_verbose": False,
            "debug": False,
            "linear_solver": "gmres",
            # "preconditioner": "lu",
            # "linear_solver": "mumps",
            "error_on_nonconvergence": False,
            "relative_tolerance": 1e-5,
            "absolute_tolerance": 1e-5,
            "maximum_iterations": 100,
            "report": False,
            "krylov_solver": {
                "nonzero_initial_guess": False,
                "absolute_tolerance": 1e-13,
                "relative_tolerance": 1e-13,
                "maximum_iterations": 1000,
                "monitor_convergence": False,
            },
            "lu_solver": {"report": False, "symmetric": True, "verbose": False},
        }

    def converged(self, r, p, i):
        res = r.norm("l2")
        print(f"Mechanics solver residual: {res}")

        if self.debug:
            if not hasattr(self, "_datacollector"):
                print("No datacollector registered with the NewtonSolver")

            else:
                self._residuals.append(res)

        return super().converged(r, p, i)

    def solver_setup(self, A, J, p, i):
        self._solver_setup_called = True
        super().solver_setup(A, J, p, i)

    def super_solve(self):
        return super().solve(self._problem, self._state.vector())

    def solve(self) -> tuple[int, bool]:
        try:
            nit, conv = super().solve(self._problem, self._state.vector())
        except RuntimeError:
            nit, conv = 0, False

        if not conv:
            logger.error("Newton solver did not converge")
            return (nit, conv)

        self._prev_state.zero()
        self._prev_state.axpy(1.0, self._state.vector())

        return (nit, conv)

    def reset(self):
        self._state.vector().zero()
        self._state.vector().axpy(1.0, self._prev_state)

    def update_solution(self, x, dx, rp, p, i):
        print(f"Updating mechanics solution with relax parameter {rp}, iteration {i}")
        super().update_solution(x, dx, rp, p, i)


class MechanicsProblem(pulse.MechanicsProblem):
    def _init_solver(self):
        if hasattr(self, "_dirichlet_bc"):
            bcs = self._dirichlet_bc
        else:
            bcs = []

        self._problem = NonlinearProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=bcs,
        )
        self._problem = ContinuationProblem(
            J=self._jacobian,
            F=self._virtual_work,
            bcs=bcs,
        )

        self.solver = NewtonSolver(
            problem=self._problem,
            state=self.state,
            active=self.material.active,
        )

    def solve(self):
        return self.solver.solve()


def iterate(problem, activation, pressure, target_activation, target_pressure):
    current_activation = float(activation)
    current_pressure = float(pressure)
    # First try with no stepping
    try:
        activation.assign(target_activation)
        pressure.assign(target_pressure)
        nit, conv = problem.solve()
        print(problem.state.vector().get_local())
        if not conv:
            raise RuntimeError("Newton solver did not converge")
    except RuntimeError:
        problem.solver.reset()
        activation.assign(current_activation)
        pressure.assign(current_pressure)
        problem.solve()
        breakpoint()
