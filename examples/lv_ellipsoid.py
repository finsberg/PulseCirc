from pathlib import Path
from dataclasses import dataclass
import cardiac_geometries.geometry
import matplotlib.pyplot as plt

import dolfin
import pulse
import numpy as np
import cardiac_geometries

import circ
import activation_model


def load_geometry(
    folder: Path,
    r_short_endo: float = 3,
    r_short_epi: float = 3.75,
    r_long_endo: float = 5.0,
    r_long_epi: float = 5.5,
    mesh_size: float = 3,
):
    if not folder.exists():
        cardiac_geometries.mesh.create_lv_ellipsoid(
            outdir=folder,
            r_short_endo=r_short_endo,
            r_short_epi=r_short_epi,
            r_long_endo=r_long_endo,
            r_long_epi=r_long_epi,
            psize_ref=mesh_size,
            mu_apex_endo=-np.pi,
            mu_base_endo=-np.arccos(r_short_epi / r_long_endo / 2),
            mu_apex_epi=-np.pi,
            mu_base_epi=-np.arccos(r_short_epi / r_long_epi / 2),
            create_fibers=True,
            fiber_angle_endo=-60,
            fiber_angle_epi=+60,
            fiber_space="P_2",
            aha=True,
        )
    geo = cardiac_geometries.geometry.Geometry.from_folder(folder)
    marker_functions = pulse.MarkerFunctions(
        cfun=geo.cfun, ffun=geo.ffun, efun=geo.efun
    )
    microstructure = pulse.Microstructure(f0=geo.f0, s0=geo.s0, n0=geo.n0)
    geometry = pulse.HeartGeometry(
        mesh=geo.mesh,
        markers=geo.markers,
        marker_functions=marker_functions,
        microstructure=microstructure,
    )
    return geometry


def default_circulation_parameters():
    return dict(
        R_ao=1.0,
        R_circ=10.0,
        C_circ=5.0,
        p_ao=10.0,
        p_dia=10.0,
    )


@dataclass
class CirculationModel:
    problem: pulse.MechanicsProblem
    outdir: Path

    def __post_init__(self):
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.outname = self.outdir / "results.xdmf"
        if self.outname.is_file():
            self.outname.unlink()
            self.outname.with_suffix(".h5").unlink()

    @property
    def activation(self) -> float:
        return float(self.problem.material.activation)

    @activation.setter
    def activation(self, value: float):
        pulse.iterate.iterate(self.problem, self.problem.material.activation, value)

    @property
    def volume(self) -> float:
        u = dolfin.split(self.problem.state)[0]
        return self.problem.geometry.cavity_volume(u=u)

    @property
    def pressure(self) -> float:
        return self.lvp.values()[0]

    @property
    def lvp(self):
        return self.problem.bcs.neumann[0].traction

    @pressure.setter
    def pressure(self, value):
        pulse.iterate.iterate(
            self.problem,
            self.problem.bcs.neumann[0].traction,
            value,
            initial_number_of_steps=15,
        )

    def dVdp(self) -> float:
        """
        Calculating the dV/dP based on FE model.

        :pulse.MechanicsProblem problem:    The mechanics problem containg the infromation on FE model.

        """
        #
        #  Backup the problem
        lvp_value_backup_dv = self.pressure
        state_backup_dv = self.problem.state.copy(deepcopy=True)
        lvp = self.lvp
        p_old = self.pressure
        v_old = self.volume
        dp0 = 0.001 * p_old
        dp = dp0
        k = 0
        flag_solved = False
        while (not flag_solved) and k < 20:
            try:
                p_new = p_old + dp
                lvp.assign(p_new)
                self.problem.solve()
                flag_solved = True
            except pulse.mechanicsproblem.SolverDidNotConverge:
                self.problem.state.assign(state_backup_dv)
                lvp.assign(lvp_value_backup_dv)
                # problem.solve()
                dp += dp0
                print(f"Derivation not Converged, increasin the dp to : {dp}")
                k += 1

        # pulse.iterate.iterate(dummy_problem, dummy_lvp, p_new, initial_number_of_steps=5)
        v_new = self.volume
        dVdp = (v_new - v_old) / (p_new - p_old)
        self.problem.state.assign(state_backup_dv)
        lvp.assign(lvp_value_backup_dv)

        return dVdp

    def save(self, t: float) -> None:
        u, p = self.problem.state.split(deepcopy=True)

        with dolfin.XDMFFile(self.outname.as_posix()) as xdmf:
            xdmf.write_checkpoint(u, "u", t, dolfin.XDMFFile.Encoding.HDF5, True)
            xdmf.write_checkpoint(p, "p", t, dolfin.XDMFFile.Encoding.HDF5, True)


def main():
    mesh_folder = Path("meshes") / "lv_ellipsoid"
    geometry = load_geometry(mesh_folder)
    print(geometry.cavity_volume())
    t_res = 1000
    results_name = "results.xdmf"

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

    systole_ind = np.where(normal_activation == 0)[0][-1] + 1
    normal_activation_systole = normal_activation[systole_ind:]
    t_eval_systole = t_eval[systole_ind:] * 1000
    tau = t_eval_systole[1] - t_eval_systole[0]

    activation = dolfin.Constant(0.0, name="gamma")

    matparams = pulse.HolzapfelOgden.default_parameters()
    material = pulse.HolzapfelOgden(
        activation=activation,
        active_model="active_stress",
        parameters=matparams,
        f0=geometry.f0,
        s0=geometry.s0,
        n0=geometry.n0,
    )

    def dirichlet_bc(W):
        V = W if W.sub(0).num_sub_spaces() == 0 else W.sub(0)
        return [
            dolfin.DirichletBC(
                V.sub(0),
                dolfin.Constant(0.0),
                geometry.ffun,
                geometry.markers["BASE"][0],
            )
        ]

    lvp = dolfin.Constant(0.0, name="LV Pressure")
    lv_marker = geometry.markers["ENDO"][0]
    lv_pressure = pulse.NeumannBC(traction=lvp, marker=lv_marker, name="lv")
    neumann_bc = [lv_pressure]

    robin_bc = [
        pulse.RobinBC(value=dolfin.Constant(1.0), marker=geometry.markers["EPI"][0]),
    ]
    # Collect boundary conditions
    bcs = pulse.BoundaryConditions(
        dirichlet=(dirichlet_bc,),
        neumann=neumann_bc,
        robin=robin_bc,
    )
    problem = pulse.MechanicsProblem(geometry, material, bcs)

    collector = circ.DataCollector(outdir=Path("results"))

    outdir = Path("results")
    outdir.mkdir(exist_ok=True, parents=True)

    circ.circ.circulation(
        model=CirculationModel(problem=problem, outdir=outdir),
        activation=normal_activation_systole,
        tau=tau,
        collector=collector,
        **default_circulation_parameters(),
    )


if __name__ == "__main__":
    main()
