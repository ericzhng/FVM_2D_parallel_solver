import time
from typing import List, Tuple, Any, Optional

import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.local_mesh import LocalMesh
from fvm_mesh.polymesh import PolyMesh

from src.solver_options import SolverOptions
from src.physics_model import PhysicsModel
from src.time_step import calculate_adaptive_dt
from src.fvm_kernels import _compute_residual, exchange_halo_data
from src.time_integrator import ExplicitEuler, MultiStageRungeKutta
from src.output import write_vtk, write_tecplot


def reconstruct_and_write(
    global_mesh: PolyMesh | None,
    mesh: LocalMesh,
    U: np.ndarray,
    comm: MPI.Comm,
    options: SolverOptions,
    equation: PhysicsModel,
    step: int,
):
    """
    Gathers simulation data from all processes and writes the output on rank 0.
    """
    rank = comm.Get_rank()

    all_U: Optional[List[np.ndarray]] = comm.gather(U, root=0)
    all_local_meshes: Optional[List[LocalMesh]] = comm.gather(mesh, root=0)

    # comm.gather returns None on non-root ranks; assert non-None before use on rank 0
    if rank == 0:
        if global_mesh is None:
            return

        assert all_U is not None and all_local_meshes is not None

        num_vars = U.shape[1]
        num_global_cells = len(global_mesh.cell_centroids)
        U_global = np.zeros((num_global_cells, num_vars))

        for i in range(len(all_U)):
            rank_U = all_U[i]
            local_mesh = all_local_meshes[i]
            if hasattr(local_mesh, "l2g_cells"):
                global_indices = local_mesh.l2g_cells[: local_mesh.num_owned_cells]
                U_global[global_indices] = rank_U[: local_mesh.num_owned_cells]

        variable_names = equation.variable_names
        filename = f"results/{options.output_filename_prefix}_{step:03d}"

        if options.output_format == "vtk":
            write_vtk(f"{filename}.vtk", global_mesh, U_global, variable_names)
        elif options.output_format == "tecplot":
            write_tecplot(f"{filename}.dat", global_mesh, U_global, variable_names)


def solve(
    equation: PhysicsModel,
    global_mesh: PolyMesh | None,
    mesh: LocalMesh,
    U: np.ndarray,
    bcs_lookup,
    comm: MPI.Comm,
    options: SolverOptions,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Main solver loop for the Finite Volume Method.
    """
    history: List[np.ndarray] = [U.copy()]
    t: float = 0.0
    n: int = 0
    dt_history: List[float] = []
    dt: float = options.dt_initial
    t_end: float = options.t_end

    if options.time_integration_method == "rk2":
        time_integrator = MultiStageRungeKutta(equation, options)
    elif options.time_integration_method == "euler":
        time_integrator = ExplicitEuler(equation, options)
    else:
        raise NotImplementedError(
            f"Time integration method '{options.time_integration_method}' is not supported."
        )

    while t < t_end:
        start_time = time.time()

        if n % options.output_interval == 0:
            reconstruct_and_write(global_mesh, mesh, U, comm, options, equation, n)

        if options.use_adaptive_dt:
            dt = calculate_adaptive_dt(mesh, U, equation, options, comm)
            dt = min(dt, t_end - t)

        exchange_halo_data(mesh, U, comm)

        U_new = time_integrator.step(mesh, U, dt, bcs_lookup, comm)

        U = U_new
        t += dt
        n += 1
        loop_time = time.time() - start_time

        history.append(U.copy())
        dt_history.append(dt)

        if comm.Get_rank() == 0:
            print(
                f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s, Step: {n}, Loop Time: {loop_time:.4f}s",
                flush=True,
            )

    reconstruct_and_write(global_mesh, mesh, U, comm, options, equation, n)
    return history, dt_history
