import time
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.local_mesh import LocalMesh

from src.solver_options import SolverOptions
from src.physics_model import PhysicsModel
from src.time_step import calculate_adaptive_dt
from src.fvm_kernels import _compute_residual, exchange_halo_data
from src.time_integrator import ExplicitEuler, MultiStageRungeKutta


def solve(
    equation: PhysicsModel,
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

    return history, dt_history
