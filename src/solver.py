import time
from typing import List, Tuple

import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.local_mesh import LocalMesh

from equation_euler import EulerEquations
from src.time_step import calculate_adaptive_dt
from src.fused_compute_residual import (
    compute_gradients_gaussian,
    compute_limiters,
    compute_fused_residual,
    LIMITERS,
)
from src.solver_options import SolverOptions


def _compute_residual(
    mesh: LocalMesh,
    U: np.ndarray,
    equation: EulerEquations,
    bcs_lookup,
    options: SolverOptions,
) -> np.ndarray:
    """
    Orchestrates the computation of the residual using the fused kernel.
    """
    nvars = U.shape[1]

    # --- 1. Gradient Computation (Parallel) ---
    gradients = compute_gradients_gaussian(
        mesh.n_cells,
        nvars,
        mesh.cell_neighbors,
        mesh.cell_face_normals,
        mesh.cell_face_areas,
        mesh.cell_centroids,
        mesh.cell_volumes,
        mesh.face_to_centroid_distances,
        mesh.cell_face_midpoints,
        U,
        options.over_relaxation,
    )

    # --- 2. Slope Limiting (Parallel) ---
    limiter_func = LIMITERS.get(options.limiter_type, LIMITERS["barth_jespersen"])
    limiters = compute_limiters(
        mesh.n_cells,
        nvars,
        mesh.cell_neighbors,
        mesh.cell_face_midpoints,
        mesh.cell_centroids,
        U,
        gradients,
        limiter_func,
    )

    # --- 3. Fused Flux Computation (Parallel) ---
    residual = compute_fused_residual(
        mesh.num_owned_cells,
        nvars,
        mesh.cell_neighbors,
        mesh.cell_face_normals,
        mesh.cell_face_areas,
        mesh.cell_face_midpoints,
        mesh.cell_centroids,
        mesh.cell_volumes,
        U,
        gradients,
        limiters,
        equation,
        bcs_lookup,
        mesh.cell_face_tags,
        options.flux_type,
    )

    return residual


def exchange_halo_data(mesh: LocalMesh, U: np.ndarray, comm: MPI.Comm) -> None:
    """
    Exchanges halo cell data between partitions using MPI.
    """
    send_requests = []
    for neighbor_rank, local_indices_to_send in mesh.send_map.items():
        data_to_send = U[local_indices_to_send, :].copy()
        req = comm.Isend(
            [data_to_send, MPI.DOUBLE], dest=neighbor_rank, tag=neighbor_rank
        )
        send_requests.append(req)

    recv_requests = []
    for neighbor_rank, local_indices_to_recv in mesh.recv_map.items():
        num_halo_cells = len(local_indices_to_recv)
        recv_buffer = np.empty((num_halo_cells, U.shape[1]), dtype=U.dtype)
        req = comm.Irecv(
            [recv_buffer, MPI.DOUBLE], source=neighbor_rank, tag=comm.Get_rank()
        )
        recv_requests.append((req, recv_buffer, local_indices_to_recv))

    # Wait for receives and update halo cells
    for req, buffer, indices in recv_requests:
        req.Wait()
        U[indices, :] = buffer

    # Wait for all sends to complete
    if send_requests:
        MPI.Request.Waitall(send_requests)


def solve(
    equation: EulerEquations,
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

    while t < t_end:
        start_time = time.time()

        if options.use_adaptive_dt:
            dt = calculate_adaptive_dt(mesh, U, equation, options, comm)
            dt = min(dt, t_end - t)

        exchange_halo_data(mesh, U, comm)

        if options.time_integration_method == "rk2":
            # Stage 1
            residual_U = _compute_residual(mesh, U, equation, bcs_lookup, options)
            U_star = U.copy()
            U_star[: mesh.num_owned_cells] -= dt * residual_U

            exchange_halo_data(mesh, U_star, comm)

            # Stage 2
            residual_U_star = _compute_residual(
                mesh, U_star, equation, bcs_lookup, options
            )
            U_new = U.copy()
            U_new[: mesh.num_owned_cells] = 0.5 * (
                U[: mesh.num_owned_cells]
                + U_star[: mesh.num_owned_cells]
                - dt * residual_U_star
            )
        elif options.time_integration_method == "euler":
            residual = _compute_residual(mesh, U, equation, bcs_lookup, options)
            U_new = U.copy()
            U_new[: mesh.num_owned_cells] -= dt * residual
        else:
            raise NotImplementedError(
                f"Time integration method '{options.time_integration_method}' is not supported."
            )

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
