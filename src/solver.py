import time
from typing import Dict, List, Tuple, Any
from dataclasses import field

import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.local_mesh import LocalMesh

from src.euler_equations import EulerEquations
from src.time_step import calculate_adaptive_dt
from src.reconstruction import compute_residual
from src.solver_options import SolverOptions


def exchange_halo_data(mesh: LocalMesh, U: np.ndarray, comm: MPI.Comm) -> None:
    """
    Exchanges halo cell data between partitions using MPI.

    This function ensures that each process has up-to-date data for its halo cells
    by communicating with neighboring processes. Non-blocking sends and receives
    are used to overlap communication with computation where possible.

    Args:
        mesh (LocalMesh): The local mesh object containing send/recv maps,
                          which define which cells to send and receive from neighbors.
        U (np.ndarray): The state vector array (conserved variables) for all cells
                        (owned and halo) on the current process. This array will
                        be updated with received halo data.
        comm (MPI.Comm): The MPI communicator used for inter-process communication.
    """
    # Prepare send buffers
    send_requests = []
    for neighbor_rank, local_indices_to_send in mesh.send_map.items():
        data_to_send = U[
            local_indices_to_send, :
        ].copy()  # .copy() is important for non-blocking send
        req = comm.Isend(
            [data_to_send, MPI.DOUBLE], dest=neighbor_rank, tag=neighbor_rank
        )
        send_requests.append(req)

    # Prepare receive buffers and initiate receives
    recv_requests = []
    recv_buffers = {}
    for neighbor_rank, local_indices_to_recv in mesh.recv_map.items():
        num_halo_cells_from_neighbor = len(local_indices_to_recv)
        num_conserved_variables = U.shape[1]
        recv_buffer = np.empty(
            (num_halo_cells_from_neighbor, num_conserved_variables), dtype=U.dtype
        )
        req = comm.Irecv(
            [recv_buffer, MPI.DOUBLE], source=neighbor_rank, tag=comm.Get_rank()
        )
        recv_requests.append(req)
        recv_buffers[(neighbor_rank, tuple(local_indices_to_recv))] = recv_buffer

    # Wait for all receive operations to complete and update U
    MPI.Request.Waitall(recv_requests)

    for (
        neighbor_rank,
        local_indices_to_recv_tuple,
    ), recv_buffer in recv_buffers.items():
        U[list(local_indices_to_recv_tuple), :] = recv_buffer

    # Wait for all send operations to complete
    MPI.Request.Waitall(send_requests)


def solve(
    equation: EulerEquations,
    mesh: LocalMesh,
    U: np.ndarray,
    bcs_lookup,
    t_end: float,
    comm: MPI.Comm,
    options: SolverOptions,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Main solver loop for the Finite Volume Method.

    This function orchestrates the time-stepping process for solving hyperbolic
    conservation laws. It uses a second-order MUSCL-Hancock scheme for spatial
    discretization and supports various time integration methods.

    Args:
        mesh (LocalMesh): The local mesh object for the current MPI process.
        U (np.ndarray): The initial conservative state vector.
        bcs_lookup: A Numba-compatible lookup array for boundary conditions.
        equation (Any): The equation system to be solved (e.g., EulerEquations).
        comm (MPI.Comm): The MPI communicator for inter-process communication.
        t_end (float): The end time of the simulation.
        options (SolverOptions, optional): Configuration options for the solver.
                                           If None, default options are used.

    Returns:
        Tuple[List[np.ndarray], List[float]]: A tuple containing:
            - history: A list of state vectors (U) at each saved time step.
            - dt_history: A list of time steps (dt) used at each saved time step.
    """
    if options is None:
        options = SolverOptions()

    history: List[np.ndarray] = [U.copy()]
    t: float = 0.0
    n: int = 0
    dt_history: List[float] = []
    dt: float = options.dt_initial

    while t < t_end:
        start_time = time.time()

        # --- Adaptive Time-Stepping ---
        if options.use_adaptive_dt:
            dt = calculate_adaptive_dt(mesh, U, equation, options, comm)
            dt = min(dt, t_end - t)

        # --- Halo Exchange ---
        exchange_halo_data(mesh, U, comm)

        # --- Time Integration ---
        if options.time_integration_method == "rk2":
            # --- Second-Order Runge-Kutta (RK2) Method ---
            # Stage 1: U_star = U - dt * R(U)
            residual_U = compute_residual(mesh, U, equation, bcs_lookup, options)
            U_star = U.copy()
            U_star[: mesh.num_owned_cells] -= dt * residual_U

            # Halo exchange for U_star before computing residual_U_star
            exchange_halo_data(mesh, U_star, comm)

            # Stage 2: U_new = 0.5 * (U + U_star - dt * R(U_star))
            residual_U_star = compute_residual(
                mesh, U_star, equation, bcs_lookup, options
            )
            U_new = U.copy()
            U_new[: mesh.num_owned_cells] = 0.5 * (
                U[: mesh.num_owned_cells]
                + U_star[: mesh.num_owned_cells]
                - dt * residual_U_star
            )

        elif options.time_integration_method == "euler":
            # --- First-Order Euler Method ---
            # U_new = U - dt * R(U)
            residual = compute_residual(mesh, U, equation, bcs_lookup, options)
            U_new = U.copy()
            U_new[: mesh.num_owned_cells] -= dt * residual

        else:
            raise NotImplementedError(
                f"Time integration method '{options.time_integration_method}' is not supported."
            )

        # Update state and time
        U = U_new
        t += dt
        n += 1

        loop_time = time.time() - start_time

        history.append(U.copy())
        dt_history.append(dt)

        # Store history and print progress (only for rank 0)
        if comm.Get_rank() == 0:
            print(
                f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s, Step: {n}, Loop Time: {loop_time:.4f}s",
                flush=True,
            )

    return history, dt_history
