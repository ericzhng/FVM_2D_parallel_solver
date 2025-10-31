from typing import Dict, List, Tuple, Any

import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.local_mesh import LocalMesh # Corrected import
from src.time_step import calculate_adaptive_dt
from src.reconstruction import compute_residual
from src.boundary import create_numba_bcs


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
        data_to_send = U[local_indices_to_send, :].copy() # .copy() is important for non-blocking send
        req = comm.Isend(data_to_send, dest=neighbor_rank, tag=neighbor_rank)
        send_requests.append(req)

    # Prepare receive buffers and initiate receives
    recv_requests = []
    recv_buffers = {}
    for neighbor_rank, local_indices_to_recv in mesh.recv_map.items():
        num_halo_cells_from_neighbor = len(local_indices_to_recv)
        num_conserved_variables = U.shape[1]
        recv_buffer = np.empty((num_halo_cells_from_neighbor, num_conserved_variables), dtype=U.dtype)
        req = comm.Irecv(recv_buffer, source=neighbor_rank, tag=comm.Get_rank())
        recv_requests.append(req)
        recv_buffers[(neighbor_rank, tuple(local_indices_to_recv))] = recv_buffer

    # Wait for all receive operations to complete and update U
    MPI.Request.Waitall(recv_requests)
    for (neighbor_rank, local_indices_to_recv_tuple), recv_buffer in recv_buffers.items():
        U[list(local_indices_to_recv_tuple), :] = recv_buffer

    # Wait for all send operations to complete
    MPI.Request.Waitall(send_requests)


def solve(
    mesh: LocalMesh,
    U: np.ndarray,
    bc_dict: Dict[str, Any],
    equation: Any,
    comm: MPI.Comm,
    t_end: float,
    limiter_type: str = "barth_jespersen",
    flux_type: str = "roe",
    over_relaxation: float = 1.2,
    use_adaptive_dt: bool = True,
    cfl: float = 0.5,
    dt_initial: float = 0.01,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Main solver loop for the Finite Volume Method.

    This function orchestrates the time-stepping process for solving hyperbolic
    conservation laws. It supports first-order Euler and second-order Runge-Kutta
    (RK2) time integration schemes, along with adaptive time-stepping based on
    the Courant-Friedrichs-Lewy (CFL) condition.

    The spatial discretization is handled by the `compute_residual` function,
    which implements a MUSCL-Hancock scheme for second-order accuracy.

    Args:
        mesh (LocalMesh): The local mesh object for the current MPI process.
        U (np.ndarray): The initial conservative state vector for all cells
                        (owned and halo) on the current process.
        bc_dict (Dict[str, Any]): A dictionary defining the boundary conditions.
        equation (Any): The equation system to be solved (e.g., EulerEquations,
                        ShallowWaterEquations). Must implement `max_eigenvalue`,
                        `roe_flux`, `hllc_flux`, and `apply_boundary_condition`.
        comm (MPI.Comm): The MPI communicator for inter-process communication.
        t_end (float): The end time of the simulation.
        limiter_type (str, optional): The type of slope limiter for MUSCL reconstruction.
                                    Defaults to "barth_jespersen".
        flux_type (str, optional): The numerical flux function (Riemann solver).
                                 Defaults to "roe".
        over_relaxation (float, optional): Over-relaxation factor for gradient computation.
                                         Defaults to 1.2.
        use_adaptive_dt (bool, optional): Whether to use adaptive time stepping.
                                        Defaults to True.
        cfl (float, optional): The Courant-Friedrichs-Lewy (CFL) number for adaptive
                             time stepping. Defaults to 0.5.
        dt_initial (float, optional): The initial time step if not adaptive.
                                    Defaults to 0.01.

    Returns:
        Tuple[List[np.ndarray], List[float]]: A tuple containing:
            - history (List[np.ndarray]): A list of state vectors (U) at each
                                          saved time step.
            - dt_history (List[float]): A list of time steps (dt) used at each
                                        saved time step.
    """
    history: List[np.ndarray] = [U.copy()]
    t: float = 0.0
    n: int = 0
    dt_history: List[float] = []
    dt: float = dt_initial
    time_integration_method: str = "rk2"  # Currently hardcoded to RK2

    # formulate bc array
    bcs_array = create_numba_bcs(bc_dict, mesh.boundary_tag_map)

    while t < t_end:
        start_time = time.time()  # Start timing the loop

        # --- Adaptive Time-Stepping ---
        if use_adaptive_dt:
            dt = calculate_adaptive_dt(mesh, U, equation, cfl, comm)
            dt = min(dt, t_end - t)

        # --- Halo Exchange ---
        exchange_halo_data(mesh, U, comm)

        # --- Time Integration ---
        if time_integration_method == "rk2":
            # --- Second-Order Runge-Kutta (RK2) Method ---
            # Stage 1: Compute intermediate state U_star
            # U_star = U - dt * R(U)
            residual_U = compute_residual(
                mesh,
                U,
                equation,
                bcs_array,
                comm,
                limiter_type,
                flux_type,
                over_relaxation,
            )
            U_star = U - dt * residual_U

            # Halo exchange for U_star before computing residual_U_star
            exchange_halo_data(mesh, U_star, comm)

            # Stage 2: Compute final state U_new = 0.5 * (U + U_star - dt * R(U_star))
            residual_U_star = compute_residual(
                mesh,
                U_star,
                equation,
                bcs_array,
                comm,
                limiter_type,
                flux_type,
                over_relaxation,
            )
            U_new = 0.5 * (U + U_star - dt * residual_U_star)

        elif time_integration_method == "euler":
            # --- First-Order Euler Method ---
            # U_new = U - dt * R(U)
            residual = compute_residual(
                mesh,
                U,
                equation,
                bcs_array,
                comm, # BUG FIX: Pass comm to compute_residual for Euler method
                limiter_type,
                flux_type,
                over_relaxation,
            )
            U_new = U - dt * residual

        else:
            raise NotImplementedError(
                f"Time integration method '{time_integration_method}' is not supported."
            )

        # Update state and time
        U = U_new
        t += dt
        n += 1

        end_time = time.time()  # End timing the loop
        loop_time = end_time - start_time

        # Store history and print progress (only for rank 0 to avoid cluttered output)
        if comm.Get_rank() == 0:
            history.append(U.copy())
            dt_history.append(dt)
            print(
                f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s, Step: {n}, Loop Time: {loop_time:.4f}s"
            )

    return history, dt_history
