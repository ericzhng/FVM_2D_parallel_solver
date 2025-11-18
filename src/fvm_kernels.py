"""
This file contains the core numerical kernels for the FVM solver.
"""

import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.local_mesh import LocalMesh
from src.solver_options import SolverOptions
from src.physics_model import PhysicsModel
from src.fused_compute_residual import (
    compute_gradients_gaussian,
    compute_limiters,
    compute_fused_residual,
    LIMITERS,
)


def _compute_residual(
    mesh: LocalMesh,
    U: np.ndarray,
    equation: PhysicsModel,
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
