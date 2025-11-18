import numpy as np
from mpi4py import MPI

from src.solver_options import SolverOptions
from src.physics_model import PhysicsModel


def calculate_adaptive_dt(
    mesh, U, equation: PhysicsModel, options: SolverOptions, comm
):
    """
    Calculates the adaptive time step for the simulation based on the CFL condition.

    The time step is limited by the maximum wave speed in the domain and the
    characteristic length of the cells to ensure numerical stability.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        equation (BaseEquation): The equation system to be solved.
        options (SolverOptions): Configuration options for the solver.
        comm (MPI.Comm): The MPI communicator.

    Returns:
        float: The calculated adaptive time step (dt).
    """
    local_min_dt = float("inf")

    # Calculate the maximum wave speed in the domain for the local mesh
    for i in range(mesh.num_owned_cells):
        value = np.sqrt(mesh.cell_volumes[i]) / equation.max_eigenvalue(U[i])
        local_min_dt = min(local_min_dt, value)

    # Perform a global allreduce to find the minimum dt across all processes
    global_min_dt = comm.allreduce(local_min_dt, op=MPI.MIN)

    return options.cfl * global_min_dt
