"""
Main driver script for the 2D Finite Volume Method (FVM) solver for Euler equations.

This script orchestrates the entire simulation process in a parallel environment using MPI.
The process includes:
1.  Setting up the MPI environment and command-line arguments.
2.  Loading and partitioning the computational mesh on the root process.
3.  Distributing the partitioned mesh to all worker processes.
4.  Setting up the initial and boundary conditions for the simulation case.
5.  Running the FVM solver to advance the solution in time.
6.  Gathering results from all processes and visualizing the final state.
"""

# --- Standard Library Imports ---
import os
import argparse
import logging

# --- Third-Party Imports ---
import numpy as np
from mpi4py import MPI
import debugpy

# --- Local Application Imports ---
from fvm_mesh.polymesh import CoreMesh, LocalMesh
from fvm_mesh.polymesh import create_local_meshes, partition_mesh
from src.case_setup import setup_case_euler
from src.euler_equations import EulerEquations
from src.visualization import plot_simulation_step, create_animation
from src.solver import solve
from src.solver_options import SolverOptions
from src.boundary import BoundaryConditions

# --- Logger Setup ---
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_mesh_and_scatter(comm):
    """
    Handles mesh loading, partitioning, and distribution on rank 0.

    Args:
        comm (MPI.Comm): The MPI communicator.

    Returns:
        tuple: A tuple containing:
            - global_mesh (CoreMesh or None): The full mesh object on rank 0, None otherwise.
            - mesh (LocalMesh): The local mesh partition for the current rank.
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    global_mesh = None
    local_meshes_list = None

    if rank == 0:
        logger.info("Initializing and reading mesh on rank 0...")
        global_mesh = CoreMesh()
        global_mesh.read_gmsh("data/euler_mesh.msh")
        global_mesh.analyze_mesh()

        logger.info(f"Partitioning mesh into {size} parts...")
        parts = partition_mesh(global_mesh, n_parts=size, method="metis")
        global_mesh.plot(
            "results/mesh_global.png", parts, show_nodes=False, show_cells=False
        )

        local_meshes_list = create_local_meshes(global_mesh, n_parts=size)

    # Distribute the mesh to all processes
    logger.info(f"Rank {rank}: Receiving scattered local mesh...")
    mesh = comm.scatter(local_meshes_list, root=0)
    mesh.plot(f"results/mesh_rank_{rank}.png", show_nodes=False, show_cells=False)
    comm.Barrier()

    return global_mesh, mesh


def reconstruct_and_visualize(
    global_mesh: CoreMesh, mesh: LocalMesh, history, dt_history, comm
):
    """
    Gathers simulation data from all processes and performs visualization on rank 0.

    Args:
        global_mesh (CoreMesh): The complete mesh, required on rank 0.
        mesh (LocalMesh): The local mesh object for the current rank.
        history (list): A list of the conservative state vectors over time for the local mesh.
        dt_history (list): A list of the time steps (dt) used in the simulation.
        comm (MPI.Comm): The MPI communicator.
    """
    rank = comm.Get_rank()

    # Gather all histories and local meshes on rank 0
    logger.info(f"Rank {rank}: Gathering results for visualization.")
    all_histories = comm.gather(history, root=0)
    all_local_meshes = comm.gather(mesh, root=0)

    if rank == 0:
        logger.info("Reconstructing global data and visualizing on rank 0...")
        if not all_histories or not all_local_meshes:
            logger.warning("No data received on rank 0 for visualization.")
            return

        # --- Reconstruct Global Data ---
        num_time_steps = len(all_histories[0])
        num_vars = all_histories[0][0].shape[1]
        num_global_cells = len(global_mesh.cell_centroids)
        global_history = []

        for t in range(num_time_steps):
            U_global = np.zeros((num_global_cells, num_vars))
            for i, (rank_history, local_mesh) in enumerate(
                zip(all_histories, all_local_meshes)
            ):
                local_U = rank_history[t]
                if hasattr(local_mesh, "l2g_cells"):
                    global_indices = local_mesh.l2g_cells[: local_mesh.num_owned_cells]
                    U_global[global_indices] = local_U[: local_mesh.num_owned_cells]
                else:
                    logger.warning(
                        "`local_mesh.l2g_cells` not found. "
                        "Visualization may be incorrect."
                    )
            global_history.append(U_global)

        # --- Visualize ---
        logger.info("Creating animation of the results...")
        create_animation(global_mesh, global_history, dt_history, variable_to_plot=0)

        logger.info("Plotting final state...")
        for k in range(num_vars):
            plot_simulation_step(
                global_mesh,
                global_history[-1],
                "Final State",
                variable_to_plot=k,
                output_dir="results",
            )


def run_fvm_solver(comm, global_mesh, mesh):
    """
    Sets up and runs the FVM solver for the given mesh and conditions.

    Args:
        comm (MPI.Comm): The MPI communicator.
        global_mesh (CoreMesh or None): The full mesh object, used by rank 0 for visualization.
        mesh (LocalMesh): The local mesh partition for the current rank.
    """
    rank = comm.Get_rank()
    logger.info(f"Rank {rank}: Setting up the simulation case...")

    # --- Simulation Parameters ---
    gamma = 1.4
    t_end = 0.25

    # --- Case Setup ---
    U_init, bc_dict = setup_case_euler(mesh, comm, gamma=gamma)
    boundary_conditions = BoundaryConditions(bc_dict, mesh.boundary_tag_map)
    bcs_lookup = boundary_conditions.to_lookup_array()
    equation = EulerEquations(gamma=gamma)

    # --- Solver Configuration ---
    solver_opts = SolverOptions(
        limiter_type="minmod",  # Options: 'barth_jespersen', 'minmod', 'superbee'
        flux_type="roe",
        over_relaxation=1.0,
        cfl=0.5,
        use_adaptive_dt=True,
        dt_initial=1e-2,
    )

    # --- Solve ---
    logger.info(f"Rank {rank}: Starting the FVM solver...")
    history, dt_history = solve(
        equation, mesh, U_init, bcs_lookup, t_end, comm, options=solver_opts
    )
    logger.info(f"Rank {rank}: Solver finished.")

    # --- Post-process and Visualize ---
    reconstruct_and_visualize(global_mesh, mesh, history, dt_history, comm)


def main():
    """
    Main driver function to parse arguments and run the solver.
    """
    parser = argparse.ArgumentParser(description="2D FVM Solver for Euler Equations")
    parser.add_argument(
        "--mpi-debug", action="store_true", help="Enable MPI debugging with debugpy."
    )
    args, _ = parser.parse_known_args()

    # --- Initialize MPI ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # --- Conditional MPI Debugging ---
    if args.mpi_debug:
        logger.info(f"Rank {rank}: PID {os.getpid()}")
        port = 5678 + rank
        debugpy.listen(("localhost", port))
        logger.info(f"Rank {rank}: Waiting for debugger on port {port}...")
        debugpy.wait_for_client()
        debugpy.breakpoint()
        logger.info(f"Rank {rank}: Debugger attached!")
        comm.Barrier()

    # --- Run Simulation ---
    global_mesh, local_mesh = setup_mesh_and_scatter(comm)
    run_fvm_solver(comm, global_mesh, local_mesh)

    if rank == 0:
        logger.info("Simulation complete.")


if __name__ == "__main__":
    main()
