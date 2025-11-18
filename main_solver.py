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
import yaml

# --- Third-Party Imports ---
from mpi4py import MPI
import debugpy

# --- Local Application Imports ---
from fvm_mesh.polymesh import PolyMesh, MeshPartitionManager, partition_mesh

from src.simulation import Simulation
from src.solver_options import SolverOptions

from src.case_setup import EulerRiemannCase
from src.case_setup import ShallowWaterRiemannCase

from src.equation_euler import EulerEquations
from src.equation_shallow_water import ShallowWaterEquations


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
        global_mesh = PolyMesh().from_gmsh("data/euler_mesh.msh")
        global_mesh.analyze_mesh()

        logger.info(f"Partitioning mesh into {size} parts...")
        parts = partition_mesh(global_mesh, size, method="metis")
        local_meshes_list = MeshPartitionManager.create_local_meshes(
            global_mesh, cell_partitions=parts
        )
        global_mesh.plot(
            "results/mesh_global.png", parts, show_nodes=False, show_cells=False
        )

    # Distribute the mesh to all processes
    logger.info(f"Rank {rank}: Receiving scattered local mesh...")
    mesh = comm.scatter(local_meshes_list, root=0)
    mesh.plot(f"results/mesh_rank_{rank}.png", show_nodes=False, show_cells=False)

    return global_mesh, mesh


def main():
    """
    Main driver function to parse arguments and run the solver.
    """
    parser = argparse.ArgumentParser(description="2D FVM Solver for Euler Equations")
    parser.add_argument(
        "--mpi-debug", action="store_true", help="Enable MPI debugging with debugpy."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to the configuration file.",
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

    # --- Load Configuration ---
    options = {}
    if rank == 0:
        logger.info(f"Loading configuration from {args.config}...")
        with open(args.config, "r") as f:
            config_dict = yaml.safe_load(f)
            options = SolverOptions.from_config(config_dict)
    options = comm.bcast(options, root=0)

    # --- Setup and Run Simulation ---
    global_mesh, local_mesh = setup_mesh_and_scatter(comm)

    equation = EulerEquations(gamma=options.gamma)

    U0, bcs_lookup = EulerRiemannCase.setup_case(local_mesh, options.gamma)

    simulation = Simulation(comm, equation, U0, bcs_lookup, options)

    simulation.run(global_mesh, local_mesh)

    if rank == 0:
        logger.info("Simulation complete.")


if __name__ == "__main__":
    main()
