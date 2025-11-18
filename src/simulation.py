"""
Defines the main Simulation class to orchestrate the FVM solver setup and execution.
"""

import logging
from mpi4py import MPI
import numpy as np

from fvm_mesh.polymesh import PolyMesh, LocalMesh
from src.physics_model import PhysicsModel
from src.solver_options import SolverOptions
from src.solver import solve
from src.visualization import reconstruct_and_visualize

logger = logging.getLogger(__name__)


class Simulation:
    """
    Orchestrates the setup, execution, and post-processing of a FVM simulation.
    """

    def __init__(
        self,
        comm: MPI.Comm,
        equation: PhysicsModel,
        U0: np.ndarray,
        bcs_lookup: np.ndarray,
        options: SolverOptions,
    ):
        self.comm = comm
        self.rank = comm.Get_rank()
        self.size = comm.Get_size()
        self.U0 = U0
        self.bcs_lookup = bcs_lookup
        self.equation = equation
        self.options = options

    def run(self, global_mesh: PolyMesh | None, mesh: LocalMesh):
        """
        Sets up and runs the FVM solver for the given mesh and conditions.

        Args:
            global_mesh (PolyMesh or None): The full mesh object, used by rank 0 for visualization.
            mesh (LocalMesh): The local mesh partition for the current rank.
        """
        # logger.info(f"Rank {self.rank}: Setting up the simulation case...")

        # --- Solve ---
        logger.info(f"Rank {self.rank}: Starting the FVM solver...")
        history, dt_history = solve(
            self.equation,
            mesh,
            self.U0,
            self.bcs_lookup,
            self.comm,
            options=self.options,
        )
        logger.info(f"Rank {self.rank}: Solver finished.")

        # --- Post-process and Visualize ---
        reconstruct_and_visualize(global_mesh, mesh, history, dt_history, self.comm)
