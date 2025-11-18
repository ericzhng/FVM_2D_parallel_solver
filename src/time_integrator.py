"""
Defines time integration schemes for the FVM solver.
"""
from abc import ABC, abstractmethod
import numpy as np

from fvm_mesh.polymesh.local_mesh import LocalMesh
from src.solver_options import SolverOptions
from src.physics_model import PhysicsModel
from src.fvm_kernels import _compute_residual, exchange_halo_data

class TimeIntegrator(ABC):
    """
    Abstract base class for time integration schemes.
    """

    def __init__(self, equation: PhysicsModel, options: SolverOptions):
        self.equation = equation
        self.options = options

    @abstractmethod
    def step(self, mesh: LocalMesh, U: np.ndarray, dt: float, bcs_lookup, comm) -> np.ndarray:
        """
        Advances the solution by one time step.

        Args:
            mesh (LocalMesh): The local mesh partition.
            U (np.ndarray): The current solution state.
            dt (float): The time step size.
            bcs_lookup: Boundary conditions lookup.
            comm: MPI communicator.

        Returns:
            np.ndarray: The new solution state.
        """
        pass

class ExplicitEuler(TimeIntegrator):
    """
    Explicit Euler time integration scheme.
    """

    def step(self, mesh: LocalMesh, U: np.ndarray, dt: float, bcs_lookup, comm) -> np.ndarray:
        """
        Advances the solution by one time step using Explicit Euler.
        """
        residual = _compute_residual(mesh, U, self.equation, bcs_lookup, self.options)
        U_new = U.copy()
        U_new[: mesh.num_owned_cells] -= dt * residual
        return U_new

class MultiStageRungeKutta(TimeIntegrator):
    """
    Multi-stage Runge-Kutta time integration scheme (rk2).
    """

    def step(self, mesh: LocalMesh, U: np.ndarray, dt: float, bcs_lookup, comm) -> np.ndarray:
        """
        Advances the solution by one time step using a 2-stage Runge-Kutta method.
        """
        # Stage 1
        residual_U = _compute_residual(mesh, U, self.equation, bcs_lookup, self.options)
        U_star = U.copy()
        U_star[: mesh.num_owned_cells] -= dt * residual_U

        exchange_halo_data(mesh, U_star, comm)

        # Stage 2
        residual_U_star = _compute_residual(mesh, U_star, self.equation, bcs_lookup, self.options)
        U_new = U.copy()
        U_new[: mesh.num_owned_cells] = 0.5 * (
            U[: mesh.num_owned_cells]
            + U_star[: mesh.num_owned_cells]
            - dt * residual_U_star
        )
        return U_new
