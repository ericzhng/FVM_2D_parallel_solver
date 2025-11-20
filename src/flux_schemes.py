from abc import ABC, abstractmethod
import numpy as np
from src.physics_model import PhysicsModel


class FluxScheme(ABC):
    """
    Abstract base class for numerical flux schemes.
    """

    @abstractmethod
    def calculate_numerical_flux(
        self,
        U_L: np.ndarray,
        U_R: np.ndarray,
        face_normal: np.ndarray,
        equation: PhysicsModel,
    ) -> np.ndarray:
        """
        Calculates the numerical flux across a face.

        Args:
            U_L (np.ndarray): Left state conservative variables.
            U_R (np.ndarray): Right state conservative variables.
            face_normal (np.ndarray): Normal vector of the face.
            equation (PhysicsModel): The physics model providing physical flux and Riemann solvers.

        Returns:
            np.ndarray: The numerical flux vector.
        """
        pass


class RiemannSolverFlux(FluxScheme):
    """
    Implements numerical flux calculation using Riemann solvers (e.g., Roe, HLLC).
    """

    def __init__(self, flux_type: str):
        if flux_type not in ["roe", "hllc"]:
            raise ValueError(
                f"Unsupported flux type: {flux_type}. Must be 'roe' or 'hllc'."
            )
        self.flux_type = flux_type

    def calculate_numerical_flux(
        self,
        U_L: np.ndarray,
        U_R: np.ndarray,
        face_normal: np.ndarray,
        equation: PhysicsModel,
    ) -> np.ndarray:
        if self.flux_type == "roe":
            return equation.roe_flux(U_L, U_R, face_normal)
        elif self.flux_type == "hllc":
            return equation.hllc_flux(U_L, U_R, face_normal)
        else:
            raise ValueError(f"Unsupported flux type: {self.flux_type}")


class CentralDifferenceFlux(FluxScheme):
    """
    Implements a simple central difference flux scheme.
    This scheme is typically used with limiters applied to the reconstructed states.
    """

    def calculate_numerical_flux(
        self,
        U_L: np.ndarray,
        U_R: np.ndarray,
        face_normal: np.ndarray,
        equation: PhysicsModel,
    ) -> np.ndarray:
        # For central difference, we average the physical fluxes from the left and right states.
        # The limiters are assumed to have been applied during the reconstruction of U_L and U_R.
        F_L = equation._compute_flux(U_L, face_normal)
        F_R = equation._compute_flux(U_R, face_normal)
        return 0.5 * (F_L + F_R)
