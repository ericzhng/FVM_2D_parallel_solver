import numpy as np


class PhysicsModel:
    """
    Base class for a physics model.

    This class defines the common interface required by the solver for any
    set of physical equations. Concrete implementations of this class must
    provide methods for flux calculations, wave speed estimation, and boundary
    condition application.
    """

    def max_eigenvalue(self, U: np.ndarray) -> float:
        """
        Calculates the maximum wave speed (eigenvalue) for a given cell state.
        This is crucial for determining the stable time step (CFL condition).

        Args:
            U (np.ndarray): The conservative state vector for a single cell.

        Returns:
            float: The maximum absolute eigenvalue.
        """
        raise NotImplementedError

    def apply_transmissive_bc(self, U_inside: np.ndarray) -> np.ndarray:
        """
        Applies a transmissive (zero-gradient) boundary condition.

        Args:
            U_inside (np.ndarray): Conservative state vector of the interior cell.

        Returns:
            np.ndarray: The conservative state vector of the ghost cell.
        """
        raise NotImplementedError

    def apply_inlet_bc(self, bc_value: np.ndarray) -> np.ndarray:
        """
        Applies an inlet boundary condition with specified primitive values.

        Args:
            bc_value (np.ndarray): The primitive variable values at the inlet.

        Returns:
            np.ndarray: The conservative state vector of the ghost cell.
        """
        raise NotImplementedError

    def apply_outlet_bc(self, U_inside: np.ndarray, bc_value: np.ndarray) -> np.ndarray:
        """
        Applies an outlet boundary condition, typically with a specified pressure or height.

        Args:
            U_inside (np.ndarray): Conservative state vector of the interior cell.
            bc_value (np.ndarray): The value(s) associated with the outlet condition.

        Returns:
            np.ndarray: The conservative state vector of the ghost cell.
        """
        raise NotImplementedError

    def apply_wall_bc(
        self, U_inside: np.ndarray, normal: np.ndarray, bc_value: np.ndarray
    ) -> np.ndarray:
        """
        Applies a wall boundary condition (slip or no-slip).

        Args:
            U_inside (np.ndarray): Conservative state vector of the interior cell.
            normal (np.ndarray): The outward-pointing normal vector of the boundary face.
            bc_value (np.ndarray): Value indicating slip (1.0) or no-slip (0.0).

        Returns:
            np.ndarray: The conservative state vector of the ghost cell.
        """
        raise NotImplementedError

    def hllc_flux(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Computes the numerical flux using the HLLC Riemann solver.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The HLLC numerical flux across the face.
        """
        raise NotImplementedError

    def roe_flux(self, U_L: np.ndarray, U_R: np.ndarray, normal: np.ndarray) -> np.ndarray:
        """
        Computes the numerical flux using the Roe approximate Riemann solver.

        Args:
            U_L (np.ndarray): Conservative state vector of the left cell.
            U_R (np.ndarray): Conservative state vector of the right cell.
            normal (np.ndarray): Normal vector of the face.

        Returns:
            np.ndarray: The Roe numerical flux across the face.
        """
        raise NotImplementedError
