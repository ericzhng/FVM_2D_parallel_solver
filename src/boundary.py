import numpy as np
from numba import njit
from src.physics_model import PhysicsModel

# --- Numba-friendly integer constants for BC types ---
TRANSMISSIVE = 0
INLET = 1
OUTLET = 2
WALL = 3


class BoundaryConditions:
    """
    Manages boundary conditions by mapping boundary names from the mesh
    to the properties defined in the configuration file.
    """

    def __init__(self, bc_map):
        self.bc_map = bc_map

    @classmethod
    def from_config(cls, bc_config, boundary_patch_map):
        """
        Factory method to create a BoundaryConditions instance from a config dict.
        """
        bc_map = {}
        for name, props in bc_config.items():
            tag_id = boundary_patch_map.get(name.lower())
            if tag_id is not None:
                bc_map[tag_id] = props
        return cls(bc_map)

    def to_lookup_array(self, n_vars):
        """
        Converts the boundary conditions to a Numba-compatible lookup array.

        This creates a 2D numpy array where each row corresponds to a boundary tag.
        - Column 0: Integer code for the boundary condition type.
        - Columns 1 to n_vars+1: Values associated with the boundary condition.
        """
        max_tag = max(self.bc_map.keys()) if self.bc_map else 0
        # Shape: (max_tag + 1, 1 (for type) + n_vars (for values))
        lookup_array = np.zeros((max_tag + 1, 1 + n_vars))

        for tag_id, props in self.bc_map.items():
            bc_type_str = props.get("type", "transmissive")
            values = np.zeros(n_vars)

            if bc_type_str == "inlet":
                bc_type_int = INLET
                # For inlet, all primitive variables are specified
                if "rho" in props:
                    values[0] = props["rho"]
                if "u" in props:
                    values[1] = props["u"]
                if "v" in props:
                    values[2] = props["v"]
                if "p" in props and n_vars > 3:
                    values[3] = props["p"]
                if "h" in props:  # For shallow water
                    values[0] = props["h"]

            elif bc_type_str == "outlet":
                bc_type_int = OUTLET
                # For outlet, typically pressure (Euler) or height (Shallow Water) is specified
                if "p" in props and n_vars > 3:
                    values[3] = props["p"]
                if "h" in props:
                    values[0] = props["h"]

            elif bc_type_str == "wall":
                bc_type_int = WALL
                # Value indicates slip (1.0) or no-slip (0.0)
                values[0] = 1.0 if props.get("slip", True) else 0.0

            else:  # transmissive
                bc_type_int = TRANSMISSIVE

            lookup_array[tag_id, 0] = bc_type_int
            lookup_array[tag_id, 1 : 1 + n_vars] = values

        return lookup_array


@njit
def apply_boundary_condition(
    U_inside: np.ndarray,
    normal: np.ndarray,
    bc_type: int,
    bc_value: np.ndarray,
    equation: PhysicsModel,
) -> np.ndarray:
    """
    Applies a boundary condition to determine the state of a ghost cell.

    This function acts as a dispatcher, calling the appropriate method on the
    physics model based on the integer boundary condition type.

    Args:
        U_inside (np.ndarray): Conservative state vector of the interior cell.
        normal (np.ndarray): The outward-pointing normal vector of the boundary face.
        bc_type (int): The integer code for the boundary condition type.
        bc_value (np.ndarray): The value(s) associated with the boundary condition.
        equation (PhysicsModel): The physics model object.

    Returns:
        np.ndarray: The conservative state vector of the ghost cell.
    """
    if bc_type == TRANSMISSIVE:
        return equation.apply_transmissive_bc(U_inside)
    elif bc_type == INLET:
        return equation.apply_inlet_bc(bc_value)
    elif bc_type == OUTLET:
        return equation.apply_outlet_bc(U_inside, bc_value)
    elif bc_type == WALL:
        return equation.apply_wall_bc(U_inside, normal, bc_value)
    else:
        # Default to transmissive for any unknown boundary condition types
        return equation.apply_transmissive_bc(U_inside)
