import numpy as np


class BoundaryConditions:
    """
    Manages boundary conditions for a simulation.
    """

    def __init__(self, bc_dict, boundary_patch_map):
        self.bc_dict = bc_dict
        self.boundary_patch_map = boundary_patch_map
        self.bc_map = self._create_bc_map()

    def _create_bc_map(self):
        """
        Creates a mapping from boundary tag ID to boundary condition properties.
        """
        bc_map = {}
        for name, props in self.bc_dict.items():
            tag_id = self.boundary_patch_map.get(name.lower())
            if tag_id is not None:
                bc_map[tag_id] = props
        return bc_map

    def to_lookup_array(self):
        """
        Converts the boundary conditions to a Numba-compatible lookup array.

        This method creates a structured numpy array that can be easily accessed
        within Numba-jitted functions. Each element in the array corresponds to a
        boundary tag and contains the boundary condition type and associated values.

        The supported boundary condition types and their values are:
        - 'inlet': Specifies fixed values for density, velocity components, and pressure.
            - values[0]: rho (density)
            - values[1]: u (x-velocity)
            - values[2]: v (y-velocity)
            - values[3]: p (pressure)
        - 'outlet': Specifies a fixed pressure at the outlet. Other variables are
          typically extrapolated from the interior.
            - values[3]: p (pressure)
        - 'wall': Represents a solid wall. Can be slip or no-slip. For Euler equations,
          slip walls are typical.
            - values[0]: 1.0 for slip wall, 0.0 for no-slip wall.
        - 'transmissive': A non-reflective boundary condition where flow properties
          are extrapolated from the interior. No specific values are needed.
        """
        max_tag = 0
        if self.boundary_patch_map:
            max_tag = max(self.boundary_patch_map.values())

        # Define a flexible dtype for the structured array
        bc_data_dtype = np.dtype(
            [
                ("type", "U20"),  # String for type
                ("values", (np.float64, 4)),  # Array of 4 floats for values
            ]
        )

        bcs_lookup = np.empty(max_tag + 1, dtype=bc_data_dtype)

        for i in range(max_tag + 1):
            bc_props = self.bc_map.get(i)
            if bc_props:
                bc_type = bc_props.get("type", "transmissive")
                values = np.zeros(4)
                if bc_type == "inlet":
                    values[0] = bc_props.get("rho", 1.0)
                    values[1] = bc_props.get("u", 0.0)
                    values[2] = bc_props.get("v", 0.0)
                    values[3] = bc_props.get("p", 1.0)
                elif bc_type == "outlet":
                    # For an outlet, pressure is typically specified
                    values[3] = bc_props.get("p", 1.0)
                elif bc_type == "wall":
                    # For an Euler solver, slip walls are the default
                    values[0] = 1.0 if bc_props.get("slip", True) else 0.0
                elif bc_type == "transmissive":
                    # No specific values needed, properties are extrapolated
                    pass

                bcs_lookup[i] = (bc_type, values)
            else:
                # Default to transmissive if a tag is not in the bc_dict
                bcs_lookup[i] = ("transmissive", np.zeros(4))

        return bcs_lookup
