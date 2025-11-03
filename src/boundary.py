
import numpy as np

class BoundaryConditions:
    """
    Manages boundary conditions for a simulation.
    """
    def __init__(self, bc_dict, boundary_tag_map):
        self.bc_dict = bc_dict
        self.boundary_tag_map = boundary_tag_map
        self.bc_map = self._create_bc_map()

    def _create_bc_map(self):
        """
        Creates a mapping from boundary tag ID to boundary condition properties.
        """
        bc_map = {}
        for name, props in self.bc_dict.items():
            tag_id = self.boundary_tag_map.get(name)
            if tag_id is not None:
                bc_map[tag_id] = props
        return bc_map

    def to_lookup_array(self):
        """
        Converts the boundary conditions to a Numba-compatible lookup array.
        """
        max_tag = 0
        if self.boundary_tag_map:
            max_tag = max(self.boundary_tag_map.values())

        # Define a flexible dtype for the structured array
        bc_data_dtype = np.dtype([
            ('type', 'U20'),  # String for type
            ('values', (np.float64, 4))  # Array of 4 floats for values
        ])

        bcs_lookup = np.empty(max_tag + 1, dtype=bc_data_dtype)

        for i in range(max_tag + 1):
            bc_props = self.bc_map.get(i)
            if bc_props:
                bc_type = bc_props.get('type', 'transmissive')
                values = np.zeros(4)
                if bc_type == 'supersonic_inlet':
                    values[0] = bc_props.get('rho', 1.0)
                    values[1] = bc_props.get('u', 0.0)
                    values[2] = bc_props.get('v', 0.0)
                    values[3] = bc_props.get('p', 1.0)
                elif bc_type == 'wall':
                    values[0] = 1.0 if bc_props.get('slip', False) else 0.0

                bcs_lookup[i] = (bc_type, values)
            else:
                # Default to transmissive if a tag is not in the bc_dict
                bcs_lookup[i] = ('transmissive', np.zeros(4))

        return bcs_lookup
