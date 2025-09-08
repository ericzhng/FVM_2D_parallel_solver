import numpy as np

# --- Sample BC Dictionary-based Definition ---
# bc_dict: Dict[str, Dict[str, Union[str, float, bool]]] = {
#     "inlet_channel": {"type": "inlet", "velocity": [1.5, 0.0]},
#     "outlet_weir": {"type": "outlet", "pressure": 0.0},
#     "downstream_boundary": {"type": "outlet", "water_depth": 0.5},
#     "solid_walls": {"type": "wall", "slip": False},
#     "open_boundary": {"type": "transmissive"},
# }


# Mapping for Boundary Condition types
BC_TYPE_MAP = {
    "inlet": 1,
    "outlet": 2,
    "wall": 3,
    "transmissive": 4,
}


def create_numba_bcs(
    bc_dict,
    boundary_tag_map,
):
    """
    Converts the human-readable dictionary into a NumPy structured array
    that can be passed to Numba.
    """
    # Define the structure (dtype) of our array.
    # We use generic 'param' fields. The meaning of each param depends on the 'bc_type'.
    # For example, for an inlet, param1=vx, param2=vy, param3=vz.
    # For an outlet, param1=pressure, param2=water_depth.
    # For a wall, param1=slip (0.0 for False, 1.0 for True).
    bc_dtype = np.dtype(
        [
            ("tag_id", np.int32),
            ("bc_type", np.int32),
            ("vars", (np.float64, 3)),
        ]
    )

    bc_list = []
    for tag_name, props in bc_dict.items():
        tag_id = boundary_tag_map.get(tag_name, -1)

        bc_type_str = props.get("type", "wall")
        bc_type_id = BC_TYPE_MAP.get(bc_type_str, 3)

        # Initialize params to a default (e.g., NaN or 0)
        vars_val = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Populate params based on BC type
        if bc_type_id == 1:  # inlet
            velocity = props.get("velocity", [0.0, 0.0, 0.0])
            vars_val = np.array(velocity, dtype=np.float64)
        elif bc_type_id == 2:  # outlet
            vars_val[0] = props.get("pressure", 0.0)
            # h = props.get("water_depth", 0.0)
        elif bc_type_id == 3:  # Wall
            vars_val[0] = 1.0 if props.get("slip", False) else 0.0

        # For transmissive, params can remain 0 as it often needs no values.
        bc_list.append((tag_id, bc_type_id, vars_val))

    return np.array(bc_list, dtype=bc_dtype)
