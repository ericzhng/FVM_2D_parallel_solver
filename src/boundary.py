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
    "supersonic_inlet": 5,  # New type for supersonic inlet
    "slip_wall": 6,  # New type for slip wall
}


def create_bcs_lookup(
    bc_dict,
    boundary_tag_map,
):
    """
    Converts the human-readable dictionary into a NumPy structured array
    that can be used as a lookup table in Numba.
    The index of the array corresponds to the boundary tag ID.
    """
    # Define the structure (dtype) of our array.
    bc_data_dtype = np.dtype(
        [
            ("bc_type", np.int32),
            ("vars", (np.float64, 4)),
        ]
    )

    max_tag = 0
    if boundary_tag_map and boundary_tag_map.values():
        max_tag = max(boundary_tag_map.values())

    # Create lookup table, initialize with a default BC (e.g., wall)
    default_bc_type = BC_TYPE_MAP.get("wall", 3)
    default_vars = np.zeros(4, dtype=np.float64)

    bcs_lookup = np.empty(max_tag + 1, dtype=bc_data_dtype)
    for i in range(max_tag + 1):
        bcs_lookup[i] = (default_bc_type, default_vars)

    for tag_name, props in bc_dict.items():
        tag_id = boundary_tag_map.get(tag_name, -1)
        if tag_id == -1:
            continue

        bc_type_str = props.get("type", "wall")
        bc_type_id = BC_TYPE_MAP.get(bc_type_str, 3)

        vars_val = np.zeros(4, dtype=np.float64)

        # Populate params based on BC type
        if bc_type_id == BC_TYPE_MAP["inlet"]:  # inlet
            velocity = props.get("velocity", [0.0, 0.0, 0.0])
            vars_val[0:3] = np.array(velocity, dtype=np.float64)
        elif bc_type_id == BC_TYPE_MAP["outlet"]:  # outlet
            vars_val[0] = props.get("pressure", 0.0)
        elif bc_type_id == BC_TYPE_MAP["wall"]:  # Wall
            vars_val[0] = 1.0 if props.get("slip", False) else 0.0
        elif bc_type_id == BC_TYPE_MAP["supersonic_inlet"]:  # Supersonic Inlet
            vars_val[0] = props.get("rho", 1.0)
            vars_val[1] = props.get("u", 0.0)
            vars_val[2] = props.get("v", 0.0)
            vars_val[3] = props.get("p", 1.0)
        elif bc_type_id == BC_TYPE_MAP["slip_wall"]:  # Slip Wall
            vars_val[0] = 1.0  # Indicate slip wall

        if tag_id <= max_tag:
            bcs_lookup[tag_id] = (bc_type_id, vars_val)

    return bcs_lookup