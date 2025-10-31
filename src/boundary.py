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
    "slip_wall": 6,         # New type for slip wall
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
    # Increased vars to 4 to accommodate rho, u, v, p for supersonic inlet.
    bc_dtype = np.dtype(
        [
            ("tag_id", np.int32),
            ("bc_type", np.int32),
            ("vars", (np.float64, 4)),  # Changed to 4 variables
        ]
    )

    bc_list = []
    for tag_name, props in bc_dict.items():
        tag_id = boundary_tag_map.get(tag_name, -1)

        bc_type_str = props.get("type", "wall")
        bc_type_id = BC_TYPE_MAP.get(bc_type_str, 3)

        # Initialize params to a default (e.g., NaN or 0)
        vars_val = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64) # Changed to 4 variables

        # Populate params based on BC type
        if bc_type_id == BC_TYPE_MAP["inlet"]:  # inlet
            velocity = props.get("velocity", [0.0, 0.0, 0.0])
            vars_val[0:3] = np.array(velocity, dtype=np.float64)
        elif bc_type_id == BC_TYPE_MAP["outlet"]:  # outlet
            vars_val[0] = props.get("pressure", 0.0)
        elif bc_type_id == BC_TYPE_MAP["wall"]:  # Wall
            vars_val[0] = 1.0 if props.get("slip", False) else 0.0
        elif bc_type_id == BC_TYPE_MAP["supersonic_inlet"]: # Supersonic Inlet
            vars_val[0] = props.get("rho", 1.0)
            vars_val[1] = props.get("u", 0.0)
            vars_val[2] = props.get("v", 0.0)
            vars_val[3] = props.get("p", 1.0)
        elif bc_type_id == BC_TYPE_MAP["slip_wall"]: # Slip Wall
            vars_val[0] = 1.0 # Indicate slip wall

        # For transmissive, params can remain 0 as it often needs no values.
        bc_list.append((tag_id, bc_type_id, vars_val))

    return np.array(bc_list, dtype=bc_dtype)
