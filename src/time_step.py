import numpy as np


def calculate_adaptive_dt(mesh, U, equation, cfl_number):
    """
    Calculates the adaptive time step for the simulation based on the CFL condition.

    The time step is limited by the maximum wave speed in the domain and the
    characteristic length of the cells to ensure numerical stability.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        calculate_wave_speed (function): A function that calculates the maximum
                                       wave speed for a given state.
        cfl_number (float): The Courant-Friedrichs-Lewy (CFL) number.

    Returns:
        float: The calculated adaptive time step (dt).
    """
    max_wave_speed = 0.0

    min_value = float("inf")
    # Calculate the maximum wave speed in the domain
    for i in range(mesh.nelem):
        value = np.sqrt(mesh.cell_volumes[i]) / equation.max_eigenvalue(U[i])
        min_value = min(min_value, value)

    return cfl_number * min_value
