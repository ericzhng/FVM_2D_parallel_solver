from typing import Dict, Union
import numpy as np
from fvm_mesh.polymesh.poly_mesh import PolyMesh


def setup_case_euler(mesh: PolyMesh, comm, gamma=1.4):
    """
    Sets up the initial conditions for a 2D Riemann problem for the Euler equations.
    The domain is split into four quadrants, each with a different initial state,
    based on the cell center coordinates (x, y).
    """
    # Define primitive variables for the four regions (p, rho, u, v)
    # p = [ 1.0,   1.0,   1.0,   1.0 ];
    # r = [ 1.0,   2.0,   1.0,   3.0 ];
    # u = [-0.75, -0.75,  0.75,  0.75];
    # v = [-0.5,   0.5,   0.5,  -0.5 ];
    p_vals = np.array([1.0, 1.0, 1.0, 1.0])
    rho_vals = np.array([1.0, 2.0, 1.0, 3.0])
    u_vals = np.array([-0.75, -0.75, 0.75, 0.75])
    v_vals = np.array([-0.5, 0.5, 0.5, -0.5])

    # Get cell centroid coordinates
    x = mesh.cell_centroids[:, 0]
    y = mesh.cell_centroids[:, 1]

    # Create boolean masks for each quadrant
    reg1 = (x >= 0.5) & (y >= 0.5)  # Top-right
    reg2 = (x < 0.5) & (y >= 0.5)  # Top-left
    reg3 = (x < 0.5) & (y < 0.5)  # Bottom-left
    reg4 = (x >= 0.5) & (y < 0.5)  # Bottom-right

    # Use masks to set initial conditions for all cells in a vectorized way
    rho = (
        rho_vals[0] * reg1
        + rho_vals[1] * reg2
        + rho_vals[2] * reg3
        + rho_vals[3] * reg4
    )
    u = u_vals[0] * reg1 + u_vals[1] * reg2 + u_vals[2] * reg3 + u_vals[3] * reg4
    v = v_vals[0] * reg1 + v_vals[1] * reg2 + v_vals[2] * reg3 + v_vals[3] * reg4
    p = p_vals[0] * reg1 + p_vals[1] * reg2 + p_vals[2] * reg3 + p_vals[3] * reg4

    # Calculate total energy per unit volume (E)
    # E = p / (gamma - 1) + 0.5 * rho * (u^2 + v^2)
    energy = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2)

    # # Calculate speed of sound (c) for each cell
    # c0 = np.sqrt(gamma * p / rho)

    # Assemble the state vector U = [rho, rho*u, rho*v, E]
    U0 = np.vstack([rho, rho * u, rho * v, energy]).T

    # Define boundary conditions - using transmissive (outlet) for all boundaries
    #   common for Riemann problem
    BCs: Dict[str, Dict[str, Union[str, float, bool]]] = {
        "top": {"type": "transmissive"},
        "bottom": {"type": "transmissive"},
        "left": {"type": "transmissive"},
        "right": {"type": "transmissive"},
    }

    return U0, BCs
