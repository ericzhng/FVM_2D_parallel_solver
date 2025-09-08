import time
import numpy as np

from src.mesh import Mesh
from src.time_step import calculate_adaptive_dt
from src.visualization import plot_simulation_step
from src.reconstruction import compute_residual
from src.euler_equations import EulerEquations
from src.shallow_water_equations import ShallowWaterEquations
from src.boundary import create_numba_bcs


def solve(
    mesh: Mesh,
    U,
    bc_dict,
    equation,
    t_end,
    limiter_type="barth_jespersen",
    flux_type="roe",
    over_relaxation=1.2,
    use_adaptive_dt=True,
    cfl=0.5,
    dt_initial=0.01,
):
    """
    Main solver loop for the Finite Volume Method.

    This function orchestrates the time-stepping process for solving hyperbolic
    conservation laws. It supports first-order Euler and second-order Runge-Kutta
    (RK2) time integration schemes, along with adaptive time-stepping based on
    the Courant-Friedrichs-Lewy (CFL) condition.

    The spatial discretization is handled by the `compute_residual` function,
    which implements a MUSCL-Hancock scheme for second-order accuracy.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The initial conservative state vector.
        boundary_conditions (dict): A dictionary defining the boundary conditions.
        equation (BaseEquation): The equation system to be solved (e.g., EulerEquations).
        t_end (float): The end time of the simulation.
        limiter_type (str, optional): The type of slope limiter for MUSCL reconstruction.
                                    Defaults to "barth_jespersen".
        flux_type (str, optional): The numerical flux function (Riemann solver).
                                 Defaults to "roe".
        over_relaxation (float, optional): Over-relaxation factor for gradient computation.
                                         Defaults to 1.2.
        use_adaptive_dt (bool, optional): Whether to use adaptive time stepping.
                                        Defaults to True.
        cfl (float, optional): The CFL number for adaptive time stepping.
                             Defaults to 0.5.
        dt_initial (float, optional): The initial time step if not adaptive.
                                    Defaults to 0.01.
        variable_to_plot (int, optional): The index of the variable to plot during simulation.
                                        Defaults to 0.

    Returns:
        tuple: A tuple containing the history of the state vector and the history
               of the time steps.
    """
    history = [U.copy()]
    t = 0.0
    n = 0
    dt_history = []
    dt = dt_initial
    time_integration_method = "rk2"

    # formulate bc array
    bcs_array = create_numba_bcs(bc_dict, mesh.boundary_tag_map)

    while t < t_end:
        start_time = time.time()  # Start timing the loop

        # --- Adaptive Time-Stepping ---
        if use_adaptive_dt:
            dt = calculate_adaptive_dt(mesh, U, equation, cfl)
            dt = min(dt, t_end - t)

        # --- Time Integration ---
        if time_integration_method == "rk2":
            # --- Second-Order Runge-Kutta (RK2) Method ---
            # Stage 1: Compute intermediate state U_star
            # U_star = U - dt * R(U)
            residual_U = compute_residual(
                mesh,
                U,
                equation,
                bcs_array,
                limiter_type,
                flux_type,
                over_relaxation,
            )
            U_star = U - dt * residual_U

            # Stage 2: Compute final state U_new using U_star
            # U_new = 0.5 * (U + U_star - dt * R(U_star))
            residual_U_star = compute_residual(
                mesh,
                U_star,
                equation,
                bcs_array,
                limiter_type,
                flux_type,
                over_relaxation,
            )
            U_new = 0.5 * (U + U_star - dt * residual_U_star)

        elif time_integration_method == "euler":
            # --- First-Order Euler Method ---
            # U_new = U - dt * R(U)
            residual = compute_residual(
                mesh,
                U,
                equation,
                bcs_array,
                limiter_type,
                flux_type,
                over_relaxation,
            )
            U_new = U - dt * residual

        else:
            raise NotImplementedError(
                f"Time integration method '{time_integration_method}' is not supported."
            )

        # Update state and time
        U = U_new
        t += dt
        n += 1

        end_time = time.time()  # End timing the loop
        loop_time = end_time - start_time

        # Store history and print progress
        history.append(U.copy())
        dt_history.append(dt)
        print(
            f"Time: {t:.4f}s / {t_end:.4f}s, dt = {dt:.4f}s, Step: {n}, Loop Time: {loop_time:.4f}s"
        )

    return history, dt_history
