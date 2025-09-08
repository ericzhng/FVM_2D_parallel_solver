from src.mesh import Mesh, plot_mesh
from src.solver import solve
from src.case_setup import setup_case_shallow_water, setup_case_euler
from src.visualization import plot_simulation_step, create_animation
from src.euler_equations import EulerEquations
from src.shallow_water_equations import ShallowWaterEquations


def main():
    """
    Main function to run the 2D Finite Volume Method (FVM) solver.

    This script orchestrates the entire simulation process:
    1.  Initializes and reads the computational mesh.
    2.  Sets up the initial and boundary conditions for the chosen physical model.
    3.  Runs the FVM solver to advance the solution in time.
    4.  Visualizes the results.
    """

    # --- 1. Initialize and Read Mesh ---
    print("Initializing and reading mesh...")
    mesh = Mesh()
    mesh.read_mesh("data/euler_mesh.msh")
    mesh.analyze_mesh()
    mesh.summary()
    # plot_mesh(mesh)  # Optional: Uncomment to visualize the mesh and check normals

    # --- 2. Set Up Case ---
    print("Setting up the simulation case...")
    equation_type = "euler"  # Choose 'shallow_water' or 'euler'

    if equation_type == "shallow_water":
        U_init, bc_dict = setup_case_shallow_water(mesh)
        equation = ShallowWaterEquations(g=9.81)
        t_end = 20
    elif equation_type == "euler":
        U_init, bc_dict = setup_case_euler(mesh)
        equation = EulerEquations(gamma=1.4)
        t_end = 0.25
    else:
        raise ValueError("Invalid equation type specified.")

    # --- 3. Solve ---
    print("Starting the FVM solver...")
    history, dt_history = solve(
        mesh,
        U_init,
        bc_dict,
        equation,
        t_end=t_end,
        limiter_type="minmod",  # Options: 'barth_jespersen', 'minmod', 'superbee'
        flux_type="roe",
        over_relaxation=1.0,
        use_adaptive_dt=True,
        cfl=0.5,
        dt_initial=1e-2,
    )
    print("Solver finished.")

    # --- 4. Visualize ---
    print("Creating animation of the results...")
    create_animation(mesh, history, dt_history, variable_to_plot=0)

    for k in range(U_init.shape[1]):
        plot_simulation_step(mesh, history[-1], "Final State", variable_to_plot=k)


if __name__ == "__main__":
    main()
