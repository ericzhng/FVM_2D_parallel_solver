from mpi4py import MPI

from fvm_mesh.polymesh.core_mesh import CoreMesh
from fvm_mesh.polymesh.poly_mesh import PolyMesh
from fvm_mesh.polymesh.local_mesh import LocalMesh

from polymesh.local_mesh import create_local_meshes
from src.solver import solve
from src.case_setup import setup_case_euler
from src.visualization import plot_simulation_step, create_animation
from src.euler_equations import EulerEquations


def main():
    """
    Main function to run the 2D Finite Volume Method (FVM) solver.

    This script orchestrates the entire simulation process:
    1.  Initializes and reads the computational mesh.
    2.  Sets up the initial and boundary conditions for the chosen physical model.
    3.  Runs the FVM solver to advance the solution in time.
    4.  Visualizes the results.
    """

    # --- 1. Initialize MPI and Read Mesh ---
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    print(f"Process {rank}/{size}: Initializing and reading mesh...")
    if rank == 0:
        global_mesh = PolyMesh.from_gmsh("data/euler_mesh.msh")
        global_mesh.analyze_mesh()
        global_mesh.print_summary()
        # Distribute the mesh to all processes
        # create_local_meshes expects a CoreMesh; provide the underlying core mesh from PolyMesh
        local_meshes_list = create_local_meshes(global_mesh, n_parts=size)
    else:
        global_mesh = None
        local_meshes_list = None

    local_mesh = comm.scatter(local_meshes_list, root=0)
    mesh = local_mesh  # Renaming for consistency with the rest of the code

    # --- 2. Set Up Case ---
    print("Setting up the simulation case...")

    U_init, bc_dict = setup_case_euler(mesh, comm)
    equation = EulerEquations(gamma=1.4)
    t_end = 0.25

    # --- 3. Solve ---
    print("Starting the FVM solver...")
    history, dt_history = solve(
        mesh,
        U_init,
        bc_dict,
        equation,
        comm,
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
