import numpy as np
from mpi4py import MPI

from fvm_mesh.polymesh.poly_mesh import PolyMesh
from fvm_mesh.polymesh.local_mesh import create_local_meshes

from src.case_setup import setup_case_euler
from src.euler_equations import EulerEquations
from src.visualization import plot_simulation_step, create_animation
from src.solver import solve
from src.boundary import BoundaryConditions


def reconstruct_and_visualize(comm, mesh, history, dt_history, U_init, global_mesh):
    """
    Gathers data from all processes and performs visualization on rank 0.
    """
    rank = comm.Get_rank()

    # Gather all histories and local meshes on rank 0
    all_histories = comm.gather(history, root=0)
    all_local_meshes = comm.gather(mesh, root=0)

    if rank == 0:
        if not all_histories or not all_local_meshes:
            print("No data received for visualization.")
            return

        # --- Reconstruct Global Data ---
        num_time_steps = len(all_histories[0])
        num_vars = all_histories[0][0].shape[1]
        num_global_cells = len(global_mesh.cell_centroids)

        global_history = []

        for t in range(num_time_steps):
            U_global = np.zeros((num_global_cells, num_vars))
            for i, (rank_history, local_mesh) in enumerate(
                zip(all_histories, all_local_meshes)
            ):
                local_U = rank_history[t]
                # NOTE: We assume local_mesh contains a mapping from local to global indices.
                # 'fvm_mesh' should provide this. Here, we assume it's `local_mesh.global_cell_indices`.
                # This part might need adjustment depending on the actual structure of `fvm_mesh.LocalMesh`.
                if hasattr(local_mesh, "global_cell_indices"):
                    global_indices = local_mesh.global_cell_indices[
                        : local_mesh.num_owned_cells
                    ]
                    U_global[global_indices] = local_U[: local_mesh.num_owned_cells]
                else:
                    # Fallback if the assumption is wrong. This will likely produce incorrect plots.
                    print(
                        "Warning: `local_mesh.global_cell_indices` not found. Visualization may be incorrect."
                    )
                    # A simple, but likely incorrect, concatenation for visualization purposes.
                    start_idx = sum(len(m.cell_centroids) for m in all_local_meshes[:i])
                    end_idx = start_idx + len(local_mesh.cell_centroids)
                    if end_idx <= num_global_cells:
                        U_global[start_idx:end_idx] = local_U

            global_history.append(U_global)

        # --- 4. Visualize ---
        print("Creating animation of the results...")
        create_animation(global_mesh, global_history, dt_history, variable_to_plot=0)

        for k in range(U_init.shape[1]):
            plot_simulation_step(
                global_mesh, global_history[-1], "Final State", variable_to_plot=k
            )


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
        # create_local_meshes expects a CoreMesh; provide the underlying core mesh from PolyMesh
        local_meshes_list = create_local_meshes(global_mesh, n_parts=size)
    else:
        global_mesh = None
        local_meshes_list = None

    # Distribute the mesh to all processes
    local_mesh = comm.scatter(local_meshes_list, root=0)
    mesh = local_mesh  # Renaming for consistency with the rest of the code

    # --- 2. Set Up Case ---
    print("Setting up the simulation case...")

    U_init, bc_dict = setup_case_euler(mesh, comm, gamma=1.4)

    boundary_conditions = BoundaryConditions(bc_dict, mesh.boundary_tag_map)
    bcs_lookup = boundary_conditions.to_lookup_array()
    equation = EulerEquations(gamma=1.4)
    t_end = 0.25

    # --- 3. Solve ---
    print("Starting the FVM solver...")
    history, dt_history = solve(
        mesh,
        U_init,
        bcs_lookup,
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
    reconstruct_and_visualize(comm, mesh, history, dt_history, U_init, global_mesh)


if __name__ == "__main__":
    main()
