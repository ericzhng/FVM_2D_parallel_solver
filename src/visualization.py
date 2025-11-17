import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpi4py import MPI

from fvm_mesh.polymesh import PolyMesh, LocalMesh


logger = logging.getLogger(__name__)


def plot_simulation_step(
    mesh: PolyMesh, U, title="", variable_to_plot=0, output_dir="."
):
    """
    Plots a specific variable from the solution on the mesh for a single time step.

    Args:
        mesh (Mesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        title (str, optional): The title of the plot. Defaults to "".
        variable_to_plot (int, optional): The index of the variable to plot.
                                           Defaults to 0 (e.g., density or water height).
    """
    node_coords = np.array(mesh.node_coords)
    x = node_coords[:, 0]
    y = node_coords[:, 1]
    var = U[:, variable_to_plot]

    # Create a triangulation for plotting
    triangles = []
    facecolors = []
    for i, conn in enumerate(mesh.cell_node_connectivity):
        node_indices = conn
        if len(node_indices) == 4:
            triangles.append([node_indices[0], node_indices[1], node_indices[2]])
            triangles.append([node_indices[0], node_indices[2], node_indices[3]])
            facecolors.extend([var[i], var[i]])
        else:
            triangles.append(node_indices)
            facecolors.append(var[i])

    plt.figure(figsize=(12, 12))
    plt.tripcolor(
        x, y, triangles=triangles, facecolors=facecolors, shading="flat", cmap="viridis"
    )
    plt.colorbar(label=f"Variable {variable_to_plot+1}")
    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(f"{output_dir}/Final_step_var{variable_to_plot+1}.png")
    # plt.show()


def create_animation(
    mesh: PolyMesh, history, dt_history, filename="simulation.gif", variable_to_plot=0
):
    """
    Creates and saves an animation of the simulation history.

    Args:
        mesh (Mesh): The mesh object.
        history (list): A list of the state vectors at each time step.
        dt_history (list): A list of the time steps.
        filename (str, optional): The filename for the output animation.
                                Defaults to "simulation.gif".
        variable_to_plot (int, optional): The index of the variable to plot.
                                           Defaults to 0.
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    node_coords = np.array(mesh.node_coords)
    x = node_coords[:, 0]
    y = node_coords[:, 1]

    # Create a triangulation for plotting
    triangles = []
    for conn in mesh.cell_node_connectivity:
        node_indices = conn
        if len(node_indices) == 4:  # Quadrilateral
            triangles.append([node_indices[0], node_indices[1], node_indices[2]])
            triangles.append([node_indices[0], node_indices[2], node_indices[3]])
        else:  # Triangle
            triangles.append(node_indices)

    # Initial plot setup
    var_initial = history[0][:, variable_to_plot]
    facecolors_initial = []
    for i, h_val in enumerate(var_initial):
        if len(mesh.cell_node_connectivity[i]) == 4:
            facecolors_initial.extend([h_val, h_val])
        else:
            facecolors_initial.append(h_val)

    tpc = ax.tripcolor(
        x,
        y,
        triangles=triangles,
        facecolors=facecolors_initial,
        shading="flat",
        cmap="viridis",
    )
    fig.colorbar(tpc, ax=ax, label=f"Variable {variable_to_plot}")
    time_text = ax.set_title(f"Simulation at t = {0.0:.4f}s")
    ax.set_xlabel("X-coordinate")
    ax.set_ylabel("Y-coordinate")
    ax.set_aspect("equal", adjustable="box")

    def update_frame(frame):
        """Updates the plot for each frame of the animation."""
        U = history[frame]
        h = U[:, 0]
        facecolors = []
        for i, h_val in enumerate(h):
            if len(mesh.cell_node_connectivity[i]) == 4:
                facecolors.extend([h_val, h_val])
            else:
                facecolors.append(h_val)
        tpc.set_array(facecolors)

        current_time = sum(dt_history[:frame])
        ax.set_title(f"Simulation at t = {current_time:.4f}s")
        return [tpc, time_text]

    # Create the animation
    anim = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(history),
        interval=100,  # Milliseconds between frames
        blit=False,
        repeat=True,
        repeat_delay=3000,
    )

    # Save or show the animation
    # anim.save(filename, writer='imagemagick', fps=10)
    plt.show()


def reconstruct_and_visualize(
    global_mesh: PolyMesh | None, mesh: LocalMesh, history, dt_history, comm
):
    """
    Gathers simulation data from all processes and performs visualization on rank 0.

    Args:
        global_mesh (PolyMesh): The complete mesh, required on rank 0.
        mesh (LocalMesh): The local mesh object for the current rank.
        history (list): A list of the conservative state vectors over time for the local mesh.
        dt_history (list): A list of the time steps (dt) used in the simulation.
        comm (MPI.Comm): The MPI communicator.
    """
    rank = comm.Get_rank()

    # Gather all histories and local meshes on rank 0
    logger.info(f"Rank {rank}: Gathering results for visualization.")
    all_histories = comm.gather(history, root=0)
    all_local_meshes = comm.gather(mesh, root=0)

    if rank == 0:
        logger.info("Reconstructing global data and visualizing on rank 0...")
        if not all_histories or not all_local_meshes:
            logger.warning("No data received on rank 0 for visualization.")
            return

        if global_mesh is None:
            logger.warning("No global mesh provided for visualization.")
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
                if hasattr(local_mesh, "l2g_cells"):
                    global_indices = local_mesh.l2g_cells[: local_mesh.num_owned_cells]
                    U_global[global_indices] = local_U[: local_mesh.num_owned_cells]
                else:
                    logger.warning(
                        "`local_mesh.l2g_cells` not found. "
                        "Visualization may be incorrect."
                    )
            global_history.append(U_global)

        # --- Visualize ---
        logger.info("Creating animation of the results...")
        create_animation(global_mesh, global_history, dt_history, variable_to_plot=0)

        logger.info("Plotting final state...")
        for k in range(num_vars):
            plot_simulation_step(
                global_mesh,
                global_history[-1],
                "Final State",
                variable_to_plot=k,
                output_dir="results",
            )
