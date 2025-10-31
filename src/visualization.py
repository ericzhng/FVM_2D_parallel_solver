import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from fvm_mesh.polymesh.poly_mesh import PolyMesh


def plot_simulation_step(mesh: PolyMesh, U, title="", variable_to_plot=0):
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
    for i, conn in enumerate(mesh.cell_connectivity):
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
    plt.colorbar(label=f"Variable {variable_to_plot}")
    plt.title(title)
    plt.xlabel("X-coordinate")
    plt.ylabel("Y-coordinate")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.savefig(f"Final_{variable_to_plot}.png")
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
    for conn in mesh.cell_connectivity:
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
        if len(mesh.cell_connectivity[i]) == 4:
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
            if len(mesh.cell_connectivity[i]) == 4:
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
