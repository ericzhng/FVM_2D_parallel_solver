"""
Handles writing simulation data to various output formats like VTK and Tecplot.
"""

import logging
from collections.abc import Mapping
import meshio
import numpy as np

from fvm_mesh.polymesh import PolyMesh


logger = logging.getLogger(__name__)


def write_vtk(filename: str, mesh: PolyMesh, U: np.ndarray, variable_names: list[str]):
    """
    Writes the solution to a VTK file.

    Args:
        filename (str): The name of the output file.
        mesh (PolyMesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        variable_names (list[str]): A list of names for the variables in U.
    """
    logger.info(f"Writing VTK file: {filename}")

    points = np.array(mesh.node_coords)

    cell_groups = {}
    for conn in mesh.cell_node_connectivity:
        n_nodes = len(conn)
        if n_nodes == 3:
            cell_type = "triangle"
        elif n_nodes == 4:
            cell_type = "quad"
        else:
            logger.warning(f"Unsupported cell type with {n_nodes} nodes.")
            continue

        if cell_type not in cell_groups:
            cell_groups[cell_type] = []
        cell_groups[cell_type].append(conn)

    cells = []
    for cell_type, conn_list in cell_groups.items():
        cells.append((cell_type, np.array(conn_list)))

    # If there is only one cell block, meshio expects the data to be in a list of length 1.
    cell_data: Mapping[str, list] = {
        name: [U[:, i]] for i, name in enumerate(variable_names)
    }

    meshio.write_points_cells(
        filename,
        points,
        cells,
        cell_data=cell_data,
        file_format="vtk",
    )


def write_tecplot(
    filename: str, mesh: PolyMesh, U: np.ndarray, variable_names: list[str]
):
    """
    Writes the solution to a Tecplot file.

    Args:
        filename (str): The name of the output file.
        mesh (PolyMesh): The mesh object.
        U (np.ndarray): The conservative state vector for all cells.
        variable_names (list[str]): A list of names for the variables in U.
    """
    logger.info(f"Writing Tecplot file: {filename}")

    points = np.array(mesh.node_coords)

    cell_groups = {}
    for conn in mesh.cell_node_connectivity:
        n_nodes = len(conn)
        if n_nodes == 3:
            cell_type = "triangle"
        elif n_nodes == 4:
            cell_type = "quad"
        else:
            logger.warning(f"Unsupported cell type with {n_nodes} nodes.")
            continue

        if cell_type not in cell_groups:
            cell_groups[cell_type] = []
        cell_groups[cell_type].append(conn)

    cells = []
    for cell_type, conn_list in cell_groups.items():
        cells.append((cell_type, np.array(conn_list)))
    # If there is only one cell block, meshio expects the data to be in a list of length 1.
    cell_data: Mapping[str, list] = {
        name: [U[:, i]] for i, name in enumerate(variable_names)
    }

    meshio.write_points_cells(
        filename,
        points,
        cells,
        cell_data=cell_data,
        file_format="tecplot",
    )
