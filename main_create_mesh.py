from fvm_mesh.meshgen.geometry import Geometry
from fvm_mesh.meshgen.mesh_generator import MeshGenerator

import gmsh


def main():
    """Create a rectangular mesh and save it to a Gmsh file."""
    # Define geometry parameters
    length = 1.0  # Length of the rectangle in the x-direction
    height = 1.0  # Height of the rectangle in the y-direction

    # Create geometry
    gmsh.initialize()
    gmsh.model.add("test_polygon")
    geom = Geometry()
    surface_tag = geom.rectangle(length, height, mesh_size=0.05)

    # synchronize the geo kernel to the model so model-level API can access entities
    gmsh.model.geo.synchronize()

    # Get the boundary entities of the surface using the model-level API
    boundary_entities = gmsh.model.getBoundary([(2, surface_tag)])

    # Identify the boundary lines by their position
    line_tags = [abs(tag) for dim, tag in boundary_entities]

    bottom_wall = -1
    right_outlet = -1
    top_wall = -1
    left_inlet = -1

    tol = 1e-9
    for line_tag in line_tags:
        bbox = gmsh.model.getBoundingBox(1, line_tag)
        if abs(bbox[1] - 0.0) < tol and abs(bbox[4] - 0.0) < tol:
            bottom_wall = line_tag
        elif abs(bbox[0] - length) < tol and abs(bbox[3] - length) < tol:
            right_outlet = line_tag
        elif abs(bbox[1] - height) < tol and abs(bbox[4] - height) < tol:
            top_wall = line_tag
        elif abs(bbox[0] - 0.0) < tol and abs(bbox[3] - 0.0) < tol:
            left_inlet = line_tag

    # Add physical groups for the boundaries
    if left_inlet != -1:
        gmsh.model.addPhysicalGroup(1, [left_inlet], name="inlet")
    if right_outlet != -1:
        gmsh.model.addPhysicalGroup(1, [right_outlet], name="outlet")
    if bottom_wall != -1 and top_wall != -1:
        gmsh.model.addPhysicalGroup(1, [bottom_wall, top_wall], name="wall")

    # Generate mesh
    output_dir = "data"
    mesher = MeshGenerator(surface_tags=surface_tag, output_dir=output_dir)
    mesh_filename = "euler_mesh.msh"
    mesh_params = {surface_tag: {"mesh_type": "quads", "char_length": 0.01}}
    mesher.generate(
        mesh_params=mesh_params,
        filename=mesh_filename,
        show_nodes=True,
        show_cells=True,
    )

    gmsh.finalize()


if __name__ == "__main__":
    main()
