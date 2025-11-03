from fvm_mesh.polymesh.poly_mesh import PolyMesh
from fvm_mesh.polymesh.local_mesh import create_local_meshes


size = 4

global_mesh = PolyMesh.from_gmsh("data/euler_mesh.msh")
global_mesh.analyze_mesh()
global_mesh.print_summary()

# create_local_meshes expects a CoreMesh; provide the underlying core mesh from PolyMesh
local_meshes_list = create_local_meshes(global_mesh, n_parts=size)

for i in range(size):
    local_mesh = local_meshes_list[i]
    print(f"Local mesh for part {i}:")
    local_mesh.print_summary()
    local_mesh.plot(
        filepath=f"trunk/local_mesh_part_{i}.png", show_cells=True, show_nodes=True
    )
