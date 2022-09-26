import open3d as o3d
import numpy as np

print("Testing mesh in Open3D...")

knot_mesh = o3d.data.KnotMesh()
mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
print(mesh)
print('Vertices:')
print(np.asarray(mesh.vertices))
print('Triangles:')
print(np.asarray(mesh.triangles))

print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0.706, 0])
print(np.asarray(mesh.triangle_normals))
o3d.visualization.draw_geometries([mesh])