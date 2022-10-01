import open3d as o3d
import numpy as np

# Calculate normal of the given face
def calculate_face_normal(face, vertices):
    v1 = vertices[face[0]]
    v2 = vertices[face[1]]
    v3 = vertices[face[2]]
    return np.cross(v2 - v1, v3 - v1)

# Calculate normals for each face of the given shape and returns the result
def calculate_shape_normals(mesh):
    normals = []
    for face in np.asarray(mesh["triangles"]):
        normals.append(calculate_face_normal(face, np.asarray(mesh["vertices"])))
    return o3d.utility.Vector3dVector(normals)

# Returns the barycenter of the shape
def get_shape_barycenter(mesh):
    return np.mean(np.asarray(mesh["vertices"]), axis=0)

# Translates the shape by the given offset and returns the result
def translate_mesh(mesh, translation):
    mesh["vertices"] = o3d.utility.Vector3dVector(np.asarray(mesh["vertices"]) + translation)
    return mesh

# Translates the shape to the given position and returns the result
def translate_mesh_to_origin(mesh):
    return translate_mesh(mesh, -get_shape_barycenter(mesh))


if __name__ == "__main__":
    import load_meshes

    # Test barycenter translation

    # Load 100 random meshes
    meshes = load_meshes.get_meshes(fromLPSB=True, fromPRIN=True, randomSample=100, returnInfoOnly=False)

    for mesh in meshes:
        # Translate the mesh to the origin
        mesh = translate_mesh_to_origin(mesh)

        # Get the barycenter
        barycenter = get_shape_barycenter(mesh)

        # Check that the barycenter is at the origin
        assert np.allclose(barycenter, np.zeros(3))

