import open3d as o3d
import numpy as np
import pyvista
import pyacvd
from pyvista import _vtk, PolyData

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

# Scales the shape by the given factor and returns the result
# Scale can be a scalar or a vector
def scale_mesh(mesh, scale):
    mesh["vertices"] = o3d.utility.Vector3dVector(np.asarray(mesh["vertices"]) * scale)
    return mesh

# Compute the shape's covariance matrix
def compute_PCA (mesh):
    # generate a matrix (3, n_points) for 3D points
    # row 0 == x-coordinate
    # row 1 == y-coordinate
    # row 2 == z-coordinate

    n_points = len(mesh["vertices"])

    # Get all x_coords, y_coords, z_coords
    x_coords = np.asarray(mesh["vertices"])[:, 0]
    y_coords = np.asarray(mesh["vertices"])[:, 1]
    z_coords = np.asarray(mesh["vertices"])[:, 2]

    A = np.zeros((3, n_points))
    A[0] = x_coords
    A[1] = y_coords
    A[2] = z_coords

    # compute the covariance matrix for A 
    # see the documentation at 
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.cov.html
    # this function expects that each row of A represents a variable, 
    # and each column a single observation of all those variables
    A_cov = np.cov(A)  # 3x3 matrix

    # computes the eigenvalues and eigenvectors for the 
    # covariance matrix. See documentation at  
    # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html 
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)

    return eigenvalues, eigenvectors

# Extract the faces from a polydata mesh and return them as an array
def get_face_list_from_polydata(mesh: PolyData):
    cells = mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = np.array(np.split(c, o[1:-1]))         # Convering an array(dtype = int64) to a list
    return faces

# Returns the amount of requested random vertices from the shape
def random_vertices(mesh, count):
    res = []
    for i in range(count):
        random = np.random.randint(0, len(mesh["vertices"]))
        res.append(mesh["vertices"][random])
    return res

# Returns the angle between the 3 given vertices
def angle_between(v1, v2, v3):
    v1 = v1 - v2
    v2 = v3 - v2
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Returns the distance between the 2 given vertices
def distance_between(v1, v2):
    return np.linalg.norm(v1 - v2)

# Returns the square root of area of triangle given by 3 random vertices
def triangle_area(v1, v2, v3):
    a = np.linalg.norm(v1 - v2)
    b = np.linalg.norm(v2 - v3)
    c = np.linalg.norm(v3 - v1)
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

# Calculates the cube root of volume of tetrahedron formed by 4 random vertices 
def tetrahedron_volume_v2(v1, v2, v3, v4):
    return np.dot(v1 - v4, np.cross(v2 - v4, v3 - v4)) / 6


# Returns the cube root of volume of tetrahedron given by 4 random vertices
def tetrahedron_volume(v1, v2, v3, v4):
    # assume the peak point is v4
    a = v1 - v4
    b = v2 - v4
    c = v3 - v4
    mixed_product = np.dot(a, np.cross(b, c))
    volume = (np.linalg.norm(mixed_product))/6
    return np.cbrt(volume)

if __name__ == "__main__":
    import load_meshes

    # Test barycenter translation

    # Load 100 random meshes
    meshes = load_meshes.get_meshes(fromLPSB=True, fromPRIN=True, randomSample=100, returnInfoOnly=False)

    for mesh in meshes:
        # Calculate the PCA
        eigenvalues, eigenvectors = compute_PCA(mesh)

        # Check that the eigenvectors are normalized
        assert np.allclose(np.linalg.norm(eigenvectors, axis=0), 1)

        # Check that the eigenvectors are orthogonal
        assert np.allclose(np.dot(eigenvectors[:, 0], eigenvectors[:, 1]), 0)

    # Test that the angle between 3 vertices is between 0 and pi
    for i in range(100):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)
        assert 0 <= angle_between(v1, v2, v3) <= np.pi

    # Test that the distance between 2 vertices is positive
    for i in range(100):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        assert distance_between(v1, v2) >= 0
    
    # Test that the area of a triangle is positive
    for i in range(100):
        v1 = np.random.rand(3)
        v2 = np.random.rand(3)
        v3 = np.random.rand(3)
        assert triangle_area(v1, v2, v3) >= 0

    assert angle_between(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 1, 0])) == np.pi / 2
    assert angle_between(np.array([0, 0, 0]), np.array([1, 0, 0]), np.array([0, 0, 1])) == np.pi / 2
    assert angle_between(np.array([0, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])) == np.pi / 2
    assert angle_between(np.array([5, 3, 2]), np.array([3, 4, 3]), np.array([5, 6, 5])) == np.pi / 2


    print("All unit tests passed")

