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

# Translates the shape to the given position and returns the result
def translate_mesh_to_origin(mesh):
    return translate_mesh(mesh, -get_shape_barycenter(mesh))

# Scales the shape by the given factor and returns the result
# Scale can be a scalar or a vector
def scale_mesh(mesh, scale):
    mesh["vertices"] = o3d.utility.Vector3dVector(np.asarray(mesh["vertices"]) * scale)
    return mesh

# Returns the shape scaled to the unit cube
def scale_mesh_to_unit(mesh):
    return scale_mesh(mesh, 1 / np.max(np.linalg.norm(np.asarray(mesh["vertices"]), axis=1)))

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


# Align the two largest eigenvectors with the x and y axis
def align_shape(mesh):
    # Do PCA
    eigenvalues, eigenvectors = compute_PCA(mesh)

    # Get the two largest eigenvectors
    largest_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    second_largest_eigenvector = eigenvectors[:, np.argsort(eigenvalues)[-2]]

    # Get the two largest eigenvectors' angles with the x and y axis
    largest_eigenvector_angle = np.arctan2(largest_eigenvector[1], largest_eigenvector[0])
    second_largest_eigenvector_angle = np.arctan2(second_largest_eigenvector[1], second_largest_eigenvector[0])

    # Get the rotation matrix for the two largest eigenvectors
    rotation_matrix = np.array([[np.cos(largest_eigenvector_angle), -np.sin(largest_eigenvector_angle), 0],
                                [np.sin(largest_eigenvector_angle), np.cos(largest_eigenvector_angle), 0],
                                [0, 0, 1]])

    # Rotate the shape
    mesh["vertices"] = o3d.utility.Vector3dVector(np.dot(rotation_matrix, np.asarray(mesh["vertices"]).T).T)
    return mesh

# Performs the flipping test and mirrors the shape if necessary
def flipping_test(mesh):
    f_x = 0
    f_y = 0
    f_z = 0

    for triangle in mesh["vertices"]:
        f_x += np.sign(triangle[0]) * (triangle[0] ** 2)
        f_y += np.sign(triangle[1]) * (triangle[1] ** 2)
        f_z += np.sign(triangle[2]) * (triangle[2] ** 2)

    return scale_mesh(mesh, np.array([np.sign(f_x), np.sign(f_y), np.sign(f_z)]))

# get remeshed face
def face_list(mesh: PolyData):
    cells = mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = np.array(np.split(c, o[1:-1]))         # Convering an array(dtype = int64) to a list
    return faces

# Remesh the shape
def resampling(mesh):
    for i in range(len(mesh["faces"])):
        mesh["faces"][i].insert(0, len(mesh["faces"][i]))

    surf = pyvista.PolyData(mesh["vertices"], mesh["faces"])    # create PolyData

    # takes a surface mesh and returns a uniformly meshed surface using voronoi clustering

    clus = pyacvd.Clustering(surf)

    clus.subdivide(3)                           # remeshing
    clus.cluster(3000)                          # vertices around 3000
    # clus.plot()                               # plot clustered mesh

    remesh = clus.create_mesh()

    vertices_new = np.array(remesh.points)      # get new vertices list
    faces_new = face_list(remesh)               # get new faces list

    mesh["vertices"] = vertices_new
    mesh["faces"] = faces_new
    mesh["numVerts"] = len(vertices_new)
    mesh["numFaces"] = len(faces_new)

    return mesh

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

        # Scale to unit cube
        mesh = scale_mesh_to_unit(mesh)

        # Check that the max distance from the origin is 1
        assert np.allclose(np.max(np.linalg.norm(np.asarray(mesh["vertices"]), axis=1)), 1)

        # Calculate the PCA
        eigenvalues, eigenvectors = compute_PCA(mesh)

        # Check that the eigenvectors are normalized
        assert np.allclose(np.linalg.norm(eigenvectors, axis=0), 1)

        # Check that the eigenvectors are orthogonal
        assert np.allclose(np.dot(eigenvectors[:, 0], eigenvectors[:, 1]), 0)

        # Align the mesh
        mesh = align_shape(mesh)

        # Perform the flipping test
        mesh = flipping_test(mesh)


    print("All unit tests passed")

