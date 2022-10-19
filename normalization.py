from util import *

# Translates the shape to the given position and returns the result
def translate_mesh_to_origin(mesh):
    return translate_mesh(mesh, -get_shape_barycenter(mesh))

# Returns the shape scaled to the unit cube
def scale_mesh_to_unit(mesh):
    return scale_mesh(mesh, 1 / np.max(np.linalg.norm(np.asarray(mesh["vertices"]), axis=1)))

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
    faces_new = get_face_list_from_polydata(remesh)               # get new faces list

    mesh["vertices"] = vertices_new
    mesh["faces"] = faces_new
    mesh["numVerts"] = len(vertices_new)
    mesh["numFaces"] = len(faces_new)

    return mesh

def normalize(mesh, doResampling=True):
    if doResampling:
        mesh = resampling(mesh)
    mesh = translate_mesh_to_origin(mesh)
    mesh = align_shape(mesh)
    mesh = flipping_test(mesh)
    mesh = scale_mesh_to_unit(mesh)
    return mesh

if __name__ == "__main__":
    import load_meshes

    # Test all normalization functions

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

        # Align the mesh
        mesh = align_shape(mesh)

        # Perform the flipping test
        mesh = flipping_test(mesh)


    print("All unit tests passed")