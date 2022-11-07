import open3d as o3d
import numpy as np
from pyvista import _vtk, PolyData
from scipy.stats import wasserstein_distance

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
    newAABB = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh["vertices"])
    mesh["aabb"] = [newAABB.get_min_bound()[0], newAABB.get_min_bound()[1], newAABB.get_min_bound()[2], newAABB.get_max_bound()[0], newAABB.get_max_bound()[1], newAABB.get_max_bound()[2]]
    return mesh

# Scales the shape by the given factor and returns the result
# Scale can be a scalar or a vector
def scale_mesh(mesh, scale):
    mesh["vertices"] = o3d.utility.Vector3dVector(np.asarray(mesh["vertices"]) * scale)
    newAABB = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh["vertices"])
    mesh["aabb"] = [newAABB.get_min_bound()[0], newAABB.get_min_bound()[1], newAABB.get_min_bound()[2], newAABB.get_max_bound()[0], newAABB.get_max_bound()[1], newAABB.get_max_bound()[2]]
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
    v1 = np.asarray(v1) - np.asarray(v2)
    v2 = np.asarray(v3) - np.asarray(v2)

    if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
        return 0

    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

# Returns the distance between the 2 given vertices
def distance_between(v1, v2):
    return np.linalg.norm(np.asarray(v1) - np.asarray(v2))

# Returns the square root of area of triangle given by 3 random vertices
def triangle_area(v1, v2, v3):
    a = np.linalg.norm(np.asarray(v1) - np.asarray(v2))
    b = np.linalg.norm(np.asarray(v2) - np.asarray(v3))
    c = np.linalg.norm(np.asarray(v3) - np.asarray(v1))
    s = (a + b + c) / 2
    return np.sqrt(s * (s - a) * (s - b) * (s - c))

# Calculates the cube root of volume of tetrahedron formed by 4 random vertices 
def tetrahedron_volume_v2(v1, v2, v3, v4):
    return np.dot(v1 - v4, np.cross(v2 - v4, v3 - v4)) / 6


# Returns the cube root of volume of tetrahedron given by 4 random vertices
def tetrahedron_volume(v1, v2, v3, v4):
    # assume the peak point is v4
    a = np.asarray(v1) - np.asarray(v4)
    b = np.asarray(v2) - np.asarray(v4)
    c = np.asarray(v3) - np.asarray(v4)
    mixed_product = np.dot(a, np.cross(b, c))
    volume = (np.linalg.norm(mixed_product))/6
    return np.cbrt(volume)

# returns the mean and standard deviation of the given single value features
def get_single_features_mean_and_sigma(features):
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    
    return mu, sigma

# returns the mean and standard deviation of the single value features of the given meshes
def get_single_features_mean_and_sigma_from_meshes(meshes):
    features = []

    for mesh in meshes:
        # check that all features are present
        if "surface_area" not in mesh or "volume" not in mesh or "compactness" not in mesh or "diameter" not in mesh or "eccentricity" not in mesh or "rectangularity" not in mesh:
            continue

        features.append([mesh['surface_area'], mesh['compactness'], mesh['volume'], mesh['diameter'], mesh['eccentricity'], mesh['rectangularity']])

    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    
    return mu, sigma

# z-score standardization
def standardize(data, mu, sigma):
    # fn = (f - favg)/fstd  prefered

    result = []
    n = 0
    for i in data:
        result.append(abs((i - mu[n]) / sigma[n]))   #abs
        n += 1
    return result

def standardize_all(meshes):
    features = []
    res = []

    for mesh in meshes:
        # check that all features are present
        if "surface_area" not in mesh or "volume" not in mesh or "compactness" not in mesh or "diameter" not in mesh or "eccentricity" not in mesh or "rectangularity" not in mesh:
            continue

        features.append([mesh['surface_area'], mesh['compactness'], mesh['volume'], mesh['diameter'], mesh['eccentricity'], mesh['rectangularity']])
        res.append({"path": mesh['path']})

    mu, sigma = get_single_features_mean_and_sigma(features)

    for i in range(len(res)):
        standardized_feature = standardize(features[i], mu, sigma)
        res[i]["surface_area_std"] = standardized_feature[0]
        res[i]["compactness_std"] = standardized_feature[1]
        res[i]["volume_std"] = standardized_feature[2]
        res[i]["diameter_std"] = standardized_feature[3]
        res[i]["eccentricity_std"] = standardized_feature[4]
        res[i]["rectangularity_std"] = standardized_feature[5]
    
    return res


# calculate the Euclidean Distance between feature vectors
# formula: d(A, B) = square root of sum((ai-bi)^2), i = 1, 2, ..., n
def get_Euclidean_Distance(vector_1, vector_2):
    # transform input feature vectors into numpy.ndarray if they are not numpy.ndarray type
    if (type(vector_1) != 'numpy.ndarray' and type(vector_2) != 'numpy.ndarray'):
        vector_1 = np.asarray(vector_1)
        vector_2 = np.asarray(vector_2)  
    Euclidean_distance = np.sqrt(np.square(vector_1 - vector_2).sum())
    return Euclidean_distance


def get_Cosine_Distance(vector_1, vector_2):
    # transform input feature vectors into numpy.ndarray if they are not numpy.ndarray type
    if (type(vector_1) != 'numpy.ndarray' and type(vector_2) != 'numpy.ndarray'):
        vector_1 = np.asarray(vector_1)
        vector_2 = np.asarray(vector_2) 

    Cosine_distance = (float(np.dot(vector_1, vector_2)) / (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))

    Cosine_distance = abs(1 - Cosine_distance)
    return Cosine_distance


def get_Earth_Mover_Distance(vector_1, vector_2):
    EMD = wasserstein_distance(vector_1, vector_2)
    return EMD

def get_feature_vector_from_mesh(mesh):
    try:
        if "surface_area_std" in mesh:
            v = [mesh["surface_area_std"], mesh["compactness_std"], mesh["volume_std"], mesh["diameter_std"], mesh["eccentricity_std"], mesh["rectangularity_std"]]
        else:
            v = [mesh["surface_area"], mesh["compactness"], mesh["volume"], mesh["diameter"], mesh["eccentricity"], mesh["rectangularity"]]
        multiValue = [mesh['A3'], mesh['D1'], mesh['D2'], mesh['D3'], mesh['D4']]

        for x in multiValue:
            for y in x:
                v.append(y)

        return v
    except:
        return None

if __name__ == "__main__":
    import load_meshes

    # Test barycenter translation

    # Load 100 random meshes
    meshes = load_meshes.get_meshes(fromLPSB=True, fromPRIN=True, fromNORM=False, randomSample=100, returnInfoOnly=False)

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

