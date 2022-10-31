import numpy as np
from scipy.stats import wasserstein_distance
import database
import ShapeDescriptors
import normalization
import os
import load_meshes
import open3d as o3d
import util

dbmngr = database.DatabaseManager()
database_length = 380
SCALARWEIGHT = 0.25
VECTORWEIGHT = 0.75

def matching_single_Feature(mesh, distance_type):
    # get features from the input (querying) mesh
    mesh = normalization.normalize(mesh)
    SurfaceAreaLoad = ShapeDescriptors.get_Surface_Area(mesh)
    CompactnessLoad = ShapeDescriptors.get_Compactness(mesh)
    VolumeLoad = ShapeDescriptors.get_Volume(mesh)
    DiameterLoad = ShapeDescriptors.get_diameter(mesh)
    EccentricityLoad = ShapeDescriptors.get_eccentricity(mesh)
    RectangularityLoad = ShapeDescriptors.get_3D_Rectangularity(mesh)

    # build feature vector of the querying mesh
    loadedMesh_feature_vector = [SurfaceAreaLoad, CompactnessLoad, VolumeLoad, DiameterLoad,
                                 EccentricityLoad, RectangularityLoad]

    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()

    # initialize an empty list
    single_Value_Feature_Vector = [[[] for i in range (2)] for i in range(database_length)]
    # combine path and 6 descriptors in to one 3-dimensional list
    # [(path, [S, C, V, D, E, R]), ..., (path, [S, C, V, D, E, R])]
    i = 0
    for mesh in meshes:
        single_Value_Feature_Vector[i][0] = mesh['path']
        single_Value_Feature_Vector[i][1] = [mesh['surface_area'], mesh['compactness'], mesh['volume'],
                                            mesh['diameter'], mesh['eccentricity'], mesh['rectangularity']]
        i = i + 1

    # standardize single_value_feature_vector
    feature_vector_Standardized = [[[] for i in range(2)] for i in range(database_length)]
    single_Value_Feature_Vectors = []
        
    # build a list to store all single_value vectors
    for i in range(database_length):
        single_Value_Feature_Vectors.append(single_Value_Feature_Vector[i][1])

    # standardization
    i = 0
    mu, sigma = util.get_single_features_mean_and_sigma(single_Value_Feature_Vectors)
    for vector in single_Value_Feature_Vector:
        feature_vector_Standardized[i][1] = util.standardize(vector[1], mu, sigma)
        feature_vector_Standardized[i][0] = vector[0]
        i += 1
    print("Descriptors Standardized from DB:", feature_vector_Standardized)

    loadedMesh_Standardized = util.standardize(loadedMesh_feature_vector, mu, sigma)
    print("LoadedMesh Standardized:", loadedMesh_Standardized)

    # calculate distance
    if distance_type == 'Euclidean':
        # Euclidean Distance Case: 
        dists = [[[] for i in range(2)] for i in range(database_length)]
        i = 0
        for vector in feature_vector_Standardized:
            dists[i][1] = get_Euclidean_Distance(loadedMesh_Standardized, vector[1])
            dists[i][0] = vector[0]
            i += 1
    elif distance_type == 'Cosine':
        # Cosine Distance Case: 
        dists = [[[] for i in range(2)] for i in range(database_length)]
        i = 0
        for vector in feature_vector_Standardized:
            dists[i][1] = get_Cosine_Distance(loadedMesh_Standardized, vector[1])
            dists[i][0] = vector[0]
            i += 1
    elif distance_type == 'EMD':
        # EMD Distance Case: 
        dists = [[[] for i in range(2)] for i in range(database_length)]
        i = 0
        for vector in feature_vector_Standardized:
            dists[i][1] = get_Earth_Mover_Distance(loadedMesh_Standardized, vector[1])
            dists[i][0] = vector[0]
            i += 1
        
    return dists


def matching_histo_Feature(mesh, distance_type, descriptor):
    # get features from the input (querying) mesh
    mesh = normalization.normalize(mesh)
    if descriptor == 'A3':
        Load = ShapeDescriptors.A3(mesh, bins=10)[0]
    elif descriptor == 'D1':
        Load = ShapeDescriptors.D1(mesh, bins=10)[0]
    elif descriptor == 'D2':
        Load = ShapeDescriptors.D2(mesh, bins=10)[0]
    elif descriptor == 'D3':
        Load = ShapeDescriptors.D3(mesh, bins=10)[0]
    elif descriptor == 'D4':
        Load = ShapeDescriptors.D4(mesh, bins=10)[0]

    print("Load", Load)
    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()

    histo_Feature_Vector = [[[] for i in range (2)] for i in range(database_length)]
    # # combine path with each histogram features
    # # [[[path] [A3]]...[[path] [A3]]]
    j = 0
    for mesh in meshes:
        histo_Feature_Vector[j][0] = mesh['path']
        if descriptor == 'A3':
            histo_Feature_Vector[j][1] = mesh['A3']
        elif descriptor == 'D1':
            histo_Feature_Vector[j][1] = mesh['D1']
        elif descriptor == 'D2':
            histo_Feature_Vector[j][1] = mesh['D2']
        elif descriptor == 'D3':
            histo_Feature_Vector[j][1] = mesh['D3']
        elif descriptor == 'D4':
            histo_Feature_Vector[j][1] = mesh['D4']
        j = j + 1

    print("Histogram Descriptors Standardized from DB", histo_Feature_Vector)

    if distance_type == 'EMD':
        i = 0
        dists = [[[] for i in range(2)] for i in range(database_length)]
        for vector in histo_Feature_Vector:
            dists[i][1] = get_Earth_Mover_Distance(Load, vector[1])
            dists[i][0] = vector[0]
            i += 1

    return dists

def sort(name, dists, distance_type, k = 5):
    dists.sort(key=lambda x: float(x[1]), reverse=False)
    print("Sorted distance", dists)
    distance = []
    for d in dists:     # remove the querying shape from searching list
        path = ''.join(d[0])
        path = path.split("\\")
        if name != path[-1]:
            distance.append(d)
    distance = distance[:k]
    return distance


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

def find_best_matches(mesh, k = 5):
    global dbmngr

    # Normalize the input mesh
    mesh = normalization.normalize(mesh)

    # get features from the input (querying) mesh
    features = ShapeDescriptors.extract_all_features(mesh)
    singleValFeatures = [features['surface_area'], features['compactness'], features['volume'],
                         features['diameter'], features['eccentricity'], features['rectangularity']]
    multiValFeatures = [features['A3'], features['D1'], features['D2'], features['D3'], features['D4']]

    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()

    # get mean and standard deviation of each feature from database
    mu, sigma = util.get_single_features_mean_and_sigma_from_meshes(meshes)

    # get features from database (cursor refresh)
    meshes = dbmngr.get_all_with_extracted_features()

    # standardize the loaded mesh's features
    singleValFeatures = util.standardize(singleValFeatures, mu, sigma)

    # compare single value features using Cosine Distance
    dists = [[[] for i in range(2)] for i in range(database_length)]
    i = 0
    for db_mesh in meshes:
        # Check if the mesh is the querying mesh, remove it if it is
        if db_mesh['path'] == mesh['path']:
            continue

        # get the meshes features
        dbSingleValFeatures = [db_mesh['surface_area'], db_mesh['compactness'], db_mesh['volume'],
                               db_mesh['diameter'], db_mesh['eccentricity'], db_mesh['rectangularity']]
        dbMultiValFeatures = [db_mesh['A3'], db_mesh['D1'], db_mesh['D2'], db_mesh['D3'], db_mesh['D4']]

        dists[i][0] = db_mesh["path"]
        dists[i][1] = get_Cosine_Distance(singleValFeatures, dbSingleValFeatures) * SCALARWEIGHT
        for j in range(len(multiValFeatures)):
            dists[i][1] += get_Earth_Mover_Distance(multiValFeatures[j], dbMultiValFeatures[j]) * VECTORWEIGHT
        i += 1

    # sort the distances
    dists.sort(key=lambda x: x[1])
    dists = dists[:k]

    res = []
    for d in dists:
        res.append(dbmngr.get_by_path(d[0]))

    return res

if __name__ == "__main__":

    # load a mesh for test
    name = "Ant/85.off"
    path = os.path.join("data/LabeledDB_new/", name)
    mesh = load_meshes.load_OFF(path)
    # name = name.split("/")

    # # single value features test
    # # Euclidean_dists = matching_single_Feature(mesh, 'Euclidean')
    # # paths = sort(name[-1], Euclidean_dists, distance_type='Euclidean', k = 5)
    # # Cosine_dists = matching_single_Feature(mesh, 'Cosine')
    # # paths = sort(name[-1], Cosine_dists, distance_type='Euclidean', k=5)
    # # EMD_dists = matching_single_Feature(mesh, 'EMD')
    # # paths = sort(name[-1], EMD_dists, distance_type='EMD', k=5)
    # # print("All distance except for the querying one:", Cosine_dists)

    # # # Histogram features test
    # # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'A3')
    # # paths = sort(name[-1], EMD_histo_dist, distance_type='A3', k=5)
    # # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D1')
    # # paths = sort(name[-1], EMD_histo_dist, distance_type='D1', k=5)
    # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D2')
    # paths = sort(name[-1], EMD_histo_dist, distance_type='D2', k=5)
    # # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D3')
    # # paths = sort(name[-1], EMD_histo_dist, distance_type='D3', k=5)
    # # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D4')
    # # paths = sort(name[-1], EMD_histo_dist, distance_type='D4', k=5)
    # print("All distance except for the querying one", EMD_histo_dist)

    # print("Retrieved", paths)
    # for i in paths:
    #     i = "".join(i[0])
    #     name = i.lstrip(i[:16])
    #     i = os.path.join("data/LabeledDB_new/", name)
    #     meshi = load_meshes.load_OFF(i)
    #     vertices = meshi['vertices']
    #     faces = meshi['faces']
    #     meshi = o3d.geometry.TriangleMesh()
    #     meshi.vertices = o3d.utility.Vector3dVector(vertices)
    #     meshi.triangles = o3d.utility.Vector3iVector(faces)
    #     meshi.paint_uniform_color([1, 0.706, 0])
    #     meshi.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([meshi])

    best = find_best_matches(mesh, k=5)

    for x in best:
        print(x["name"] + "-" + x["path"] + "-" + x["class"])