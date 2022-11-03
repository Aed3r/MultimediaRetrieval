import time
import numpy as np
from scipy.stats import wasserstein_distance
import database
import ShapeDescriptors
import normalization
import os
import load_meshes
import open3d as o3d
import util
from tqdm import tqdm

dbmngr = database.DatabaseManager()
database_length = 380
# SCALARWEIGHT = 0.25
# VECTORWEIGHT = 0.75
SCALARWEIGHT = 0.25/6
VECTORWEIGHT = 0.75/5
# SCALARWEIGHT = 0.25/6
# VECTORWEIGHT = [0.05, 0.05, 0.25, 0.25, 0.15]


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

# Uses the loaded meshe's features to find the k closest matches in the database (in theory..)
# Assumes the mesh already has features extracted, not necessarily standardized though
# Use k to specify how many results to return
def find_best_matches(mesh, k = 5):
    global dbmngr

    print("Finding best matches for mesh: ", mesh["name"])

    singleValFeatures = [mesh['surface_area'], mesh['compactness'], mesh['volume'],
                         mesh['diameter'], mesh['eccentricity'], mesh['rectangularity']]
    multiValFeatures = [mesh['A3'], mesh['D1'], mesh['D2'], mesh['D3'], mesh['D4']]

    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()

    # get mean and standard deviation of each feature from database
    print("Calculating mean and standard deviation of database features...")
    start = time.time()
    mu, sigma = util.get_single_features_mean_and_sigma_from_meshes(meshes)
    end = time.time()
    print("Calculating mean and standard deviation of database features done in ", end - start, " seconds")

    # get features from database (cursor refresh)
    meshes = dbmngr.get_all_with_extracted_features()

    # standardize the loaded mesh's features
    print("Standardizing the loaded meshe's single value features...")
    start = time.time()
    singleValFeatures = util.standardize(singleValFeatures, mu, sigma)
    end = time.time()
    print("Standardizing the loaded meshe's single value features done in ", end - start, " seconds")

    # compare single value features using Cosine Distance
    dists = [[[] for i in range(2)] for i in range(database_length)]

    i = 0
    for db_mesh in tqdm(meshes, desc='Comparing meshes', ncols=150, total=dbmngr.get_mesh_count()):
        # Check if the mesh is the querying mesh, remove it if it is
        if db_mesh["name"] == mesh["name"]:
            continue

        if "surface_area_std" not in db_mesh:
            raise Exception("Error: the database meshes have not been standardized. Please run `python3 main.py standardizeDB` first.")

        # get the meshes features
        dbSingleValFeatures = [db_mesh['surface_area_std'], db_mesh['compactness_std'], db_mesh['volume_std'],
                               db_mesh['diameter_std'], db_mesh['eccentricity_std'], db_mesh['rectangularity_std']]
        dbMultiValFeatures = [db_mesh['A3'], db_mesh['D1'], db_mesh['D2'], db_mesh['D3'], db_mesh['D4']]

        #dbSingleValFeatures = util.standardize(dbSingleValFeatures, mu, sigma)
        dists[i][0] = db_mesh["path"]
        dists[i][1] = get_Cosine_Distance(singleValFeatures, dbSingleValFeatures) * SCALARWEIGHT
        for j in range(len(multiValFeatures)):
            dists[i][1] += get_Earth_Mover_Distance(multiValFeatures[j], dbMultiValFeatures[j]) * VECTORWEIGHT
        # try Distance weighting
        # dists[i][1] = get_Euclidean_Distance(singleValFeatures, dbSingleValFeatures)
        # for j in range(len(multiValFeatures)):
        #      D1[i][j] = float(get_Earth_Mover_Distance(multiValFeatures[j], dbMultiValFeatures[j]))

        i += 1

    # standardize the distance
    # dmu = np.mean(D1, axis=0) #why it doesn't work?
    # dsigma = np.std(D1, axis=0)
    # i=0
    # for n in D1:
    #     dists[i][1] += util.standardize(n, dmu, dsigma)
    #     i += 1
    # print("dists", dists)

    #sort the distances
    dists.sort(key=lambda x: x[1])
    dists = dists[:k]

    res = []
    print("Results:")
    for i in range(k):
        res.append(dbmngr.get_by_path(dists[i][0]))
        res[i]["distance"] = dists[i][1]
        print(" #" + str(i + 1) + " " + res[i]["name"] + " distance:" + str(dists[i][1]))

    return res

if __name__ == "__main__":

    # load a mesh for test
    name = "Fish/222.off"
    path = os.path.join("data/LabeledDB_new/", name)
    mesh = load_meshes.load_OFF(path)
    #name = name.split("/")

    # # single value features test
    # Euclidean_dists = matching_single_Feature(mesh, 'Euclidean')
    # paths = sort(name[-1], Euclidean_dists, distance_type='Euclidean', k = 5)
    # Cosine_dists = matching_single_Feature(mesh, 'Cosine')
    # paths = sort(name[-1], Cosine_dists, distance_type='Euclidean', k=5)
    # # EMD_dists = matching_single_Feature(mesh, 'EMD')
    # # paths = sort(name[-1], EMD_dists, distance_type='EMD', k=5)
    # # print("All distance except for the querying one:", Cosine_dists)

    # # # Histogram features test
    # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'A3')
    # paths = sort(name[-1], EMD_histo_dist, distance_type='A3', k=5)
    # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D1')
    # paths = sort(name[-1], EMD_histo_dist, distance_type='D1', k=5)
    # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D2')
    # paths = sort(name[-1], EMD_histo_dist, distance_type='D2', k=5)
    # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D3')
    # paths = sort(name[-1], EMD_histo_dist, distance_type='D3', k=5)
    # EMD_histo_dist = matching_histo_Feature(mesh, 'EMD', 'D4')
    # paths = sort(name[-1], EMD_histo_dist, distance_type='D4', k=5)
    # print("All distance except for the querying one", EMD_histo_dist)
    # print("Retrieved", paths)

    mesh = normalization.normalize(mesh)
    mesh = ShapeDescriptors.extract_all_features(mesh, True)
    best = find_best_matches(mesh, k=5)

    for x in best:
        print(x["name"] + "-" + x["path"] + "-" + x["class"])

    for i in best:
    #for i in paths:
        # i = "".join(i[0])
        # name = i.lstrip(i[:16])
        name = i["class"]+ "/" + i["name"]
        i = os.path.join("data/LabeledDB_new/", name)
        meshi = load_meshes.load_OFF(i)
        vertices = meshi['vertices']
        faces = meshi['faces']
        meshi = o3d.geometry.TriangleMesh()
        meshi.vertices = o3d.utility.Vector3dVector(vertices)
        meshi.triangles = o3d.utility.Vector3iVector(faces)
        meshi.paint_uniform_color([1, 0.706, 0])
        meshi.compute_vertex_normals()
        o3d.visualization.draw_geometries([meshi])