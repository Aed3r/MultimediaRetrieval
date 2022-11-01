import numpy as np
from scipy.stats import wasserstein_distance
from tomlkit import string
import database
import ShapeDescriptors
import normalization
import os
import load_meshes

dbmngr = database.DatabaseManager()
database_length = 380

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
    loadedMesh_feature_vector = []
    loadedMesh_feature_vector.append(SurfaceAreaLoad)
    loadedMesh_feature_vector.append(CompactnessLoad)
    loadedMesh_feature_vector.append(VolumeLoad)
    loadedMesh_feature_vector.append(DiameterLoad)
    loadedMesh_feature_vector.append(EccentricityLoad)
    loadedMesh_feature_vector.append(RectangularityLoad)
    # print("loadedMesh_feature_vector: ")
    # print(loadedMesh_feature_vector)


    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()

    # initialize an empty list
    single_Value_Feature_Vector = [[[] for i in range (2)] for i in range(database_length)]
    # combine path and 6 descriptors in to one 3-dimensional list
    # [[[path] [S, C, V, D, E, R]]...[[path] [S, C, V, D, E, R]]]
    i = 0
    for mesh in meshes:
        single_Value_Feature_Vector[i][0].append(mesh['path'])
        single_Value_Feature_Vector[i][1].append(mesh['surface_area'])
        single_Value_Feature_Vector[i][1].append(mesh['compactness'])
        single_Value_Feature_Vector[i][1].append(mesh['volume'])
        single_Value_Feature_Vector[i][1].append(mesh['diameter'])
        single_Value_Feature_Vector[i][1].append(mesh['eccentricity'])
        single_Value_Feature_Vector[i][1].append(mesh['rectangularity'])
        i = i + 1
    # print(single_Value_Feature_Vector)

    
    # standardize single_value_feature_vector
    feature_vector_Standardized = []
    single_Value_Feature_Vectors = []
        
    # build a list to store all single_value vectors
    for i in range(database_length):
        single_Value_Feature_Vectors.append(single_Value_Feature_Vector[i][1])
    
    # standardization
    for vector in single_Value_Feature_Vectors:
        feature_vector_Norm = standardization(vector)
        feature_vector_Standardized.append(feature_vector_Norm)
    # print("feature_vector_Standardized: ")
    # print(feature_vector_Standardized)
    
    # calculate distance
    if distance_type == 'Euclidean':
        # Euclidean Distance Case: 
        dists = []
        for vector in feature_vector_Standardized:
            dist_between_feature_vectors = get_Euclidean_Distance(loadedMesh_feature_vector, vector)
            dists.append(dist_between_feature_vectors)
    elif distance_type == 'Cosine':
        # Cosine Distance Case: 
        dists = []
        for vector in feature_vector_Standardized:
            dist_between_feature_vectors = get_Cosine_Distance(loadedMesh_feature_vector, vector)
            dists.append(dist_between_feature_vectors)
    elif distance_type == 'EMD':
        # EMD Distance Case: 
        dists = []
        for vector in feature_vector_Standardized:
            dist_between_feature_vectors = get_Earth_Mover_Distance(loadedMesh_feature_vector, vector)
            dists.append(dist_between_feature_vectors)
        
    return dists, single_Value_Feature_Vector

def matching_histo_Feature(mesh, distance_type):
    # get features from the input (querying) mesh
    mesh = normalization.normalize(mesh)
    A3Load = ShapeDescriptors.A3(mesh, bins=10)[0]
    D1Load = ShapeDescriptors.D1(mesh, bins=10)[0]
    D2Load = ShapeDescriptors.D2(mesh, bins=10)[0]
    D3Load = ShapeDescriptors.D3(mesh, bins=10)[0]
    D4Load = ShapeDescriptors.D4(mesh, bins=10)[0]

    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()

    histo_Feature_Vector = [[[] for i in range (6)] for i in range(database_length)]
    # # combine path with each histogram features 
    # # [[[path] [A3]]...[[path] [A3]]]
    j = 0
    for mesh in meshes:
        histo_Feature_Vector[j][0] = mesh['path']
        histo_Feature_Vector[j][1] = mesh['A3']
        histo_Feature_Vector[j][2] = mesh['D1']
        histo_Feature_Vector[j][3] = mesh['D2']
        histo_Feature_Vector[j][4] = mesh['D3']
        histo_Feature_Vector[j][5] = mesh['D4']
        j = j + 1
    # print(histo_Feature_Vector)

    # build a list to store all histogram vectors
    histo_Feature_Vectors_A3 = []
    histo_Feature_Vectors_D1 = []
    histo_Feature_Vectors_D2 = []
    histo_Feature_Vectors_D3 = []
    histo_Feature_Vectors_D4 = []
    for i in range(database_length):
        histo_Feature_Vectors_A3.append(histo_Feature_Vector[i][1])
        histo_Feature_Vectors_D1.append(histo_Feature_Vector[i][2])
        histo_Feature_Vectors_D2.append(histo_Feature_Vector[i][3])
        histo_Feature_Vectors_D3.append(histo_Feature_Vector[i][4])
        histo_Feature_Vectors_D4.append(histo_Feature_Vector[i][5])

    # histogram features: A3-D4 do not need normalization

    if distance_type == 'EMD_histo_A3':
        # A3
        dists = []
        for vector in histo_Feature_Vectors_A3:
            dist_between_feature_vectors = get_Earth_Mover_Distance(A3Load, vector)
            dists.append(dist_between_feature_vectors)
    elif distance_type == 'EMD_histo_D1':
        # D1
        dists = []
        for vector in histo_Feature_Vectors_D1:
            dist_between_feature_vectors = get_Earth_Mover_Distance(D1Load, vector)
            dists.append(dist_between_feature_vectors)
    elif distance_type == 'EMD_histo_D2':
        # D2
        dists = []
        for vector in histo_Feature_Vectors_D2:
            dist_between_feature_vectors = get_Earth_Mover_Distance(D2Load, vector)
            dists.append(dist_between_feature_vectors)
    elif distance_type == 'EMD_histo_D3':
        # D3
        dists = []
        for vector in histo_Feature_Vectors_D3:
            dist_between_feature_vectors = get_Earth_Mover_Distance(D3Load, vector)
            dists.append(dist_between_feature_vectors)
    elif distance_type == 'EMD_histo_D4':
        # D4
        dists = []
        for vector in histo_Feature_Vectors_D4:
            dist_between_feature_vectors = get_Earth_Mover_Distance(D4Load, vector)
            dists.append(dist_between_feature_vectors)


    return dists, histo_Feature_Vector


def sorting(dists, feature_vector_with_path, distance_type, k = 5):
    # sort top K values in dists[]
    # from small to large values
    dists = np.asarray(dists) # transform to ndarray type
    sorted_index = sorted(range(len(dists)), key=lambda k: dists[k]) # index of sorted array
    sorted_dists = dists[sorted_index] # sorted array
    print("%d Smallest %s distance between loaded mesh's feature vector and feature vectors in database" %(k, distance_type))
    print(sorted_dists[:k])
    print("index of sorted array: ")
    print(sorted_index[:k])

    # find the corresponding paths/files of top k meshes
    print("5 smallest distance meshes corresponding paths: ")
    index = sorted_index[:k]
    paths = []
    for i in index:
        if distance_type == 'Euclidean' or distance_type == 'Cosine' or distance_type == 'EMD':
            print(feature_vector_with_path[i][0]) # single_Value_Feature_Vector
            path = feature_vector_with_path[i][0]
            paths.append(path)
        elif distance_type == 'EMD_histo_A3' or 'EMD_histo_D1' or 'EMD_histo_D2' or 'EMD_histo_D3' or 'EMD_histo_D4':
            print(feature_vector_with_path[i][0]) # histo_Feature_Vector
            path = feature_vector_with_path[i][0]
            paths.append(path)
    return dists, sorted_dists[:k], sorted_index[:k], paths


def minmaxNormalization(meshf, dbf):
    # fn = (f - fmin)/(fmax - fmin)
    meshfNorm = abs((meshf - min(dbf)) / (max(dbf) - min(dbf)))
    return meshfNorm

def standardNormalization(meshf, dbf):
    #fn = (f - favg)/fstd  prefered
    meshfNorm = abs((meshf - np.mean(dbf)) / np.std(dbf))
    return meshfNorm

# z-score standardization
def standardization(data):
    mu = np.mean(data, axis=0) 
    sigma = np.std(data, axis=0)
    result = (data - mu) / sigma
    return result


# run this function to save normalized features into database
def saveNormFeatures():
    meshes = dbmngr.get_all_with_extracted_features()
    SurfaceArea = []
    Compactness = []
    Volume = []
    Diameter = []
    Eccentricity = []
    Rectangularity = []
    for mesh in meshes:
        SurfaceArea.append(mesh['surface_area'])
        Compactness.append(mesh['compactness'])
        Volume.append(mesh['volume'])       #!Rectangularity
        Diameter.append(mesh['diameter']) 
        Eccentricity.append(mesh['eccentricity'])
        Rectangularity.append(mesh['rectangularity'])

    meshes = dbmngr.get_all_with_extracted_features()
    for mesh in meshes:
        mesh['surface_area'] = standardNormalization(mesh['surface_area'], SurfaceArea)
        mesh['compactness'] = standardNormalization(mesh['compactness'], Compactness)
        mesh['volume'] = standardNormalization(mesh['volume'], Volume)
        mesh['diameter'] = standardNormalization(mesh['diameter'], Diameter)
        mesh['eccentricity'] = standardNormalization(mesh['eccentricity'], Eccentricity)
        mesh['rectangularity'] = standardNormalization(mesh['rectangularity'], Rectangularity)
        dbmngr.update_one(mesh)
    pass

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

    Cosine_distance = (float(np.dot(vector_1, vector_2)) / 
               (np.linalg.norm(vector_1) * np.linalg.norm(vector_2)))

    # Sometimes using value ranges in [0, 1] to indicate the distance, so I normalize it here in case we need this
    normalized_Cosine_dist = (1 - Cosine_distance)/ 2.0

    return normalized_Cosine_dist

def get_Earth_Mover_Distance(vector_1, vector_2):

    EMD = wasserstein_distance(vector_1, vector_2)
    return EMD

if __name__ == "__main__":

    # load a mesh for test
    name = "Ant/82.off"
    # name = "Hand/181.off"
    path = os.path.join("data/LabeledDB_new/", name)
    mesh = load_meshes.load_OFF(path)
 
    # single value features test
    Euclidean_dists, single_Value_Feature_Vector = matching_single_Feature(mesh, 'Euclidean')
    _, _ , _, paths = sorting(Euclidean_dists, single_Value_Feature_Vector, distance_type='Euclidean', k = 5)
    Cosine_dists, single_Value_Feature_Vector = matching_single_Feature(mesh, 'Cosine')
    _, _ , _, paths = sorting(Cosine_dists, single_Value_Feature_Vector, distance_type='Cosine', k = 5)
    EMD_dists, single_Value_Feature_Vector = matching_single_Feature(mesh, 'EMD')
    _, _ , _, paths = sorting(EMD_dists, single_Value_Feature_Vector, distance_type='EMD', k = 5)


    # Histogram features test

    EMD_histo_dist, histo_Feature_Vector = matching_histo_Feature(mesh, 'EMD_histo_A3')
    _, _ , _, paths = sorting(EMD_histo_dist, histo_Feature_Vector, distance_type='EMD_histo_A3', k = 5)

    EMD_histo_dist, histo_Feature_Vector = matching_histo_Feature(mesh, 'EMD_histo_D1')
    _, _ , _, paths = sorting(EMD_histo_dist, histo_Feature_Vector, distance_type='EMD_histo_D1', k = 5)

    EMD_histo_dist, histo_Feature_Vector = matching_histo_Feature(mesh, 'EMD_histo_D2')
    _, _ , _, paths = sorting(EMD_histo_dist, histo_Feature_Vector, distance_type='EMD_histo_D2', k = 5)

    EMD_histo_dist, histo_Feature_Vector = matching_histo_Feature(mesh, 'EMD_histo_D3')
    _, _ , _, paths = sorting(EMD_histo_dist, histo_Feature_Vector, distance_type='EMD_histo_D3', k = 5)

    EMD_histo_dist, histo_Feature_Vector = matching_histo_Feature(mesh, 'EMD_histo_D4')
    _, _ , _, paths = sorting(EMD_histo_dist, histo_Feature_Vector, distance_type='EMD_histo_D4', k = 5)

    

