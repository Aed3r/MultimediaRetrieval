import numpy as np
from scipy.stats import wasserstein_distance
import database
import ShapeDescriptors
import normalization
import os
import load_meshes

dbmngr = database.DatabaseManager()
database_length = 380

def getFeatures(mesh, distance_type):
    # get features from the input (querying) mesh
    mesh = normalization.normalize(mesh)
    SurfaceAreaLoad = ShapeDescriptors.get_Surface_Area(mesh)
    CompactnessLoad = ShapeDescriptors.get_Compactness(mesh)
    VolumeLoad = ShapeDescriptors.get_Volume(mesh)
    DiameterLoad = ShapeDescriptors.get_diameter(mesh)
    EccentricityLoad = ShapeDescriptors.get_eccentricity(mesh)
    RectangularityLoad = ShapeDescriptors.get_3D_Rectangularity(mesh)

    A3Load = ShapeDescriptors.A3(mesh)
    D1Load = ShapeDescriptors.D1(mesh)
    D2Load = ShapeDescriptors.D2(mesh)
    D3Load = ShapeDescriptors.D3(mesh)
    D4Load = ShapeDescriptors.D4(mesh)

    loadedMesh_feature_vector = []
    loadedMesh_feature_vector.append(SurfaceAreaLoad)
    loadedMesh_feature_vector.append(CompactnessLoad)
    loadedMesh_feature_vector.append(VolumeLoad)
    loadedMesh_feature_vector.append(DiameterLoad)
    loadedMesh_feature_vector.append(EccentricityLoad)
    loadedMesh_feature_vector.append(RectangularityLoad)
    print("loadedMesh_feature_vector: ")
    print(loadedMesh_feature_vector)

    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()
    SurfaceArea = []
    Compactness = []
    Volume = []
    Diameter = []
    Eccentricity = []
    Rectangularity = []
    # build feature vectors
    single_Value_Feature_Vector = [[[] for i in range (2)] for i in range(database_length)]
    # print(single_Value_Feature_Vector)

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

    '''
    @author Sichun
    '''
    # for mesh in meshes:
    #     SurfaceArea.append(mesh['surface_area'])
    #     Compactness.append(mesh['compactness'])
    #     Volume.append(mesh['volume'])       #!Rectangularity
    #     Diameter.append(mesh['diameter'])
    #     Eccentricity.append(mesh['eccentricity'])
    #     Rectangularity.append(mesh['rectangularity'])

    # # normalize features of selected shape
    # SurfaceAreaNorm = standardNormalization(SurfaceAreaLoad, SurfaceArea)
    # CompactnessNorm = standardNormalization(CompactnessLoad, Compactness)
    # VolumeNorm = standardNormalization(VolumeLoad, Volume)
    # DiameterNorm = standardNormalization(DiameterLoad, Diameter)
    # EccentricityNorm = standardNormalization(EccentricityLoad, Eccentricity)
    # RectangularityNorm = standardNormalization(RectangularityLoad, Rectangularity)
    
    
    '''
    @author Nisha
    '''
    feature_vector_Standardized = []
    single_Value_Feature_Vectors = []
    
    # build a list to store all single_value vectors
    for i in range(database_length):
        single_Value_Feature_Vectors.append(single_Value_Feature_Vector[i][1])
    
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
        # print("Euclidean distance:")
        # print(dists)

    elif distance_type == 'Cosine':
        # Euclidean Distance Case: 
        dists = []
        for vector in feature_vector_Standardized:
            dist_between_feature_vectors = get_Cosine_Distance(loadedMesh_feature_vector, vector)
            dists.append(dist_between_feature_vectors)
        # print("Cosine distance:")
        # print(dists)
    
    # sort top K values in dists[]
    # from small to large values
    dists = np.asarray(dists) # transform to ndarray type
    sorted_index = sorted(range(len(dists)), key=lambda k: dists[k]) # index of sorted array
    sorted_dists = dists[sorted_index] # sorted array
    k = 5
    print("%d Smallest %s distance between loaded mesh's feature vector and feature vectors in database" %(k, distance_type))
    print(sorted_dists[:k])
    print("index of sorted array: ")
    print(sorted_index[:k])

    # find the corresponding paths/files of top k meshes
    print("5 smallest distance meshes corresponding paths: ")
    index = sorted_index[:k]
    paths = []
    for i in index:
        print(single_Value_Feature_Vector[i][0])
        path = single_Value_Feature_Vector[i][0]
        paths.append(path)

    return dists, sorted_dists[:k], sorted_index[:k], paths
    '''
    @author Sichun
    '''
    # calculate distance
    SurfaceAreaDis = [[None for i in range(2)] for j in range(len(SurfaceArea))]
    # print(SurfaceAreaDis)
    # print(len(SurfaceAreaDis)) # 380
    CompactnessDis = [[None for i in range(2)] for j in range(len(Compactness))]
    VolumeDis = [[None for i in range(2)] for j in range(len(Volume))]
    DiameterDis = [[None for i in range(2)] for j in range(len(Diameter))]
    EccentricityDis = [[None for i in range(2)] for j in range(len(Eccentricity))]
    RectangularityDis = [[None for i in range(2)] for j in range(len(Rectangularity))]


    meshes = dbmngr.get_all_with_extracted_features()
    n=0

    if distance_type == 'Euclidean':
        # Euclidean Distance Case: 
        for mesh in meshes:
            SurfaceAreaDis[n][0] = mesh['path']
            SurfaceAreaDis[n][1] = get_Euclidean_Distance(SurfaceAreaNorm, mesh['surface_area'])
            CompactnessDis[n][0] = mesh['path']
            CompactnessDis[n][1] = get_Euclidean_Distance(CompactnessNorm, mesh['compactness'])
            VolumeDis[n][0] = mesh['path']
            VolumeDis[n][1] = get_Euclidean_Distance(VolumeNorm, mesh['volume'])
            DiameterDis[n][0] = mesh['path']
            DiameterDis[n][1] = get_Euclidean_Distance(DiameterNorm, mesh['diameter'])
            EccentricityDis[n][0] = mesh['path']
            EccentricityDis[n][1] = get_Euclidean_Distance(EccentricityNorm, mesh['eccentricity'])
            RectangularityDis[n][0] = mesh['path']
            RectangularityDis[n][1] = get_Euclidean_Distance(RectangularityNorm, mesh['rectangularity'])
            n += 1
    elif distance_type == 'Cosine':
        # Cosine Distance Case:
        for mesh in meshes:
            SurfaceAreaDis[n][0] = mesh['path']
            SurfaceAreaDis[n][1] = get_Cosine_Distance(SurfaceAreaNorm, mesh['surface_area'])
            CompactnessDis[n][0] = mesh['path']
            CompactnessDis[n][1] = get_Cosine_Distance(CompactnessNorm, mesh['compactness'])
            VolumeDis[n][0] = mesh['path']
            VolumeDis[n][1] = get_Cosine_Distance(VolumeNorm, mesh['volume'])
            DiameterDis[n][0] = mesh['path']
            DiameterDis[n][1] = get_Cosine_Distance(DiameterNorm, mesh['diameter'])
            EccentricityDis[n][0] = mesh['path']
            EccentricityDis[n][1] = get_Cosine_Distance(EccentricityNorm, mesh['eccentricity'])
            RectangularityDis[n][0] = mesh['path']
            RectangularityDis[n][1] = get_Cosine_Distance(RectangularityNorm, mesh['rectangularity'])
            n += 1

    return SurfaceAreaDis, CompactnessDis, VolumeDis, DiameterDis, EccentricityDis, RectangularityDis


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
    # # test Euclidean Distance
    # list1 = [1,1,1]
    # list2 = [2,3,4]
    # vector1 = np.asarray(list1)
    # vector2 = np.asarray(list2)
    #euclidean_dist = get_Euclidean_Distance(vector1, vector2)

    name = "Ant/82.off"
    # name = "Hand/181.off"
    path = os.path.join("data/LabeledDB_new/", name)
    mesh = load_meshes.load_OFF(path)
    euclidean_dist = getFeatures(mesh, 'Euclidean')
    # cosine_dist = getFeatures(mesh, 'Cosine')
    



    # EMD_dist = getFeatures(mesh, 'EMD')



    #print('Euclidean distance: %.5f' %i)
    #print("get_Euclidean_Distance test finished!")
    #
    # cosine_dist = get_Cosine_Distance(vector1, vector2)
    # print('Cosine distance: %.5f' %cosine_dist)
    # print("get_Cosine_Distance test finished!")
    #
    # EMD = get_Earth_Mover_Distance(vector1, vector2)
    # print("Earth Mover distance: %.5f" %EMD)
    # print("get_Earth_Mover_Distance test finished!")
