import numpy as np
from scipy.stats import wasserstein_distance
import database
import ShapeDescriptors
import normalization
import os
import load_meshes

dbmngr = database.DatabaseManager()

def getFeatures(mesh):
    mesh = normalization.normalize(mesh)
    SurfaceAreaLoad = ShapeDescriptors.get_Surface_Area(mesh)
    CompactnessLoad = ShapeDescriptors.get_Compactness(mesh)
    # RectangularityLoad = ShapeDescriptors.get_3D_Rectangularity(mesh)
    VolumeLoad = ShapeDescriptors.get_Volume(mesh)
    DiameterLoad = ShapeDescriptors.get_diameter(mesh)
    EccentricityLoad = ShapeDescriptors.get_eccentricity(mesh)

    # get features from database
    meshes = dbmngr.get_all_with_extracted_features()
    SurfaceArea = []
    Compactness = []
    Volume = []
    Diameter = []
    Eccentricity = []
    for mesh in meshes:
        SurfaceArea.append(mesh['surface_area'])
        Compactness.append(mesh['compactness'])
        Volume.append(mesh['volume'])       #!Rectangularity
        Diameter.append(mesh['diameter'])
        Eccentricity.append(mesh['eccentricity'])

    # normalize features of selected shape
    SurfaceAreaNorm = standardNormalization(SurfaceAreaLoad, SurfaceArea)
    CompactnessNorm = standardNormalization(CompactnessLoad, Compactness)
    VolumeNorm = standardNormalization(VolumeLoad, Volume)
    DiameterNorm = standardNormalization(DiameterLoad, Diameter)
    EccentricityNorm = standardNormalization(EccentricityLoad, Eccentricity)

    # calculate distance
    SurfaceAreaDis = [[None for i in range(2)] for j in range(len(SurfaceArea))]
    CompactnessDis = [[None for i in range(2)] for j in range(len(Compactness))]
    VolumeDis = [[None for i in range(2)] for j in range(len(Volume))]
    DiameterDis = [[None for i in range(2)] for j in range(len(Diameter))]
    EccentricityDis = [[None for i in range(2)] for j in range(len(Eccentricity))]
    meshes = dbmngr.get_all_with_extracted_features()
    n=0
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
        n += 1

    return SurfaceAreaDis, CompactnessDis, VolumeDis, DiameterDis, EccentricityDis


def minmaxNormalization(meshf, dbf):
    # fn = (f - fmin)/(fmax - fmin)
    meshfNorm = abs((meshf - min(dbf)) / (max(dbf) - min(dbf)))
    return meshfNorm

def standardNormalization(meshf, dbf):
    #fn = (f - favg)/fstd  prefered
    meshfNorm = abs((meshf - np.mean(dbf)) / np.std(dbf))
    return meshfNorm

# run this function to save normalized features into database
def saveNormFeatures():
    meshes = dbmngr.get_all_with_extracted_features()
    SurfaceArea = []
    Compactness = []
    Volume = []
    Diameter = []
    Eccentricity = []
    for mesh in meshes:
        SurfaceArea.append(mesh['surface_area'])
        Compactness.append(mesh['compactness'])
        Volume.append(mesh['volume'])       #!Rectangularity
        Diameter.append(mesh['diameter'])
        Eccentricity.append(mesh['eccentricity'])

    meshes = dbmngr.get_all_with_extracted_features()
    for mesh in meshes:
        mesh['surface_area'] = standardNormalization(mesh['surface_area'], SurfaceArea)
        mesh['compactness'] = standardNormalization(mesh['compactness'], Compactness)
        mesh['volume'] = standardNormalization(mesh['volume'], Volume)
        mesh['diameter'] = standardNormalization(mesh['diameter'], Diameter)
        mesh['eccentricity'] = standardNormalization(mesh['eccentricity'], Eccentricity)
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
    normalized_dist = (1 - Cosine_distance)/ 2.0

    return Cosine_distance

def get_Earth_Mover_Distance(vector_1, vector_2):

    EMD = wasserstein_distance(vector1, vector2)
    return EMD

if __name__ == "__main__":
    # test Euclidean Distance
    list1 = [1,1,1]
    list2 = [2,3,4]
    vector1 = np.asarray(list1)
    vector2 = np.asarray(list2)
    #euclidean_dist = get_Euclidean_Distance(vector1, vector2)

    name = "Ant/82.off"
    path = os.path.join("data/LabeledDB_new/", name)
    mesh = load_meshes.load_OFF(path)
    euclidean_dist = getFeatures(mesh)

    print(euclidean_dist[2])

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
