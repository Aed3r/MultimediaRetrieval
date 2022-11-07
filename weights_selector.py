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
import distance_functions

dbmngr = database.DatabaseManager()
meshes = dbmngr.get_all_with_extracted_features()

k = 5

mu, sigma = util.get_single_features_mean_and_sigma_from_meshes(meshes)
print("mu: ", mu)
print("sigma: ", sigma)

meshes = dbmngr.get_all_with_extracted_features()
#meshes = dbmngr.get_x_with_extracted_features(1)

feature_weights = {
    "surface_area": 0.0,
    "volume": 0.0,
    "compactness": 0.0,
    "diameter": 0.0,
    "eccentricity": 0.0,
    "rectangularity": 0.0,
    "A3": 0.0,
    "D1": 0.0,
    "D2": 0.0,
    "D3": 0.0,
    "D4": 0.0
}

tot = dbmngr.get_mesh_count_with_features()
for comp_mesh in tqdm(meshes, ncols=150, total=tot):
    dists = [[[] for i in range(14)] for i in range(tot)]

    singleValFeatures = [comp_mesh['surface_area'], comp_mesh['compactness'], comp_mesh['volume'],
                         comp_mesh['diameter'], comp_mesh['eccentricity'], comp_mesh['rectangularity']]
    multiValFeatures = [comp_mesh['A3'], comp_mesh['D1'], comp_mesh['D2'], comp_mesh['D3'], comp_mesh['D4']]

    i = 0
    db_meshes = dbmngr.get_all_with_extracted_features()
    for db_mesh in db_meshes:
        # Check if the mesh is the querying mesh, remove it if it is
        if db_mesh["name"] == comp_mesh["name"]:
            continue

        if "surface_area_std" not in db_mesh:
            raise Exception("Error: the database meshes have not been standardized. Please run `python3 main.py standardizeDB` first.")

        # get the meshes features
        dbSingleValFeatures = [db_mesh['surface_area_std'], db_mesh['compactness_std'], db_mesh['volume_std'],
                               db_mesh['diameter_std'], db_mesh['eccentricity_std'], db_mesh['rectangularity_std']]
        dbMultiValFeatures = [db_mesh['A3'], db_mesh['D1'], db_mesh['D2'], db_mesh['D3'], db_mesh['D4']]

        #dbSingleValFeatures = util.standardize(dbSingleValFeatures, mu, sigma)
        k = 0
        dists[i][0] = db_mesh["path"]
        dists[i][1] = distance_functions.get_Euclidean_Distance(singleValFeatures, dbSingleValFeatures) * distance_functions.SCALARWEIGHT
        for j in range(len(multiValFeatures)):
            dists[i][1] += distance_functions.get_Earth_Mover_Distance(multiValFeatures[j], dbMultiValFeatures[j]) * distance_functions.VECTORWEIGHT
        k = 2
        for j in range(len(singleValFeatures)):
            dists[i][k] = distance_functions.get_Euclidean_Distance(singleValFeatures[j], dbSingleValFeatures[j]) * distance_functions.SCALARWEIGHT
            k += 1
        for j in range(len(multiValFeatures)):
            dists[i][k] = distance_functions.get_Earth_Mover_Distance(multiValFeatures[j], dbMultiValFeatures[j]) * distance_functions.VECTORWEIGHT
            k += 1
        i += 1

    #sort the distances
    dists.sort(key=lambda x: x[1])
    bestDists = dists[:k]

    for i in range(k):
        res = dbmngr.get_by_path(bestDists[i][0])
        #res[i]["distance"] = dists[i][1]

        if comp_mesh["class"] == res["class"]:
            mult = 1
        else:
            mult = -1

        feature_weights["surface_area"] += mult * dists[i][2]
        feature_weights["compactness"] += mult * dists[i][3]
        feature_weights["volume"] += mult * dists[i][4]
        feature_weights["diameter"] += mult * dists[i][5]
        feature_weights["eccentricity"] += mult * dists[i][6]
        feature_weights["rectangularity"] += mult * dists[i][7]
        feature_weights["A3"] += mult * dists[i][8]
        feature_weights["D1"] += mult * dists[i][9]
        feature_weights["D2"] += mult * dists[i][10]
        feature_weights["D3"] += mult * dists[i][11]
        feature_weights["D4"] += mult * dists[i][12]

print("feature_weights: ")
print("- surface_area: ", str(feature_weights["surface_area"]))
print("- compactness: ", str(feature_weights["compactness"]))
print("- volume: ", str(feature_weights["volume"]))
print("- diameter: ", str(feature_weights["diameter"]))
print("- eccentricity: ", str(feature_weights["eccentricity"]))
print("- rectangularity: ", str(feature_weights["rectangularity"]))
print("- A3: ", str(feature_weights["A3"]))
print("- D1: ", str(feature_weights["D1"]))
print("- D2: ", str(feature_weights["D2"]))
print("- D3: ", str(feature_weights["D3"]))
print("- D4: ", str(feature_weights["D4"]))

# Export data to txt file
with open("data/feature_weights.txt", "w") as f:
    f.write("feature_weights: \n")
    f.write("- surface_area: " + str(feature_weights["surface_area"]) + "\n")
    f.write("- compactness: " + str(feature_weights["compactness"]) + "\n")
    f.write("- volume: " + str(feature_weights["volume"]) + "\n")
    f.write("- diameter: " + str(feature_weights["diameter"]) + "\n")
    f.write("- eccentricity: " + str(feature_weights["eccentricity"]) + "\n")
    f.write("- rectangularity: " + str(feature_weights["rectangularity"]) + "\n")
    f.write("- A3: " + str(feature_weights["A3"]) + "\n")
    f.write("- D1: " + str(feature_weights["D1"]) + "\n")
    f.write("- D2: " + str(feature_weights["D2"]) + "\n")
    f.write("- D3: " + str(feature_weights["D3"]) + "\n")
    f.write("- D4: " + str(feature_weights["D4"]) + "\n")

    
# - eccentricity:  -2241.0246355515646
# - rectangularity:  -64.99672346741919
# - A3:  -11.124653094570562
# - D1:  -17.600144437692897
# - D2:  -10.435040404289847
# - D3:  -28.309109462658554
