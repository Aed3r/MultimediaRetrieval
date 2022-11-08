import time
import os
import util
from tqdm import tqdm
from annoy import AnnoyIndex
import numpy as np

DEFAULTANNEXPORT = os.path.join("data", "annoy_index.ann")
SCALARWEIGHT = 0.25
VECTORWEIGHT = 0.75
#WEIGHTS = [SCALARWEIGHT/6 for x in range(6)] + [VECTORWEIGHT/50 for x in range(50)]
WEIGHTS = [0.04284498, 0.03419309, 0.0450837 , 0.04429373, 0.02135738, 0.04512191] + [0.15362412 for x in range(10)] + [0.15341157 for x in range(10)] + [0.15363206 for x in range(10)] + [0.15307127 for x in range(10)] + [0.15336618 for x in range(10)]
FEATUREVECTORLENGTH = 56
NUMTREES = 500
ANNMETRIC = "angular"

# Creates the Annoy index from the given meshes
def create_ann(meshes, exportPath=DEFAULTANNEXPORT):
    # Create the Annoy index
    t = AnnoyIndex(FEATUREVECTORLENGTH, metric=ANNMETRIC) # Metric can be "angular", "euclidean", "manhattan", "hamming", or "dot"

    # Add the meshes to the index
    tot = len(list(meshes.clone()))
    for i, mesh in enumerate(tqdm(meshes, desc="Adding meshes to index", ncols=150, total=tot)):
        v = util.get_feature_vector_from_mesh(mesh, WEIGHTS)
        t.add_item(i, vector=v)
    
    # Build the index
    print("Building index...")
    start = time.time()
    t.build(NUMTREES)
    print(f"Index built successfully. ({round(time.time() - start)}s)")

    # Check if the export path exists
    if not os.path.exists(os.path.dirname(exportPath)):
        os.makedirs(os.path.dirname(exportPath))

    # Save the index
    t.save(exportPath)

    # Unload the index
    t.unload()

# Loads the Annoy index from the given path
def load_ann(importPath=DEFAULTANNEXPORT):
    # Load the index
    t = AnnoyIndex(FEATUREVECTORLENGTH, metric=ANNMETRIC)
    try:
        t.load(importPath)
    except:
        raise Exception("Error: the Annoy index does not exist. Please run `python3 main.py create_ann` first.")

    return t

# Returns the k nearest neighbors of the query mesh
# Assumes that the querymesh already has standardized features extracted
def get_knn(queryMesh, k, importPath=DEFAULTANNEXPORT):
    # Load the index
    t = load_ann(DEFAULTANNEXPORT)

    # Get the query vector
    v = util.get_feature_vector_from_mesh(queryMesh, WEIGHTS)

    # Get the k nearest neighbors
    nn = t.get_nns_by_vector(v, k+1, include_distances=True)

    # Remove the query mesh from the results
    for i, dist in enumerate(nn[1]):
        if np.isclose(dist, 0):
            nn[0].pop(i)
            nn[1].pop(i)
            break

    # Unload the index
    t.unload()

    return nn

# Returns the meshes with a distance less than the given threshold r from the query mesh
# Assumes that the querymesh already has standardized features extracted
def get_rnn(queryMesh, r, importPath=DEFAULTANNEXPORT):
    # Load the index
    t = load_ann(DEFAULTANNEXPORT)

    # Get the query vector
    v = util.get_feature_vector_from_mesh(queryMesh, WEIGHTS)

    # Get the k nearest neighbors
    nn = t.get_nns_by_vector(v, t.get_n_items(), include_distances=True)

    # Unload the index
    t.unload()

    # Filter the results
    filtered = []
    for i, dist in enumerate(nn[1]):
        if dist <= r:
            filtered.append((nn[0][i], dist))

    return filtered

if __name__ == "__main__":
    import database

    start = time.time()

    dbmngr = database.DatabaseManager()

    query = dbmngr.get_all_with_extracted_features()[0]
    #query = dbmngr.query({'class': 'Bird'})[0]

    # Get the k nearest neighbors
    nn = get_knn(queryMesh=query, k=10)

    for i, x in enumerate(nn[0]):
        mesh = dbmngr.get_all_with_extracted_features()[x]
        print("#" + str(i) + ": " + mesh["path"] + " - " + mesh["class"] + "(dist=" + str(nn[1][i]) + ")")

    print("Time: " + str(time.time() - start))