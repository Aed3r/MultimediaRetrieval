import os
import time
import load_meshes
import open3d as o3d
from tqdm import tqdm
import math
import util
import numpy as np
import normalization
import matplotlib.pyplot as plt
from tqdm.contrib.concurrent import process_map

FEATUREPLOTSPATH = "data/featurePlots/"

# Calculate the distance between barycenter and origin
def get_barycenter_origin_distance(mesh):
    barycenter = get_shape_barycenter(mesh)
    origin = np.asarray([0, 0, 0])
    barycenter_origin_distance = distance_between(barycenter, origin)

    return barycenter_origin_distance


# Calculate the absolute cosine similarity between 2 vectors
def get_Cosine_similarity(mesh):

    eigenvalues, eigenvectors = compute_PCA(mesh)
    largest_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]
    vec1 = largest_eigenvector
    # v2: x-axis
    vec2 = [1, 0, 0]
    vec1 = np.asarray(vec1)
    vec2 = np.asarray(vec2)
    # cosine_similarity = vec1.dot(vec2) / np.linalg.norm(vec1) * np.linalg.norm(vec2)
    cosine_similarity = (float(np.dot(vec1, vec2)) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    absolute_cosine_similarity = abs(cosine_similarity)

    return absolute_cosine_similarity


# Calculate the length of the longest AABB edge
def get_longest_AABB_edge(mesh):
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(mesh['vertices']))
    longest_AABB_edge = aabb.get_max_extent()
    
    return longest_AABB_edge

# Calculate the surface area of a mesh in the data list
def get_Surface_Area(data):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']), o3d.cpu.pybind.utility.Vector3iVector(data['faces']))
    surface_area = o3d.geometry.TriangleMesh.get_surface_area(mesh)
    return surface_area

# Calculate the volume of meshes in the data list
def get_Volume(data):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']), o3d.cpu.pybind.utility.Vector3iVector(data['faces']))
    volume = o3d.geometry.TriangleMesh.get_volume(mesh)
    return volume

def get_Compactness(data):
    SurfaceArea = get_Surface_Area(data)
    Volume = get_Volume(data)
    S_3 = SurfaceArea * SurfaceArea * SurfaceArea  # cannot use np.power() here, or the list would transfer to array automatically
    V_2 = Volume * Volume
    comP = S_3 / (36 * (math.pi) * V_2)
    return comP

# Calculate the 3D rectangularity of meshes in the data list
def get_3D_Rectangularity(data):
    shape_volume = get_Volume(data) # return a shape_volume list
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(data['vertices']))
    OBB = o3d.geometry.AxisAlignedBoundingBox.get_oriented_bounding_box(aabb)
    Rectangularity = shape_volume/OBB.volume()
    return Rectangularity

# Calculate the diameter of a mesh list
# definition: largest distance between any two surface points in a mesh
def get_diameter(data):
    mesh = data
    baryCenter = util.get_shape_barycenter(mesh)

    distance_array1 = [] # the array to store all distances between vertex 1 and barycenter
    for j in range(len(mesh['vertices'])):
        distance_barycenter_vertex1 = util.distance_between(mesh['vertices'][j], baryCenter)
        distance_array1.append(distance_barycenter_vertex1)
    distance1_Maximum = max(distance_array1)
    max_index1 = distance_array1.index(distance1_Maximum) # return the index of the maximum value in distances, indicates the index of its vertex
    vertex1 = mesh['vertices'][max_index1]
    distance_array2 = [] # the array to store all distances between vertex 1 and vertex 2
    for n in range(len(mesh['vertices'])):
        distance_vertex1_vertex2 = util.distance_between(mesh['vertices'][n], vertex1)
        distance_array2.append(distance_vertex1_vertex2)
    distance2_Maximum = max(distance_array2)

    return distance2_Maximum

# Calculate the eccentricity of a mesh list
def get_eccentricity(data):
    mesh = data
    eigenvalues, eigenvectors = util.compute_PCA(mesh)
    Ecc = max(eigenvalues)/min(eigenvalues)
    return Ecc

# Runs the sampling function func for 2 vertices
def sample2Verts(mesh, func):
    k = pow(mesh["numVerts"], 1.0 / 2.0)
    res = []

    i = 0
    while i < k:
        # Get random number from 0 to n
        vi = util.random_vertices(mesh, 1)[0]
        j = 0
        while j < k:
            vj = util.random_vertices(mesh, 1)[0]
            if (set(vi) == set(vj)):
                continue

            res.append(func(vi, vj))
            j += 1
        i += 1

    return res

# Runs the sampling function func for 3 vertices
def sample3Verts(mesh, func):
    k = pow(mesh["numVerts"], 1.0 / 3.0)
    res = []

    i = 0
    while i < k:
        # Get random number from 0 to n
        vi = util.random_vertices(mesh, 1)[0]
        j = 0
        while j < k:
            vj = util.random_vertices(mesh, 1)[0]
            if (set(vi) == set(vj)):
                continue
            l = 0
            while l < k:
                vl = util.random_vertices(mesh, 1)[0]
                if (set(vl) == set(vi) or set(vl) == set(vj)):
                    continue

                res.append(func(vi, vj, vl))
                l += 1
            j += 1
        i += 1 
    
    return res

# Runs the sampling function func for 4 vertices
def sample4Verts(mesh, func):
    k = pow(mesh["numVerts"], 1.0 / 4.0)
    res = []

    i = 0
    while i < k:
        # Get random number from 0 to n
        vi = util.random_vertices(mesh, 1)[0]
        j = 0
        while j < k:
            vj = util.random_vertices(mesh, 1)[0]
            if (set(vi) == set(vj)):
                continue
            l = 0
            while l < k:
                vl = util.random_vertices(mesh, 1)[0]
                if (set(vl) == set(vi) or set(vl) == set(vj)):
                    continue
                m = 0
                while m < k:
                    vm = util.random_vertices(mesh, 1)[0]
                    if (set(vm) == set(vi) or set(vm) == set(vj) or set(vm) == set(vl)):
                        continue

                    res.append(func(vi, vj, vl, vm))
                    m += 1
                l += 1
            j += 1
        i += 1 

    return res

# Normalize the given histogram
def normalize_histogram(histogram, bins):
    histogram = list(histogram / np.sum(histogram))
    return histogram, bins

# Sample the angle of 3 random vertices in the mesh
def A3(mesh, bins=25):
    res = sample3Verts(mesh, util.angle_between)

    # Calculate histogram of angles
    histogram, bins = np.histogram(res, bins, range=(0, math.pi), density=True)
    
    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return normalize_histogram(histogram, bins)

# Sample the distance between barycenter and random vertex in the mesh
def D1(mesh, bins=22):
    barycenter = np.array(util.get_shape_barycenter(mesh))
    res = []

    for i in range(mesh["numVerts"]):
        res.append(util.distance_between(barycenter, np.array(mesh["vertices"][i])))

    # Calculate histogram of distances
    histogram, bins = np.histogram(res, bins, range=(0,1), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return normalize_histogram(histogram, bins)

# Sample the distance between two random vertices in the mesh
def D2(mesh, bins=23):
    res = sample2Verts(mesh, util.distance_between)

    # Calculate histogram of distances
    histogram, bins = np.histogram(res, bins, range=(0, math.sqrt(3)), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return normalize_histogram(histogram, bins)

# Sample the square root of area of triangle given by 3 random vertices 
def D3(mesh, bins=25):
    res = sample3Verts(mesh, util.triangle_area)

    # Calculate histogram of areas
    histogram, bins = np.histogram(res, bins, range=(0, (math.sqrt(3) / 2) ** (1/2) ), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return normalize_histogram(histogram, bins)

# Sample the cube root of volume of tetrahedron formed by 4 random vertices 
def D4(mesh, bins=29):
    res = sample4Verts(mesh, util.tetrahedron_volume)

    # Calculate histogram of volumes
    histogram, bins = np.histogram(res, bins, range=(0,(1/3) ** (1/3)), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return normalize_histogram(histogram, bins)

# Generate plots for the A3, D1, D2, D3 and D4 features for the given meshes
# Use doNormalization=False if the meshes are already normalized
# The function takes care of normalizing the histograms
def genFeaturePlots(meshes, doNormalization=False):
    features = {}

    print("Generating feature plots...")
    startTime = time.time()

    if doNormalization:
        for i in tqdm(range(len(meshes)), desc="Normalizing meshes", ncols=150):
            meshes[i] = normalization.normalize(meshes[i])

    features["A3"] = {}
    for i in tqdm(range(len(meshes)), desc="Calculating A3 shape descriptor", ncols=150):
        meshClass = meshes[i]["class"]
        if meshClass not in features["A3"]:
            features["A3"][meshClass] = {}
    
        features["A3"][meshClass][meshes[i]["name"]] = A3(meshes[i])

    features["D1"] = {}
    for i in tqdm(range(len(meshes)), desc="Calculating D1 shape descriptor", ncols=150):
        meshClass = meshes[i]["class"]
        if meshClass not in features["D1"]:
            features["D1"][meshClass] = {}

        features["D1"][meshClass][meshes[i]["name"]] = D1(meshes[i])

    features["D2"] = {}
    for i in tqdm(range(len(meshes)), desc="Calculating D2 shape descriptor", ncols=150):
        meshClass = meshes[i]["class"]
        if meshClass not in features["D2"]:
            features["D2"][meshClass] = {}
        
        features["D2"][meshClass][meshes[i]["name"]] = D2(meshes[i])

    features["D3"] = {}
    for i in tqdm(range(len(meshes)), desc="Calculating D3 shape descriptor", ncols=150):
        meshClass = meshes[i]["class"]
        if meshClass not in features["D3"]:
            features["D3"][meshClass] = {}
        
        features["D3"][meshClass][meshes[i]["name"]] = D3(meshes[i])

    features["D4"] = {}
    for i in tqdm(range(len(meshes)), desc="Calculating D4 shape descriptor", ncols=150):
        meshClass = meshes[i]["class"]
        if meshClass not in features["D4"]:
            features["D4"][meshClass] = {}
        
        features["D4"][meshClass][meshes[i]["name"]] = D4(meshes[i])

    for descriptor in tqdm(features, desc="Generating plots", ncols=150):
        # Verify that the save location exists
        saveLocation = os.path.join(FEATUREPLOTSPATH, descriptor)
        if not os.path.exists(saveLocation):
            os.makedirs(saveLocation)

        for shapeClass in features[descriptor]:
            plt.figure()
            plt.title(f"{descriptor} shape descriptor for the '{shapeClass}' shape class")
            for shape in features[descriptor][shapeClass]:
                feature = features[descriptor][shapeClass][shape]
                plt.plot(feature[1][:-1], feature[0])

            plt.savefig(os.path.join(saveLocation, f"{shapeClass}.png"))
            plt.close()
    
    print(f"Feature plots generated in {time.time() - startTime} seconds")

def extract_all_features(mesh, returnFullMesh=False):
    if returnFullMesh:
        res = mesh
    else:
        res = {}
        res["path"] = mesh["path"]

    try:
        res["volume"] = get_Volume(mesh)
        res["surface_area"] = get_Surface_Area(mesh)
        res["compactness"] = get_Compactness(mesh)
        res["diameter"] = get_diameter(mesh)
        res["eccentricity"] = get_eccentricity(mesh)
        res["rectangularity"] = get_3D_Rectangularity(mesh)
        res["A3"] = A3(mesh, bins=10)[0]
        res["D1"] = D1(mesh, bins=10)[0]
        res["D2"] = D2(mesh, bins=10)[0]
        res["D3"] = D3(mesh, bins=10)[0]
        res["D4"] = D4(mesh, bins=10)[0]
    except:
        print("Error in extracting features for mesh: ", mesh["path"])
    return res

def extract_all_features_from_meshes(meshes):
    return process_map(extract_all_features, meshes, desc="Extracting features", ncols=150)

if __name__ == '__main__':
    import statistics
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, fromNORM=False, randomSample=-1, returnInfoOnly=False)
    #data = 'data/LabeledDB_new/Bird/256.off'
    #mesh = load_meshes.load_OFF(data)

    normalized_data = load_meshes.get_meshes(fromLPSB=False, fromPRIN=False, fromNORM=True, randomSample=-1, returnInfoOnly=False)

    barycenter_origin_distance = []
    absolute_cosine_similarity = []
    longest_AABB_edge = []
    
    for i in range(len(data)):
        # barycenter_origin_distance.append(util.get_barycenter_origin_distance(data[i]))
        absolute_cosine_similarity.append(util.get_Cosine_similarity(data[i]))
        longest_AABB_edge.append(util.get_longest_AABB_edge(data[i]))
        

    for i in range(len(data)):
        print("The %dth origin data feature: " %(i+1))
        print("barycenter_origin_distance: %.5f" %barycenter_origin_distance[i])
        print("absolute_cosine_similarity: %.5f" %absolute_cosine_similarity[i])
        print("AABB_edge_0: %.5f" %longest_AABB_edge[i])


    
    # store in sheet and histogram
    # statistics.draw_histogram(barycenter_origin_distance, 'barycenter distance')
    # statistics.save_Excel(barycenter_origin_distance, 'baryCenter_origin_distance_I')
    # statistics.draw_histogram(absolute_cosine_similarity, 'cosine similarity')
    # statistics.save_Excel(absolute_cosine_similarity, 'absolute_cosine_similarity_I')
    statistics.draw_histogram(longest_AABB_edge, 'AABB')
    statistics.save_Excel(longest_AABB_edge, 'longest_AABB_edge_I')

    barycenter_origin_distance_Normalized = []
    absolute_cosine_similarity_Normalized = []
    longest_AABB_edge_Normalized = []


    for i in range(len(normalized_data)):
        # barycenter_origin_distance_Normalized.append(util.get_barycenter_origin_distance(normalized_data[i]))
        # absolute_cosine_similarity_Normalized.append(util.get_Cosine_similarity(normalized_data[i]))
        longest_AABB_edge_Normalized.append(util.get_longest_AABB_edge(normalized_data[i]))
        

    # for i in range(len(data)):
        # print("The %dth origin data feature: " %(i+1))
        # print("barycenter_origin_distance_normalized: %.5f" %barycenter_origin_distance_Normalized[i])
        # print("absolute_cosine_similarity_normalized: %.5f" %absolute_cosine_similarity_Normalized[i])
        # print("longest_AABB_edge_normalized: %.5f" %longest_AABB_edge_Normalized[i])
    # statistics.draw_histogram(barycenter_origin_distance_Normalized, 'barycenter distance')
    # statistics.save_Excel(barycenter_origin_distance_Normalized, 'baryCenter_origin_distance_N')
    # statistics.draw_histogram(absolute_cosine_similarity_Normalized, 'cosine similarity')
    # statistics.save_Excel(absolute_cosine_similarity_Normalized, 'absolute_cosine_similarity_N')
    statistics.draw_histogram(longest_AABB_edge_Normalized, 'AABB')
    statistics.save_Excel(longest_AABB_edge_Normalized, 'longest_AABB_edge_N')


    # # get normalized mesh
    # util_data = []
    # for mesh in data:
    #     util_data.append(normalization.normalize(mesh))
    # #util_data.append(normalization.normalize(mesh))

    # Compactness = []
    # Rectangularity = []
    # SurfaceArea = []
    # Volume = []
    # Diameter = []
    # Eccentricity = []
    # for i in range(len(util_data)):
    #     Volume.append(get_Volume(util_data[i]))
    #     SurfaceArea.append(get_Surface_Area(util_data[i]))
    #     Compactness.append(get_Compactness(util_data[i]))
    #     Rectangularity.append(get_3D_Rectangularity(util_data[i]))
    #     Diameter.append(get_diameter(util_data[i]))
    #     Eccentricity.append(get_eccentricity(util_data[i]))

    # for i in range(len(util_data)):
    #     print("The %dth data feature: " %(i+1))
    #     print("Volume: %.20f" %Volume[i])
    #     print("Surface Area: %.5f" %SurfaceArea[i])
    #     print("Compactness: %.5f" %Compactness[i])
    #     print("Rectangularity: %.5f" %Rectangularity[i])
    #     print("Diameter: %.5f" %Diameter[i])
    #     print("Eccentricity: %.5f" %Eccentricity[i])

    #genFeaturePlots()
