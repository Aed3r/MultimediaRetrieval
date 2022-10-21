import os
import time
import load_meshes
import open3d as o3d
from tqdm import tqdm
import pymeshfix
import math
import util
import numpy as np
import normalization
import matplotlib.pyplot as plt

FEATUREPLOTSPATH = "data/featurePlots/"

# Calculate the surface area of a mesh in the data list
def get_Surface_Area(data):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data['vertices']), o3d.cpu.pybind.utility.Vector3iVector(data['faces']))
    surface_area = o3d.geometry.TriangleMesh.get_surface_area(mesh)
    return surface_area

# only work for certain shapes  https://pymeshfix.pyvista.org/
def hole_stitching(data):
    errorNumber = 1
    vertices = np.asarray(data['vertices'])
    faces = data['faces']

    try:
        meshfix = pymeshfix.MeshFix(vertices, faces)
        # meshfix.plot()      # Plot input
        # Repair input mesh
        meshfix.repair(verbose=True)
        # meshfix.plot()      # View the repaired mesh (requires vtkInterface)
        # Access the repaired mesh with vtk
        data['vertices'] = meshfix.v
        data['faces'] = meshfix.f
    except:
        print('Not suitable' + errorNumber)
        errorNumber += 1

    return data

# Calculate the volume of meshes in the data list
def get_Volume(data):
    data = hole_stitching(data)
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

# Sample the angle of 3 random vertices in the mesh
def A3(mesh):
    res = sample3Verts(mesh, util.angle_between)

    # Calculate histogram of angles
    histogram, bins = np.histogram(res, bins=25, range=(0, math.pi), density=True)
    
    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram, bins

# Sample the distance between barycenter and random vertex in the mesh
def D1(mesh):
    barycenter = np.array(util.get_shape_barycenter(mesh))
    res = []

    for i in range(mesh["numVerts"]):
        res.append(util.distance_between(barycenter, np.array(mesh["vertices"][i])))

    # Calculate histogram of distances
    histogram, bins = np.histogram(res, bins=22, range=(0,1), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram, bins

# Sample the distance between two random vertices in the mesh
def D2(mesh):
    res = sample2Verts(mesh, util.distance_between)

    # Calculate histogram of distances
    histogram, bins = np.histogram(res, bins=23, range=(0, math.sqrt(3)), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram, bins

# Sample the square root of area of triangle given by 3 random vertices 
def D3(mesh):
    res = sample3Verts(mesh, util.triangle_area)

    # Calculate histogram of areas
    histogram, bins = np.histogram(res, bins=25, range=(0, (math.sqrt(3) / 2) ** (1/2) ), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram, bins

# Sample the cube root of volume of tetrahedron formed by 4 random vertices 
def D4(mesh):
    res = sample4Verts(mesh, util.tetrahedron_volume)

    # Calculate histogram of volumes
    histogram, bins = np.histogram(res, bins=29, range=(0,(1/3) ** (1/3)), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram, bins


def genFeaturePlots():
    meshes = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=-1, returnInfoOnly=False)
    features = {}

    print("Generating feature plots...")
    startTime = time.time()

    for i in tqdm(range(len(meshes)), desc="Normalizing meshes", ncols=150):
        meshes[i] = normalization.normalize(meshes[i], doResampling=False)

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

if __name__ == '__main__':
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=2, returnInfoOnly=False)
    #data = 'data/LabeledDB_new/Bird/256.off'
    #mesh = load_meshes.load_OFF(data)
    # get normalized mesh
    util_data = []
    for mesh in data:
        util_data.append(normalization.normalize(mesh))
    #util_data.append(normalization.normalize(mesh))

    Compactness = []
    Rectangularity = []
    SurfaceArea = []
    Volume = []
    Diameter = []
    Eccentricity = []
    for i in range(len(util_data)):
        Volume.append(get_Volume(util_data[i]))
        SurfaceArea.append(get_Surface_Area(util_data[i]))
        Compactness.append(get_Compactness(util_data[i]))
        Rectangularity.append(get_3D_Rectangularity(util_data[i]))
        Diameter.append(get_diameter(util_data[i]))
        Eccentricity.append(get_eccentricity(util_data[i]))

    for i in range(len(util_data)):
        print("The %dth data feature: " %(i+1))
        print("Volume: %.20f" %Volume[i])
        print("Surface Area: %.5f" %SurfaceArea[i])
        print("Compactness: %.5f" %Compactness[i])
        print("Rectangularity: %.5f" %Rectangularity[i])
        print("Diameter: %.5f" %Diameter[i])
        print("Eccentricity: %.5f" %Eccentricity[i])

    #genFeaturePlots()

