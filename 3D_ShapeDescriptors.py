import load_meshes
import numpy as np
import open3d as o3d
import util
from tqdm import tqdm
import pymeshfix
import math
import util
import numpy as np
import normalization

def triangle_area(tri):
    # triangle
    if isinstance(tri, list):
        tri = np.array(tri)
    # all edges
    edges = tri[1:] - tri[0:1]  # v1-v0, v2-v0
    # row wise cross product
    cross_product = np.cross(edges[:-1], edges[1:], axis=1)  # (v1-v0) X (v2-v0)
    # area of all triangles
    area = np.linalg.norm(cross_product, axis=1) / 2  # compute the area

    return sum(area)


def get_SurfaceArea(data):
    Surface_area = []

    for i in tqdm(range(len(data)), desc="Computing", ncols=100):  # get each shape
        Area = 0
        for j in range(len(data[i]['faces'])):  # get each face
            v_id = data[i]['faces'][j]  # get the vertices of one face
            v1 = data[i]['vertices'][v_id[0]]
            v2 = data[i]['vertices'][v_id[1]]
            v3 = data[i]['vertices'][v_id[2]]
            Area += triangle_area([v1, v2, v3])  # compute the area of triangle
        Surface_area.append(Area)
    return Surface_area


def get_Volume(data):
    volume = []
    data = hole_stitching(data)
    for i in range(len(data)):
        v_total = 0
        o = util.get_shape_barycenter(data[i])
        for j in range(len(data[i]['vertices'])):
            v = data[i]['vertices'][j]
            v1 = v[0] - o
            v2 = v[1] - o
            v3 = v[2] - o
            v_total += np.linalg.norm(np.cross(v1,v2)*v3)
        volume.append(v_total/6)
    return volume

# only work for certain shapes  https://pymeshfix.pyvista.org/
def hole_stitching(data):
    errorNumber = 1
    for i in range(len(data)):
        # Create object from vertex and face arrays
        vertices = np.asarray(data[i]['vertices'])
        faces = data[i]['faces']

        try:
            meshfix = pymeshfix.MeshFix(vertices, faces)
            # meshfix.plot()      # Plot input
            # Repair input mesh
            meshfix.repair(verbose=True)
            # meshfix.plot()      # View the repaired mesh (requires vtkInterface)
            # Access the repaired mesh with vtk
            data[i]['vertices'] = meshfix.v
            data[i]['faces'] = meshfix.f
        except:
            print('Not suitable' + errorNumber)
            errorNumber += 1

    return data


def get_Compactness(data):
    Compactness = []

    for i in tqdm(range(len(data)), desc = "Computing", ncols = 100): # get each shape
        comP = 0
        SurfaceArea = get_SurfaceArea(data)
        Volume = get_Volume(data)
        S_3 = SurfaceArea[i]*SurfaceArea[i]*SurfaceArea[i] # cannot use np.power() here, or the list would transfer to array automatically
        V_2 = Volume[i]*Volume[i]
        comP = S_3/(36*(math.pi)*V_2)
        Compactness.append(comP) 
    return Compactness

def get_aabbVolume(data):
    aabbVolume = []
    for i in range(len(data)):
        aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(data[i]['vertices']))
        aabbVolume.append(aabb.volume())
    return aabbVolume

# Runs the sampling function func for 2 vertices
def sample2Verts(mesh, func):
    k = pow(mesh["numVerts"], 1.0 / 2.0)
    res = []

    i = 0
    while i < k:
        # Get random number from 0 to n
        vi = util.random_vertices(mesh, 1)
        j = 0
        while j < k:
            vj = util.random_vertices(mesh, 1)
            if (vi == vj):
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
        vi = util.random_vertices(mesh, 1)
        j = 0
        while j < k:
            vj = util.random_vertices(mesh, 1)
            if (vi == vj):
                continue
            l = 0
            while l < k:
                vl = util.random_vertices(mesh, 1)
                if (vl == vi or vl == vj):
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
        vi = util.random_vertices(mesh, 1)
        j = 0
        while j < k:
            vj = util.random_vertices(mesh, 1)
            if (vi == vj):
                continue
            l = 0
            while l < k:
                vl = util.random_vertices(mesh, 1)
                if (vl == vi or vl == vj):
                    continue
                m = 0
                while m < k:
                    vm = util.random_vertices(mesh, 1)
                    if (vm == vi or vm == vj or vm == vl):
                        continue

                    res.append(func(vi, vj, vl, vm))
                    m += 1
                l += 1
            j += 1
        i += 1 

# Sample the angle of 3 random vertices in the mesh
def A3(mesh):
    res = sample3Verts(mesh, util.angle_between)

    # Calculate histogram of angles
    histogram, _ = np.histogram(res, bins=25, range=(0, math.pi), density=True)
    
    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram

# Sample the distance between barycenter and random vertex in the mesh
def D1(mesh):
    barycenter = np.array(util.get_shape_barycenter(mesh))
    res = []

    for i in range(mesh["numVerts"]):
        res.append(util.distance_between(barycenter, np.array(mesh["vertices"][i])))

    # Calculate histogram of distances
    histogram, _ = np.histogram(res, bins=22, range=(0,1), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram

# Sample the distance between two random vertices in the mesh
def D2(mesh):
    res = sample2Verts(mesh, util.distance_between)

    # Calculate histogram of distances
    histogram, _ = np.histogram(res, bins=23, range=(0, math.sqrt(3)), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram

# Sample the square root of area of triangle given by 3 random vertices 
def D3(mesh):
    res = sample3Verts(mesh, util.triangle_area)

    # Calculate histogram of areas
    histogram, _ = np.histogram(res, bins=25, range=(0, (math.sqrt(3) / 2) ** (1/2) ), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram

# Sample the cube root of volume of tetrahedron formed by 4 random vertices 
def D4(mesh):
    res = sample4Verts(mesh, util.tetrahedron_volume)

    # Calculate histogram of volumes
    histogram, _ = np.histogram(res, bins=29, range=(0,(1/3) ** (1/3)), density=True)

    # Calculate the mean of the histogram
    histogram = list(histogram / np.sum(histogram))

    return histogram


if __name__ == '__main__':
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=1, returnInfoOnly=False)

    # get normalized mesh
    util_data = []
    for mesh in data:
        util_data.append(normalization.normalize_mesh(mesh))

    SurfaceArea = get_SurfaceArea(util_data)
    Volume = get_Volume(util_data)
    Compactness = get_Compactness(util_data)
    aabbVolume = get_aabbVolume(util_data)


    for i in range(len(Compactness)):
        print("The %dth data feature: " %(i+1))
        print("Volume: %.20f" %Volume[i]) # The volume is too small sometimes
        print("Surface Area:%.5f" %SurfaceArea[i])
        print("Compactness: %.5f" %Compactness[i])
        print("Axis-aligned bounding-box volume: %.5f" %aabbVolume[i])


    # for i in range(len(RepairedMesh)):
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(RepairedMesh[i]['vertices'])
    #     mesh.triangles = o3d.utility.Vector3iVector(RepairedMesh[i]['faces'])
    #     mesh.paint_uniform_color([1, 0.706, 0])
    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([mesh])




