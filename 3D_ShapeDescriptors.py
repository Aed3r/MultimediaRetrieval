import load_meshes
import numpy as np
import open3d as o3d
import util
from tqdm import tqdm
import pymeshfix
import math
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


# def get_SurfaceArea(data):
#     Surface_area = []

#     for i in tqdm(range(len(data)), desc="Computing", ncols=100):  # get each shape
#         Area = 0
#         for j in range(len(data[i]['faces'])):  # get each face
#             v_id = data[i]['faces'][j]  # get the vertices of one face
#             v1 = data[i]['vertices'][v_id[0]]
#             v2 = data[i]['vertices'][v_id[1]]
#             v3 = data[i]['vertices'][v_id[2]]
#             Area += triangle_area([v1, v2, v3])  # compute the area of triangle
#         Surface_area.append(Area)
#     return Surface_area


# def get_Volume(data):
#     volume = []
#     data = hole_stitching(data)
#     for i in range(len(data)):
#         v_total = 0
#         o = util.get_shape_barycenter(data[i])
#         for j in range(len(data[i]['vertices'])):
#             v = data[i]['vertices'][j]
#             v1 = v[0] - o
#             v2 = v[1] - o
#             v3 = v[2] - o
#             v_total += np.linalg.norm(np.cross(v1,v2)*v3)
#         volume.append(v_total/6)
#     return volume

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
        SurfaceArea = o3d_Get_Surface_Area(data)
        Volume = o3d_Get_Volume(data)
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



def o3d_Get_Surface_Area(data):
    # surface_area = mesh.get_surface_area()
    SA = []
    for i in range(len(data)):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data[i]['vertices']), o3d.cpu.pybind.utility.Vector3iVector(data[i]['faces']))
        surface_area = o3d.geometry.TriangleMesh.get_surface_area(mesh)
        # print("Surface Area:")
        # print(surface_area)
        SA.append(surface_area)
    return SA


def o3d_Get_Volume(data):
    V = []
    for i in range(len(data)):
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data[i]['vertices']), o3d.cpu.pybind.utility.Vector3iVector(data[i]['faces']))
        volume = o3d.geometry.TriangleMesh.get_volume(mesh)
        V.append(volume)
    return V


def get_diameter(data):
    Diameter = []
    for i in range(len(data)):
        distances = []
        #mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(data[i]['vertices']), o3d.cpu.pybind.utility.Vector3iVector(data[i]['faces']))
        mesh = data[i]
        #print(np.asarray(mesh['vertices']))
        print(len(data[i]['vertices']))
        for j in range(len(mesh['vertices'])): # The calculation is too complex!!!! need to reimplement it
            for n in range(len(mesh['vertices'])):
                vertex1 = mesh['vertices'][j]
                vertex2 = mesh['vertices'][n]
                print(vertex1)
                print(vertex2)
                dist = util.distance_between(vertex1, vertex2)
                distances.append(dist)
        # calculate the distance between all 2 vertices
        # find the maximum
        diameter = max(distances)
        Diameter.append(diameter)
    return Diameter

# Calculate the eccentricity of a mesh list
def get_eccentricity(data):
    #Eccentricity = eigenvalue1 / eigenvalue3
    eccentricity = []
    for i in range(len(data)):
        mesh = data[i]
        eigenvalues, eigenvectors = util.compute_PCA(mesh)
        Ecc = eigenvalues[0] / eigenvalues[2]
        eccentricity.append(Ecc)
    return eccentricity


if __name__ == '__main__':
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=1, returnInfoOnly=False)
    
    # get normalized mesh
    util_data = []
    for mesh in data:
        # mesh = util.resampling(mesh)
        mesh = normalization.translate_mesh_to_origin(mesh)
        mesh = normalization.align_shape(mesh)
        mesh = normalization.flipping_test(mesh)
        mesh = normalization.scale_mesh_to_unit(mesh)
        util_data.append(mesh)

    Compactness = get_Compactness(util_data)
    aabbVolume = get_aabbVolume(util_data)
    SurfaceArea2 = o3d_Get_Surface_Area(util_data)
    Volume2 = o3d_Get_Volume(util_data)
    # Diameter = get_diameter(util_data)
    Eccentricity = get_eccentricity(util_data)

    for i in range(len(Compactness)):
        print("The %dth data feature: " %(i+1))
        print("Volume2: %.20f" %Volume2[i])
        print("Surface Area2:%.5f" %SurfaceArea2[i])
        print("Compactness: %.5f" %Compactness[i])
        print("Axis-aligned bounding-box volume: %.5f" %aabbVolume[i])
        # print("Diameter: %.5f" %Diameter[i])
        print("Eccentricity: %.5f" %Eccentricity[i])


    # for i in range(len(RepairedMesh)):
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(RepairedMesh[i]['vertices'])
    #     mesh.triangles = o3d.utility.Vector3iVector(RepairedMesh[i]['faces'])
    #     mesh.paint_uniform_color([1, 0.706, 0])
    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([mesh])




