import load_meshes
import numpy as np
import open3d as o3d
import util
from tqdm import tqdm
import pymeshfix
import math

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




if __name__ == '__main__':
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=5, returnInfoOnly=False)

    # get normalized mesh
    util_data = []
    for mesh in data:
        # mesh = util.resampling(mesh)
        mesh = util.translate_mesh_to_origin(mesh)
        mesh = util.align_shape(mesh)
        mesh = util.flipping_test(mesh)
        mesh = util.scale_mesh_to_unit(mesh)
        util_data.append(mesh)

    SurfaceArea = get_SurfaceArea(util_data)
    Volume = get_Volume(util_data)
    Compactness = get_Compactness(util_data)
    for i in range(len(Compactness)):
        print("The %dth data feature: " %(i+1))
        print("Volume: %.20f" %Volume[i])
        print("Surface Area:%.5f" %SurfaceArea[i])
        print("Compactness: %.5f" %Compactness[i])

    # for i in range(len(RepairedMesh)):
    #     mesh = o3d.geometry.TriangleMesh()
    #     mesh.vertices = o3d.utility.Vector3dVector(RepairedMesh[i]['vertices'])
    #     mesh.triangles = o3d.utility.Vector3iVector(RepairedMesh[i]['faces'])
    #     mesh.paint_uniform_color([1, 0.706, 0])
    #     mesh.compute_vertex_normals()
    #     o3d.visualization.draw_geometries([mesh])




