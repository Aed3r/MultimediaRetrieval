import load_meshes
import pyacvd
import pyvista
from pyvista import _vtk, PolyData
import numpy as np
from numpy import split
import open3d as o3d
import pymongo
from tqdm import tqdm
from pymongo import MongoClient
import visualizer
import statistics

# get remeshed face
def face_list(mesh: PolyData):
    cells = mesh.GetPolys()
    c = _vtk.vtk_to_numpy(cells.GetConnectivityArray())
    o = _vtk.vtk_to_numpy(cells.GetOffsetsArray())
    faces = np.array(split(c, o[1:-1]))         # Convering an array(dtype = int64) to a list
    return faces

def resampling(file):
    # get a mesh
    mesh = load_meshes.load_OFF(file)
    vertices = mesh['vertices']
    faces = mesh['faces']

    for i in range(len(faces)):
        faces[i].insert(0, len(faces[i]))

    surf = pyvista.PolyData(vertices, faces)    # create PolyData

    # takes a surface mesh and returns a uniformly meshed surface using voronoi clustering

    # I can't run the visualizer, so use the following to check the shape
    # surf.plot(show_edges=True, color='w')     # plot original mesh
    clus = pyacvd.Clustering(surf)

    clus.subdivide(3)                           # remeshing
    clus.cluster(3000)                          # vertices around 3000
    # clus.plot()                               # plot clustered mesh

    remesh = clus.create_mesh()

    # I can't run the visualizer, so use the following to check the shape:
    # remesh.plot(color='w', show_edges=True)   # plot uniformly remeshed shape

    vertices_new = np.array(remesh.points)      # get new vertices list
    faces_new = face_list(remesh)               # get new faces list
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(vertices_new))

    numVerts_new = len(vertices_new)
    numFaces_new = len(faces_new)
    aabb_new = [aabb.get_min_bound()[0], aabb.get_min_bound()[1], aabb.get_min_bound()[2], aabb.get_max_bound()[0], aabb.get_max_bound()[1], aabb.get_max_bound()[2]]

    return numVerts_new, numFaces_new, aabb_new


def Stat(r_meshes):
    rm = r_meshes.find()
    LDB_verticesData = []
    LDB_facesData = []
    PSB_verticesData = []
    PSB_facesData = []
    rm_data = []

    # get vertices and faces number in remeshed database
    for data in rm:
        if data['path'][5:14] == "LabeledDB":
            LDB_verticesData = np.append(LDB_verticesData, data['numVerts'])
            LDB_facesData = np.append(LDB_facesData, data['numFaces'])
            del data['_id']
            rm_data = np.append(rm_data, data)
        else:
            PSB_verticesData = np.append(PSB_verticesData, data['numVerts'])
            PSB_facesData = np.append(PSB_facesData, data['numFaces'])
            del data['_id']
            rm_data = np.append(rm_data, data)

    # show statistics of remeshed database
    statistics.draw_histogram(LDB_verticesData, 'vertex')
    statistics.draw_histogram(LDB_facesData, 'face')
    statistics.draw_histogram(PSB_verticesData, 'vertex')
    statistics.draw_histogram(PSB_verticesData, 'face')

    statistics.save_Excel(rm_data, 'remeshed_PSB')  # save to Excel


if __name__ == '__main__':
    # get mesh from database
    # NOTICE: To test the function, change randomSample to a small number!  Use -1 to load all meshes.
    # this seems strange, maybe we should take data from the database?
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=True, randomSample=5, returnInfoOnly=True)
    data_new = []                               # store remeshed data

    db = MongoClient("mongodb://localhost:27017/")['mmr']
    if not 'refined_meshes' in db.list_collection_names():
        db.create_collection('refined_meshes')  # Create the collection
    r_meshes = db["refined_meshes"]

    # resampling mesh with too little/many vertices
    for i in tqdm(range(len(data)), desc="Inserting meshes into database", ncols=150):
        if int(data[i]['numVerts']) < 500 or int(data[i]['numVerts']) > 50000:

            data_new = resampling(data[i]['path'])
            data[i]['numVerts'] = data_new[0]
            data[i]['numFaces'] = data_new[1]
            data[i]['aabb'] = data_new[2]

            try:
                r_meshes.insert_one(data[i])
            except pymongo.errors.DuplicateKeyError:
                print('duplicate mesh found and ignored')

        else:
            try:
                r_meshes.insert_one(data[i])
            except pymongo.errors.DuplicateKeyError:
                print('duplicate mesh found and ignored')

    # Stat(r_meshes)                                # show statistics
