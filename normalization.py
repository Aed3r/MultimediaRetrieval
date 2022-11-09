import os
from util import *
from tqdm import tqdm
import pymeshfix
import trimesh
import pyvista
import pyacvd

MINRESAMPLINGTHRESHOLD = 500
MAXRESAMPLINGTHRESHOLD = 50000

# Translates the shape to the given position and returns the result
def translate_mesh_to_origin(mesh):
    return translate_mesh(mesh, -get_shape_barycenter(mesh))

# Returns the shape scaled to the unit cube
def scale_mesh_to_unit(mesh):
    (xMax, yMax, zMax) = list(np.max(mesh["vertices"], axis=0))
    (xMin, yMin, zMin) = list(np.min(mesh["vertices"], axis=0))

    maxDist = max(abs(xMax - xMin), abs(yMax - yMin), abs(zMax - zMin))

    return scale_mesh(mesh, 1 / maxDist)

# Align the two largest eigenvectors with the x and y axis
def align_shape(mesh):
    # Do PCA
    eigenvalues, eigenvectors = compute_PCA(mesh)
    rot_matrix = np.array(eigenvectors)
    
    # Rotate the shape
    mesh["vertices"] = o3d.utility.Vector3dVector(np.dot(rot_matrix, np.asarray(mesh["vertices"]).T).T)

    newAABB = o3d.geometry.AxisAlignedBoundingBox.create_from_points(mesh["vertices"])
    mesh["aabb"] = [newAABB.get_min_bound()[0], newAABB.get_min_bound()[1], newAABB.get_min_bound()[2], newAABB.get_max_bound()[0], newAABB.get_max_bound()[1], newAABB.get_max_bound()[2]]
    
    return mesh

# Performs the flipping test and mirrors the shape if necessary
def flipping_test(mesh):
    f_x = 0
    f_y = 0
    f_z = 0

    for triangle in mesh["vertices"]:
        f_x += np.sign(triangle[0]) * (triangle[0] ** 2)
        f_y += np.sign(triangle[1]) * (triangle[1] ** 2)
        f_z += np.sign(triangle[2]) * (triangle[2] ** 2)

    return scale_mesh(mesh, np.array([np.sign(f_x), np.sign(f_y), np.sign(f_z)]))

# Remesh the shape
def resampling(mesh):
    for i in range(len(mesh["faces"])):
        mesh["faces"][i].insert(0, len(mesh["faces"][i]))

    surf = pyvista.PolyData(mesh["vertices"], mesh["faces"])    # create PolyData

    # takes a surface mesh and returns a uniformly meshed surface using voronoi clustering

    clus = pyacvd.Clustering(surf)

    clus.subdivide(3)                           # remeshing
    clus.cluster(3000)                          # vertices around 3000
    # clus.plot()                               # plot clustered mesh

    remesh = clus.create_mesh()

    vertices_new = np.array(remesh.points)      # get new vertices list
    faces_new = get_face_list_from_polydata(remesh)               # get new faces list

    mesh["vertices"] = vertices_new
    mesh["faces"] = faces_new
    mesh["numVerts"] = len(vertices_new)
    mesh["numFaces"] = len(faces_new)

    return mesh

def is_watertight(mesh):
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mesh['vertices']),
                                     o3d.cpu.pybind.utility.Vector3iVector(mesh['faces']))
    is_watertight = mesh.is_watertight()
    return is_watertight

# only work for certain shapes  https://pymeshfix.pyvista.org/
def hole_stitching(data, verbose=False):
    errorNumber = 0
    vertices = np.asarray(data['vertices'])
    faces = data['faces']

    try:
        meshfix = pymeshfix.MeshFix(vertices, faces)
        #meshfix.plot()      # Plot input
        # Repair input mesh
        meshfix.repair(verbose=verbose)
        #meshfix.plot()      # View the repaired mesh (requires vtkInterface)
        # Access the repaired mesh with vtk
        data['vertices'] = meshfix.v
        data['faces'] = meshfix.f
        data['numVerts'] = len(meshfix.v)
        data['numFaces'] = len(meshfix.f)
    except:
        print('Not suitable' + errorNumber)
        errorNumber += 1

    return data


# new hole filling in trimesh.repair
def hole_filling(mesh):
    errorNumber = 1
    # works for 3 meshes that hole_stitching cannot deal with:
    # data/ply/LabeledDB_new/Glasses/42.off
    # data/ply/LabeledDB_new/Mech/331.off
    # data/ply/LabeledDB_new/Table/160.off

    try:
        mesh = trimesh.Trimesh(vertices=mesh['vertices'], faces=mesh['faces'])
        mesh = trimesh.repair.fill_holes(mesh)
    except:
        print('Not suitable: %d'  %errorNumber)
        errorNumber = errorNumber + 1  

    return mesh

# Applies all the normalization steps to the given mesh
def normalize(mesh):
    if mesh['numVerts'] < MINRESAMPLINGTHRESHOLD or mesh['numVerts'] > MAXRESAMPLINGTHRESHOLD:
        mesh = resampling(mesh)
    if not is_watertight(mesh):
        mesh = hole_stitching(mesh)
    if not is_watertight(mesh):     #especially for "data/LabeledDB_new/Bearing/349.off"
        mesh["faces"] = mesh["faces"].tolist()
        mesh = resampling(mesh)
        mesh = hole_stitching(mesh)
    mesh = translate_mesh_to_origin(mesh)
    mesh = align_shape(mesh)
    mesh = flipping_test(mesh)
    mesh = scale_mesh_to_unit(mesh)
    return mesh

# Normalizes and returns all the meshes in the 'meshes' array
def normalize_all(meshes):
    res = []

    for mesh in tqdm(meshes, desc="Normalizing meshes", ncols=150):
        res.append(normalize(mesh))

    return res

if __name__ == "__main__":
    import load_meshes
    import statistics

    # Test all normalization functions

    # load all 380 meshes in LPSB
    meshes = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, fromNORM=False, randomSample=-1, returnInfoOnly=False)
    barycenters_0 = []
    barycenters_1 = []
    for mesh in meshes:
        # check the barycenter histogram before centering 
        barycenter_0 = get_shape_barycenter(mesh)
        barycenters_0.append(barycenter_0)
        # Translate the mesh to the origin
        mesh = translate_mesh_to_origin(mesh)

        # Get the barycenter
        barycenter = get_shape_barycenter(mesh)

        # Check that the barycenter is at the origin
        assert np.allclose(barycenter, np.zeros(3))

        # check the barycenter histogram after centering
        barycenters_1.append(barycenter.copy()) # barycenters list


    statistics.draw_histogram([i[0] for i in barycenters_0], 'centering') # histogram before centering- check x axis
    statistics.draw_histogram([i[0] for i in barycenters_1], 'centering') # histogram after centering   
    statistics.draw_histogram([i[1] for i in barycenters_0], 'centering') # histogram before centering- check y axis
    statistics.draw_histogram([i[1] for i in barycenters_1], 'centering') # histogram after centering   
    statistics.draw_histogram([i[2] for i in barycenters_0], 'centering') # histogram before centering- check z axis
    statistics.draw_histogram([i[2] for i in barycenters_1], 'centering') # histogram after centering   
    print (barycenters_0)
    print(barycenters_1)  

    print("Centering tests passed")
        # # Scale to unit cube
        # mesh = scale_mesh_to_unit(mesh)

        # # Check that the max distance from the origin is 1
        # assert np.allclose(np.max(np.linalg.norm(np.asarray(mesh["vertices"]), axis=1)), 1)

        # # Align the mesh
        # mesh = align_shape(mesh)

        # # Perform the flipping test
        # mesh = flipping_test(mesh)


    print("All unit tests passed")
