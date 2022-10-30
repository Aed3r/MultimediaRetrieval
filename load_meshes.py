from multiprocessing import resource_tracker
import os
import random
from tqdm import tqdm
import open3d as o3d

LABELED_PSB_DB = os.path.join("data", "LabeledDB_new")
PRINCETON_DB = os.path.join("data","psb_v1", "benchmark")
NORMALIZEDMESHESFOLDER = os.path.join("data", "normalized")

"""Load a mesh from an OFF file.
    Use returnInfoOnly=True to return the number of vertices and faces, and the type of the faces.
"""
def load_OFF (path, returnInfoOnly=False):
    lineCount = 0
    vertexCount = 0
    faceCount = 0
    readCounts = False
    readVertices = False
    vertices = []
    faces = []
    faceType = None

    # Check if the file exists
    if not os.path.isfile(path):
        raise Exception('File ' + path + ' not found!')

    with open(path, 'r') as f:
        for line in f:
            lineCount += 1
            if line[0] == '#' or not line.strip() or (lineCount == 1 and line.strip() == 'OFF'):
                continue

            # Separate line into tokens
            tokens = line.split()

            if len(tokens) == 3:
                if not readCounts:
                    numVerts = int(tokens[0])
                    numFaces = int(tokens[1])

                    vertices = [None] * numVerts
                    faces = [None] * numFaces

                    readCounts = True
                    continue

                if readVertices:
                    raise Exception('Unexpected vertex line in faces section')

                if vertexCount >= numVerts:
                    raise Exception('More vertices in file than expected!')

                # Read vertex coordinates
                vertices[vertexCount] = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
                vertexCount += 1
                continue

            if len(tokens) == 4:
                if not readVertices:
                    readVertices = True

                if faceCount >= numFaces:
                    raise Exception('More faces in file than expected!')

                if not readCounts:
                    raise Exception('File line counts missing!')

                if int(tokens[0]) == 3:
                    # Read triangle 
                    faces[faceCount] = [int(tokens[1]), int(tokens[2]), int(tokens[3])]

                    # Set face type
                    if faceType is None:
                        faceType = 'tri'
                    elif faceType == 'quad':
                        faceType = 'mixed'
                else:
                    raise Exception('Unsupported face type!')
                
                faceCount += 1
                continue

            if len(tokens) == 5:
                if not readVertices:
                    readVertices = True

                if faceCount >= numFaces:
                    raise Exception('More faces in file than expected!')

                if not readCounts:
                    raise Exception('File line counts missing!')
    
                if int(tokens[0]) == 4:
                    # Read quad
                    faces[faceCount] = [int(tokens[1]), int(tokens[2]), int(tokens[3])]
                    faces[faceCount + 1] = [int(tokens[1]), int(tokens[3]), int(tokens[4])]

                    # Set face type
                    if faceType is None:
                        faceType = 'quad'
                    elif faceType == 'tri':
                        faceType = 'mixed'
                else:
                    raise Exception('Unsupported face type!')
                
                faceCount += 1
                continue

            raise Exception('Unexpected line in file!')

    # Calculate the AABB
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(vertices))

    # Extract the file name from the path
    fileName = os.path.basename(path)

    result = {
        "path": path, 
        "numVerts": len(vertices), 
        "numFaces": len(faces), 
        "faceType": faceType,
        "aabb": [aabb.get_min_bound()[0], aabb.get_min_bound()[1], aabb.get_min_bound()[2], aabb.get_max_bound()[0], aabb.get_max_bound()[1], aabb.get_max_bound()[2]],
        "name": fileName}
    
    if not returnInfoOnly:
        result["vertices"] = vertices
        result["faces"] = faces

    return result

"""Load a mesh from an PLY file.
    Use returnInfoOnly=True to return the number of vertices and faces, and the type of the faces.
"""
def load_PLY (path, returnInfoOnly=False):
    lineCount = 0
    vertexCount = 0
    faceCount = 0
    readVertices = False
    vertices = []
    faces = []
    faceType = None

    # Check if the file exists
    if not os.path.isfile(path):
        raise Exception('File ' + path + ' not found!')

    with open(path, 'r') as f:
        for line in f:
            lineCount += 1
            if line[0:7] == "comment" or not line.strip() or (lineCount == 1 and line.strip() == 'ply') or line.strip() == 'end_header' or line[0:8] == 'property':
                continue

            # Detect format line
            if line[0:6] == 'format':
                if line[7:12] != 'ascii':
                    raise Exception('Unsupported file format!')
                continue

            # Detect element line
            if line[0:7] == 'element':
                tokens = line.split()
                if tokens[1] == 'vertex':
                    numVerts = int(tokens[2])
                    vertices = [None] * numVerts
                elif tokens[1] == 'face':
                    numFaces = int(tokens[2])
                    faces = [None] * numFaces
                continue

            # Separate line into tokens
            tokens = line.split()

            if len(tokens) == 3:
                if readVertices:
                    raise Exception('Unexpected vertex line in faces section')

                if vertexCount >= numVerts:
                    raise Exception('More vertices in file than expected!')

                # Read vertex coordinates
                vertices[vertexCount] = [float(tokens[0]), float(tokens[1]), float(tokens[2])]
                vertexCount += 1
                continue

            if len(tokens) == 4:
                if not readVertices:
                    readVertices = True

                if faceCount >= numFaces:
                    raise Exception('More faces in file than expected!')

                if int(tokens[0]) == 3:
                    # Read triangle 
                    faces[faceCount] = [int(tokens[1]), int(tokens[2]), int(tokens[3])]

                    # Set face type
                    if faceType is None:
                        faceType = 'tri'
                    elif faceType == 'quad':
                        faceType = 'mixed'
                else:
                    raise Exception('Unsupported face type!')
                
                faceCount += 1
                continue

            if len(tokens) == 5:
                if not readVertices:
                    readVertices = True

                if faceCount >= numFaces:
                    raise Exception('More faces in file than expected!')

                if int(tokens[0]) == 4:
                    # Read quad
                    faces[faceCount] = [int(tokens[1]), int(tokens[2]), int(tokens[3])]
                    faces[faceCount + 1] = [int(tokens[1]), int(tokens[3]), int(tokens[4])]

                    # Set face type
                    if faceType is None:
                        faceType = 'quad'
                    elif faceType == 'tri':
                        faceType = 'mixed'
                else:
                    raise Exception('Unsupported face type!')
                
                faceCount += 1
                continue

            raise Exception('Unexpected line in file!')

    # Calculate the AABB
    aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(vertices))

    # Extract the file name from the path
    fileName = os.path.basename(path)

    result = {
        "path": path, 
        "numVerts": len(vertices), 
        "numFaces": len(faces), 
        "faceType": faceType,
        "aabb": [aabb.get_min_bound()[0], aabb.get_min_bound()[1], aabb.get_min_bound()[2], aabb.get_max_bound()[0], aabb.get_max_bound()[1], aabb.get_max_bound()[2]],
        "name": fileName}
    
    if not returnInfoOnly:
        result["vertices"] = vertices
        result["faces"] = faces

    return result

# Get the shape classification for the Princeton Shape Benchmark
def get_princeton_shape_classes():
    classesFile = os.path.join(PRINCETON_DB, "classification", "v1", "coarse1", "coarse1Train.cla")
    lineCount = 0
    classCount = 0
    shapeCount = 0
    classCounter = 0
    shapeCounter = 0
    classes = []

    # Check if the file exists
    if not os.path.isfile(classesFile):
        raise Exception('Failed to find the Princeton class shapes file at \'' + classesFile + '\'!')
        
    with open(classesFile, 'r') as f:
        for line in f:
            lineCount += 1

            if (not line.strip()) or lineCount == 1:
                continue
            
            # Get number of classes and shapes
            if lineCount == 2:
                tokens = line.split()
                classCount = int(tokens[0])
                shapeCount = int(tokens[1])
                continue
                
            # Separate line into tokens
            tokens = line.split()

            # Check if line is a class name
            if not tokens[0].isdigit():
                if classCounter >= classCount:
                    raise Exception('More classes in file than expected!')

                classes.append({'name': tokens[0], 'shapes': []})
                classCounter += 1
                continue

            # Check if line is a shape number
            if tokens[0].isdigit():
                if shapeCounter >= shapeCount:
                    raise Exception('More shapes in file than expected!')

                classes[classCounter - 1]['shapes'].append(int(tokens[0]))
                shapeCounter += 1
                continue
            
            raise Exception('Unexpected line in file!')

    return classes

def load_mesh(path, returnInfoOnly=False):
    if path.endswith('.off'):
        return load_OFF(path, returnInfoOnly)
    elif path.endswith('.ply'):
        return load_PLY(path, returnInfoOnly)
    else:
        raise Exception('Unsupported file format!')

# Load meshes from the Labeled PSB dataset (fromLPSB=True), Princeton Shape Benchmark (fromPRINC=True) or normalized meshes directory (fromNORM=True)
# Use returnInfoOnly=True to return the number of vertices and faces, and the type of the faces only.
# Use randomSample to randomly sample a subset of the meshes of both datasets. Use -1 to load all meshes.
def get_meshes(fromLPSB=False, fromPRIN=False, fromNORM=True, randomSample=200, returnInfoOnly=True):
    directories = [] # Array of all directories to find meshes in, with their associated shape class: [(directory, shapeClass), ...]
    files = [] # Array of all files to load meshes from: [(filename, shapeClass), ...]
    results = []

    # Charge Labeled PSB dataset if required
    if fromLPSB:
        # Check if the database exists
        if not os.path.isdir(LABELED_PSB_DB):
            raise Exception('Failed to find the Labeled PSB database at \'' + LABELED_PSB_DB + '\'!')

        directories.extend([(os.path.join(LABELED_PSB_DB, d), d) for d in os.listdir(LABELED_PSB_DB) if os.path.isdir(os.path.join(LABELED_PSB_DB, d))])

    # Charge the Princeton Shape Benchmark dataset if required
    if fromPRIN:
        classes = get_princeton_shape_classes()

        for c in classes:
            for s in c['shapes']:
                if s < 100:
                    folder = os.path.join(PRINCETON_DB, "db", "0", 'm' + str(s))
                else:
                    folder = os.path.join(PRINCETON_DB, "db", str(s)[0:len(str(s))-2], 'm' + str(s))
                directories.append((folder, c['name']))

    # Charge Normalized meshes if required
    if fromNORM:
        # Check if the database exists
        if not os.path.isdir(NORMALIZEDMESHESFOLDER):
            raise Exception('Failed to find the Normalized meshes directory at \'' + NORMALIZEDMESHESFOLDER + '\'!')

        res = [(os.path.join(NORMALIZEDMESHESFOLDER, d), d) for d in os.listdir(NORMALIZEDMESHESFOLDER) if os.path.isdir(os.path.join(NORMALIZEDMESHESFOLDER, d))]
        directories.extend(res)

    # Find all files located in the loaded directories with file extension .ply or .off
    for d in directories:
        for f in os.listdir(d[0]):
            if f.endswith('.ply') or f.endswith('.off'):
                files.append((os.path.join(d[0], f), d[1]))

    # Randomly select meshes if needed
    if randomSample > 0:
        files = random.sample(files, randomSample)

    # Load the meshes
    if returnInfoOnly:
        lbl = 'Loading info of ' + str(len(files)) + ' mesh'
    else:
        lbl = 'Loading ' + str(len(files)) + ' mesh'
    if len(files) == 1:
        lbl += '...'
    else:
        lbl += 'es...'
    
    for f in tqdm(files, desc=lbl, ncols=150):
        res = load_mesh(f[0], returnInfoOnly)
        res["class"] = f[1]
        results.append(res)
    
    return results

# Saves the given mesh to the given directory
# Automatically creates and saves in a subdirectory with the shape class name
def save_mesh(mesh, directory):
    directory = os.path.join(directory, mesh["class"])

    # Check if the output directory exists
    if not os.path.isdir(directory):
        os.makedirs(directory)

    # Save the mesh as .off
    geometry = o3d.geometry.TriangleMesh()
    geometry.vertices = o3d.utility.Vector3dVector(mesh["vertices"])
    geometry.triangles = o3d.utility.Vector3iVector(mesh["faces"])
    o3d.io.write_triangle_mesh(os.path.join(directory, mesh['name']), geometry, write_ascii=True)

# Saves the given meshes to the given directory
# Automatically creates and saves in subdirectories with the shape class names
def save_all_meshes(meshes, directory):
    # Check if the output directory exists
    if not os.path.isdir(directory):
        os.makedirs(directory)

    for m in tqdm(meshes, desc='Saving meshes', ncols=150):
        save_mesh(m, directory)

def load_thumbnail(path):
    # Check if the thumbnail exists
    if not os.path.isfile(path):
        raise Exception('Failed to find the thumbnail at \'' + path + '\'!')

    # Load the thumbnail
    return o3d.io.read_image(path)

if __name__ == '__main__':
    import sys
    import numpy as np
    import open3d as o3d

    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        #filename = 'data/LabeledDB_new/Airplane/61.off'
        filename = 'data/ply/airplane.ply'

    if filename[-4:] == '.ply':
        mesh = load_PLY(filename, False)
        vertices = mesh['vertices']
        faces = mesh['faces']
    elif filename[-4:] == '.off':
        mesh = load_OFF(filename)
        vertices = mesh['vertices']
        faces = mesh['faces']
    else:
        raise Exception('Unsupported file format!')

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([1, 0.706, 0])
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    test = get_meshes(fromLPSB=True, fromPRIN=True, fromNORM=False, randomSample=200, returnInfoOnly=True) # Place a breakpoint here to test the function