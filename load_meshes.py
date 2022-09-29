"""Load a mesh from an OFF file.
    Use returnInfoOnly=True to return the number of vertices and faces, and the type of the faces.
"""
def load_OFF (filename, returnInfoOnly=False):
    lineCount = 0
    vertexCount = 0
    faceCount = 0
    readCounts = False
    readVertices = False
    vertices = []
    faces = []
    faceType = None

    with open(filename, 'r') as f:
        for line in f:
            if line[0] == '#' or not line.strip() or (lineCount == 0 and line.strip() == 'OFF'):
                continue

            # Separate line into tokens
            tokens = line.split()

            if len(tokens) == 3:
                if not readCounts:
                    numVerts = int(tokens[0])
                    numFaces = int(tokens[1])
                    numEdges = int(tokens[2])

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

    if returnInfoOnly:
        return numVerts, numFaces, faceType
    else:
        return vertices, faces

"""Load a mesh from an PLY file.
    Use returnInfoOnly=True to return the number of vertices and faces, and the type of the faces.
"""
def load_PLY (filename, returnInfoOnly=False):
    lineCount = 0
    vertexCount = 0
    faceCount = 0
    readVertices = False
    vertices = []
    faces = []
    faceType = None

    with open(filename, 'r') as f:
        for line in f:
            if line[0:7] == "comment" or not line.strip() or (lineCount == 0 and line.strip() == 'ply') or line.strip() == 'end_header' or line[0:8] == 'property':
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

    if returnInfoOnly:
        return numVerts, numFaces, faceType
    else:
        return vertices, faces


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
        vertices, faces = load_PLY(filename)
    elif filename[-4:] == '.off':
        vertices, faces = load_OFF(filename)
    else:
        raise Exception('Unsupported file format!')

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh])