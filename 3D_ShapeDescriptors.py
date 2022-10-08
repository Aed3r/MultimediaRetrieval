import load_meshes
import numpy as np
import open3d as o3d
from tqdm import tqdm

def triangle_area(tri):
    # triangle
    if isinstance(tri, list):
        tri = np.array(tri)
    # all edges
    edges = tri[1:] - tri[0:1]                    # v1-v0, v2-v0
    # row wise cross product
    cross_product = np.cross(edges[:-1], edges[1:], axis=1)     # (v1-v0) X (v2-v0)
    # area of all triangles
    area = np.linalg.norm(cross_product, axis=1)/2              # compute the area

    return sum(area)

def get_SurfaceArea(data):
    Surface_area = []

    for i in tqdm(range(len(data)), desc="Computing", ncols=100):   # get each shape
        Area = 0
        for j in range(len(data[i]['faces'])):      # get each face
            v_id = data[i]['faces'][j]              # get the vertices of one face
            v1 = data[i]['vertices'][v_id[0]]
            v2 = data[i]['vertices'][v_id[1]]
            v3 = data[i]['vertices'][v_id[2]]
            Area += triangle_area([v1,v2,v3])       # compute the area of triangle
        Surface_area.append(Area)
    return Surface_area

if __name__ == '__main__':
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=4, returnInfoOnly=False)
    SA = get_SurfaceArea(data)
    print(SA)




