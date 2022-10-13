import random
import util
import numpy as np

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
def sampling4Verts(mesh, func):
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
    return sample3Verts(mesh, util.angle_between)

# Sample the distance between barycenter and random vertex in the mesh
def D1(mesh):
    barycenter = np.array(util.get_shape_barycenter(mesh))
    res = []

    for i in range(mesh["numVerts"]):
        res.append(util.distance_between(barycenter, np.array(mesh["vertices"][i])))

    return res

# Sample the distance between two random vertices in the mesh
def D2(mesh):
    return sample2Verts(mesh, util.distance_between)

# Sample the square root of area of triangle given by 3 random vertices 
def D3(mesh):
    return sample3Verts(mesh, util.triangle_area)