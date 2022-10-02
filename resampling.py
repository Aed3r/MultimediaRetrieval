# from pyvista import examples
# import pyvista as pv
#
import numpy as np
import matlab as ml

import load_meshes as lm
# import numpy as np
#
# # download cow mesh
# cow = examples.download_cow()
# filename = 'data/psb_v1/benchmark/db/2/m200/m200.off'
# mesh = lm.load_OFF(filename)
# vertices = np.array(mesh['vertices'])
# faces = np.hstack(mesh['faces'])
# surf = pv.PolyData(vertices, faces)
# print(cow)
# # cow.plot(show_edges=True, color='w')
# print(surf)
# # surf.plot(show_edges=True, color='w')

# plot original mesh
# cow.plot(show_edges=True, color='w')
#
# clus = pyacvd.Clustering(cow)
# # mesh is not dense enough for uniform remeshing
# clus.subdivide(3)
# clus.cluster(20000)
#
# # plot clustered cow mesh
# clus.plot()
#
# remesh = clus.create_mesh()
#
# # plot uniformly remeshed cow
# remesh.plot(color='w', show_edges=True)

# def subdivide(UnstructuredGrid3D):
filename = 'data/LabeledDB_new/Airplane/61.off'
mesh = lm.load_OFF(filename)
vertices = mesh['vertices']
faces = mesh['faces']

numVerts = len(vertices)
numFaces = len(faces)

i = 0
j = 0
p = [None] * numVerts
f = [None] * numFaces
area_sorted_cells = [None] * numFaces

while i < numVerts:
     p[i] = vertices[i][0], vertices[i][1], vertices[i][2]
     #print(p[i])
     i += 1

while j < numFaces:
     f[j] = faces[j][0], faces[j][1], faces[j][2]
     x = np.mat(p[f[j][0]])
     y = np.mat(p[f[j][1]])
     z = np.mat(p[f[j][2]])
     cross_prod = np.cross(y - x, z - x)
     c = np.mat(cross_prod).A
     area = ml.norm(c)/ 2
     area_sorted_cells[j] = area, j
     # print(area_sorted_cells[j])
     j += 1


area_sorted_cells.sort()
# print(area_sorted_cells)
min_area = 0.00015

# p2 = p;
# f2 = 0;
# Nsubdivided = 0;
# faces2 = [None] * numFaces
n = 0

while n < numFaces:
      area = area_sorted_cells[n][0]
      it = area_sorted_cells[n][1]
      if (area > min_area):
          x = np.mat(p[f[n][0]])
          y = np.mat(p[f[n][1]])
          z = np.mat(p[f[n][2]])
          center = x+y+z
          #center = np.array(center).ravel()

      n += 1

# print(area)
