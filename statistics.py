import imp
from unicodedata import name
from matplotlib.lines import VertexSelector
import matplotlib.pyplot as plt
import load_meshes
import open3d as o3d
import numpy as np
import pandas as pd
import xlsxwriter


verticesData = []
facesData = []

# this is the function to draw histogram
def draw_histogram(numberCounts, type):
    
    plt.hist2d(numberCounts,bins = 10000)# bins value is the minor interval in histogram x-axis
    if type == 'vertex':
        plt.xlabel('number of vertices')
        #plt.axis([0, 30000, 0, 10]) # x-axis, [0,30000], y-axis, [0,10]
    elif type == 'face':
        plt.xlabel('number of faces')
        #plt.axis([0, 40000, 0, 25])
    plt.ylabel('frequency')
    plt.title('Histogram')
    plt.show()


def save_Excel(dataset, filename):
    df = pd.DataFrame(dataset)
    writer = pd.ExcelWriter(filename+'.xlsx', engine='xlsxwriter')
    df.to_excel(writer, sheet_name = filename, index=False)
    writer.save()

if __name__ == '__main__':
    # load the data
    # data = load_meshes.get_meshes(fromLPSB=False, fromPRIN=True, randomSample=-1, returnInfoOnly=True)
    data = load_meshes.get_meshes(fromLPSB=True, fromPRIN=False, randomSample=-1, returnInfoOnly=True)
    print(data)
    print(type(data)) # list
    print(data[1])
    print(type(data[1])) # dict
    print(data[1]['numVerts'])
    print(data[1]['numFaces'])
    # store data of vertices and faces into np.array
    for d in data:
        verticesData = np.append(verticesData, d['numVerts'])
        facesData = np.append(facesData, d['numFaces'])
    
    # draw the histogram
    draw_histogram(verticesData, 'vertex')
    draw_histogram(facesData, 'face')

    # Export to an excel sheet
    #save_Excel(data, 'Princeton')
    save_Excel(data, 'LabeledDB_new')