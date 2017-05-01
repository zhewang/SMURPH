import random

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean
from sklearn.decomposition import KernelPCA

import smurph
import linear


def plotPoints(points):
    parray = np.array(points)
    plt.figure(figsize=(6,6))
    plt.scatter(parray[:,0], parray[:,1], s = 3)
    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

    plt.show()

def dataset_diffHoles():

    # generate a grid 100*100
    grid = []
    for i in range(41):
        for j in range(41):
            if euclidean((i,j), (20,20)) < 20:
                grid.append((i, j))

    radius = [2,3,4,5]
    points_list = []
    classes = []

    # bites
    center = (20, 0)
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, center) > r:
                points.append(p)
        points_list.append(points)
        classes.append(1)

    # 1 hole
    center = (20, 20)
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, center) > r:
                points.append(p)
        points_list.append(points)
        classes.append(2)

    # 2 holes
    centers = [(10, 20), (30, 20)]
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, centers[0]) > r and euclidean(p, centers[1]) > r:
                points.append(p)
        points_list.append(points)
        classes.append(3)

    # 3 holes
    centers = [(12, 12), (20, 30), (28, 12)]
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, centers[0]) > r and euclidean(p, centers[1]) > r and euclidean(p, centers[2]) > r:
                points.append(p)
        points_list.append(points)
        classes.append(4)

    return points_list, classes


# load FirstMM_object_data [M. Neumann 2013]
def loadPoints(fpath):
    m = sio.loadmat(fpath)
    pcloud = m['pointCloudObjectFrame']
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(pcloud[0], pcloud[1], pcloud[2], s=1)
    # print(len(pcloud[0]))
    # plt.show()

    points = []
    for i in range(len(pcloud[0])):
        points.append((pcloud[0][i], pcloud[1][i], pcloud[2][i]))

    return points

# calculate kernel for FirstMM_object_data
def calculateKernelforDB():
    pc_list = []
    with open('./DB/list.txt', 'r') as f:
        for line in f.readlines():
            pc_list.append(loadPoints('./DB/'+line.rstrip('\n')))
    k = smurph.kernelMP(pc_list, [0.1], 10, 100, 1)
    np.savetxt('kernel.txt', k)

def plot2D(kernel, classes, markers, colors):
    U, s, V = np.linalg.svd(kernel, full_matrices=True)
    result = U.dot(np.diag(np.sqrt(s)))
    x = result[:,0]
    y = result[:,1]

    for xp, yp, c in zip(x, y, classes):
        plt.scatter(xp, yp, marker=markers[c], c=colors[c])
    plt.show()

def exp_DB():
    # calculateKernelforDB()

    kernel = np.loadtxt('kernel.txt')
    # 1: long bottle, 2: bowl, 3: knife; 4: small can; 5:mug, 6:glass
    # 7: pan with handle
    classes = np.array([
        1,1,2,2,2, 3,3,3,3,4, 4,4,1,3,6,
        3,7,6,6,3, 3,6,5,5,5, 7,7,7,7,3,
        3,3,3,3,3, 3,7,3,3,1, 6
    ])
    markers = {1:'^', 2:'h', 3:'8', 4:'*', 5:'D', 6:'o', 7:'s'}
    colors = { 1:'#e41a1c',2:'#377eb8',3:'#4daf4a',4:'#984ea3',
              5:'#ff7f00',6:'#ffff33',7:'#a65628'}
    plot2D(kernel, classes, markers, colors)

def exp_DB_linear():
    pc_list = []
    with open('./DB/list.txt', 'r') as f:
        for line in f.readlines():
            pc_list.append(loadPoints('./DB/'+line.rstrip('\n')))
    k = linear.kernel(pc_list, 100)
    np.savetxt('kernel_linear.txt', k)

    kernel = np.loadtxt('kernel_linear.txt')
    # 1: long bottle, 2: bowl, 3: knife; 4: small can; 5:mug, 6:glass
    # 7: pan with handle
    classes = np.array([
        1,1,2,2,2, 3,3,3,3,4, 4,4,1,3,6,
        3,7,6,6,3, 3,6,5,5,5, 7,7,7,7,3,
        3,3,3,3,3, 3,7,3,3,1, 6
    ])
    markers = {1:'^', 2:'h', 3:'8', 4:'*', 5:'D', 6:'o', 7:'s'}
    colors = { 1:'#e41a1c',2:'#377eb8',3:'#4daf4a',4:'#984ea3',
              5:'#ff7f00',6:'#ffff33',7:'#a65628'}
    plot2D(kernel, classes, markers, colors)


if __name__ == '__main__':
    # exp_DB()
    # exp_DB_linear()
    points_list, classes = dataset_diffHoles()
    k = smurph.kernelMP(points_list, [40,20,10], 20, 100, 1)
    markers = {1:'^', 2:'h', 3:'8', 4:'*', 5:'D', 6:'o', 7:'s'}
    colors = { 1:'#e41a1c',2:'#377eb8',3:'#4daf4a',4:'#984ea3',
              5:'#ff7f00',6:'#ffff33',7:'#a65628'}
    plot2D(kernel, classes, markers, colors)

