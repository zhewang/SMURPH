import random

import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import KernelPCA

import smurph

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


if __name__ == '__main__':
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
