import random
import math

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

from scipy.spatial.distance import euclidean
from sklearn.decomposition import KernelPCA

import smurph
import linear
import hod


def plotPoints(points):
    parray = np.array(points)
    plt.figure(figsize=(6,6))
    plt.scatter(parray[:,0], parray[:,1], s = 3)
    # plt.axis('off')
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

    # bites
    center = (20, 0)
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, center) > r:
                points.append(p)
        points_list.append(points)

    # 1 hole
    center = (20, 20)
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, center) > r:
                points.append(p)
        points_list.append(points)

    # 2 holes
    centers = [(10, 20), (30, 20)]
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, centers[0]) > r and euclidean(p, centers[1]) > r:
                points.append(p)
        points_list.append(points)

    # 3 holes
    centers = [(12, 12), (20, 30), (28, 12)]
    for r in radius:
        points = []
        for p in grid:
            if euclidean(p, centers[0]) > r and euclidean(p, centers[1]) > r and euclidean(p, centers[2]) > r:
                points.append(p)
        points_list.append(points)

    return points_list

def dataset_multiscale():
    points_list = []

    points = []
    step = 0.3
    for i in np.arange(0, 41, step):
        for j in np.arange(0, 41, step):
            if euclidean((i,j), (20,20)) < 20 and euclidean((i,j), (20,20)) > 15:
                points.append((i, j))

    points_list.append(points)

    # generate centers
    center_count = 32
    centers = []
    for t in np.arange(0, 2*np.pi, 2*np.pi/center_count):
        centers.append((math.cos(t)*17.5+20, math.sin(t)*17.5+20))

    points = []
    for p in points_list[0]:
        valid = True
        for c in centers:
            if euclidean(p,c) < 1:
                valid = False
                break
        if valid is True:
            points.append(p)

    points_list.append(points)

    # double the above two
    doubled_list = []
    for points in points_list:
        new_points = []
        for p in points:
            new_points.append(p)
            new_points.append( (p[0]+40, p[1]))
        doubled_list.append(new_points)


    # scale the above four
    scaled_list = []
    for points in points_list+doubled_list:
        for alpha in [0.6, 0.2]:
            scaled = []
            for p in points:
                scaled.append( [x*alpha for x in p] )
            scaled_list.append(scaled)

    final_list = points_list+scaled_list+doubled_list
    return final_list

################################################################################

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

def plot2D(kernel, markers, colors):
    U, s, V = np.linalg.svd(kernel, full_matrices=True)
    result = U.dot(np.diag(np.sqrt(s)))
    x = result[:,0]
    y = result[:,1]

    for xp, yp, m, c in zip(x, y, markers, colors):
        plt.scatter(xp, yp, marker=m, c=c, alpha = 0.6)
    plt.show()

def exp_DB(kernelfunc, args):
    pc_list = []
    with open('./DB/list.txt', 'r') as f:
        for line in f.readlines():
            pc_list.append(loadPoints('./DB/'+line.rstrip('\n')))
    # k = kernelfunc(pc_list, [0.1], 10, 100, 1)
    k = kernelfunc(pc_list, *args)
    np.savetxt('kernel.txt', k)


def exp_multiholes(kernelfunc, args):
    points_list = dataset_diffHoles()
    k = kernelfunc(points_list, *args)
    # k = kernelfunc(points_list, [10], 5, 2000, 1)
    np.savetxt('kernel.txt', k)

def exp_multiscale(kernelfunc, args):
    points_list = dataset_multiscale()
    k = kernelfunc(points_list, *args)
    # k = smurph.kernelMP(points_list, [40, 10, 5], 5, 300, 1)
    # k = hod.kernel(points_list)
    np.savetxt('kernel.txt', k)

################################################################################

def plot2DPCA_DB(kernel_file_path):
    kernel = np.loadtxt(kernel_file_path)
    # 1: long bottle, 2: bowl, 3: knife; 4: small can; 5:mug, 6:glass
    # 7: pan with handle
    classes = np.array([
        1,1,2,2,2, 3,3,3,3,4, 4,4,1,3,6,
        3,7,6,6,3, 3,6,5,5,5, 7,7,7,7,3,
        3,3,3,3,3, 3,7,3,3,1, 6
    ])
    # markers = {1:'^', 2:'h', 3:'8', 4:'*', 5:'D', 6:'o', 7:'s'}
    markers = ['o' for i in range(len(classes))]
    colors_map = { 1:'#e41a1c',2:'#377eb8',3:'#4daf4a',4:'#984ea3',
              5:'#ff7f00',6:'#ffff33',7:'#a65628'}
    colors = []
    for c in classes:
        colors.append(colors_map[c])

    plot2D(kernel, markers, colors)

def plot2DPCA_Multiholes(kernel_file_path):
    k = np.loadtxt('kernel.txt')

    # style book
    holes = {0: 'o', 1: '^', 2: 's', 3: 'p'}
    holeSize = {'s':'#ffffb2', 'm': '#fecc5c' , 'l': '#fd8d3c', 'xl': '#e31a1c'}
    markers = [
        holes[0],holes[0],holes[0],holes[0],
        holes[1],holes[1],holes[1],holes[1],
        holes[2],holes[2],holes[2],holes[2],
        holes[3],holes[3],holes[3],holes[3]
    ]
    colors = [
        holeSize['s'],holeSize['m'],holeSize['l'],holeSize['xl'],
        holeSize['s'],holeSize['m'],holeSize['l'],holeSize['xl'],
        holeSize['s'],holeSize['m'],holeSize['l'],holeSize['xl'],
        holeSize['s'],holeSize['m'],holeSize['l'],holeSize['xl']
    ]
    plot2D(k, markers, colors)

def plot2DPCA_Multiscale(kernel_file_path):
    k = np.loadtxt('kernel.txt')

    # shape: O: ^, OO: *
    # scale: L-M-S:
    #    solid :#2ca25f, #99d8c9, #e5f5f9
    #    dotted:#e34a33, #fdbb84, #fee8c8
    shape = {'o': '^', 'oo': '*'}
    scale = {
        'solid-L': '#2ca25f', 'solid-M': '#99d8c9', 'solid-S': '#e5f5f9', # green
        'dotted-L': '#e34a33', 'dotted-M': '#fdbb84', 'dotted-S': '#fee8c8' # red
    }

    markers = [
        shape['o'],shape['o'],
        shape['o'],shape['o'],shape['o'],shape['o'],
        shape['oo'],shape['oo'],shape['oo'],shape['oo'],
        shape['oo'],shape['oo'],
    ]
    colors = [
        scale['solid-L'],scale['dotted-L'],
        scale['solid-M'],scale['solid-S'],scale['dotted-M'],scale['dotted-S'],
        scale['solid-M'],scale['solid-S'],scale['dotted-M'],scale['dotted-S'],
        scale['solid-L'],scale['dotted-L'],
    ]
    plot2D(k, markers, colors)

################################################################################

if __name__ == '__main__':
    # exp_DB(hod.kernel, args = ())
    # exp_DB(linear.kernel, args = (2000,))
    # exp_DB(smurph.kernelMP, args=([0.1], 10, 100, 1))
    # plot2DPCA_DB('kernel_DB_10_100.txt')
    # plot2DPCA_DB('kernel_DB_20_350.txt')

    # exp_multiholes(hod.kernel, args = ())
    # plot2DPCA_Multiholes('kernel_multiholes_[40_20_10]_20_100.txt')

    exp_multiscale(hod.kernel, args = ())
    # plot2DPCA_Multiscale('kernel_multiscale_[40_10_5]_5_300_1.txt')
