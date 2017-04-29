import random

import scipy.io as sio
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

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


if __name__ == '__main__':
    points1 = loadPoints('./DB/freeform_beer_bottle.mat')
    points2 = loadPoints('./DB/freeform_hunting_knife.mat')
    points3 = loadPoints('./DB/freeform_pan_long.mat')

    k = smurph.kernelMP([points1, points2, points3], [0.1], 1, 100, 1)
    print(k)

