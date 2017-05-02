import math
import numpy as np
import random

from scipy.spatial import distance_matrix

def points_inner(points1, points2):
    assert(len(points1) == len(points2))
    result = 0.0
    for p1, p2 in zip(points1, points2):
        result += np.inner(p1, p2)
    return result / len(points1)

def calHOD(points):
    distM = distance_matrix(points, points)
    hist, edges = np.histogram(distM, 10)
    return hist*1.0 / sum(hist)

def kernel(points_list):
    features_list = []
    for points in points_list:
        features_list.append(calHOD(points))

    # calculate kernel
    k = np.zeros(shape=(len(points_list), len(points_list)), dtype='f8')
    for i in range(len(points_list)):
        for j in range(i, len(points_list)):
            print('calculating inner product of <{}, {}>'.format(i, j))
            inner_product = np.inner(features_list[i], features_list[j])
            k[i][j] = inner_product
            k[j][i] = inner_product

    return k


if __name__ == '__main__':

    p1 = np.loadtxt('./data/mesh.xy', delimiter=',').tolist()
    p2 = np.loadtxt('./data/rect.xy', delimiter=',').tolist()

    print(kernel([p1,p2]))
