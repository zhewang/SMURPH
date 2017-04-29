import copy
import math
import numpy as np
import json
import random

from multiprocessing import Pool
from subprocess import Popen, PIPE
from scipy.integrate import quad
from scipy.spatial import distance_matrix


################################################################################
# inner product functions

def lfunc(s, l, pd):
    st = [ ((e[0]+e[1])/2.0,  (e[1]-e[0])/2.0) for e in pd]
    t = [ max(e[1]-abs(s-e[0]), 0.0) for e in st]
    t.sort()

    if l > len(t):
        return 0.0
    else:
        return t[-l]

def lfunc_multiscale(s, l, pds):
    avg = 0.0
    for pd in pds:
        avg += lfunc(s, l, pd)
    return avg / len(pds)

def integrand(s, l, pd1, pd2):
    return lfunc(s, l, pd1)*lfunc(s, l, pd2)

def pd_inner(pd1, pd2, l_bound):
    sigma = 0
    for i in range(1, l_bound+1):
        I = quad(integrand, -np.inf, np.inf, args=(i,pd1,pd2), limit=100)
        sigma += I[0]
    return sigma

def f_inner(f1, f2, l_bound):
    sigma = 0
    for i in range(1, l_bound+1):
        ifunc = lambda x : f1(x, i)*f2(x, i)
        I = quad(ifunc, -np.inf, np.inf, limit=100)
        sigma += I[0]
    return sigma


def pdsvec_inner(pdsvec1, pdsvec2, l_bound, weights):
    assert(len(pdsvec1) == len(pdsvec2) == len(weights))
    result = 0.0
    for pds1, pds2, w in zip(pdsvec1, pdsvec2, weights):
        f1 = lambda s, l : lfunc_multiscale(s, l, pds1)
        f2 = lambda s, l : lfunc_multiscale(s, l, pds2)
        result += w*f_inner(f1, f2, l_bound)
    return result

def flist_inner(flist1, flist2, l_bound, weights):
    assert(len(flist1) == len(flist2) == len(weights))
    result = 0.0
    for f1, f2, w in zip(flist1, flist2, weights):
        result += w*f_inner(f1,f2, l_bound)
    return result

################################################################################
# helper functions

def calPD(points):
    assert(len(points) > 0)
    str_data = json.dumps(points)
    dim = len(points[0])
    p = Popen(['./persistence/bin/cal_pd', str(dim), str_data], stdout=PIPE)
    out, err = p.communicate()

    pd = []
    for line in out.split('\n'):
        line = line.split()
        if len(line) == 3:
            pd.append((float(line[1]), float(line[2])))
    return set(pd)

# calculte the maximum pair-wise distance
def maxDist(metric_space):
    maxdist = 0
    for i in range(metric_space.shape[0]):
        for j in range(metric_space.shape[1]):
            if metric_space[i][j] > maxdist:
                maxdist = metric_space[i][j]
    return maxdist

# return a list of points within the given ball
def pointsInBall(points, metric_space, center_idx, r):
    inBall = []
    for i in range(metric_space.shape[0]):
        if metric_space[i][center_idx] <= r:
            inBall.append(points[i])
    return inBall

################################################################################
# kernels

def calRepresentation(args):
    points = args[0]
    metric_space = args[1]
    radius = args[2]
    m = args[3]
    b = args[4]
    s = args[5]
    pointsID = args[6]

    print('calculating representation for point cloud {}'.format(pointsID))
    pds_all_r = []
    for r in radius:
        pds_at_r = []
        for i in range(m):
            center_idx = random.randint(0, len(points)-1)
            for j in range(s):
                ball = pointsInBall(points, metric_space, center_idx, r)
                bootstrap = []
                if len(ball) <= b:
                    bootstrap = ball
                else:
                    bootstrap = random.sample(ball, b)
                pd = calPD(bootstrap)
                pds_at_r.append(pd)
        pds_all_r.append(pds_at_r)
    return pds_all_r

# SMURPH kernel multiprocessing
def kernelMP(points_list, radius, m, b, s):

    p = Pool(4)
    ms_list = [distance_matrix(X, X) for X in points_list]
    args_list = zip(
        points_list, ms_list,
        [radius for i in range(len(points_list))],
        [m for i in range(len(points_list))],
        [b for i in range(len(points_list))],
        [s for i in range(len(points_list))],
        [i for i in range(len(points_list))]
    )
    pds_all_r_list = p.map(calRepresentation, args_list)

    # calculate kernel
    weights = [(radius[0] / r)**3 for r in radius]
    k = np.zeros(shape=(len(points_list), len(points_list)), dtype='f8')
    for i in range(len(points_list)):
        for j in range(i, len(points_list)):
            l_bound = s
            print('calculating inner product of <{}, {}>'.format(i, j))
            inner_product = pdsvec_inner(pds_all_r_list[i], pds_all_r_list[j], l_bound, weights)
            k[i][j] = inner_product
            k[j][i] = inner_product

    return k

# SMURPH kernel
def kernel(points_list, radius, m, b, s):
    pds_all_r_list = []
    ms_list = [distance_matrix(X, X) for X in points_list]
    weights = [(radius[0] / r)**3 for r in radius]
    xcount = 0
    for X, metric_space in zip(points_list, ms_list):
        print('calculating representation for point cloud {}'.format(xcount))
        xcount += 1
        pds_all_r = []
        for r in radius:
            pds_at_r = []
            for i in range(m):
                center_idx = random.randint(0, len(X)-1)
                for j in range(s):
                    ball = pointsInBall(X, metric_space, center_idx, r)
                    bootstrap = []
                    if len(ball) <= b:
                        bootstrap = ball
                    else:
                        bootstrap = random.sample(ball, b)
                    pd = calPD(bootstrap)
                    pds_at_r.append(pd)
            pds_all_r.append(pds_at_r)
        pds_all_r_list.append(pds_all_r)

    # calculate kernel
    k = np.zeros(shape=(len(points_list), len(points_list)), dtype='f8')
    for i in range(len(points_list)):
        for j in range(i, len(points_list)):
            l_bound = s
            print('calculating inner product of <{}, {}>'.format(i, j))
            inner_product = pdsvec_inner(pds_all_r_list[i], pds_all_r_list[j], l_bound, weights)
            k[i][j] = inner_product
            k[j][i] = inner_product

    return k

# global kernel as described in the paper, used for validating
def kernel_global(points_list):
    pd_list = []
    for X in points_list:
        pd_list.append(calPD(X))

    k = np.zeros(shape=(len(points_list), len(points_list)), dtype='f8')
    for i in range(len(points_list)):
        for j in range(i, len(points_list)):
            l_bound = max(len(points_list[i]), len(points_list[j]))
            inner_product = pd_inner(pd_list[i], pd_list[j], l_bound)
            k[i][j] = inner_product
            k[j][i] = inner_product
    return k


if __name__ == '__main__':

    p1 = np.loadtxt('./data/mesh.xy', delimiter=',').tolist()
    p2 = np.loadtxt('./data/rect.xy', delimiter=',').tolist()
    p3 = np.loadtxt('./data/torus.xyz').tolist()

    # print(kernel_global([p1,p2,p3]))
    # print(kernel([p1,p2,p3], [20], 1, 300, 1))
