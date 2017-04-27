import math
import numpy as np
import json

from subprocess import Popen, PIPE
from scipy.integrate import quad


def lfunc(s, l, pd):
    st = [ ((e[0]+e[1])/2.0,  (e[1]-e[0])/2.0) for e in pd]
    t = [ max(e[1]-abs(s-e[0]), 0.0) for e in st]
    t.sort()

    if l > len(t):
        return 0.0
    else:
        return t[-l]

def integrand(s, l, pd1, pd2):
    return lfunc(s, l, pd1)*lfunc(s, l, pd2)

def inner(pd1, pd2):
    sigma = 0
    for i in range(1, max(len(pd1), len(pd2))):
        I = quad(integrand, -np.inf, np.inf, args=(i,pd1,pd2), limit=100)
        sigma += I[0]
    return sigma

def calPD(points):
    str_data = json.dumps(points)
    p = Popen(['./persistence/bin/cal_pd', '2', str_data], stdout=PIPE)
    out, err = p.communicate()

    pd = []
    for line in out.split('\n'):
        line = line.split()
        if len(line) == 3:
            pd.append((float(line[1]), float(line[2])))
    return pd

if __name__ == '__main__':

    p1 = np.loadtxt('./data/mesh.xy', delimiter=',').tolist()
    p2 = np.loadtxt('./data/rect.xy', delimiter=',').tolist()

    pd1 = calPD(p1)
    pd2 = calPD(p2)
    print(inner(pd1, pd2))

