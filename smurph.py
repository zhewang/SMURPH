import math
import numpy as np
import json

from subprocess import Popen
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

if __name__ == '__main__':
    pd1 = [(1.0, 1.2), (1.0, 1.2), (1.0, 1.2), (1.3, 2.0), (2.0, 2.0)]
    pd2 = [(2.0, 5.0), (1.0, 2.0), (2.0, 8.0), (4.0, 9.0), (9.0, 11.0)]

    print(inner(pd1, pd2))

    data = [{'px':0, 'py':0},{'px':1, 'py':1},{'px':2, 'py':2}]
    str_data = json.dumps(data)
    p = Popen(['./persistence/bin/cal_pd', '2', str_data])
    p.communicate()

