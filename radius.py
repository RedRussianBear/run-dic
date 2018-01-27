import numpy as np
from math import pi, sin, cos, sqrt

import sys
from scipy.optimize import minimize
from re import split


def find_radius_error(directory, img_prefix, num):
    known = 28.620

    fin = open('%s/%d/%s-0000_0.csv' % (directory, num, img_prefix))
    fin.readline()
    line = fin.readline()

    m = []
    while line:
        nums = [float(x) for x in split(',\W*?', line)]
        m.append(np.array([nums[0], nums[1], nums[2]]))
        line = fin.readline()
        if line == '\n':
            line = None
    fin.close()

    def error(p):
        py, px, pz, th, ph, r = p
        d = np.array([cos(th) * sin(ph), sin(th) * sin(ph), cos(ph)])
        p0 = np.array([py, px, pz])
        to_p0 = [x - p0 for x in m]
        dists = [(sqrt(x.dot(x) - (x.dot(d)) ** 2) - r) ** 2 for x in to_p0]
        return sum(dists)

    bounds = ((None, None), (None, None), (None, None), (-pi, pi), (0, pi), (0, None))
    solve = minimize(error, np.array([0, 0, 175, pi / 2, pi / 2, 30]), bounds=bounds, method='SLSQP')
    print(solve)
    res = solve['x'].tolist()

    radius = res[5]

    percent = (radius - known) / known

    return percent


if __name__ == "__main__":
    print(find_radius_error(sys.argv[1], sys.argv[2], sys.argv[3]))
