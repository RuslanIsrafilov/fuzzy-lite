import numpy as np

def trimf(x, abc):
    a, b, c = np.r_[abc]
    y = np.zeros(len(x))

    if a != b:
        idx = np.nonzero(np.logical_and(a < x, x < b))[0]
        y[idx] = (x[idx] - a) / float(b - a)

    if b != c:
        idx = np.nonzero(np.logical_and(b < x, x < c))[0]
        y[idx] = (c - x[idx]) / float(c - b)

    idx = np.nonzero(x == b)
    y[idx] = 1
    return y

def trapmf(x, abcd):
    a, b, c, d = np.r_[abcd]
    y = np.ones(len(x))

    idx = np.nonzero(x <= b)[0]
    y[idx] = trimf(x[idx], np.r_[a, b, b])

    idx = np.nonzero(x >= c)[0]
    y[idx] = trimf(x[idx], np.r_[c, c, d])

    idx = np.nonzero(x < a)[0]
    y[idx] = np.zeros(len(idx))

    idx = np.nonzero(x > d)[0]
    y[idx] = np.zeros(len(idx))

    return y


def interp_membership(x, xmf, xx):
    x1 = x[x <= xx][-1]
    x2 = x[x >= xx][0]

    idx1 = np.nonzero(x == x1)[0][0]
    idx2 = np.nonzero(x == x2)[0][0]

    xmf1 = xmf[idx1]
    xmf2 = xmf[idx2]

    if x1 == x2:
        xxmf = xmf[idx1]
    else:
        slope = (xmf2 - xmf1) / float(x2 - x1)
        xxmf = slope * (xx - x1) + xmf1
    return xxmf


def operator_min(x, y):
    return min(x, y)

def operator_max(x, y):
    return max(x, y)

def operator_prod(x, y):
    return x * y

def operator_centroid(universe, xmf):
    sum1 = np.sum(np.array(universe) * np.array(xmf))
    sum2 = np.sum(np.array(xmf))
    return sum1 / sum2