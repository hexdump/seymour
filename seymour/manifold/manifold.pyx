from libc.math cimport abs, sqrt, sin, floor, tan, exp

def sig(double x):
    cdef double e
    #x = round(x, 3)
    try:
        e = exp(-x)
    except OverflowError:
        return 0
    return ((1 / (1 + e)) - 0.5) * 2

def cossim(v1, v2):
    cdef double num = 0
    cdef double v1_dist_squared = 0
    cdef double v2_dist_squared = 0
    for i in range(len(v1)):
        num += v1[i] * v2[i]
        v1_dist_squared += v1[i] ** 2
        v2_dist_squared += v2[i] ** 2
    return num / (sqrt(v1_dist_squared) * sqrt(v2_dist_squared) + 0.1)

cdef rpd(est, act):
    return abs(est - act)

def list_rpd(x1, x2):
    err = 0
    for x, y in zip(x1, x2):
        if isinstance(x, list):
            err += list_rpd(x, y)
        else:
            err += rpd(x, y)
    return err    

def layer(c, *argv):
    cdef double total = 0
    for arg in argv:
        x = arg
        total += x + c
#        total += sin(x * c + c) / len(argv)
#        total += sin(tan(arg + c)) / len(argv)
    return sin(total)
