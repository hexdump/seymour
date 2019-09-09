#!/usr/bin/env python3

#
# [common.py]
#
# Common math utilities for Seymour.
# Copyright (C) 2019, Liam Schumm
#

import numpy as np
from libc.math cimport abs

deviate = np.random.normal

cdef rpd(double est, double act):
    # relative percent difference
    return abs(est - act)
#    if est == act:
#        return 0
#    else:
#        return abs(2 * (est - act) / (abs(est) + abs(act)))

def clist_rpd(est, act, length):
    total = 0
    i = 0

    length = len(est)
    while i < length:
        if isinstance(est[i], list):
            total += clist_rpd(est[i], act[i], len(est[i]))
#        total += rpd(est[i], act[i])
        i += 1

    return total
    
def list_rpd(est, act):
    total = 0
    
    if isinstance(est, np.float64) or isinstance(est, float):
        total += rpd(est, act)
    else:
        est = est.flatten()
        act = act.flatten()
        for i in range(len(est)):
            total += list_rpd(est[i], act[i])
    return total
    
