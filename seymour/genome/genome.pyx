#!/usr/bin/env python3

#
# [genome.py]
#
# Genome utilities for Seymour.
# Copyright (C) 2019, Liam Schumm
#

from random import random
from seymour.common import deviate, list_rpd
#from libc.stdlib cimport malloc

def mutate_genome(genome, err,
                  sd_err_coeff, conv_pow):
    return [deviate(x, sd_err_coeff * abs(err ** conv_pow))
            for x in genome]

def mix_genomes(l1, l2):
    assert len(l1) == len(l2)
    return [l1[i] if random() > 0.5 else l2[i]
            for i in range(len(l1))]



# cdef generate_genome(int size, double err, double mean):
#     cdef double *genome = <double *> malloc(size * sizeof(double))
#     if not genome:
#         raise MemoryError('unable to allocate space for genome.')

    
def generate_genome(size, mean=0, err=1):
    return [deviate(mean, err) for _ in range(size)]


def asexually_reproduce(genome, error,
                        sd_err_coeff, conv_pow):
    return mutate_genome(genome, error, sd_err_coeff, conv_pow)

def sexually_reproduce(genome1, error1,
                       genome2, error2,
                       sd_err_coeff, conv_pow):
    return [mutate_genome(mix_genomes(genome1, genome2),
                          0.5 * (error1 + error2))
            for _ in range(2)]

genome_difference = list_rpd
