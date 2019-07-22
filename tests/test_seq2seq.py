#!/usr/bin/env python3

#
# [tests/test_seq2seq.py]
#
# A simple seq2seq model trained using seymour.
# Copyright (C) 2019, Liam Schumm
#

import seymour.net as net
import seymour.ga as ga

word_indices = {
    'START': .0,
    'hello': .1,
    'how': .2,
    'are': .3,
    'you': .4,
    'bye': .5,
    'good': .6,
    'see': .7,
    'soon': .8,
    'EOF': .9,
}

word_indices_reversed = {v: k for k, v in word_indices.items()}

INPUTS = [
    'START hello how are you EOF',
    'START bye EOF',
]

OUTPUTS = [
    'good how are you EOF',
    'see you soon EOF'
]

inputs = [[word_indices[word] for word in sentence.split()] for sentence in INPUTS]
outputs = [[word_indices[word] for word in sentence.split()] for sentence in OUTPUTS]

def from_sentence(sentence):
    return [word_indices[word] for word in sentence.split()]

def fix(x):
    if x < 0: return 0
    if x > .9: return .9
    else: return int(x * 10) / 10

def to_sentence(vector):
    return [word_indices_reversed[fix(val)] for val in vector]

THOUGHT_SIZE = 5

import math

def sig(x):
    #x = round(x, 3)
    try:
        e = math.exp(-x)
    except OverflowError:
        return 0
    return 1 / (1 + e)

def hetero_arrdif(x, y):
    score = 0
    if len(x) > len(y):
        score += len(x) - len(y)
        x = x[:len(y)]

    if len(y) > len(x):
        score += len(y) - len(x)
        y = y[:len(x)]

    for (i, j) in zip(x, y):
        score += i - j

    return score

import numpy as np
def to_vec(c):
    return list(np.reshape(c, (len(c))))

class Seq2Seq(ga.Individual):

    def __init__(self, genome=None):

        enc_ni = THOUGHT_SIZE + 1
        encoder_genome_size = (enc_ni ** 2 + enc_ni) * 5 + enc_ni * THOUGHT_SIZE

        dec_ni = THOUGHT_SIZE
        decoder_genome_size = (dec_ni ** 2 + dec_ni) * 5 + enc_ni * (THOUGHT_SIZE + 1)

        self.genome = genome
        self.genome_size = encoder_genome_size + decoder_genome_size

        self.encoder_genome_size = encoder_genome_size
        self.decoder_genome_size = decoder_genome_size
        super().__init__()

    def reproduce(self, genome):
        return Seq2Seq(genome)

    def evaluate(self, inp, debug=False):
        
        assert(len(inp) > 1)

        self.encoder = net.Network([[0] * 6], [[0] * 5], 5, self.genome[:self.encoder_genome_size]) 
        self.decoder = net.Network([[0] * 5], [[0] * 6], 5, self.genome[self.encoder_genome_size:self.encoder_genome_size + self.decoder_genome_size])
        
        thought = [0] * THOUGHT_SIZE
        
        for char in inp:
            thought = to_vec(self.encoder.evaluate([char] + thought))

        i = 0
        out = [0]
        while (out[-1] < .9) and (i < 30):
            [c, *thought] = to_vec(self.decoder.evaluate(thought))
#            if debug:
#                print('IS')
#                print(c)
            #print(to_vec(self.decoder.evaluate(thought)))
            out.append(c)
            i += 1
        return out

    def fitness_function(self):
        score = 0
        for input in inputs:
            exp = self.evaluate(input)
            score += hetero_arrdif(exp, input)
        return sig(score)
            
gt = ga.GeneticTrainer(Seq2Seq, ())
s = gt.train(10)

#print(s.genome)
print(s.evaluate(inputs[0], True))
