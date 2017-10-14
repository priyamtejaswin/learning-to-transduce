#!/usr/bin/env python

## catch22 - a simple rnn in numpy

import numpy as np
import sys
import ipdb
from itertools import izip

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)


def generate_data(length, size):
    """
    Generate a random sequence of 1's and 0's.
    Inputs: length of sequences, number of samples
    Outputs: list of sequences, sum of sequences
    """

    print "Generating %d samples of length %d."%(size, length)
    data, target = [], []

    for i in xrange(size):
        row = np.zeros(length)
        n_ones = np.random.randint(low=0, high=length+1)
        n_indices = np.random.random_integers(low=0, high=length-1, size=n_ones)
        row[n_indices] = 1

        data.append(row)
        target.append(np.sum(row))

    assert len(data)==len(target), "-- data, target lengths mis-match --"
    return np.array(data), np.array(target)


def relu(x):
    """
    ReLU activation,
    Inputs: any number
    Outputs: ReLU'd number
    """

    return np.max((0, x))


def forward_pass((Wi, Wh, Wo), x):
    """
    Run single instance of a forward pass.
    Inputs: (weights), x_input
    Outputs: S_input, S_activation, y
    """

    aS = np.zeros(x.shape[0])
    bS = np.zeros_like(aS)

    for t in range(x.shape[0]):
        aS[t] = (Wi * x[t]) + (Wh * bS[t-1]) # right now everything is one number
        bS[t] = relu(aS[t])

    y = Wo * bS[t]

    return aS, bS, y


def main():
    """
    Main code for the program.
    """

    LENGTH = 10
    SIZE = 25
    data, target = generate_data(length=LENGTH, size=SIZE)

    SCALE = 0.1
    Wi = np.random.rand() * SCALE
    Wh = np.random.rand() * SCALE
    Wo = np.random.rand() * SCALE

    for _d, _t in izip(data, target):
        aS, bS, y = forward_pass((Wi, Wh, Wo), _d)
        print bS, _t, y

    return


if __name__ == '__main__':
    main()
