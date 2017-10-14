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


def deriv_relu(x):
    """
    Derivative function for ReLU.
    """

    if x>0:
        return 1
    else:
        return 0


def forward_pass((Wi, Wh, Wo), x):
    """
    Run single instance of a forward pass.
    Inputs: (weights), x_input
    Outputs: S_input, S_activation, y
    """

    aS = np.zeros(x.shape[0] + 1)
    bS = np.zeros_like(aS)

    for t in range(x.shape[0]):
        aS[t] = (Wi * x[t]) + (Wh * bS[t-1]) # right now everything is one number
        bS[t] = relu(aS[t])

    y = Wo * bS[t]

    return aS, bS, y


def rsme_loss(y, z):
    """
    Returns rmse error.
    Inputs: output, target
    Outputs: rmse
    """

    return 0.5 * np.square(y-z)


def backward_pass((Wi, Wh, Wo), aS, bS, x, y, z):
    """
    Returns updates for weights.
    Inputs: (weights), aS, bS, input, output, target
    Returns: del_Wi, del_Wh, del_Wo
    """
    LENGTH = x.shape[0]

    del_h = np.zeros(aS.shape[0]) # delta for hidden units

    del_Wi, del_Wh, del_Wo = 0, 0, 0

    for t in range(LENGTH)[::-1]:
        if t==LENGTH-1:
            del_h[t] = deriv_relu(aS[t]) * ((y - z)*Wo + del_h[t+1]*Wh)
        else:
            del_h[t] = deriv_relu(aS[t]) * del_h[t+1]*Wh

    # (y - z) == d_Loss/d_aOutputUnit
    del_Wo = (y - z) * bS[LENGTH-1] # del_O x bS[t_LAST]
    for t in range(LENGTH):
        del_Wh += del_h[t] * bS[t]
        del_Wi += del_h[t] * x[t]

    return del_Wi, del_Wh, del_Wo


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

    # for _d, _t in izip(data, target):
    #     aS, bS, y = forward_pass((Wi, Wh, Wo), _d)
    #     print bS, _t, y

    x, z = np.array([1, 1, 1, 1, 1]), 5

    aS, bS, y = forward_pass((Wi, Wh, Wo), x)
    print backward_pass((Wi, Wh, Wo), aS, bS, x, y, z)

    return


if __name__ == '__main__':
    main()
