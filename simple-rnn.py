#!/usr/bin/env python

## catch22 - a simple rnn in numpy

import numpy as np
import sys
import ipdb
from itertools import izip

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

## data - counting 1's in a sequence
def generate_data(length, size):
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


## main
def main():
    T = 10
    SIZE = 50
    data, target = generate_data(length=T, size=SIZE)

    for _d, _t in izip(data, target):
        print _d, _t

    return


if __name__ == '__main__':
    main()
