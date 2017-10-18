#!/usr/bin/env python

## A slightly more complex rnn in numpy.

import numpy as np
import sys
import ipdb

RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)


def generate_data(size, bit_size):
    """
    Generate random data for binary addition.
    Inputs: number of samples, number of bits
    Outputs: 2 unit input sequences, 1 unit output sequences
    """

    assert isinstance(bit_size, int), "-- bit_size is not int --"
    print "Generating %d samples of length %d."%(size, bit_size)
    data, target = [], []

    unwrap = lambda s: map(int, s[::-1])

    max_num = 2**bit_size - 1
    for _ix in xrange(size):
        a = np.random.randint(low=0, high=max_num+1)
        b = np.random.randint(low=0, high=max_num-a+1)
        c = a + b

        bin_a = unwrap(np.binary_repr(a, bit_size))
        bin_b = unwrap(np.binary_repr(b, bit_size))
        bin_c = unwrap(np.binary_repr(c, bit_size))

        data.append([bin_a, bin_b])
        target.append(bin_c)

    return np.array(data), np.array(target)


def main():
    """
    Main code.
    """

    data, target = generate_data(50, 5)
    for _ix in range(data.shape[0]):
        print data[_ix]
        print np.atleast_2d(target[_ix])
        print "----------------------\n"

    return


if __name__ == '__main__':
    main()
