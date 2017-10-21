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


def relu(x):
    """
    ReLU activation.
    Inputs: any number
    Outputs: ReLU'd number
    """

    return np.max((0, x))


def deriv_relu(x):
    """
    ReLU activation derivative.
    Inputs: any number
    Outputs: 1 if x>0 else 0
    """

    return 1 if x>0 else 0


def forward_pass((Wi, Wh, Wo), x):
    """
    Run single instance of a forward pass.
    Inputs: (weights), x_input
    Outputs: S_input, S_activation, y
    """

    LENGTH = x.shape[1]
    INPUT_DIM = x.shape[0]
    assert Wi.shape == x[:, 0].shape, "-- Wi, input shape mis-match --"

    aS = np.zeros(LENGTH+1)
    bS = np.zeros_like(aS)
    y = np.zeros(LENGTH)

    for t in range(LENGTH):
        aS[t] = np.sum(x[:, t] * Wi, axis=0) + (bS[t-1] * Wh)
        bS[t] = relu(aS[t])
        y[t] = bS[t] * Wo

    return aS, bS, y


def loss(y, z):
    """
    Implements mse loss function.
    Inputs: network output, target output
    Outputs: single value(normed for the entire batch)
    """

    assert y.shape == z.shape, "-- y, z shape mis-match -- %s %s"\
        %(str(y.shape), str(z.shape))
    return 0.5 * np.sum(np.square(z - y), axis=0)


def backward_pass((Wi, Wh, Wo), x, z, aS, bS, y):
    """
    Returns updates for parameters.
    Inputs: (parameters), input, target, aState, bState, output
    Outputs: gradients wrt parameters
    """

    assert y.shape == z.shape, "-- y, z shape mis-match -- %s %s"\
        %(str(y.shape), str(z.shape))
    LENGTH = x.shape[1]

    del_Wi = np.zeros_like(Wi)
    del_Wh = 0
    del_Wo = 0

    del_output = y - z

    for t in range(LENGTH)[::-1]:
        del_Wo += del_output[t] * bS[t]

        hfac, ifac = 1, 1
        hcon, icon = del_output[t] * Wo, del_output[t] * Wo

        for i in range(t+1)[::-1]:
            htime = deriv_relu(aS[i]) * bS[i-1]
            del_Wh += hcon * hfac * htime
            hfac *= deriv_relu(aS[i]) * Wh

            itime = deriv_relu(aS[i]) * x[:, i]
            del_Wi += icon * ifac * itime
            ifac *= deriv_relu(aS[i]) * Wh

    return del_Wi, del_Wh, del_Wo


def gradient_check():
    """
    Run a numerical gradient check for all model parameters.
    """

    print "\nRunning numerical gradient check...\n"
    SMALL_VAL = 1e-5
    x = np.array([[1, 1, 0, 1, 1], [0, 1, 0, 0, 0]])
    z = np.array([1, 0, 1, 1, 1])
    print "Input:\n", x
    print "Target:\n", '', z

    Wi = np.random.rand(2)
    Wh = np.random.rand()
    Wo = np.random.rand()

    aS, bS, y = forward_pass((Wi, Wh, Wo), x)
    print "\nLoss:\n", loss(y, z)

    del_Wi, del_Wh, del_Wo = backward_pass((Wi, Wh, Wo), x, z, aS, bS, y)

    ## del_Wo
    _, _, y = forward_pass((Wi, Wh, Wo+SMALL_VAL), x)
    plus = loss(y, z)
    _, _, y = forward_pass((Wi, Wh, Wo-SMALL_VAL), x)
    minus = loss(y, z)
    num_grad = (plus - minus)/(2*SMALL_VAL)
    assert np.allclose(del_Wo, num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: % --f"%(del_Wo, num_grad)

    ## del_Wh
    _, _, y = forward_pass((Wi, Wh+SMALL_VAL, Wo), x)
    plus = loss(y, z)
    _, _, y = forward_pass((Wi, Wh-SMALL_VAL, Wo), x)
    minus = loss(y, z)
    num_grad = (plus - minus)/(2*SMALL_VAL)
    assert np.allclose(del_Wh, num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: % --f"%(del_Wh, num_grad)

    ## del_Wi_0
    _, _, y = forward_pass((Wi + np.array([SMALL_VAL, 0]), Wh, Wo), x)
    plus = loss(y, z)
    _, _, y = forward_pass((Wi - np.array([SMALL_VAL, 0]), Wh, Wo), x)
    minus = loss(y, z)
    num_grad = (plus - minus)/(2*SMALL_VAL)
    assert np.allclose(del_Wi[0], num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: % --f"%(del_Wi[0], num_grad)

    ## del_Wi_1
    _, _, y = forward_pass((Wi + np.array([0, SMALL_VAL]), Wh, Wo), x)
    plus = loss(y, z)
    _, _, y = forward_pass((Wi - np.array([0, SMALL_VAL]), Wh, Wo), x)
    minus = loss(y, z)
    num_grad = (plus - minus)/(2*SMALL_VAL)
    assert np.allclose(del_Wi[1], num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: % --f"%(del_Wi[1], num_grad)

    print "\nPASSED\n"
    return


def main():
    """
    Main code.
    """

    NUM_SAMPLES, BIT_SIZE, EPOCHS = 1000, 4, 25
    TEST_SAMPLES = 100
    ALPHA = 0.0005

    data, target = generate_data(NUM_SAMPLES, BIT_SIZE)
    test_data, test_target = generate_data(TEST_SAMPLES, BIT_SIZE)

    Wi = np.random.rand(2) * 0.1
    Wh = np.random.rand() * 0.1
    Wo = np.random.rand() * 0.1

    to_number = lambda a: a.dot(2**np.arange(a.size))

    for _ep in xrange(EPOCHS):
        for _ix in xrange(NUM_SAMPLES):
            x, z = data[_ix], target[_ix]
            aS, bS, y = forward_pass((Wi, Wh, Wo), x)
            del_Wi, del_Wh, del_Wo = backward_pass((Wi, Wh, Wo), x, z, aS, bS, y)

            del_Wi = np.clip(del_Wi, -0.25, 0.25)
            del_Wh = np.clip(del_Wh, -0.25, 0.25)
            del_Wo = np.clip(del_Wo, -0.25, 0.25)

            Wi -= (ALPHA * del_Wi)
            Wh -= (ALPHA * del_Wh)
            Wo -= (ALPHA * del_Wo)

        TEST_ACC, TEST_LOSS = [], []
        for _ix in xrange(TEST_SAMPLES):
            x, z = test_data[_ix], test_target[_ix]
            _, _, y = forward_pass((Wi, Wh, Wo), x)

            TEST_LOSS.append(loss(y, z))
            TEST_ACC.append(to_number(z) == to_number(np.round(y)))

        print "Epoch %d, Loss %.2f, Accuracy %.2f"\
                %(_ep+1, np.mean(TEST_LOSS), np.mean(TEST_ACC))

    return


if __name__ == '__main__':
    gradient_check()
    main()
