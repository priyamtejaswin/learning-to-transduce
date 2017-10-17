#!/usr/bin/env python

## A simple rnn in numpy - the REAL catch22.

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

    assert len(data) == len(target), "-- data, target lengths mis-match --"
    return np.array(data), np.array(target)


def relu(x):
    """
    ReLU activation,
    Inputs: any number
    Outputs: ReLU'd number
    """

    return np.maximum(0, x)


def deriv_relu(x):
    """
    Derivative function for ReLU.
    """

    g = np.zeros_like(x)
    g[x>0] = 1
    return g


def forward_pass((Wi, Wh, Wo), x):
    """
    Run single instance of a forward pass.
    Inputs: (weights), x_input
    Outputs: S_input, S_activation, y
    """

    LENGTH = x.shape[1] # LENGTH of sequence
    aS = np.zeros((x.shape[0], LENGTH+1)) # BSIZE, LENGTH+1
    bS = np.zeros_like(aS)

    for t in range(LENGTH):
        aS[:, t] = (x[:, t] * Wi) + (bS[:, t-1] * Wh) # right now everything is one number
        bS[:, t] = relu(aS[:, t])

    y = bS[:, t] * Wo

    return aS, bS, y


def mse_loss(y, z):
    """
    Returns mse error.
    Inputs: output, target
    Outputs: mse
    """
    # ipdb.set_trace()
    assert y.shape == z.shape, "-- y, z shape mis-match --"
    assert len(y.shape) == 1, "-- y shape is incorrect --"
    return 0.5 * np.mean(np.square(z - y), axis=0)


def backward_pass((Wi, Wh, Wo), x, z, aS, bS, y):
    """
    Returns updates for weights.
    Inputs: (weights), input, target, aS, bS, output
    Returns: del_Wi, del_Wh, del_Wo
    """

    LENGTH = x.shape[1] # LENGTH of sequence
    BSIZE = x.shape[0]

    del_Wi = 0
    del_Wh = 0

    ## del_Wo only has one term since
    ## Wo is only dependent on one activation.
    assert y.shape == z.shape, "-- y, z shape mis-match --"
    assert len(y.shape) == 1, "-- y shape incorrect --"
    del_output = y - z
    del_Wo = del_output * bS[:, LENGTH-1]

    hidden_factor = 1
    hidden_constant = del_output * Wo

    input_factor = 1
    input_constant = del_output * Wo

    ## del_Wh, del_Wi require only 1 loop because
    ## the next layer has only one activation
    ## which occurs at the final timestep.
    for t in range(LENGTH)[::-1]:
        hidden_time_derivative = deriv_relu(aS[:, t]) * bS[:, t-1]
        del_Wh += hidden_constant * hidden_factor * hidden_time_derivative
        hidden_factor *= deriv_relu(aS[:, t]) * Wh ## dS(t)/dS(t-1)

        input_time_derivative = deriv_relu(aS[:, t]) * x[:, t]
        del_Wi += input_constant * input_factor * input_time_derivative
        input_factor *= deriv_relu(aS[:, t]) * Wh ## dS(t)/dS(t-1)

    return np.mean(del_Wo, axis=0) , np.mean(del_Wh, axis=0), np.mean(del_Wi, axis=0)


def gradient_check():
    """
    Do a numerical gradient check for the entire model.
    """

    print "Running numerical gradient check..."

    SMALL_VAL = 1e-5
    SCALE = 0.1
    Wi = np.random.rand() * SCALE
    Wh = np.random.rand() * SCALE
    Wo = np.random.rand() * SCALE

    x = np.array([
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0]
    ])
    z = np.array([5, 4])

    aS, bS, y = forward_pass((Wi, Wh, Wo), x)
    del_Wo, del_Wh, del_Wi = backward_pass((Wi, Wh, Wo), x, z, aS, bS, y)

    ## del_Wo
    _, _, upper = forward_pass((Wi, Wh, Wo+SMALL_VAL), x)
    lupper = mse_loss(upper, z)
    _, _, lower = forward_pass((Wi, Wh, Wo-SMALL_VAL), x)
    llower = mse_loss(lower, z)

    num_grad = (lupper - llower)/(2*SMALL_VAL)
    assert np.allclose(del_Wo, num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: % --f"%(num_grad, del_Wo)

    ## del_Wh
    _, _, upper = forward_pass((Wi, Wh+SMALL_VAL, Wo), x)
    lupper = mse_loss(upper, z)
    _, _, lower = forward_pass((Wi, Wh-SMALL_VAL, Wo), x)
    llower = mse_loss(lower, z)

    num_grad = (lupper - llower)/(2*SMALL_VAL)
    assert np.allclose(del_Wh, num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: %f --"%(num_grad, del_Wh)

    ## del_Wi
    _, _, upper = forward_pass((Wi+SMALL_VAL, Wh, Wo), x)
    lupper = mse_loss(upper, z)
    _, _, lower = forward_pass((Wi-SMALL_VAL, Wh, Wo), x)
    llower = mse_loss(lower, z)

    num_grad = (lupper - llower)/(2*SMALL_VAL)
    assert np.allclose(del_Wi, num_grad, rtol=1e-4), \
        "-- Mismatch numerical: %f, analytical: %f --"%(num_grad, del_Wi)

    print "PASSED"
    return


def main():
    """
    Main code.
    """

    LENGTH, SAMPLES = 10, 1000
    TEST_SAMPLES = 100
    EPOCHS = 5
    SCALE = 0.1
    ALPHA = 0.001

    ## parameters
    Wi = np.random.rand() * SCALE
    Wh = np.random.rand() * SCALE
    Wo = np.random.rand() * SCALE

    ## training data
    data, target = generate_data(length=LENGTH, size=SAMPLES)
    ## testing data
    test_data, test_target = generate_data(length=LENGTH, size=TEST_SAMPLES)

    log = "Epoch %d. train_loss: %.2f, train_acc: %.2f, test_loss: %.2f, test_acc: %.2f"

    print "\nStart training.\n"
    for _ep in xrange(EPOCHS):
        test_acc, test_loss = [], []
        train_acc, train_loss = [], []

        for _ix in xrange(SAMPLES):
            x, z = np.atleast_2d(data[_ix]), np.array([target[_ix]])
            aS, bS, y = forward_pass((Wi, Wh, Wo), x)
            loss = mse_loss(y, z)

            del_Wo, del_Wh, del_Wi = backward_pass((Wi, Wh, Wo), x, z, aS, bS, y)

            Wi -= ALPHA * del_Wi
            Wh -= ALPHA * del_Wh
            Wo -= ALPHA * del_Wo

            train_acc.append(np.round(y)==z)
            train_loss.append(loss)

        for _ix in xrange(TEST_SAMPLES):
            x, z = np.atleast_2d(data[_ix]), np.array([target[_ix]])
            aS, bS, y = forward_pass((Wi, Wh, Wo), x)
            loss = mse_loss(y, z)

            test_acc.append(np.round(y)==z)
            test_loss.append(loss)

        print log%(_ep+1, np.mean(train_loss), np.mean(train_acc),
                    np.mean(test_loss), np.mean(test_acc))

    print "\nComplete.\n"
    print "Trained parameters:\nWi %f\nWh %f\nWo %f\n"%(Wi, Wh, Wo)
    return


if __name__ == '__main__':
    gradient_check()
    main()

    # x = np.array([
    # [1, 1, 1, 1, 1],
    # [0, 0, 0, 0, 0]
    # ])
    # z = np.array([5, 0])
    #
    # SCALE = 0.1
    # Wi = np.random.rand() * SCALE
    # Wh = np.random.rand() * SCALE
    # Wo = np.random.rand() * SCALE
    #
    # aS, bS, y = forward_pass((Wi, Wh, Wo), x)
    # print backward_pass((Wi, Wh, Wo), x, z, aS, bS, y)
