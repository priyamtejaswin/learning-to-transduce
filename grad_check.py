from __future__ import print_function
import numpy as np
from ltt.layers import Dense, ReLU, Tanh, Sigmoid, MSE
from copy import deepcopy
from itertools import chain
from ltt.models import Model
from ltt.layers import RNN

def gradient_check(model):
    """
    Perform the gradient check on given model
    """

    # dummy data
    x = np.random.randn(4, model.feature_size)
    y_true = np.ones_like(model.do_forward(x))

    # analytic grads
    m_output = model.do_forward(x)

    stepped_output = []
    stepped_hidden = [np.zeros((1, model.layers["rnn"].n_hidden))]
    model.layers["rnn"].RNNTIME = 0

    for i in range(x.shape[0]):
        s_h, s_o = model.layers["rnn"].step_forward(x[i][np.newaxis, :], stepped_hidden[-1])
        stepped_hidden.append(s_h)
        stepped_output.append(s_o)

    assert np.array_equal(np.vstack(stepped_output), m_output), "step_forward and forward mismatch"
    print("\n\t\t\tdef:step_forward and def:forward outputs match. PASSED.\n")

    m_loss   = model.do_loss(y_true)
    _op_grad = model.loss_grad 
    # model.do_backward() #grads cached in each layer
    for i in range(x.shape[0])[::-1]:
        gr = model.layers["rnn"].step_backward(_op_grad[i,:].reshape(1,-1))

    # gradient check
    SMALL_VAL = 1e-5
    for layer_name, layer_obj in model.layers.items():

        print("layer: ", layer_name)

        for wt, anal_grad in zip(layer_obj.weights_iter(), layer_obj.grads_iter()):

            # lupper
            wt[...] = wt + SMALL_VAL
            yupper = model.do_forward(x)
            lupper = model.do_loss(y_true)

            # llower
            wt[...] = wt - (2.0*SMALL_VAL)
            ylower = model.do_forward(x)
            llower = model.do_loss(y_true)

            # reset param
            wt[...] = wt + SMALL_VAL

            num_grad = (lupper - llower) / (2*SMALL_VAL)

            assert np.allclose(num_grad, anal_grad, rtol=1e-4), \
                "-- Mismatch numerical: %f, analytical: % --f"%(num_grad, anal_grad)

            print("check", np.allclose(num_grad, anal_grad, rtol=1e-4))

        print("PASSED")

    print("def:step_forward and def:forward outputs match. PASSED.\n")
    print("\nAll parameter gradient checks completed")


if __name__ == '__main__':

    # create objects
    # l1 = Dense(n_in=3, n_out=4, name="dense_1")
    # loss = MSE("mse_loss")

    # model = Model(name="grad_check_model", loss_layer=loss)
    # model.add(l1)
    # model.feature_size = 3

    # gradient_check(model)

    l1 = RNN(n_in=3, n_out=2, n_hidden=5, LENGTH=4, name="rnn")
    loss = MSE("mse_loss")
    model = Model(name="gradient_check", loss_layer=loss)
    model.add(l1)
    model.feature_size = 3
    gradient_check(model)
