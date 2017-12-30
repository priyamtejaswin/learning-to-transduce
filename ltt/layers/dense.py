from __future__ import print_function, absolute_import
from .abstract_layer import AbstractLayer
from ..utils import array_init
import numpy as np
from copy import deepcopy
from itertools import chain

class Dense(AbstractLayer):
    """ Fully connected layer """

    def __init__(self, n_in, n_out, name):
        self.weights    = array_init((n_in, n_out), vtype="rand") # np.random.randn(n_in, n_out) * 0.01
        self.bias       = array_init(n_out)
        self.name       = name

    def forward(self, x):
        assert x.shape[1] == self.weights.shape[0]
        self.input = x
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, current_error):
        """
        current_error = dL_dy
        output = dL_dx
        also save, dL_dW, dL_db
        """
        assert current_error.shape == self.output.shape
        self.output_grad    = current_error
        self.input_grad     = np.dot(self.output_grad, self.weights.T)
        self.weights_grad   = np.dot(self.input.T, self.output_grad)
        self.bias_grad      = self.output_grad.sum(axis=0, keepdims=True)
        self.check_grad_shapes() # built in self check on variables and their gradients
        return self.input_grad

    def check_grad_shapes(self):
        assert self.input.shape == self.input_grad.shape, "input and input_grad shapes do not match"
        assert self.output.shape == self.output_grad.shape, "output and output_grad shapes do not match"
        assert self.weights.shape == self.weights_grad.shape, "weights and weights_grad shapes do not match"
        assert self.bias.shape == self.bias_grad.shape, "bias and bias_grad shapes do not match"

    def return_weights(self):
        return (self.weights, self.bias)

    def return_grads(self):
        return (self.weights_grad, self.bias_grad)

    def weights_iter(self):
        iter_weights = np.nditer(self.weights, op_flags=["readwrite"])
        iter_bias = np.nditer(self.bias, op_flags=["readwrite"])
        return chain(iter_weights, iter_bias)

    def grads_iter(self):
        iter_grad_weights = np.nditer(self.weights_grad, op_flags=["readonly"])
        iter_grad_bias = np.nditer(self.bias_grad, op_flags=["readonly"])
        return chain(iter_grad_weights, iter_grad_bias)

def dense_test():
    x = np.random.rand(5, 10) * 0.1
    d = Dense(n_in=10, n_out=20, name="Dense1")
    # import ipdb; ipdb.set_trace()
    f = d.forward(x)
    error = np.zeros_like(f) + 1.0
    d.backward(error)
    print("PASSED")
