from __future__ import print_function
from __future__ import absolute_import
from .abstract_layer import AbstractLayer
import numpy as np
from copy import deepcopy

class Tanh(AbstractLayer):
    """ Tanh activation function """

    def __init__(self, name):
        self.name = name 

    def forward(self, x):
        self.input = x 
        self.output = np.tanh(self.input) 
        return self.output

    def backward(self, current_error):
        self.output_grad = current_error 
        dy_dx = 1 - np.power(self.output, 2)
        self.input_grad = np.multiply(self.output_grad, dy_dx) 
        self.check_grad_shapes()
        return self.input_grad

    def check_grad_shapes(self):
        assert self.input.shape == self.input_grad.shape, "input and input_grad shapes do not match"
        assert self.output.shape == self.output_grad.shape, "output and output_grad shapes do not match"

    def return_weights(self):
        return None

    def return_grads(self):
        return None

    def weights_iter(self):
        return np.nditer(np.array([]), op_flags=['readwrite'], flags=["zerosize_ok"])

def tanh_test():

    import ipdb; ipdb.set_trace()
    x = np.random.randn(5,10) 
    th = Tanh("tanh") 
    y = th.forward(x) 
    y_grad = np.zeros_like(y) + 1.0 
    x_grad = th.backward(y_grad)
    print("PASSED")
