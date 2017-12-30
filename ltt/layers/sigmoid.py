from __future__ import print_function
from __future__ import absolute_import
from .abstract_layer import AbstractLayer
import numpy as np
from copy import deepcopy

class Sigmoid(AbstractLayer):
    """ Sigmoid activation function """

    def __init__(self, name):
        self.name = name 

    def forward(self, x):
        self.input = x 
        self.output = 1 / (1 + np.exp(-self.input)) 
        return self.output

    def backward(self, current_error):
        self.output_grad = current_error 
        dy_dx = np.multiply( self.output , (1.0 - self.output)) 
        self.input_grad  = self.output_grad * dy_dx  
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
        return np.nditer(np.array([]), op_flags=["readwrite"], flags=["zerosize_ok"])

    def grads_iter(self):
        return np.nditer(np.array([]), op_flags=["readonly"], flags=["zerosize_ok"])

def sigmoid_test():

    # import ipdb; ipdb.set_trace()
    x = np.random.randn(5,10) 
    sigm = Sigmoid("sigmoid") 
    y = sigm.forward(x) 
    y_grad = np.zeros_like(y) + 1.0 
    x_grad = sigm.backward(y_grad)
    print("PASSED")

