from __future__ import print_function
from __future__ import absolute_import
from .abstract_layer import AbstractLayer
import numpy as np
from copy import deepcopy

class ReLU(AbstractLayer):
    """ ReLU activation function """

    def __init__(self, name):
        self.name = name

    def forward(self, x):
        self.input = x
        self.output = np.maximum(self.input, 0)
        return self.output

    def backward(self, current_error):
        assert self.output.shape == current_error.shape, "output and error shapes do not match"

        self.output_grad = current_error
        mask = 1.0 * (self.input > 0)
        self.input_grad = np.multiply(mask, self.output_grad)

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

def relu_test():
    x = np.random.randn(5, 10) * 0.1
    rel = ReLU(name="relu1")
    y = rel.forward(x)
    y_grad = np.zeros_like(y) + 1.0
    x_grad = rel.backward(y_grad)

    print("PASSED")
