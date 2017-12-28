from __future__ import print_function
from abstract_layer import AbstractLayer
import numpy as np
from copy import deepcopy

class Dense(AbstractLayer):
    """ Fully connected layer """

    def __init__(self, n_in, n_out):
        self.weights    = np.random.randn(n_in, n_out) * 0.01
        self.bias       = np.zeros(n_out)
        self.weights_grad = None ## why None? why initialise at all?
        self.bias_grad  = None ## why not let them be created during backward pass?

    def forward(self, x):
        self.input = deepcopy(x) ## is this really required?
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward(self, current_error):
        """
        current_error = dL_dy
        output = dL_dx
        also save, dL_dW, dL_db
        """
        assert self.current_error.shape == self.output.shape, "incorrect error shape" ## assert before starting?

        self.output_grad    = deepcopy(current_error) ## is this really required?
        self.input_grad     = np.dot(self.output_grad, self.weights.T) ## del_error * del_local
        self.weights_grad   = np.dot(self.input.T, self.output_grad)
        self.bias_grad      = self.output_grad.sum(axis=0)

        self.check_grad_shapes() # built in self check on variables and their gradients
        return self.input_grad

    def check_grad_shapes(self):
        assert self.input.shape == self.input_grad.shape, "input and input_grad shapes do not match"
        assert self.output.shape == self.output_grad.shape, "output and output_grad shapes do not match"
        assert self.weights.shape == self.weights_grad.shape, "weights and weights_grad shapes do not match"
        assert self.bias.shape == self.bias_grad.shape, "bias and bias_grad shapes do not match"

    def return_weights(self):
        return [self.weights, self.bias]

    def return_grads(self):
        return [self.weights_grad, self.bias_grad]

if __name__ == '__main__':

    try:
        x = np.random.rand(5, 10) * 0.1
        d = Dense(n_in=10, n_out=20)
        # import ipdb; ipdb.set_trace()
        f = d.forward(x)
        error = np.zeros_like(f) + 1.0
        d.backward(error)
    except:
        print("Backward pass failed")
    else:
        print("Backward pass shapes check passed")
