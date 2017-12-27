from abstract_layer import AbstractLayer 
import numpy as np
from copy import deepcopy

class Dense(AbstractLayer):
    """ Fully connected layer """ 

    def __init__(self, n_in, n_out):
        self.weights    = np.random.randn(n_in, n_out) * 0.01 
        self.bias       = np.zeros(n_out)
        self.weights_grad = None 
        self.bias_grad  = None 

    def forward(self, x):
        self.input = deepcopy(x) 
        self.output = np.dot(self.input, self.weights) + self.bias 
        return self.output 

    def backward(self, current_error):
        """
        current_error = dL_dy 
        output = dL_dx 
        also save, dL_dW, dL_db 
        """
        self.output_grad    = deepcopy(current_error)
        self.input_grad     = np.dot(self.output_grad, self.weights) 
        self.weights_grad   = np.dot(self.output_grad, self.input) 
        self.bias_grad      = self.output_grad * 1.0 
        return self.input_grad

    def return_weights(self):
        return [self.weights, self.bias] 
    
    def return_grads(self):
        return [self.weights_grad, self.bias_grad]

