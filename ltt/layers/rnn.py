from __future__ import print_function, absolute_import
from .abstract_layer import AbstractLayer
from ..utils import array_init
import numpy as np
from copy import deepcopy
from itertools import chain 

class RNN(AbstractLayer):
    """ Recurrent nn layer """ 
    
    def __init__(self, n_in, n_out, n_hidden, LENGTH, name="rnn"):
        self.name = name 
        self.n_in = n_in 
        self.n_out = n_out 
        self.n_hidden = n_hidden
        self.LENGTH = LENGTH # length of 1 sequence i.e number of items in 1 sequence
        self.Wi = array_init((n_in, n_hidden))
        self.Wh = array_init((n_hidden, n_hidden))
        self.Wo = array_init((n_hidden, n_out)) 
        print(" self.Wo: {} | self.Wi: {} | self.Wh: {}".format(self.Wo.shape, self.Wi.shape, self.Wh.shape))

    def set_weights(self, weight_tuple):
        self.Wi = deepcopy(weight_tuple[2]) 
        self.Wh = deepcopy(weight_tuple[1])
        self.Wo = deepcopy(weight_tuple[0])
        return 

    def return_weights(self):
        return (deepcopy(self.Wo), deepcopy(self.Wh), deepcopy(self.Wi))

    def weights_iter(self):
        iter_Wo = np.nditer(self.Wo, op_flags=["readwrite"])
        iter_Wh = np.nditer(self.Wh, op_flags=["readwrite"])
        iter_Wi = np.nditer(self.Wi, op_flags=["readwrite"])
        return chain(iter_Wo, iter_Wh, iter_Wi)

    def grads_iter(self):
        iter_Wo_grad = np.nditer(self.del_Wo, op_flags=["readwrite"])
        iter_Wh_grad = np.nditer(self.del_Wh, op_flags=["readwrite"])
        iter_Wi_grad = np.nditer(self.del_Wi, op_flags=["readwrite"])
        return chain(iter_Wo_grad, iter_Wh_grad, iter_Wi_grad)

    def return_grads(self):
        return (deepcopy(self.del_Wo), deepcopy(self.del_Wh), deepcopy(self.del_Wi))

    def forward(self, x):
        """ 
        perform forward pass 
        curerntly input is of the shape = (num of sequences, n_in)
        eventually in batched rnn input_shape = (batch, num_of_sequences, n_in)
        """
        
        assert x.shape[1] == self.Wi.shape[0] , "dimensions of input are different from n_in"
        assert x.shape[0] == self.LENGTH, "LENGTH of sequence is different from the one defined"
        
        # LENGTH = x.shape[0]  # length of 1 sequence i.e number of items in 1 sequence 
        INPUT_DIM = x.shape[1] # dimension of 1 item of 1 sequence
        
        aS = np.zeros((self.LENGTH+1, self.n_hidden))
        bS = np.zeros_like(aS) 
        
        y  = np.zeros((self.LENGTH, self.n_out)) # output for each item in sequence
        
        for t in range(self.LENGTH):
            aS[t,:] = np.dot(x[t,:], self.Wi) + np.dot(bS[t-1], self.Wh)  
            bS[t,:] = np.tanh(aS[t, :]) 
            y[t, :] = np.dot(bS[t, :], self.Wo)

        # cache values 
        self.aS = aS 
        self.bS = bS 
        self.output = y 
        self.input = x 
    
        return self.output 

    def backward(self, current_error):
        
        assert current_error.shape[0] == self.output.shape[0],\
                "current_error (dL/dy) shape does not match y shape"
        
        self.del_Wo = np.zeros_like(self.Wo)
        self.del_Wh = np.zeros_like(self.Wh)
        self.del_Wi = np.zeros_like(self.Wi) 
        self.del_input = np.zeros_like(self.input)
        
        for t in range(self.LENGTH)[::-1]:
            self.del_Wo += np.dot(self.bS[t,:].reshape(-1,1), current_error[t,:].reshape(1,-1))
            
            delta_t = np.dot ( current_error[t,:].reshape(1,-1), self.Wo.T )
            delta_t = delta_t * ( 1 - (self.bS[t,:]**2) )
            assert delta_t.shape == (1,self.n_hidden), "delta_t incorrect shape"
            
            for i in range(t+1)[::-1]:
                # Update del_Wh
                self.del_Wh += np.outer(delta_t, self.bS[i-1,:]).T
                # Update delWi 
                self.del_Wi += np.outer(delta_t, self.input[i,:]).T

                # Update self.del_input 
                self.del_input[i,:] += np.dot(delta_t, self.Wi.T)[0]
                
                # update delta 
                delta_t = np.dot(delta_t, self.Wh.T) * (1-self.bS[i-1]**2)
        
        return self.del_input

def rnn_test():

    # data 
    # sequence length = 10
    x = np.random.randn(8, 3) 
    y = np.ones((8, 2))

    # forward pass test 
    rnn_obj = RNN(n_in=3, n_out=2, n_hidden=5, LENGTH=8, name="rnn")
    _ = rnn_obj.forward(x)
    current_error = np.ones_like(y) + 2.0 
    rnn_obj.backward(current_error) 


