from ..utils import array_init 
from copy import deepcopy 
import numpy as np 

class NeuralStack():
    """A neural stack implementation. It should only accept and return the updated states."""
    
    def __init__(self, name="nstack"):
        self.name = name 
        self.timestep = -1 # init timestep is -1 

    def V_t(self, V_prev, v_t):
        
        # checks and balances 
        assert V_prev.shape[1] = v_t.shape[1] 
        assert v_t.shape[0] = 1 
        # Concatenate v_t to V_prev, this now is the V_t
        return np.concatenate([V_prev, v_t])

    def s_t(self):
        pass 

    def r_t(self):
        pass 
