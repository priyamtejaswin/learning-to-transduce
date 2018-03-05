from ..utils import array_init 
from copy import deepcopy 
import numpy as np 

class NeuralStack():
    """A neural stack implementation. It should only accept and return the updated states."""
    
    def __init__(self, name="nstack"):
        self.name = name 

    def V_t(self, V_prev, v_t):
        
        # checks and balances 
        assert V_prev.shape[1] = v_t.shape[1] 
        assert v_t.shape[0] = 1 
        # Concatenate v_t to V_prev, this now is the V_t
        return np.concatenate([V_prev, v_t])

    def s_t(self, s_prev, u_t, d_t):
        """
        Generate s_t from s_prev and current push signal "dt" and pop signal "u_t". 
        """
        # infer timestep based on length of s_prev 
        CURTIMESTEP = len(s_prev) + 1  
        
        print("Current timestep: ", CURTIMESTEP)
        
        # abstraction for convenience 
        def s_t_i(i):
            if i == CURTIMESTEP - 1:
                return d_t 
            else:
                return np.maximum(0, s_prev[i] - np.maximum(0, u_t - np.sum(s_prev[i+1:])))
        
        s_curr = [] 
        for i in range(CURTIMESTEP):
            s_curr.append(s_t_i(i))
        
        # checks and balances 
        assert len(s_curr) == CURTIMESTEP 
        return np.array(s_curr)


    def r_t(self):
        pass 
