from __future__ import absolute_import
from .base_memory import BaseMemory
from ..utils import array_init
from copy import deepcopy
import numpy as np

class NeuralStack(BaseMemory):
    """A neural stack implementation"""

    def __init__(self,
            embedding_size = 17,
            max_time = 5,
            name="nstack"
        ):
        self.name = name
        self.max_time = max_time
        self.embedding_size = embedding_size

        self.v_mat = array_init((self.max_time, self.embedding_size), vtype="zeros")
        self.s_vec = array_init(max_time, vtype="zeros")
        self.timestep = -1 ## stack is empty

    def update_s(self, u, d):
        if self.timestep == -1:
            raise AttributeError("timestep is still -1. Something went wrong.")

        s_prev = deepcopy(self.s_vec)

        for i in range(self.timestep, -1, -1):
            if i == self.timestep:
                self.s_vec[0, i] = d ## [0, i] because s_vec is a row vector shape(1, max_time)
            else:
                self.s_vec = np.maximum(0,
                    s_prev[0, i] - np.maximum(0, u - np.sum(s_prev[0, i+1:self.timestep]))
                )

    def forward(self, (v, u, d)):
        assert v.shape == (1, self.embedding_size), "v vector shape and embedding_size do not match"
        if self.timestep >= (self.max_time-1):
            raise AttributeError("Stack timestep has reached max_time: %d"%self.max_time)

        self.timestep += 1

        self.v_mat[self.timestep] = v
        self.update_s(u, d)

        return

    def backward(self):
        return


def stack_test():
    ns = NeuralStack(embedding_size=1, max_time=3)
    print ns.name

    print "PASSED"

if __name__ == '__main__':
    pass
