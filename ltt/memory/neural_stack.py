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

    def print_stack(self):
        if self.timestep == -1:
            raise AttributeError("timestep is still -1. Something went wrong.")
        print "\nStack at timestep %d"%self.timestep
        for ix, (v,s) in enumerate(zip(self.v_mat, self.s_vec[0])):
            print ix, (v, s)
        return

    def update_s(self, u, d):
        if self.timestep == -1:
            raise AttributeError("timestep is still -1. Something went wrong.")

        s_prev = deepcopy(self.s_vec)

        for i in range(self.timestep, -1, -1):
            if i == self.timestep:
                self.s_vec[0, i] = d ## [0, i] because s_vec is a row vector shape(1, max_time)
            else:
                self.s_vec[0, i] = np.maximum(0,
                    s_prev[0, i] - np.maximum(0, u - np.sum(s_prev[0, i+1:self.timestep]))
                )
        self.s_prev = s_prev
        return

    def update_r(self):
        if self.timestep == -1:
            raise AttributeError("timestep is still -1. Something went wrong.")

        rvals = [] ## should be a list of tuples
        for i in range(self.timestep+1):
            rvals.append(
                (np.minimum(
                    self.s_vec[0, i],
                    np.maximum(0, 1.0 - np.sum(self.s_vec[0, i+1:self.timestep+1]))
                    ) , self.v_mat[i]))

        assert isinstance(rvals[0], tuple)
        assert len(rvals[0]) == 2

        self.r_val = np.sum(map(lambda t: t[0] * t[1], rvals), keepdims=True)

    def forward(self, v, u, d):
        """
        The stack is represented by (v_mat, s_vec).
        The stack is expected to return its update state(v_mat`, s_vec`) EVERY TIME.
        """
        self.ut = u

        assert v.shape == (1, self.embedding_size), "v vector shape and embedding_size do not match"
        if self.timestep >= (self.max_time-1):
            raise AttributeError("Stack timestep has reached max_time: %d"%self.max_time)

        self.timestep += 1

        self.v_mat[self.timestep] = v
        self.update_s(u, d)
        self.update_r()

    def backward(self, true_rt):
        """
        The stack is represented by (v_mat, s_vec).
        The stack expects a gradient wrt its state(v_mat, s_vec) EVERY TIME.
        """
        del_rt = self.r_val - true_rt
        del_ut = 0

        self.del_s_vec = np.zeros_like(self.s_vec) ## this will have dL/ds[n] = dL/dr * dr/ds[n]

        for i in range(self.timestep+1):
            if i<self.timestep:
                self.del_s_vec[0, i] += del_rt * self.grad_rt_stn(i)
                del_ut += self.del_s_vec[0, i] * self.grad_sti_ut(i)

            else:
                del_dt = del_rt * self.grad_rt_stn(i) * self.grad_sti_dt(i)

        return del_ut, del_dt


    def grad_rt_stn(self, n):
        s = []
        temp_strength_sum = lambda ix: np.sum(self.s_vec[0, ix+1: self.timestep+1])

        for i in range(0, self.timestep+1):

            if self.s_vec[0, i] <= max(0, 1 - temp_strength_sum(i)):
                s.append(int(i==n) * self.v_mat[i])

            elif ( (i<n) and (self.s_vec[0, i]>=max(0, 1 - temp_strength_sum(i))) and (temp_strength_sum(i)<=1) ):
                s.append(-1 * self.v_mat[i])

            else:
                s.append(0)

        return np.sum(s)

    def grad_sti_dt(self, i):
        return int(self.timestep==i)

    def grad_sti_ut(self, i):
        prev_strength_sum = lambda ix: np.sum(self.s_prev[0, ix+1: self.timestep])

        if ( (i<self.timestep) and (self.s_vec[0, i]>0) and ((self.ut - prev_strength_sum(i)) > 0) ):
            return -1
        else:
            return 0


def stack_test():
    import numpy as np
    ns = NeuralStack(embedding_size=1, max_time=600)
    print ns.name

    ns.forward(np.array([[4]]), 0, 1)
    ns.forward(np.array([[8]]), 0, 1)
    ns.forward(np.array([[2]]), 0, 1)

    ns.print_stack()
    print "r_val", ns.r_val

    u, d = 0.17, 0.39

    for ix in range(50):
        print "\n\t\tTraining iter:%d"%ix

        ns.forward(np.array([[0.137]]), u, d) ## rt should be 2
        print ns.r_val
        du, dt = ns.backward(2)

        u -= 0.5 * du
        d -= 0.5 * dt

        ns.forward(np.array([[0.567]]), u, d) ## rt should be 8
        print ns.r_val
        du, dt = ns.backward(2)

        u -= 0.5 * du
        d -= 0.5 * dt

        ns.forward(np.array([[0.111]]), u, d) ## rt should be 4
        print ns.r_val
        du, dt = ns.backward(2)

        u -= 0.5 * du
        d -= 0.5 * dt

        print du, dt

    ## ns.forward(np.array([[99]]), 0.9, 0.9) should raise AttributeError
    import ipdb; ipdb.set_trace()
    print "PASSED"

if __name__ == '__main__':
    pass
