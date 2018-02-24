from __future__ import absolute_import
from .base_memory import BaseMemory
from ..utils import array_init
from copy import deepcopy
import numpy as np

class NeuralStack(BaseMemory):
    """A neural stack implementation. It should only accept and return the updated states."""

    def __init__(self, name="nstack"):
        self.name = name
        self.timestep = -1

    def update_s(self, strength_prev, d, u):
        if self.timestep == -1:
            raise AttributeError("timestep is still -1. Something went wrong.")

        s_prev = deepcopy(strength_prev)
        s_new = np.hstack((s_prev, [[d]]))

        TIME = s_new.shape[1]

        for i in range(1, TIME-1):
            s_new[0, i] = np.maximum(0,
                s_prev[0, i] - np.maximum(0, u - np.sum(s_prev[0, i+1:TIME-1]))
            )

        return s_new

    def update_r(self, values_t, strength_t):
        if self.timestep == -1:
            raise AttributeError("timestep is still -1. Something went wrong.")

        rvals = [] ## should be a list of tuples
        TIME = strength_t.shape[1]

        for i in range(TIME):
            rvals.append(
                (np.minimum(
                    strength_t[0, i],
                    np.maximum(0, 1.0 - np.sum(strength_t[0, i+1:TIME]))
                    ) , values_t[i]))

        assert isinstance(rvals[0], tuple)
        assert len(rvals[0]) == 2

        r_t = np.sum(map(lambda t: t[0] * t[1], rvals), keepdims=True)
        return r_t

    def forward(self, previous_state, input_state):
        """
        The stack will not store anything.

        The stack will accept two inputs:
        1. previous_state: (values_prev, strength_prev)
        2. input_state: (dt, ut, vt)

        The stack will return two outputs:
        1. next_state: (values_t, strength_t)
        2. output: r_t
        """
        self.timestep += 1

        values_prev, strength_prev = previous_state
        dt, ut, vt = input_state

        assert len(vt.shape) == 2
        assert len(values_prev.shape) == 2
        assert len(strength_prev.shape) == 2
        assert values_prev.shape[1] == vt.shape[1]
        assert strength_prev.shape[1] == values_prev.shape[0]

        values_t = np.vstack((values_prev, vt))

        strength_t = self.update_s(strength_prev, dt, ut)

        r_t = self.update_r(values_t, strength_t)

        return (values_t, strength_t), r_t


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
    """
    The stack will not store anything.

    The stack will accept two inputs:
    1. previous_state: (values_prev, strength_prev)
    2. input_state: (dt, ut, vt)

    The stack will return two outputs:
    1. next_state: (values_t, strength_t)
    2. output: r_t
    """

    import numpy as np
    ns = NeuralStack(name="MyStack")
    print ns.name

    v_next, s_next = np.array([[0.0]]), np.array([[0.0]])
    import ipdb; ipdb.set_trace()

    (v_next, s_next), r_t = ns.forward( (v_next, s_next), ( 0.8, 0.0, np.array([[1]]) ) )
    print v_next, "\n", s_next, r_t
    import ipdb; ipdb.set_trace()

    (v_next, s_next), r_t = ns.forward( (v_next, s_next), ( 0.5, 0.1, np.array([[2]]) ) )
    print v_next, "\n", s_next, r_t
    import ipdb; ipdb.set_trace()

    (v_next, s_next), r_t = ns.forward( (v_next, s_next), ( 0.9, 0.9, np.array([[3]]) ) )
    print v_next, "\n", s_next, r_t
    import ipdb; ipdb.set_trace()

    # print "r_val", ns.r_val
    #
    # u, d = 0.17, 0.39
    #
    # for ix in range(50):
    #     print "\n\t\tTraining iter:%d"%ix
    #
    #     ns.forward(np.array([[0.137]]), u, d) ## rt should be 2
    #     print ns.r_val
    #     du, dt = ns.backward(2)
    #
    #     u -= 0.5 * du
    #     d -= 0.5 * dt
    #
    #     ns.forward(np.array([[0.567]]), u, d) ## rt should be 8
    #     print ns.r_val
    #     du, dt = ns.backward(2)
    #
    #     u -= 0.5 * du
    #     d -= 0.5 * dt
    #
    #     ns.forward(np.array([[0.111]]), u, d) ## rt should be 4
    #     print ns.r_val
    #     du, dt = ns.backward(2)
    #
    #     u -= 0.5 * du
    #     d -= 0.5 * dt
    #
    #     print du, dt
    #
    # ## ns.forward(np.array([[99]]), 0.9, 0.9) should raise AttributeError
    # import ipdb; ipdb.set_trace()
    # print "PASSED"

if __name__ == '__main__':
    pass
