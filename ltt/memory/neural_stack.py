from ..utils import array_init
from copy import deepcopy
import numpy as np
from ..layers import MSE

class NeuralStack():
    """A neural stack implementation. It should only accept and return the updated states."""

    def __init__(self, name="nstack"):
        self.name = name

    def V_t(self, V_prev, v_t):

        # checks and balances
        assert V_prev.shape[1] == v_t.shape[1]
        assert v_t.shape[0] == 1
        # Concatenate v_t to V_prev, this now is the V_t
        return np.concatenate([V_prev, v_t])

    def s_t(self, s_prev, u_t, d_t):
        """
        Generate s_t from s_prev and current push signal "dt" and pop signal "u_t".
        """
        # infer timestep based on length of s_prev
        CURTIMESTEP = len(s_prev) + 1

        # print("Current timestep: ", CURTIMESTEP)

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


    def r_t(self, s_t, V_t):
        # infer current time step
        CURTIMESTEP = len(s_t)
        EMBEDDINGSIZE = V_t.shape[1]
        # print("Current timestep: ", CURTIMESTEP)
        # print("Embedding size: ", EMBEDDINGSIZE)

        # checks and balances
        assert len(s_t) == len(V_t)

        # abstraction for convenience
        def r_t_i(i):
            return V_t[i] * (np.minimum(s_t[i], np.maximum(0, 1 - np.sum(s_t[i+1:]))))

        # looping
        weighted_r = []
        for i in range(CURTIMESTEP):
            weighted_r.append(r_t_i(i))
        weighted_r = np.array(weighted_r)

        # checks and balances
        assert np.shape(weighted_r) == (CURTIMESTEP, EMBEDDINGSIZE)

        # do the weighted sum
        r_curr = np.sum(weighted_r, axis=0, keepdims=True)
        assert np.shape(r_curr) == (1, EMBEDDINGSIZE)

        return r_curr

    def BACK_r_t(self, del_r_t, s_t, V_t):
        """
        inputs:
        del_r_t: gradient on r_t (should be same shape as r_t)
        s_t : to compute intermediates
        V_t : to compute intermediates

        outputs:
        del_V_t : gradient on V_t
        del_s_t : gradient on s_t
        """
        del_s_t = np.zeros_like(s_t)
        del_V_t = np.zeros_like(V_t)

        def BACK_r_t_i(i):
            c = np.sum(s_t[i+1:])
            b = np.maximum(0, 1 - c)
            del_V_t[i] += del_r_t[0] * np.minimum(s_t[i], b)
            if s_t[i] < b:
                del_s_t[i] += np.sum(del_r_t * V_t[i])
            else:
                # s_t[i] > b
                del_b = np.sum(del_r_t * V_t[i])
                if (1 - c) > 0.0:
                    del_c = -1.0 * del_b
                    del_s_t[i+1:] += del_c

        # infer current time step
        CURTIMESTEP = len(s_t)
        EMBEDDINGSIZE = V_t.shape[1]
        # print("Current timestep: ", CURTIMESTEP)
        # print("Embedding size: ", EMBEDDINGSIZE)

        # checks and balances
        assert len(s_t) == len(V_t)
        for i in range(CURTIMESTEP):
            BACK_r_t_i(i)

        return del_V_t, del_s_t

    def BACK_s_t(self, del_s_t, s_prev, u_t, d_t):
        """
        inputs:
        del_s_t: gradient on s_t (must be same shape as s_t)
        s_prev: to compute intermediates
        u_t, d_t: to compute intermediates

        outputs:
        del_s_prev: gradient on s_prev
        del_u_t, del_d_t: gradient on u_t, d_t
        """
        assert len(s_prev) == len(del_s_t) - 1

        CURTIMESTEP = len(del_s_t)

        del_u_t = np.zeros_like([u_t])
        del_d_t = np.zeros_like([d_t])
        del_s_prev = np.zeros_like(s_prev)

        # convenience , this function will be called in a for loop
        def BACK_s_t_i(i):
            if i==CURTIMESTEP-1:
                del_d_t[0] += del_s_t[i]
            else:
                d = np.sum(s_prev[i+1:])
                c = u_t - d
                b = s_prev[i] - np.maximum(0, c)
                if b > 0:
                    del_b = del_s_t[i]
                    del_s_prev[i] += del_b * 1.0 # del_b * db/dst-1[i]
                    if c > 0:
                        del_c = del_b * -1.0 # del_c = del_b * db/dc ; db/dc = -1.0
                        del_u_t[0] += del_c * 1.0 # del_u_t = del_c * dc/du_t ; dc/du_t = 1.0
                        del_d = del_c * -1.0 # del_d = del_c * dc/dd ; dc/dd = -1.0 ; since c = u_t - d
                        del_s_prev[i+1:] += del_d

        # call BACK_s_t_i for each i
        for i in range(CURTIMESTEP):
            BACK_s_t_i(i)

        return del_s_prev, del_u_t[0], del_d_t[0]

    def BACK_V_t(self, del_r_t, s_curr, d_t, V_curr):
        CURTIMESTEP = len(s_curr)
        assert len(V_curr.shape) == 2
        assert V_curr.shape[0] == CURTIMESTEP
        assert del_r_t[0].shape == V_curr[-1].shape
        assert del_r_t.shape[0] == 1

        del_r_t = del_r_t[0].copy()

        del_V_curr = np.zeros_like(V_curr)

        del_V_prev = np.zeros((V_curr.shape[0]-1, V_curr.shape[1]))

        def BACK_V_t_n(n):
            if n==(CURTIMESTEP - 1):
                del_V_curr[n] = d_t * del_r_t
            else:
                del_V_curr[n] += np.minimum(s_curr[n], np.maximum(0, 1 - np.sum(s_curr[n+1:]))) * del_r_t

            for i in range(CURTIMESTEP-1):
                if i == n:
                    del_V_prev[i] += del_V_curr[n]

        for n in range(CURTIMESTEP):
            BACK_V_t_n(n)

        return del_V_prev, del_V_curr[-1]

    def forward(self, V_prev, s_prev, d_t, u_t, v_t):
        """
        Wrapper over ns.V_t, ns.s_t and ns.r_t 
        """
        V_t = self.V_t(V_prev, v_t)
        s_t = self.s_t(s_prev, u_t, d_t)
        r_t = self.r_t(s_t, V_t) 
        return V_t, s_t, r_t

    def backward(self, grad_r_t, s_prev, d_t, u_t, V_t, s_t):
        """ 
        Wrapper over ns.BACK_r_t, ns.BACK_s_t, ns.BACK_V_t
        """
        grad_V_t, grad_s_t = self.BACK_r_t(grad_r_t, s_t, V_t)
        grad_s_prev, grad_u_t, grad_d_t = self.BACK_s_t(grad_s_t, s_prev, u_t, d_t)
        grad_V_prev, grad_v_t = self.BACK_V_t(grad_r_t, s_t, d_t, V_t)
        return grad_V_prev, grad_s_prev, grad_d_t, grad_u_t, grad_v_t

def test_stack_forward():
    EMBEDDINGSIZE = 3
    V = {}
    s = {}
    r = {}
    V[0] = np.empty(shape=(0, EMBEDDINGSIZE))
    s[0] = np.array([])
    r[0] = np.array([])
    ns = NeuralStack()

    vts = np.eye(3)

    # t = 1
    V[1] = ns.V_t( V[0], vts[0].reshape(1,-1) )
    s[1] = ns.s_t( s[0], 0, 0.8 )
    r[1] = ns.r_t( s[1], V[1] )

    # t = 2
    V[2] = ns.V_t( V[1], vts[1].reshape(1,-1) )
    s[2] = ns.s_t( s[1], 0.1, 0.5 )
    r[2] = ns.r_t( s[2], V[2] )

    # t = 3
    V[3] = ns.V_t( V[2], vts[2].reshape(1,-1) )
    s[3] = ns.s_t( s[2], 0.9, 0.9 )
    r[3] = ns.r_t( s[3], V[3] )

    print("Last read vector: ", r[3])
    print("Last stack state: \n", V[3])
    print("Last strength vector: ", s[3])

def test_r_t_grad_check():
    EMBEDDINGSIZE = 3
    CURTIMESTEP = 4
    ns = NeuralStack()
    loss = MSE("mse_loss")

    vts = np.eye(3)

    # forward pass
    V_t = np.random.randn(CURTIMESTEP, EMBEDDINGSIZE)
    s_t = np.random.randn(CURTIMESTEP, )

    # gradient checking s_t
    for k in range(4):
        print("gradient check for s_t[{}]".format(k))

        # numer grad
        delta = 1e-5
        # up
        s_t[k] += delta
        r_t_up = ns.r_t(s_t, V_t)
        loss_up = loss.forward(r_t_up, np.ones((1,3)))
        # low
        s_t[k] -= 2.0*delta
        r_t_low = ns.r_t(s_t, V_t)
        loss_low = loss.forward(r_t_low, np.ones((1,3)))
        # reset
        s_t[k] += delta
        numer_grad = (loss_up - loss_low) / (2*delta)

        # analytical grad
        # fwd
        rt_preds = ns.r_t(s_t, V_t)
        # bwd
        grad_rt_preds = loss.backward(rt_preds, np.ones((1,3)))
        grad_V_t, grad_s_t = ns.BACK_r_t(grad_rt_preds, s_t, V_t)

        # check
        print(np.allclose(grad_s_t[k], numer_grad))
        print(numer_grad, grad_s_t[k])
        rel_error = np.abs(grad_s_t[k] - numer_grad) / np.maximum(np.abs(grad_s_t[k]), np.abs(numer_grad))
        print("Relative error: {}".format(rel_error))

    # gradient checking V_t
    it = np.nditer(V_t, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:

        print("Performing gradient check for V_t location: {}".format(it.multi_index))

        # numer grad
        delta = 1e-6
        # up
        it[0] += delta
        r_t_up = ns.r_t(s_t, V_t)
        loss_up = loss.forward(r_t_up, np.ones((1,3)))
        # low
        it[0] -= 2.0*delta
        r_t_low = ns.r_t(s_t, V_t)
        loss_low = loss.forward(r_t_low, np.ones((1,3)))
        # reset
        it[0]  += delta
        numer_grad = (loss_up - loss_low) / (2*delta)

        # analytical grad
        # fwd
        rt_preds = ns.r_t(s_t, V_t)
        # bwd
        grad_rt_preds = loss.backward(rt_preds, np.ones((1,3)))
        grad_V_t, grad_s_t = ns.BACK_r_t(grad_rt_preds, s_t, V_t)

        # check
        print(np.allclose(grad_V_t[it.multi_index], numer_grad))
        rel_error = np.abs(grad_V_t[it.multi_index] - numer_grad) / np.maximum(np.abs(grad_V_t[it.multi_index]), np.abs(numer_grad))
        print("Relative error: {}".format(rel_error))
        print(grad_V_t[it.multi_index], numer_grad)

        # move on to next
        it.iternext()

def test_s_t_grad_check():
    """ Test the BACK_s_t function """

    u_t = 0.0
    d_t = 0.9
    s_prev = np.random.rand(3,).astype(np.float64)
    CURTIMESTEP = 4
    ns = NeuralStack()
    loss = MSE("mse_loss")
    # gradient checking s_prev
    it = np.nditer(s_prev, flags=["c_index"], op_flags=['readwrite'])
    while not it.finished:

        print("Performing gradient check for s_prev location: {}".format(it.index))

        # numer grad
        delta = 1e-5
        # up
        it[0] += delta
        s_t_up = ns.s_t(s_prev, u_t, d_t)
        loss_up = loss.forward(s_t_up.reshape(-1,1), np.ones(shape=(4,1)))
        # low
        it[0] -= 2.0*delta
        s_t_low = ns.s_t(s_prev, u_t, d_t)
        loss_low = loss.forward(s_t_low.reshape(-1,1), np.ones(shape=(4,1)))
        # reset
        it[0]  += delta
        numer_grad = (loss_up - loss_low) / (2*delta)

        # analytical grad
        # fwd
        st_preds = ns.s_t(s_prev, u_t, d_t)
        # bwd
        grad_st_preds = loss.backward(st_preds.reshape(-1,1), np.ones(shape=(4,1)))
        grad_s_prev, grad_u_t, grad_d_t = ns.BACK_s_t(grad_st_preds, s_prev, u_t, d_t)

        # check
        print(np.allclose(grad_s_prev[it.index], numer_grad))
        rel_error = np.abs(grad_s_prev[it.index] - numer_grad) / np.maximum(np.abs(grad_s_prev[it.index]), np.abs(numer_grad))
        print("Relative error: {}".format(rel_error))
        print(grad_s_prev[it.index], numer_grad)

        # move on to next
        it.iternext()

def test_V_t_grad_check():
    EMBEDDINGSIZE = 3
    V = {}
    s = {}
    r = {}
    V[0] = np.empty(shape=(0, EMBEDDINGSIZE))
    s[0] = np.array([])
    r[0] = np.array([])
    ns = NeuralStack()
    loss = MSE("mse_loss")

    vts = np.eye(3)

    # t = 1
    V[1] = ns.V_t( V[0], vts[0].reshape(1,-1) )
    s[1] = ns.s_t( s[0], 0, 0.8 )
    r[1] = ns.r_t( s[1], V[1] )

    # t = 2
    V[2] = ns.V_t( V[1], vts[1].reshape(1,-1) )
    s[2] = ns.s_t( s[1], 0.1, 0.5 )
    r[2] = ns.r_t( s[2], V[2] )

    del_r_t = loss.backward( r[2], np.ones_like(r[2]) )
    import ipdb; ipdb.set_trace()
    del_V_prev, del_v_t = ns.BACK_V_t( del_r_t, s[2], 0.5, V[2] )

    ak_del_V, ak_del_s = ns.BACK_r_t(del_r_t, s[2], V[2])

    assert np.all(ak_del_V[-1] == del_v_t)
    print("V_t_grad passed.")
