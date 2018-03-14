import numpy as np
import ipdb
from ltt.memory import NeuralStack
from ltt.layers import MSE

def main():
    """
    Script to test stack pushing operation.
    """

    ns = NeuralStack()
    # push, push, push, push, pop, pop.
    u_ts = np.array([np.nan, 0, 0, 0, 0, 1, 1])
    d_ts = np.array([np.nan, 1, 1, 1, 1, 0.77, 0.87])
    EMBEDDING_SIZE = 2
    TOTALTIMESTEPS = 6

    sequence = np.array([
        [0.1, 0], # push
        [0.2, 0], # push
        [0.3, 0], # push
        [0.5, 0], # push
        [0.234, 0], # doesn't matter
        [0.756, 0] # doesn't matter
    ])

    expected_output = np.zeros((7, 2))
    expected_output[5, 0] = 0.3 # This will be the READ value after the 1st pop signal.
    expected_output[6, 0] = 0.2 # READ value afte the 2nd pop signal.

    for epochs in xrange(50000):
        V, s, r = {}, {}, {}

        V[0] = np.empty((0, EMBEDDING_SIZE))
        s[0] = np.array([])

        d_grads = np.zeros_like(d_ts)
        u_grads = np.zeros_like(u_ts)

        loss = MSE("loss_mse")

        ## Forward pass
        for ts in range(1, TOTALTIMESTEPS+1):
            V[ts] = ns.V_t(V[ts-1], sequence[ts-1].reshape(1,-1))
            s[ts] = ns.s_t(s[ts-1], u_ts[ts], d_ts[ts]).astype(np.float64)
            r[ts] = ns.r_t(s[ts], V[ts]).astype(np.float64)

        ## Backward pass
        for ts in range(6, 4, -1): # backprop error to only what you want to learn.
            grad_r_t = r[ts] - expected_output[ts]
            grad_V_t, grad_s_t = ns.BACK_r_t(grad_r_t, s[ts], V[ts])
            grad_s_prev, grad_u_t, grad_d_t = ns.BACK_s_t(grad_s_t, s[ts-1], u_ts[ts], d_ts[ts])
            u_grads[ts] += grad_u_t
            d_grads[ts] += grad_d_t


        d_ts[5] -= 0.05 * d_grads[5]
        d_ts[6] -= 0.05 * d_grads[6]

        np.clip(d_ts, 0, 1, out=d_ts)

        if epochs%500 == 0:
            print("u: ", d_ts)
            print("r: ", r)

if __name__ == '__main__':
    main()
