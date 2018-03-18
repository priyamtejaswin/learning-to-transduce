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
    u_ts = np.array([np.nan, 0, 0.24, 0.21, 0.89, 1])
    d_ts = np.array([np.nan, 1, 0.04, 0.76, 0.11, 0])
    EMBEDDING_SIZE = 2
    TOTALTIMESTEPS = 5

    sequence = np.array([
        [0.1, 0], # push
        [0.2, 0], # push
        [0.3, 0], # push
        [0.0, 0], # push
        [0.0, 0] # doesn't matter
    ])

    expected_output = np.zeros((6, 2))
    expected_output[1,0]  = 0.1 
    expected_output[2,0]  = 0.2 
    expected_output[3,0]  = 0.3 
    expected_output[4,0]  = 0.2 
    expected_output[5, 0] = 0.1 

    for epochs in range(50000):
        V, s, r = {}, {}, {}

        V[0] = np.empty((0, EMBEDDING_SIZE))
        s[0] = np.array([])

        d_grads = np.zeros_like(d_ts)
        u_grads = np.zeros_like(u_ts)

        loss = MSE("loss_mse")

        ## Forward pass
        for ts in range(1, TOTALTIMESTEPS+1):
            V[ts], s[ts], r[ts] = ns.forward(V[ts-1], s[ts-1], d_ts[ts], u_ts[ts], sequence[ts-1].reshape(1,-1))

        ## Backward pass
        for ts in range(TOTALTIMESTEPS, 0, -1): # backprop error to only what you want to learn.
            grad_r_t = r[ts] - expected_output[ts]
            grad_v_prev, grad_s_prev, grad_d_t, grad_u_t, grad_v_t = ns.backward(grad_r_t, s[ts-1], d_ts[ts], u_ts[ts], V[ts], s[ts])
            u_grads[ts] += grad_u_t
            d_grads[ts] += grad_d_t


        for bw_step in range(TOTALTIMESTEPS, 0, -1):
            d_ts[bw_step] -= 0.02 * d_grads[bw_step]
            u_ts[bw_step] -= 0.02 * u_grads[bw_step] 

        # clipping 
        np.clip(d_ts, 0, 1, out=d_ts)
        np.clip(u_ts, 0, 1, out=u_ts)

        if epochs%500 == 0:
            print("u: ", u_ts)
            print("d: ", d_ts)
            print("r: ", r)

if __name__ == '__main__':
    main()
