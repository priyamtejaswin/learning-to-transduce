import numpy as np
from ltt.memory import NeuralStack
from ltt.layers import MSE

def main():
    """
    Script to test stack popping operation.
    """

    ns = NeuralStack()
    u_ts = np.array([np.NaN, 0, 0, 0, 0, 0.4, 0.03])
    d_ts = np.array([np.NaN, 1, 1, 1, 1, 0, 0])
    EMBEDDING_SIZE = 2
    TOTALTIMESTEPS = 6

    sequence = np.array([ [0.1, 0], [0.2,0], [0.3,0] , [0.5, 0], [0.6,0], [0.7,0] ])
    expected_output = np.zeros(shape=(7,2)).astype(np.float64)
    expected_output[5,0] = 0.30
    expected_output[6,0] = 0.20

    for epochs in range(50000):

        V = {} # storage for each time step
        s = {} # strength vectors for each time step
        r = {} # read vectors out of the stack
        V[0] = np.empty(shape=(0, EMBEDDING_SIZE))
        s[0] = np.array([])

        d_grads = np.zeros_like(d_ts)
        u_grads = np.zeros_like(u_ts)

        loss = MSE("loss_mse")

        # import ipdb; ipdb.set_trace()
        # forward pass
        for ts in range(1, TOTALTIMESTEPS+1):
            # print("TS:",ts)

            # forward pass
            V[ts] = ns.V_t(V[ts-1], sequence[ts-1].reshape(1,-1))
            s[ts] = ns.s_t(s[ts-1], u_ts[ts], d_ts[ts]).astype(np.float64)
            r[ts] = ns.r_t(s[ts], V[ts]).astype(np.float64)
            # print("read vec:", r[ts])

        # import ipdb; ipdb.set_trace()
        # Backward pass
        for ts in range(6, 4, -1):
            # print("Backward pass TS: ", ts)
            # print("expected: {}, read: {}".format(expected_output[ts], r[ts]))
            # grad_r_t = loss.backward(r[ts], sequence[ts-3-1].reshape(1,-1))
            grad_r_t = r[ts] - expected_output[ts]
            grad_V_t, grad_s_t = ns.BACK_r_t(grad_r_t, s[ts], V[ts])
            grad_s_prev, grad_u_t, grad_d_t = ns.BACK_s_t(grad_s_t, s[ts-1], u_ts[ts], d_ts[ts])
            u_grads[ts] += grad_u_t
            d_grads[ts] += grad_d_t


        # u_ts[4] -= 0.01 * u_grads[4]
        u_ts[5] -= 0.01 * u_grads[5]
        u_ts[6] -= 0.01 * u_grads[6]

        np.clip(u_ts, 0, 1, out=u_ts)

        if epochs%500 == 0:
            print("u: ", u_ts)
            print("r: ", r)

        # sgd update
        # alpha = 0.001
        # for ts in range(1, TOTALTIMESTEPS+1):
            # u[ts] -= alpha * u_grads[ts]
            # d[ts] -= alpha * d_grads[ts]
        # if epochs%50==0:
            # print("u: ", u)
            # print("d: ", d)





if __name__ == "__main__":
    main()
