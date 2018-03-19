import numpy as np
np.random.seed(1234)

from ltt.layers import RNN
from ltt.layers import Dense
from ltt.layers import Sigmoid
from ltt.layers import Tanh
from ltt.memory import NeuralStack

n_in = 4
n_hid = 3
n_out = 5
EMBED_SIZE = 2

thernn = RNN(n_in=n_in, n_hidden=n_hid, n_out=n_out, LENGTH=10, name="recurrent_layer")

cnt_input = (
                np.zeros((1, n_hid)), # h_t-1
                np.zeros((1, EMBED_SIZE)), # r_t-1
                (
                    np.zeros((0, EMBED_SIZE)), # V_t-1
                    np.array([]) # s_t-1
                )
            )

sequence_input = [
    np.array([[0.5, 0]])
]

rnn_input = np.hstack((sequence_input[0], cnt_input[1])) # current sequence_input and previous stack_read.

rnn_hidden = cnt_input[0]

current_hidden, current_output = thernn.step_forward(rnn_input, rnn_hidden)

dense_dt = Dense(n_in=n_out, n_out=1, name="dense_dt")
dense_ut = Dense(n_in=n_out, n_out=1, name="dense_ut")
dense_vt = Dense(n_in=n_out, n_out=EMBED_SIZE, name="dense_vt")
dense_ot = Dense(n_in=n_out, n_out=EMBED_SIZE, name="dense_ot")

sigd_dt = Sigmoid(name="sigd_dt")
sigd_ut = Sigmoid(name="sigd_ut")
tanh_vt = Tanh(name="tanh_vt")
tanh_ot = Tanh(name="tanh_ot")

nstack = NeuralStack(name="nstack")

import ipdb; ipdb.set_trace()

nstack.forward(V_prev=cnt_input[2][0], s_prev=cnt_input[2][1],
               d_t = sigd_dt.forward(dense_dt.forward(current_output)).squeeze(),
               u_t = sigd_ut.forward(dense_ut.forward(current_output)).squeeze(),
               v_t = tanh_vt.forward(dense_vt.forward(current_output))
)

print("PASSED.")
