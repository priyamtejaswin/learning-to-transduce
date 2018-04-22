import numpy as np
np.random.seed(1234)

from ltt.layers import RNN
from ltt.layers import Dense
from ltt.layers import Sigmoid
from ltt.layers import Tanh
from ltt.memory import NeuralStack

# n_in = 4
# n_hid = 3
# n_out = 5
# EMBED_SIZE = 2

# thernn = RNN(n_in=n_in, n_hidden=n_hid, n_out=n_out, LENGTH=10, name="recurrent_layer")

# cnt_input = (
                # np.zeros((1, n_hid)), # h_t-1
                # np.zeros((1, EMBED_SIZE)), # r_t-1
                # (
                    # np.zeros((0, EMBED_SIZE)), # V_t-1
                    # np.array([]) # s_t-1
                # )
            # )

# sequence_input = [
    # np.array([[0.5, 0]])
# ]

# rnn_input = np.hstack((sequence_input[0], cnt_input[1])) # current sequence_input and previous stack_read.

# rnn_hidden = cnt_input[0]

# current_hidden, current_output = thernn.step_forward(rnn_input, rnn_hidden)

# dense_dt = Dense(n_in=n_out, n_out=1, name="dense_dt")
# dense_ut = Dense(n_in=n_out, n_out=1, name="dense_ut")
# dense_vt = Dense(n_in=n_out, n_out=EMBED_SIZE, name="dense_vt")
# dense_ot = Dense(n_in=n_out, n_out=EMBED_SIZE, name="dense_ot")

# sigd_dt = Sigmoid(name="sigd_dt")
# sigd_ut = Sigmoid(name="sigd_ut")
# tanh_vt = Tanh(name="tanh_vt")
# tanh_ot = Tanh(name="tanh_ot")

# nstack = NeuralStack(name="nstack")

# import ipdb; ipdb.set_trace()

# V_t, s_t, r_t = nstack.forward(V_prev=cnt_input[2][0], s_prev=cnt_input[2][1],
               # d_t = sigd_dt.forward(dense_dt.forward(current_output)).squeeze(),
               # u_t = sigd_ut.forward(dense_ut.forward(current_output)).squeeze(),
               # v_t = tanh_vt.forward(dense_vt.forward(current_output))
# )

#### Main controller class.
class Controller:
    """ Main Controller class. """

    def __init__(self, controller_in, rnn_hid, rnn_out, LENGTH, EMBED_SIZE, name="controller"):
        """ Initializes the rnn, stack and all dense+nonlinear objects. """

        self.controller_in = controller_in
        rnn_in = controller_in + EMBED_SIZE # concat( input, r_t-1 ) --> input to the RNN
        self.rnn = RNN(n_in=rnn_in, n_hidden=rnn_hid, n_out=rnn_out, LENGTH=LENGTH, name="recurrent_layer")

        self.dense_dt = Dense(n_in=rnn_out, n_out=1, name="dense_dt")
        self.dense_ut = Dense(n_in=rnn_out, n_out=1, name="dense_ut")
        self.dense_vt = Dense(n_in=rnn_out, n_out=EMBED_SIZE, name="dense_vt")
        self.dense_ot = Dense(n_in=rnn_out, n_out=EMBED_SIZE, name="dense_ot")

        self.sigd_dt = Sigmoid(name="sigd_dt")
        self.sigd_ut = Sigmoid(name="sigd_ut")
        self.tanh_vt = Tanh(name="tanh_vt")
        self.tanh_ot = Tanh(name="tanh_ot")

        self.nstack = NeuralStack(name="nstack")

        # Caching
        self.TS =0
        self.CACHE = {} 
        self.CACHE[self.TS] = {
            "V_t": np.zeros((0, EMBED_SIZE)),
            "s_t": np.array([]),  
            "r_t": np.zeros((1,EMBED_SIZE)), 
            "h_t": np.zeros((1, rnn_hid)),   ### RNN initial state is ? 
            "oprime_t": None, 
            "o_t": None, 
            "d_t": None, 
            "u_t": None, 
            "v_t": None
        }
        # self.CACHE[self.TS] = {
                                # "h_prev":np.zeros((1,n_hid)),
                                # "rt_prev":
        
        # (

                        # np.zeros((1, n_hid)), # h_t-1
                        # np.zeros((1, EMBED_SIZE)), # r_t-1
                        # (
                            # np.zeros((0, EMBED_SIZE)), # V_t-1
                            # np.array([]) # s_t-1
                        # )
                    # )

    def step_forward(self, H_prev, seq_input):
        """
        cnt_input is the tuple as defined in the paper.
        seq_input is a (1, EMBED_SIZE) vector.
        """

        rnn_h_prev, stack_r_prev, V_prev, s_prev = H_prev

        rnn_i_curr = np.hstack((seq_input, stack_r_prev))
        rnn_h_curr, rnn_o_curr = self.rnn.step_forward(xin=rnn_i_curr, hp=rnn_h_prev)

        d_t = self.sigd_dt.forward(self.dense_dt.forward(rnn_o_curr)).squeeze(),
        u_t = self.sigd_ut.forward(self.dense_ut.forward(rnn_o_curr)).squeeze(),
        v_t = self.tanh_vt.forward(self.dense_vt.forward(rnn_o_curr))
        o_t = self.tanh_ot.forward(self.dense_ot.forward(rnn_o_curr))

        V_curr, s_curr, r_curr = self.nstack.forward(
                V_prev=V_prev, 
                s_prev=s_prev,
                d_t = d_t,
                u_t = u_t,
                v_t = v_t
        )

        return (rnn_h_curr, r_curr, V_curr, s_curr, d_t, u_t, v_t, o_t, rnn_o_curr)

    def forward(self, cnt_inputs):
        """
        Wrapper over step_forward 
        """ 
        assert cnt_inputs.ndim == 2, "Input to the controller should be (total timesteps, input_dim) dim matrix" 
        assert cnt_inputs.shape[1] == self.controller_in, "Incorrect shape for input to the controller" 

        total_timesteps = np.shape(cnt_inputs)[0] 
        print("Total timesteps: {}".format(total_timesteps)) 

        for loop_idx in range(total_timesteps):

            self.TS += 1 

            H_prev = ( self.CACHE[self.TS-1]["h_t"], self.CACHE[self.TS-1]["r_t"], self.CACHE[self.TS-1]["V_t"], self.CACHE[self.TS-1]["s_t"] ) 
            ops = self.step_forward(H_prev, cnt_inputs[loop_idx].reshape(1,-1)) 

            # unpack ops and store in cache 
            self.CACHE[self.TS] = {} 
            self.CACHE[self.TS]["h_t"], self.CACHE[self.TS]["r_t"], self.CACHE[self.TS]["V_t"] = ops[:3] 
            self.CACHE[self.TS]["s_t"], self.CACHE[self.TS]["d_t"], self.CACHE[self.TS]["u_t"] = ops[3:6]
            self.CACHE[self.TS]["v_t"], self.CACHE[self.TS]["o_t"], self.CACHE[self.TS]["oprime_t"] = ops[6:9]
            assert len(self.CACHE[self.TS]) == 9, "We are storing 9 items"

            print("finished: {}".format(self.TS))


    # def step_backward(
        # self, del_o_t, del_r_curr,
        # s_prev, d_t, u_t, V_t, s_t
        # ):
        # """
        # Expects error w.r.t. o_t for RNN.backward.
        # Expects error w.r.t. r_curr for NS.backward.

        # The 2nd row of inputs is for the stack.backward
        # ** o_t and r_curr are defined in self.step_forward **

        # For the last timestep(t==n), the error signal will only come from \
        # the output o_t since at t==n, error for r_t has not been computed.

        # For all other timesteps, del_r_curr will come from error w.r.t. the \
        # RNN's inputs ==> concat(i_t, r_t`)
        # """

        # del_O = np.zeros((1, self.rnn.n_out))
        # del_O += self.dense_ot(self.tanh_ot(del_o_t))

        # _, _, del_d_t, del_u_t, del_v_t = self.nstack.backward(del_r_curr, s_prev, d_t, u_t, V_t, s_t)

        # del_O += self.dense_dt(self.sigd_dt(del_d_t))
        # del_O += self.dense_ut(self.sigd_ut(del_u_t))
        # del_O += self.dense_vt(self.tanh_vt(del_v_t))


#### Testing.
import ipdb; ipdb.set_trace()

CTRL = Controller(controller_in=2, rnn_hid=3, rnn_out=5, LENGTH=10, EMBED_SIZE=2)
sequence_input = np.array([
    [0.1, 0], 
    [0.2, 0], 
    [0.3, 0],
    [0.4, 0],
])

CTRL.forward(sequence_input) 

# next_state, cnt_output = CTRL.step_forward(cnt_input, sequence_input[0])

# assert np.array_equal(V_t, next_state[2][0])
# assert np.array_equal(s_t, next_state[2][1])
# assert np.array_equal(r_t, next_state[1])
# print("PASSED")
