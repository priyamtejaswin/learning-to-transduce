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
