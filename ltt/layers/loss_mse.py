from __future__ import absolute_import, print_function
from .abstract_layer import AbstractLayer
import numpy as np

class MSE(AbstractLayer):
    """MSE loss layer"""

    def __init__(self, name):
        self.name = name

    def forward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "preds, target shape does not match"
        self.loss = 0.5 * np.mean(
                np.sum(np.square(y_pred - y_true), axis=1)
        )
        return self.loss

    def backward(self, y_pred, y_true):
        assert y_pred.shape == y_true.shape, "preds, target shape does not match"
        self.output_grad = (y_pred - y_true)/y_pred.shape[0]
        self.input_grad = self.output_grad ## since its the loss layer
        return self.input_grad

    def return_weights(self):
        return None

    def set_weights(self):
        return None

    def return_grads(self):
        return None

    def weights_iter(self):
        return np.nditer(np.array([]), op_flags=["readwrite"], flags=["zerosize_ok"])

    def grads_iter(self):
        return np.nditer(np.array([]), op_flags=["readonly"], flags=["zerosize_ok"])


def mse_test():
    mse = MSE("loss1")
    y_pred = np.zeros((3, 4))
    y_true = np.random.rand(3, 4)

    print(mse.forward(y_pred=y_pred, y_true=y_true))
    print(mse.backward(y_pred, y_true))
    print("PASSED")
