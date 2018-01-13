from __future__ import absolute_import
from ..layers import AbstractLayer
import numpy as np

class Model(object):
    """
    A simple sequential model.
    self.sequence stores the order in which ops were added.
    self.layers stores the layers against names.

    Forward pass and loss are separate.
    def:forward will just return the prediction.
    """

    def __init__(self, name, loss_layer=None, optimizer=None):
        if loss_layer is not None:
            assert isinstance(loss_layer, AbstractLayer), "loss is not AbstractLayer object"

        self.optimizer = optimizer
        self.name = name
        self.loss_layer = loss_layer
        self.sequence = []
        self.layers = {}

    def add(self, layer):
        assert isinstance(layer, AbstractLayer), "object is not AbstractLayer object"
        assert layer.name not in self.layers, "layer name already in model"

        # if len(self.sequence) == 0:
            # self.feature_size = layer.weights.shape[0]

        self.layers[layer.name] = layer
        self.sequence.append(layer.name)
        return

    def do_forward(self, x):
        self.batch_size = x.shape[0] * 1.0

        mlimit = len(self.layers) - 1
        for ix, lname in enumerate(self.sequence):
            layer = self.layers[lname]
            y = layer.forward(x)
            if ix==mlimit:
                break
            x = y

        self.output = y
        return self.output

    def do_loss(self, target):
        assert target.shape == self.output.shape, "output and target shapes do not match"
        self.loss = self.loss_layer.forward(self.output, target)
        self.loss_grad = self.loss_layer.backward(self.output, target)
        return self.loss

    def do_backward(self):
        del_error = self.loss_grad
        for ix, lname in list(enumerate(self.sequence))[::-1]:
            del_error = self.layers[lname].backward(del_error)

        return

    def do_update(self):
        self.optimizer.update(self)
        pass


def model_test():
    from ..layers import Dense, MSE
    from ..optimizers import SGD

    d1 = Dense(n_in=3, n_out=4, name="d1")
    l1 = MSE("loss1")
    sgd = SGD()

    model = Model(name="m1", loss_layer=l1, optimizer=sgd)
    model.add(d1)

    print model.sequence
    print model.layers

    x = np.random.rand(2, 3)
    t = np.ones((2, 4))

    m_output = model.do_forward(x)
    m_loss = model.do_loss(target=t)
    model.do_backward()
    model.do_update()

    print "PASSED"
