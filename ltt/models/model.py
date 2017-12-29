from __future__ import absolute_import
from ..layers import AbstractLayer

class Model(object):
    """
    A simple sequential model.
    self.sequence stores the order in which ops were added.
    self.layers stores the layers against names.

    Forward pass and loss are separate.
    def:forward will just return the prediction.
    """

    def __init__(self, name, loss_layer):
        self.name = name
        self.loss_layer = loss_layer
        self.sequence = []
        self.layers = {}

    def add(self, layer):
        assert isinstance(layer, AbstractLayer), "object is not AbstractLayer object"
        assert layer.name not in self.layers, "layer name already in model"

        self.layers[layer.name] = layer
        self.sequence.append(layer.name)
        return

    def do_forward(self, x):
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
        self.loss_grad = self.loss_layer.backward(target)
        return self.loss

    def do_backward(self):
        del_error = self.loss_grad
        for ix, lname in reversed(enumerate(self.sequence)):
            del_error = self.layers[lname].backward(del_error)

        return

    def do_update(self):
        """HARDCODED AS sgd FOR NOW!!!"""
        pass


def model_test():
    from ..layers import Dense
    d1 = Dense(n_in=5, n_out=6, name="d1")
    model = Model(name="m1")

    model.add(d1)
    print model.sequence
    print model.layers

    # model.add(d1) ## should fail.
    print "PASSED"
