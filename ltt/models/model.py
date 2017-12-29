from __future__ import absolute_import
from ..layers import AbstractLayer

class Model(object):
    """
    A simple sequential model.
    self.sequence stores the order in which ops were added.
    self.layers stores the layers against names.
    """

    def __init__(self, name):
        self.name = name
        self.sequence = []
        self.layers = {}

    def add(self, layer):
        assert isinstance(layer, AbstractLayer), "object is not AbstractLayer object"
        assert layer.name not in self.layers, "layer name already in model"

        self.layers[layer.name] = layer
        self.sequence.append(layer.name)
        return

def model_test():
    from ..layers import Dense
    d1 = Dense(n_in=5, n_out=6, name="d1")
    model = Model(name="m1")

    model.add(d1)
    print model.sequence
    print model.layers

    # model.add(d1) ## should fail.
    print "PASSED"
