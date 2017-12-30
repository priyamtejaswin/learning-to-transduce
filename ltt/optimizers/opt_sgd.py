from __future__ import absolute_import
from ..models import Model
from ..layers import Dense, MSE
import itertools
import numpy as np

class SGD(object):
    """
    Stochastic Gradient Descent optimiser.
    """

    def __init__(self, alpha=0.001, name="opt_sgd"):
        self.alpha = alpha
        self.name = name
        self.counter = 0

    def update(self, model):
        assert isinstance(model, Model)
        import ipdb; ipdb.set_trace()

        for lname, layer in model.layers.iteritems():
            weights = layer.return_weights()
            grads = layer.return_grads()

            if weights is None:
                continue

            new_weights = []
            for w, g in itertools.izip(weights, grads):
                assert w.shape == g.shape, "weights and grads shape do not match during update"
                w -= g * self.alpha
                new_weights.append(w)

            layer.set_weights(new_weights)

        self.counter += 1

def sgd_test():
    mmodel = Model("mmodel", loss_layer=MSE("mse_loss"))
    d1 = Dense(3, 4, "d1")

    x = np.random.rand(2, 3)
    t = np.ones((2, 4))

    mmodel.add(d1)

    mmodel.do_forward(x)
    mmodel.do_loss(t)
    mmodel.do_backward()

    sgd = SGD()

    import ipdb; ipdb.set_trace()
    sgd.update(mmodel)
