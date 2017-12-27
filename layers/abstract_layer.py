import abc

class AbstractLayer(object):
    """Abstract class for layers."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def backward(self, current_error):
        return

    @abc.abstractmethod
    def return_weights(self):
        return
