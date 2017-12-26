from abc import ABCMeta, abstractmethod

class AbstractLayer(object):
    """Abstract class for layers."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def forward(self, x):
        return

    @abstractmethod
    def backward(self, current_error):
        return
