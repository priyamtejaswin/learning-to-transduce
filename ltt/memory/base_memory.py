import abc

class BaseMemory(object):
    """Abstract class for nerual memory structures."""

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def forward(self, x):
        return

    @abc.abstractmethod
    def backward(self):
        return

if __name__ == '__main__':
    pass
