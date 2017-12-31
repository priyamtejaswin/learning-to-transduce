from __future__ import absolute_import
from .base_memory import BaseMemory
from ..utils import array_init

class NeuralStack(BaseMemory):
    """A neural stack implementation"""

    def __init__(self, name="nstack"):
        self.name = name

    def forward(self, x):
        return

    def backward(self):
        return


def stack_test():
    ns = NeuralStack()
    print ns.name
    print "PASSED"

if __name__ == '__main__':
    pass
