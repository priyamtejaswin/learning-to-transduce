import numpy as np

def array_init(shape, vtype="rand", seed=1234):
    """
    Init ndarray as ones, zeros or rand.
    If shape is a int, it will return (shape, 1).
    """

    if seed is not False:
        np.random.seed(seed)

    if type(shape) not in (int, tuple, list):
        raise AttributeError("shape has to be int, tuple or list. Input:%s"%str(type(shape)))
    else:
        if isinstance(shape, int):
            shape = (shape, 1)

        if vtype == "zeros":
            return np.zeros(shape)
        elif vtype == "ones":
            return np.ones(shape)
        elif vtype == "rand":
            return np.random.rand(*shape) * 0.1 # Need to expand the shape.
        else:
            raise AttributeError("vtype is not ones, zeros or rand. Input:%s"%str(vtype))

if __name__ == '__main__':
    print array_init(5) # random
    print array_init((5, 3), "ones")
    print array_init([3, 5], "zeros")
    print array_init([3, 3], "mmmm")
