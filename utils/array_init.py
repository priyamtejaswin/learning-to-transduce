import numpy as np

def array_init(shape, vtype="zeros", seed=None):
    if seed is not None:
        np.random.seed(seed)

    if type(shape) not in (int, tuple, list):
        raise AttributeError("shape has to be int, tuple or list. Input:", str(type(shape)))
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
            raise AttributeError("vtype is not ones, zeros or rand. Input:", str(vtype))

if __name__ == '__main__':
    pass
