# Learning to Transduce ... using a modular, extensible framework that we created.
*- by Priyam Tejaswin and Akshay Chawla*

## Introduction
Writing your own framework is every ML programmer's dream. But don't just take our word for it! One need not look beyond the countless libraries in every language imaginable from Crystal to Ruby. Why not add another framework which no one will ever hear of to the mix?

All kidding aside, developing a simple ML framework which can be used for different projects from scratch really is one of our goals. We believe it is the ultimate test of theory, design and programming skills. Inspired from [Keras](https://github.com/fchollet/keras/), repository tracks our attempts towards building an extensible, modular ML framework. As a poc, we chose to implement the [Learning to Transduce Deepmind paper](https://pdfs.semanticscholar.org/b6e5/7009cd5a6ce0825b3dd1fb1fa535e52c4f3d.pdf) using this framework. 

**NOTE**: *This is a work in progress. Scroll down to the ends to see which features are pending.*

## Codebase
**master** is the core deployment branch. The code is organised as follows.
```
learning-to-transduce/
├── README.md
├── grad_check.py ## Script for numerical gradient checking.
├── ltt
│   ├── __init__.py
│   ├── layers ## Various layers to build models.
│   │   ├── README.md
│   │   ├── __init__.py
│   │   ├── abstract_layer.py ## AbstractBaseClass for a layer object.
│   │   ├── dense.py
│   │   ├── loss_mse.py
│   │   ├── relu.py
│   │   ├── rnn.py
│   │   ├── sigmoid.py
│   │   ├── tanh.py
│   ├── memory ## External memory objects to augment a model.
│   │   ├── __init__.py
│   │   ├── base_memory.py ## AbstractBaseClass for a memory object.
│   │   ├── neural_stack.py ## A neural stack object.
│   ├── models ## Different types of model(API) implementations.
│   │   ├── __init__.py
│   │   ├── model.py ## Basic sequential model.
│   ├── optimizers ## Optimizers.
│   │   ├── __init__.py
│   │   ├── opt_sgd.py ## SGD optimizer. Accepts a MODEL object.
│   └── utils ## Utility classes and functions.
│       ├── __init__.py
│       ├── initializers.py ## Initialize numpy arrays consistently.
├── mnist_test.py ## Small example of using our framework's API.
├── my_tests.py ## Ignored testig file.
├── not-so-simple-rnn.py ## A crude RNN for binary addition. Reference.
└── simple-rnn.py ## Simple implementation for reference.
```

## Basic usage
Lets walk through a simple example and build a multi-layer perceptron using a sequential model API.

1) Begin by importing libraries to load, pre-process datasets and visualise results.
```
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits 
import sklearn.preprocessing 
from sklearn.utils import shuffle
```

2) Import the required layer objects and the model object from `ltt`. 
```
from ltt.models import Model 
from ltt.layers import Dense, Tanh, Sigmoid, ReLU, MSE
from ltt.optimizers import SGD
```

3) Load and pre-process the mnist data as dense numpy array. Our framework does not support sparse objects.
```
    data, target = load_digits(return_X_y=True)
    data, target = shuffle(data, target)
    target = target.reshape(len(target), 1)
    enc = sklearn.preprocessing.OneHotEncoder() 
    enc.fit(target)
    target = enc.transform(target).toarray()
    data = data / 16.0
```

4. Let's create a model. Here's the constructor for a `Model` object. 
```
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
```
A new model object expects a loss layer and an optimizer and with them create a model.
```
loss = MSE("mse_loss")
sgd_optimizer = SGD(alpha=0.1)
model = Model(name="mnist_test", 
				loss_layer=loss, 
				optimizer=sgd_optimizer) 
```

5. The architecture we'll follow is `INPUT[64] -> HIDDEN1[32] -> SIGMOID -> HIDDEN[10] -> SIGMOID -> LOSS.` Let's create and add the required layer objects. 
```
model.add(Dense(n_in=64, n_out=32, name="dense1"))
model.add(Sigmoid(name="act1"))
model.add(Dense(n_in=32, n_out=10, name="dense2"))
model.add(Sigmoid(name="act2"))
```
The `.add` method of `model` will take a `Layer` object append it sequentially for execution.

6. Start training with data.
```
for epoch in range(500):
    print("Epoch: {}".format(epoch))
    epoch_loss = [] 
    for start_idx in range(0, len(data), 25):

        ## batching
        end_idx = min(len(data), start_idx + 25)
        batch_x = data[start_idx:end_idx, :] 
        batch_y = target[start_idx:end_idx, :]

        ## forward --> loss --> backward
        model.do_forward(batch_x) 
        batch_loss = model.do_loss(batch_y)
        model.do_backward() 
        model.do_update() 

        epoch_loss.append(batch_loss)

    print("Loss: {}".format(sum(epoch_loss)/len(epoch_loss)))
```
7. Get predictions and accuracy
```
data_test, target_test = data[:200], target[:200] 
y_preds = model.do_forward(data_test) 
target_test = np.argmax(target_test, axis=1)
y_preds     = np.argmax(y_preds, axis=1) 
print((y_preds==target_test).mean())
```

## How it works

Every layer object is required to have defined the following methods.
```
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

    @abc.abstractmethod
    def return_grads(self):
        return

    @abc.abstractmethod
    def weights_iter(self):
        return

    @abc.abstractmethod
    def grads_iter(self):
        return

    @abc.abstractmethod
    def set_weights(self):
        return
```
After declaring a model, `model.do_forward(x)` runs a forward pass with `x` as input. Since this is a sequential model, it will call the `.forward(x)` method for every layer in the order of addition. Here's the code for `model.do_forward(x)`
```
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
```

You are then required to explicitly calculate the loss since `model.do_forward(x)` will only update the model with the output of the final layer BEFORE the `loss_layer`. With the loss calculated, you can run `model.do_backward()` to start the backward pass. Again, the model will call every layer's `.backward(current_error)` in reverse order. Concretely
```
def do_backward(self):
    del_error = self.loss_grad
    for ix, lname in list(enumerate(self.sequence))[::-1]:
        del_error = self.layers[lname].backward(del_error)

    return
```
This will update the gradient placeholders for every layer. To update the weights using an optimizer, call `model.do_update()`. Internally, that method will call `model.optimizer.update()`. The `SGD` optimizer then updates the model weights using the `layer.set_weights(args)` as follows
```
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

        for lname, layer in model.layers.iteritems():
            weights = layer.return_weights()
            grads = layer.return_grads()

            if weights is None:
                continue

            for w, g in itertools.izip(weights, grads):
                assert w.shape == g.shape, "weights and grads shape do not match during update"
                w -= g * self.alpha

            layer.set_weights(deepcopy(weights))

        self.counter += 1
```

The complete example is in `./mnist_test.py` . Feel free to play around with different activation functions or add your own loss layers. Just follow the base classes and `./ltt/layers/abstract_layer.py` for reference.

## Progress
TODO
