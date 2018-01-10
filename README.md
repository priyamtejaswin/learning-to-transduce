# Learning to Transduce ... using a modular, extensible framework that we created.
*- by Priyam Tejaswin and Akshay Chawla*

## Introduction
Writing your own framework is every ML programmer's dream. But don't just take our word for it! One need not look beyond the countless libraries in every language imaginable from Crystal to Ruby. Why not add another framework which no one will ever hear of to the mix?

All kidding aside, developing a simple ML framework which can be used for different projects from scratch really is one of our goals. We believe it is the ultimate test of theory, design and programming skills. This repository tracks our attempts towards building an extensible, modular ML framework. As a poc, we chose to implement the Learning to Transduce Deepmind paper using this framework. This is a work in progress. Scroll down to the ends to see which features are pending.

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
