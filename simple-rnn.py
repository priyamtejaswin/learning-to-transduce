#!/usr/bin/env python

## catch22 - a simple rnn in numpy

import numpy as np
import sys
import ipdb

RANDOM_SEED = 1234
np.random.seed(RANDOM_SEED)

## data - counting 1's in a sequence
