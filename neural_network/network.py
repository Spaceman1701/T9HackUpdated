import numpy as np
from neural_network.matrix import Matrix

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [Matrix(next_layer, current_layer) for next_layer, current_layer in zip(sizes[1:], sizes[:-1])]


def calc_error(activation, expected):
    return np.subtract(activation, expected);


def calc_sigmoid(activation):
    print(activation)
    return 1.0 / (1.0 + np.exp(-activation))


def calc_d_sigmoid(value):
    return value * (1.0 - value)


n = Network([4, 3, 2])

for w in n.weights:
    print(w, end="\n \n")








