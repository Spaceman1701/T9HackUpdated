from neural_network.matrix import Matrix, Vector
import neural_network.matrix as matrix
import math

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [Matrix(next_layer, current_layer).set_randomize()
                        for next_layer, current_layer in zip(sizes[1:], sizes[:-1])]
        #  weights[layer][origin][target]
        self.biases = [Vector(size).set_randomize() for size in sizes[1:]]
        #  biases[layer][node]

    def calc_activation(self, layer, inputs):
        return self.weights[layer] * inputs + self.biases[layer]

    def feed_forward(self, inputs):
        inputs = Vector.from_list(inputs)
        for layer in range(self.num_layers - 1):
            inputs = calc_sigmoid(self.calc_activation(layer, inputs))
        return inputs


def calc_error(activation, expected):
    return activation - expected


def calc_sigmoid(activation):
    return 1.0 / (1.0 + matrix.exp(-activation))

def calc_d_sigmoid(value):
    return value * (1.0 - value)


n = Network([4, 3, 2])

print(n.feed_forward([1, 1, 1, 1]))








