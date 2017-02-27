from neural_network.layer import Layer
import numpy as np

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = np.array([[[1 for weight in range(sizes[layer])]  # weights[layer][node_fom][node_to]
                                for node in range(size)]
                                 for layer, size in enumerate(sizes[1:])])
        self.biases = np.array([[0 for bias in range(size)] for size in sizes])  # bias[layer][node]

        #print(self.biases)

    def calc_activation(self, layer, prev_activation):
        layer_weights = self.weights[layer]
        output = []
        for node in layer_weights:
            output.append(np.dot(node, prev_activation))
        print("output activation: {0}, {1}, bias = {2}".format(len(output), output, self.biases[layer + 1]))
        print(len(output))
        return np.array(output) + self.biases[layer + 1]

    def feed_forward(self, inputs):
        for layer in range(self.num_layers - 1):
            inputs = self.calc_sigmoid(self.calc_activation(layer, inputs))
        return inputs

    def print_weights(self):
        print("Network Weights:")
        for layer_num, layer in enumerate(self.weights):
            print("layer {0}: ".format(layer_num), end="")
            print(layer)


def calc_sigmoid(activation):
    return 1.0 / (1.0 + np.exp(-activation))


def calc_d_sigmoid(sigmoid):
    return sigmoid * (1.0 - sigmoid)


n = Network([1, 1, 1])
print(n.feed_forward([1]))
print(calc_sigmoid(np.array([1])))




