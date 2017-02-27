from neural_network.layer import Layer
import numpy as np

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = np.array([[[np.random.uniform() for weight in range(sizes[layer])]  # weights[layer][node_fom][node_to]
                                for node in range(size)]
                                 for layer, size in enumerate(sizes[1:])])
        self.biases = np.array([[np.random.uniform() for bias in range(size)] for size in sizes])  # bias[layer][node]

        ##print(self.weights)

    def calc_activation(self, layer, prev_activation):
        layer_weights = self.weights[layer]
        output = []
        for node in layer_weights:
            output.append(np.dot(node, prev_activation))
        print("output activation: {0}, {1}, bias = {2}".format(len(output), output, self.biases))
        return output + self.biases

    def calc_sigmoid(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def feed_forward(self, inputs):
        for layer in range(self.num_layers):
            inputs = self.calc_sigmoid(self.calc_activation(layer, inputs))

    def print_weights(self):
        print("Network Weights:")
        for layer_num, layer in enumerate(self.weights):
            print("layer {0}: ".format(layer_num), end="")
            print(layer)


n = Network([3, 1, 2])
n.feed_forward([1, 1, 1])




