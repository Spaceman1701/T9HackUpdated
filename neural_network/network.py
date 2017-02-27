from neural_network.layer import Layer
import numpy as np

class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = np.array([[[np.random.uniform() for weight in range(sizes[layer + 1])]  # weights[layer][node_fom][node_to]
                                for node in range(size)]
                                 for layer, size in enumerate(sizes[:-1])])
        self.biases = np.array([[np.random.uniform() for bias in range(size)] for size in sizes])  # bias[layer][node]

    def calc_activation(self, layer, prev_activation):
        layer_weights = self.weights[layer]
        print(len(layer_weights))
        output = []
        for node in layer_weights:

            output.append(np.dot(node, prev_activation))
        print(len(output))
        return output + self.biases

    def calc_sigmoid(self, activation):
        return 1.0 / (1.0 + np.exp(-activation))

    def feed_forward(self, inputs):
        for layer in range(self.num_layers):
            inputs = self.calc_sigmoid(self.calc_activation(layer, inputs))


#l = [3, 2, 1]
#print(l[:-1])
n = Network([1, 3, 1])
n.feed_forward([1])




