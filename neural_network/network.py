from neural_network.layer import Layer
import numpy as np
import copy


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = np.array(([([([np.random.uniform() for weight in range(sizes[layer])])  # weights[layer][node_fom][node_to]
                                for node in range(size)])
                                 for layer, size in enumerate(sizes[1:])]))
        self.biases = np.array([[np.random.uniform() for bias in range(size)] for size in sizes])  # bias[layer][node]

    def create_weights_array(self):
        return np.array([[[0 for weight in range(self.sizes[layer])]  # weights[layer][node_fom][node_to]
                                for node in range(size)]
                                 for layer, size in enumerate(self.sizes[1:])])

    def create_bias_aray(self):
        return np.array([[0 for bias in range(size)] for size in self.sizes])

    def calc_activation(self, layer, prev_activation):
        layer_weights = self.weights[layer]
        output = []
        for node in layer_weights:
            output.append(np.dot(node, prev_activation))
        return np.array(output) + self.biases[layer + 1]

    def feed_forward(self, inputs):
        for layer in range(self.num_layers - 1):
            inputs = calc_sigmoid(self.calc_activation(layer, inputs))
        return inputs

    def dot_activation_weight(self, layer_weights, a):  # dot each set of weights with the activation to produce a new list
        output = []
        for node in layer_weights:
            output.append(np.dot(node, a))
        return np.array(output)

    def backpropigate(self, inputs, outputs):
        weight_offset = self.create_weights_array()
        bias_offset = self.create_bias_aray()

        activations = [self.calc_activation(0, inputs)]
        activation_sigmoid = [calc_sigmoid(activations[-1])]
        d_activation = [calc_d_sigmoid(activation_sigmoid[-1])]

        for layer in range(1, self.num_layers - 1):
            activations.append(self.calc_activation(layer, activation_sigmoid[-1]))
            activation_sigmoid.append(calc_sigmoid(activations[-1]))
            d_activation.append(calc_d_sigmoid(activation_sigmoid[-1]))

        activations = np.array(activations)
        activation_sigmoid = np.array(activation_sigmoid)
        d_activation = np.array(d_activation)

        error = calc_error(activation_sigmoid[-1], outputs) * calc_d_sigmoid(activations[-1])
        print(len(activation_sigmoid[-1]))

        weight_offset[-1] = self.dot_activation_weight(error, activation_sigmoid[-1])
        bias_offset[-1] = error

        #  TODO: fix inconsistent numpy-ing and implement backpropigation for layers that aren't the last one

        return weight_offset, bias_offset

    def transpose_dot_weight(self, a, b):
        output = []
        for node in a:
            output.append(np.dot(a, b))
        return np.array(output)

    def train(self, inputs, outputs, rate):
        tempw = copy.deepcopy(self.weights)
        result = self.backpropigate(inputs, outputs)
        print(result[0])
        for i in range(len(result[0])):
            for j in range(len(result[0][i])):
                for k in range(len(result[0][i][j])):
                    self.weights[i][j][k] -= result[0][i][j][k] * rate

        for i in range(len(result[1])):
            for j in range(len(result[1][i])):
                self.biases[i][j] -= result[1][i][j] * rate


    def print_weights(self):
        print("Network Weights:")

        for layer_num, layer in enumerate(self.weights):
            print("layer {0}: ".format(layer_num), end="")
            print(layer)


def calc_error(activation, expected):
    return np.subtract(activation, expected);


def calc_sigmoid(activation):
    print(activation)
    return 1.0 / (1.0 + np.exp(-activation))


def calc_d_sigmoid(value):
    return value * (1.0 - value)







