from neural_network.matrix import Matrix
import neural_network.matrix as matrix
import math


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.num_layers = len(sizes)
        self.weights = [Matrix(next_layer, current_layer).set_randomize()
                        for next_layer, current_layer in zip(sizes[1:], sizes[:-1])]
        #  weights[layer][origin][target]
        self.biases = [Matrix(size, 1).set_randomize() for size in sizes[1:]]
        #  biases[layer][node]

    def calc_activation(self, layer, inputs):
        result = (self.weights[layer] * inputs) + self.biases[layer]
        return result

    def feed_forward(self, inputs):
        inputs_mat = Matrix.from_list(inputs)
        for layer in range(self.num_layers - 1):
            inputs_mat = calc_sigmoid(self.calc_activation(layer, inputs_mat))
        return inputs_mat

    def backpropigate(self, inputs, outputs):
        weights_offset = [Matrix(next_layer, current_layer).set_zero()
                        for next_layer, current_layer in zip(self.sizes[1:], self.sizes[:-1])]
        biases_offset = [Matrix(size, 1).set_zero() for size in self.sizes[1:]]

        inputs_mat = Matrix.from_list(inputs)
        outputs_mat = Matrix.from_list(outputs)

        activations = [inputs_mat]
        activations_sigmoid = [calc_sigmoid(inputs_mat)]
        d_activation = [calc_d_sigmoid(inputs_mat)]

        for layer in range(1, self.num_layers):
            activations.append(self.calc_activation(layer, activations_sigmoid[-1]))
            activations_sigmoid.append(calc_sigmoid(activations[-1]))
            d_activation.append(calc_d_sigmoid(activations_sigmoid[-1]))

        error = calc_error(activations_sigmoid[-1], outputs_mat).entrywise_product(d_activation[-1])
        print(type(error))
        print(type(d_activation[-2]))
        weights_offset[-1] = error.transpose() * d_activation[-2]
        biases_offset[-1] = error




def calc_error(activation, expected):
    return activation - expected


def calc_sigmoid(activation):
    assert type(activation) == Matrix
    return 1.0 / (1.0 + matrix.exp(-activation))


def calc_d_sigmoid(value):
    onem = 1.0 - value
    return value.entrywise_product(onem)


n = Network([4, 3, 2])

#print(n.feed_forward([1, 1, 1, 1]))

print()

n.backpropigate([1, 1, 1, 1], [0, 0])









