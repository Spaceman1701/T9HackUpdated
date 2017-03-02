from neural_network.matrix import Matrix
import neural_network.matrix as matrix
import random


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

    def backpropagate(self, inputs, outputs):
        weights_offset = [Matrix(next_layer, current_layer).set_zero()
                        for next_layer, current_layer in zip(self.sizes[1:], self.sizes[:-1])]
        biases_offset = [Matrix(size, 1).set_zero() for size in self.sizes[1:]]

        inputs_mat = Matrix.from_list(inputs)
        outputs_mat = Matrix.from_list(outputs)

        activation_transfers = [inputs_mat]
        d_activations = [calc_d_sigmoid(inputs_mat)]

        for layer in range(self.num_layers - 1):
            activation = self.calc_activation(layer, activation_transfers[-1])
            activation_transfers.append(calc_sigmoid(activation))
            d_activations.append(calc_d_sigmoid(activation_transfers[-1]))  # pycharm complains about this, but it's not wrong

        error = calc_error(activation_transfers[-1], outputs_mat).entrywise_product(d_activations[-1])
        weights_offset[-1] = error * d_activations[-2].transpose()
        biases_offset[-1] = error

        for layer in range(self.num_layers - 2, 1, -1):
            d_activation = d_activations[layer]
            error = (self.weights[layer + 1].transpose() * error).entrywise_product(d_activation)
            w_off = error * d_activations[layer - 1].transpose()
            weights_offset[layer] = w_off
            biases_offset[layer] = error
        return weights_offset, biases_offset

    def train(self, data, learning_rate, epochs, sample_size):
        if sample_size < 1:
            raise ValueError("sample size must be positive")
        for e in range(epochs):
            random.shuffle(data)
            samples = [data[i:i + sample_size] for i in range(epochs - sample_size)]
            for sample in samples:
                for data_item in sample:
                    weight_delta, bias_delta = self.backpropigate(data_item[0], data_item[1])

                self.weights = [weight - (delta * learning_rate / sample_size)
                               for weight, delta in zip(self.weights, weight_delta)]
                self.biases = [bias - (delta * learning_rate / sample_size)
                               for bias, delta in zip(self.biases, bias_delta)]


def calc_error(activation, expected):
    result = activation - expected
    assert type(result) == Matrix
    return result


def calc_sigmoid(activation):
    assert type(activation) == Matrix
    result = 1.0 / (1.0 + matrix.exp(-activation))
    assert type(result) == Matrix
    return result


def calc_d_sigmoid(value):
    assert type(value) == Matrix
    result = value.entrywise_product(1.0 - value)
    assert type(result) == Matrix
    return result










