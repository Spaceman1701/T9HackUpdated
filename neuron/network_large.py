import math
import random


class Network:
    def __init__(self, sizes):
        self.sizes = sizes
        self.nodes = []
        prev = None
        for size in sizes:
            layer = [Node(prev) for _ in range(size)]
            self.nodes.append(layer)
            prev = layer

    def feed_forward(self, inputs):
        for value, node in zip(inputs, self.nodes[0]):  # input layer
            node.value = value
        for layer in self.nodes:
            for node in layer:
                node.eval()
        return [n.value for n in self.nodes[-1]]

    def clear_network(self):
        for node in self.all_nodes():
            node.value = None
            node.error = None

    def __str__(self):
        output = "Network"
        for layer in self.nodes:
            output += '\n'
            for node in layer:
                output += '['
                for e in node.ins:
                    output += str(e) + ', '
                output += '], '

        return output

    def all_nodes(self):
        for layer in self.nodes:
            for node in layer:
                yield node

    def back_propigate(self, inputs, outputs, learning_rate):
        self.feed_forward(inputs)
        assert len(self.nodes[-1]) == len(outputs)
        for layer in reversed(self.nodes):
            for node, value in zip(layer, outputs):
                node.calc_error(value[0])
        for node in self.nodes[1]:
            node.update_weights(learning_rate, 1)

        self.clear_network()

    def back_propigate_batch(self, batch, learning_rate, sample_size=1):
        for inputs, outputs in batch:
            self.feed_forward(inputs)
            assert len(self.nodes[-1]) == len(outputs)
            for layer in reversed(self.nodes):
                for node, value in zip(layer, outputs):
                    node.calc_error(value[0])
        for node in self.nodes[1]:
            node.update_weights(learning_rate, sample_size)

        self.clear_network()

    def train(self, training_data, epochs, batch_sides, learning_rate):
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            batches = [training_data[k:k + batch_sides] for k in range(0, n, batch_sides)]
            for batch in batches:
                self.back_propigate_batch(batch, learning_rate, batch_sides)
            print("Epoch \{0} finished.", j)


class Edge:
    def __init__(self, origin, target, weight):
        self.weight = weight
        self.origin = origin
        self.target = target

    def __str__(self):
        return str(self.weight)

class BiasEdge(Edge):
    def __str__(self):
        self.origin =

class BiasNode(Node):

class Node:
    def __init__(self, prev):
        self.outs = []
        self.ins = []
        self.bias = random.uniform(0.0, 1.0)
        self.value = None
        self.error = None
        if prev:
            for node in prev:
                e = Edge(node, self, random.uniform(0.0, 1.0))
                node.outs.append(e)
                self.ins.append(e)

    def eval(self):
        activation = sum([weight * input_value for weight, input_value in zip_edge_value(self.ins)]) + self.bias
        activation = 1.0 / (1.0 + math.exp(-activation))
        self.value = activation
        return activation

    def calc_error(self, expected):
        if self.error:
            return self.error
        if not self.outs:
            self.error = expected - self.value
        else:
            z = sum([e.weight * e.target.value for e in self.ins]) + self.bias
            delta = (expected - self.value) * d_eval_func(z)
            self.error = sum([e.weight * e.target.calc_error(expected) for e in self.outs]) + self.bias
        return self.error

    def update_weights(self, learning_rate, sample_size):
        for e in self.ins:
            assert self.value is not None
            assert self.error is not None
            assert e.origin.value is not None
            e.weight += (learning_rate * d_eval_func(self.value) * self.error * e.origin.value) / sample_size
        self.bias -= (self.calc_error(None) * d_eval_func(self.value) * learning_rate) / sample_size
        for out in self.outs:
            out.target.update_weights(learning_rate, sample_size)


def zip_edge_value(inputs):
    for edge in inputs:
        yield edge.weight, edge.origin.value


def d_eval_func(output):
    return output * (1 - output)


if __name__ == '__main__':
    in_data = [[1, 2] * 100] * 1000
    out_data = [[[0.2], [0.4], [0.6]]] * 1000
    data = []
    for v_i, v_o in zip(in_data, out_data):
        data.append((v_i, v_o))

    n = Network([200, 15, 3])
    print(n.feed_forward([1, 2] * 100))
    n.train(data, 50, 10, 0.9)
    res = n.feed_forward([1, 2] * 100)
    print(res)
    if any([x > 1.1 for x in res]):
        print("bad mapping")