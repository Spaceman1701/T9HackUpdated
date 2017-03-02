from neural_network.network import Network
from loader.data_loader import load_data_wrapper


def check_number(output, expected):
    o_value = 0
    for index, value in enumerate(output):
        if value > output[o_value]:
            o_value = index
    e_value = 0
    for index, value in enumerate(expected):
        if value > expected[e_value]:
            e_value = index
    return o_value == e_value

n = Network([784, 30, 10])


mnist_data = load_data_wrapper()


training_data = mnist_data[0]

print("starting training")
print(training_data)

data_list = []
for inputs, results in training_data:
    fixed_results = []
    for r in results:
        fixed_results.append(r[0])
    assert len(fixed_results) == 10
    data_list.append((inputs, fixed_results))

suc = 0
for d in data_list[:2000]:
    if check_number(n.feed_forward(d[0]).as_vertical_list()[0], d[1]):
        suc += 1

print(suc / 2000)

n.train(data_list[:20000], 3, 30, 15, max_samples=10, test_data=data_list[20000:21000])
print("finished training")

suc = 0
for d in data_list[:2000]:
    if check_number(n.feed_forward(d[0]).as_vertical_list()[0], d[1]):
        suc += 1

print(suc / 2000)

