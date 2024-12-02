import json
import argparse
from neural_network import NeuralNetwork


def main():
    parser = argparse.ArgumentParser(description='Process two integers')
    parser.add_argument('b', type=int, help='coefficient b')
    parser.add_argument('c', type=int, help='coefficient c')
    args = parser.parse_args()
    b = args.b
    c = args.c

    with open('network.json', 'r') as file:
        data = json.load(file)

    metadata = data['metadata']
    min_values = metadata['min_values']
    max_values = metadata['max_values']
    inputs = metadata['inputs']
    layers = metadata['layers']
    activation_function = metadata['activation_function']
    weights = data['weights']

    network = NeuralNetwork(inputs=inputs, layers=layers, activation_function=activation_function)

    for i, layer in enumerate(network.network):
        for j, perceptron in enumerate(layer):
            perceptron.weights = weights[i][j]

    network.print_weights()

    print('Solve equation:')
    roots = network.run([
        normalize(b, 'b', min_values, max_values),
        normalize(c, 'c', min_values, max_values)
    ])
    print(denormalize(roots[0], 'x1', min_values, max_values))
    print(denormalize(roots[1], 'x2', min_values, max_values))

def normalize(value, label, mins, maxes):
    min_value = mins[label]
    max_value = maxes[label]
    return (value - min_value)/(max_value - min_value)

def denormalize(value, label, mins, maxes):
    min_value = mins[label]
    max_value = maxes[label]
    return (value * (max_value - min_value)) + min_value

if __name__ == '__main__':
    main()
