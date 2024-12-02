import csv
import pandas as pd
from neural_network import NeuralNetwork
import json

EPOCHS_NUMBER = 10

def main():
    inputs = 2
    layers = [10, 2]
    activation_function = 'relu'
    network = NeuralNetwork(inputs=inputs, layers=layers, activation_function=activation_function)

    with open('dataset.csv', mode='r') as file:
        df = pd.read_csv(file)

    row_count = len(df)
    min_values = df.min()
    max_values = df.max()

    for epoch in range(EPOCHS_NUMBER):
        mse = 0.0
        with open('dataset.csv', mode='r') as file:
            reader = csv.reader(file)
            header = next(reader)  # Skip the header row
            row_number = 0
            for row in reader:
                row_number += 1
                mse += network.propagate_back(
                    [
                        normalize(float(row[0]), 'b', min_values, max_values),
                        normalize(float(row[1]), 'c', min_values, max_values),
                    ],
                    [
                        normalize(float(row[2]), 'x1', min_values, max_values),
                        normalize(float(row[3]), 'x2', min_values, max_values),
                    ]
                ) / row_count
        if epoch % 1 == 0:
            print('Epoch #' + str(epoch) + ' MSE=' + str(mse))

    network.print_weights()

    data = {
        "metadata": {"inputs": inputs, "layers": layers,"activation_function": activation_function,
                     "min_values": min_values.to_dict(), "max_values": max_values.to_dict()},
        "weights": build_weight_matrix(network.network),
    }

    # Write the dictionary to a JSON file
    with open('network.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

def build_weight_matrix(network):
    return [[perceptron.weights.tolist() for perceptron in layer] for layer in network]

def normalize(value, label, mins, maxes):
    min_value = mins[label]
    max_value = maxes[label]
    return (value - min_value)/(max_value - min_value)

if __name__ == '__main__':
    main()
