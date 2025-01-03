import csv
import pandas as pd
from neural_network import NeuralNetwork
import json

EPOCHS_NUMBER = 5

def main():
    with open('dataset.csv', mode='r') as file:
        df = pd.read_csv(file)

    row_count = len(df)
    #TODO: define mins and maxes for every row
    min_values = df.min()
    max_values = df.max()

    header = df.columns.tolist()
    input_labels = [col for col in header if col.startswith('i')]
    output_labels = [col for col in header if col.startswith('o')]

    layers = [10, len(output_labels)]
    activation_function = 'relu'
    network = NeuralNetwork(inputs=len(input_labels), layers=layers, activation_function=activation_function)

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
                        normalize(float(row[0]), 'i1', min_values, max_values),
                        normalize(float(row[1]), 'i2', min_values, max_values),
                    ],
                    [
                        normalize(float(row[2]), 'o1', min_values, max_values),
                        normalize(float(row[3]), 'o2', min_values, max_values),
                    ]
                ) / row_count
        if epoch % 1 == 0:
            print('Epoch #' + str(epoch) + ' MSE=' + str(mse))

    network.print_weights()

    data = {
        "metadata": {
            "input_labels": input_labels, "output_labels": output_labels,"layers": layers,
            "activation_function": activation_function, "min_values": min_values.to_dict(),
            "max_values": max_values.to_dict()}, "weights": build_weight_matrix(network.network),
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
