import csv
import pandas as pd
from neural_network import NeuralNetwork

EPOCHS_NUMBER = 3

def main():
    network = NeuralNetwork(inputs=3, layers=[3, 8, 8, 4], activation_function='relu')

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
                        normalize(float(row[0]), 'a', min_values, max_values),
                        normalize(float(row[1]), 'b', min_values, max_values),
                        normalize(float(row[2]), 'c', min_values, max_values),
                    ],
                    [
                        normalize(float(row[3]), 'x1', min_values, max_values),
                        normalize(float(row[4]), 'x2', min_values, max_values),
                        normalize(float(row[5]), 'x3', min_values, max_values),
                        normalize(float(row[6]), 'x4', min_values, max_values),
                    ]
                ) / row_count
        if epoch % 1 == 0:
            print(mse)

    network.print_weights()

    print('Solve equation:')
    roots = network.run([
        normalize(1, 'a', min_values, max_values),
        normalize(2, 'b', min_values, max_values),
        normalize(3, 'c', min_values, max_values)
    ])
    print(denormalize(roots[0], 'x1', min_values, max_values))
    print(denormalize(roots[1], 'x2', min_values, max_values))
    print(denormalize(roots[2], 'x3', min_values, max_values))
    print(denormalize(roots[3], 'x4', min_values, max_values))

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

'''
xavier init
play with learning rate
play with smaller datasets
'''