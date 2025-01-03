import random
import math
import csv

DATASET_SIZE = 100000
MIN_VALUE=-10.0
MAX_VALUE=10.0

def main():
    with open(r'dataset.csv', 'w', newline='') as csvfile:
        csvfile.truncate()
        fieldnames = ['i1', 'i2', 'o1', 'o2']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'i1': 'i1', 'i2': 'i2', 'o1': 'o1', 'o2': 'o2'})

    entries_generated = 0

    while entries_generated < DATASET_SIZE:
        a = 1  # random.uniform(MIN_VALUE, MAX_VALUE)
        b = random.uniform(MIN_VALUE, MAX_VALUE)
        c = random.uniform(MIN_VALUE, MAX_VALUE)
        discriminant = b ** 2 - 4 * a * c
        if discriminant >= 0:
            discriminant_root = math.sqrt(discriminant)
            x1 = (-b + discriminant_root) / (2 * a)
            x2 = (-b - discriminant_root) / (2 * a)
        else:
            continue
        with open(r'dataset.csv', 'a', newline='') as csvfile:
            fieldnames = ['i1', 'i2', 'o1', 'o2']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({'i1': b, 'i2': c, 'o1': x1, 'o2': x2})
        entries_generated += 1

if __name__ == '__main__':
    main()
