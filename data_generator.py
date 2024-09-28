import random
import math
import csv

DATASET_SIZE = 10000
MIN_VALUE=-10.0
MAX_VALUE=10.0

with open(r'dataset.csv', 'w', newline='') as csvfile:
    csvfile.truncate()
    fieldnames = ['a', 'b', 'c', 'x1', 'x2', 'x3', 'x4']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writerow({'a': 'a', 'b': 'b', 'c': 'c', 'x1': 'x1', 'x2': 'x2', 'x3': 'x3', 'x4': 'x4'})

for i in range(DATASET_SIZE):
    a = random.uniform(MIN_VALUE, MAX_VALUE)
    b = random.uniform(MIN_VALUE, MAX_VALUE)
    c = random.uniform(MIN_VALUE, MAX_VALUE)
    discriminant = b ** 2 - 4 * a * c
    if discriminant > 0:
        discriminant_root = math.sqrt(discriminant)
        x1 = (-b + discriminant_root)/ (2 * a)
        x2 = 0
        x3 = (-b - discriminant_root)/ (2 * a)
        x4 = 0
    else:
        discriminant_root = math.sqrt(-discriminant)
        x1 = -b / (2 * a)
        x2 = discriminant_root / (2 * a)
        x3 = -b / (2 * a)
        x4 = - discriminant_root / (2 * a)
    with open(r'dataset.csv', 'a', newline='') as csvfile:
        fieldnames = ['a', 'b', 'c', 'x1', 'x2', 'x3', 'x4']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writerow({'a': a, 'b': b, 'c': c, 'x1': x1, 'x2': x2, 'x3': x3, 'x4': x4})
