import numpy as np
from cpmpy import *

def decompose_sum_constraint_to_binary(X, target_sum):
    # Assuming X is a list of integer variables, each ranging from 0 to some maximum value d
    d = max([var.ub for var in X])  # Find the maximum upper bound of the variables

    # Create binary variables
    binary_vars = []
    for i, var in enumerate(X):
        binary_vars.append([boolvar(name=f'b_{i}_{j}') for j in range(d + 1)])

    # Create the model
    model = Model()

    # Ensure each integer variable is represented by exactly one binary variable being 1
    for i, b in enumerate(binary_vars):
        model += [sum(b) == 1]
        model += [X[i] == sum(j * b[j] for j in range(d + 1))]

    # Define the binary sum constraint
    binary_sum_constraint = sum(sum(j * binary_vars[i][j] for j in range(d + 1)) for i in range(len(X))) >= target_sum

    # Print the binary sum constraints
    for i in range(len(X)):
        for j in range(d + 1):
            print(f'{j} * b_{i}_{j}', end=' + ' if j < d else '')
        print()

    # Print the overall sum constraint
    print(f'Sum of all binary constraints >= {target_sum}')

    # Add the binary sum constraint to the model
    model += [binary_sum_constraint]

    return model, binary_vars

# Define the range for variables (e.g., 0 to 2)
domain = range(3)  # Values from 0 to 2

# Create the variables
X = [intvar(0, 2, name=f'X{i+1}') for i in range(3)]

# Target sum
target_sum = 3

# Decompose the sum constraint into binary constraints
model, binary_vars = decompose_sum_constraint_to_binary(X, target_sum)

# Solve the model
if model.solve():
    print("Solution found:")
    for xi in X:
        print(f'{xi.name}: {xi.value()}')
    for i, b in enumerate(binary_vars):
        for j, bv in enumerate(b):
            print(f'{bv.name}: {bv.value()}')
else:
    print("No solution found")
