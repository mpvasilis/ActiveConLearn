import json
import os
import shutil
from datetime import datetime
import matplotlib.pyplot as plt
from cpmpy import *
from cpmpy.solvers import CPM_ortools
from cpmpy.expressions.variables import intvar
from typing import List, Dict, Tuple

class Constraint:
    def __init__(self, name, values, indexes, value=None, operator=None):
        self.name = name
        self.values = values
        self.indexes = indexes
        self.value = value
        self.operator = operator

    def __str__(self):
        return f"Constraint(name={self.name}, values={self.values}, indexes={self.indexes}, value={self.value}, operator={self.operator})"

class Problem:
    def __init__(self, problem_type, instance, size, solutions, input_data):
        self.problem_type = problem_type
        self.instance = instance
        self.size = size
        self.solutions = solutions
        self.input_data = input_data

class CATFly:
    def __init__(self, instance_file: str, constraints_to_check: List[str]):
        self.instance_file = instance_file
        self.constraints_to_check = constraints_to_check
        self.learned_constraints = []
        self.use_active_learning = False
        self.num_of_solutions_to_use = 0
        self.show_prints = True
        self.problem = None

    def set_active_learning(self, use_active_learning: bool):
        self.use_active_learning = use_active_learning

    def read_instance_file(self):
        with open(self.instance_file, 'r') as f:
            instance_data = json.load(f)
        self.problem = Problem(
            problem_type=instance_data['problemType'],
            instance=instance_data['instance'],
            size=instance_data['size'],
            solutions=instance_data['solutions'],
            input_data=instance_data['inputData']
        )

    def handle_array_solution(self, solution):
        grid2d = solution['array']
        size_y = len(grid2d)
        size_x = len(grid2d[0]) if size_y > 0 else 0
        indexes2d = [[(i * size_x) + j for j in range(size_x)] for i in range(size_y)]

        self.generate_row_constraints(grid2d, indexes2d)
        self.generate_column_constraints(grid2d, indexes2d)
        self.generate_diagonal_constraints(grid2d, indexes2d)
        self.generate_sliding_window_constraints(grid2d, indexes2d)

    def generate_row_constraints(self, grid2d, indexes2d):
        for row_idx, row in enumerate(grid2d):
            row_values = row
            row_indexes = indexes2d[row_idx]
            self.learned_constraints.append(Constraint('row', row_values, row_indexes))

    def generate_column_constraints(self, grid2d, indexes2d):
        for col_idx in range(len(grid2d[0])):
            col_values = [grid2d[row_idx][col_idx] for row_idx in range(len(grid2d))]
            col_indexes = [indexes2d[row_idx][col_idx] for row_idx in range(len(grid2d))]
            self.learned_constraints.append(Constraint('column', col_values, col_indexes))

    def generate_diagonal_constraints(self, grid2d, indexes2d):
        size_y = len(grid2d)
        size_x = len(grid2d[0])
        primary_diag_values = [grid2d[i][i] for i in range(min(size_y, size_x))]
        primary_diag_indexes = [indexes2d[i][i] for i in range(min(size_y, size_x))]
        self.learned_constraints.append(Constraint('primary_diagonal', primary_diag_values, primary_diag_indexes))

        secondary_diag_values = [grid2d[i][size_x - i - 1] for i in range(min(size_y, size_x))]
        secondary_diag_indexes = [indexes2d[i][size_x - i - 1] for i in range(min(size_y, size_x))]
        self.learned_constraints.append(Constraint('secondary_diagonal', secondary_diag_values, secondary_diag_indexes))

    def generate_sliding_window_constraints(self, grid2d, indexes2d):
        window_size = 2  # Example window size, can be parameterized
        size_y = len(grid2d)
        size_x = len(grid2d[0])
        for row in range(size_y - window_size + 1):
            for col in range(size_x - window_size + 1):
                window_values = [grid2d[row + i][col + j] for i in range(window_size) for j in range(window_size)]
                window_indexes = [indexes2d[row + i][col + j] for i in range(window_size) for j in range(window_size)]
                self.learned_constraints.append(Constraint('sliding_window', window_values, window_indexes))

    def generate_constraints(self):
        for solution in self.problem.solutions:
            if 'array' in solution:
                self.handle_array_solution(solution)
            elif 'list' in solution:
                self.handle_list_solution(solution)

    def handle_list_solution(self, solution):
        lst = solution['list']
        indexes = list(range(len(lst)))
        self.learned_constraints.append(Constraint('list', lst, indexes))

    def run(self):
        if self.show_prints:
            print("Starting Global Constraint Acquisition Tool")
            print("Reading instance file:", self.instance_file)
        self.read_instance_file()
        self.generate_constraints()
        if self.show_prints:
            print("Learned constraints:")
            for constraint in self.learned_constraints:
                print(constraint)

if __name__ == "__main__":
    instance_file = "path_to_instance_file.json"
    constraints_to_check = ["allDifferent", "sum", "arithm"]
    catfly = CATFly(instance_file, constraints_to_check)
    catfly.set_active_learning(True)
    catfly.run()
