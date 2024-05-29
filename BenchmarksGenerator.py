import random
import cpmpy as cp
from cpmpy import *
from cpmpy.expressions.utils import all_pairs
from cpmpy.transformations.normalize import toplevel_list
import json
import numpy as np
import concurrent.futures

def save_solution_to_json(grid, file_name, format_template):
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"formatTemplate": format_template, "solutions": []}

    solution = {"array": grid.value().tolist()}
    data["solutions"].append(solution)

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def generate_solutions(model_func, json_filename, max_solutions=100):
    grid, C_T, model, format_template = model_func()
    solutions_found = 0
    while solutions_found < max_solutions and model.solve():
        save_solution_to_json(grid, json_filename, format_template)
        solutions_found += 1
        model += (grid != grid.value())  # Add constraint to find a different solution

def run_benchmarks_in_parallel():
    benchmarks = [
        (_construct_4sudoku, '4sudoku_solution.json'),
        (_construct_9sudoku, '9sudoku_solution.json'),
        (lambda: _construct_nurse_rostering(5, 3, 7), 'nurse_rostering_solution.json'),
        (lambda: _construct_nurse_rostering_advanced(5, 3, 2, 7), 'nurse_rostering_advanced_solution.json'),
        (lambda: _construct_examtt_simple(), 'examtt_simple_solution.json'),
        (lambda: _construct_examtt_advanced(), 'examtt_advanced_solution.json'),
        (_construct_jsudoku, 'jsudoku_solution.json'),
        (_construct_murder_problem, 'murder_problem_solution.json')
    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_solutions, func, filename, 10) for func, filename in benchmarks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Benchmark failed with exception: {e}")

# Helper functions to construct benchmarks
def _construct_4sudoku():
    grid = intvar(1, 4, shape=(4, 4), name="grid")
    model = Model()
    for row in grid:
        model += AllDifferent(row).decompose()
    for col in grid.T:
        model += AllDifferent(col).decompose()
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            model += AllDifferent(grid[i:i + 2, j:j + 2]).decompose()
    C = list(model.constraints)
    C_T = set(toplevel_list(C))
    format_template = {"array": [[{"high": 4, "low": 1, "type": "dvar"} for _ in range(4)] for _ in range(4)]}
    return grid, C_T, model, format_template

def _construct_9sudoku():
    grid = intvar(1, 9, shape=(9, 9), name="grid")
    model = Model()
    for row in grid:
        model += AllDifferent(row).decompose()
    for col in grid.T:
        model += AllDifferent(col).decompose()
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            model += AllDifferent(grid[i:i + 3, j:j + 3]).decompose()
    C = list(model.constraints)
    C_T = set(toplevel_list(C))
    format_template = {"array": [[{"high": 9, "low": 1, "type": "dvar"} for _ in range(9)] for _ in range(9)]}
    return grid, C_T, model, format_template

def _construct_nurse_rostering(num_nurses, shifts_per_day, num_days):
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day), name="shifts")
    model = Model()
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day,:]).decompose()
    for day in range(num_days - 1):
        model += (roster_matrix[day, shifts_per_day - 1] != roster_matrix[day + 1, 0])
    C = list(model.constraints)
    C_T = set(toplevel_list(C))
    format_template = {"array": [[{"high": num_nurses, "low": 1, "type": "dvar"} for _ in range(shifts_per_day)] for _ in range(num_days)]}
    return roster_matrix, C_T, model, format_template

def _construct_nurse_rostering_advanced(num_nurses, shifts_per_day, nurses_per_shift, num_days):
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="shifts")
    model = Model()
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day,...]).decompose()
    for day in range(num_days - 1):
        model += AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()
    C = list(model.constraints)
    C_T = set(toplevel_list(C))
    format_template = {"array": [[{"high": num_nurses, "low": 1, "type": "dvar"} for _ in range(nurses_per_shift)] for _ in range(shifts_per_day)] for _ in range(num_days)}
    return roster_matrix, C_T, model, format_template

def _construct_examtt_simple(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())
    model = Model()
    model += AllDifferent(all_courses).decompose()
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()
    C = list(model.constraints)
    C_T = set(toplevel_list(C))
    format_template = {"array": [[{"high": total_slots, "low": 1, "type": "dvar"} for _ in range(courses_per_semester)] for _ in range(NSemesters)]}
    return courses, C_T, model, format_template

def _construct_examtt_advanced(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14, NProfessors=30):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())
    model = Model()
    model += AllDifferent(all_courses).decompose()
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()
    C = list(model.constraints)
    assert NProfessors <= total_courses
    courses_per_professor = total_courses // NProfessors
    remaining_courses = total_courses % NProfessors
    pcon_close = 0.3
    Prof_courses = list()
    for i in range(NProfessors):
        prof_courses = list()
        for j in range(courses_per_professor):
            prof_courses.append(all_courses.pop())
        if i < remaining_courses:
            prof_courses.append(all_courses.pop())
        Prof_courses.append(prof_courses)
        if len(prof_courses) > 1:
            r = random.uniform(0, 1)
            if r < pcon_close:
                for c1, c2 in all_pairs(prof_courses):
                    model += abs(c1 - c2) // slots_per_day <= 2
    C_T = set(toplevel_list(C))
    format_template = {"array": [[{"high": total_slots, "low": 1, "type": "dvar"} for _ in range(courses_per_semester)] for _ in range(NSemesters)]}
    return courses, C_T, model, format_template


def _construct_murder_problem():
    grid = intvar(1, 5, shape=(4, 5), name="grid")
    model = Model()

    # Constraints on rows and columns
    model += [AllDifferent(row).decompose() for row in grid]

    # Additional constraints of the murder problem
    model += [grid[0, 1] == grid[1, 2]]
    model += [grid[0, 2] != grid[1, 4]]
    model += [grid[3, 2] != grid[1, 4]]
    model += [grid[0, 2] != grid[1, 0]]
    model += [grid[0, 2] != grid[3, 4]]
    model += [grid[3, 4] == grid[1, 3]]
    model += [grid[1, 1] == grid[2, 1]]
    model += [grid[2, 3] == grid[0, 3]]
    model += [grid[2, 0] == grid[3, 3]]
    model += [grid[0, 0] != grid[2, 4]]
    model += [grid[0, 0] != grid[1, 4]]
    model += [grid[0, 0] == grid[3, 0]]

    C_T = list(model.constraints)
    C_T = toplevel_list(C_T)

    format_template = {
        "array": [
            [{"high": 5, "low": 1, "type": "dvar"} for _ in range(5)] for _ in range(4)
        ]
    }

    return grid, C_T, model, format_template

def _construct_jsudoku():
    # Variables
    grid = intvar(1, 9, shape=(9, 9), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # The 9 blocks of squares in the specific instance of jsudoku
    blocks = [
        [grid[0, 0], grid[0, 1], grid[0, 2], grid[1, 0], grid[1, 1], grid[1, 2], grid[2, 0], grid[2, 1], grid[2, 2]],
        [grid[0, 3], grid[0, 4], grid[0, 5], grid[0, 6], grid[1, 3], grid[1, 4], grid[1, 5], grid[1, 6], grid[2, 4]],
        [grid[0, 7], grid[0, 8], grid[1, 7], grid[1, 8], grid[2, 8], grid[3, 8], grid[4, 8], grid[5, 7], grid[5, 8]],
        [grid[4, 1], grid[4, 2], grid[4, 3], grid[5, 1], grid[5, 2], grid[5, 3], grid[6, 1], grid[6, 2], grid[6, 3]],
        [grid[2, 3], grid[3, 2], grid[3, 3], grid[3, 4], grid[4, 4], grid[5, 4], grid[5, 5], grid[5, 6], grid[6, 5]],
        [grid[2, 5], grid[2, 6], grid[2, 7], grid[3, 5], grid[3, 6], grid[3, 7], grid[4, 5], grid[4, 6], grid[4, 7]],
        [grid[3, 0], grid[3, 1], grid[4, 0], grid[5, 0], grid[6, 0], grid[7, 0], grid[7, 1], grid[8, 0], grid[8, 1]],
        [grid[6, 4], grid[7, 2], grid[7, 3], grid[7, 4], grid[7, 5], grid[8, 2], grid[8, 3], grid[8, 4], grid[8, 5]],
        [grid[6, 6], grid[6, 7], grid[6, 8], grid[7, 6], grid[7, 7], grid[7, 8], grid[8, 6], grid[8, 7], grid[8, 8]]
    ]

    # Constraints on blocks
    for i in range(0, 9):
        model += AllDifferent(blocks[i][:]).decompose()  # python's indexing

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(model.constraints))

    format_template = {
        "array": [
            [{"high": 9, "low": 1, "type": "dvar"} for _ in range(9)] for _ in range(9)
        ]
    }

    print(len(C_T))

    return grid, C_T, model, format_template


run_benchmarks_in_parallel()