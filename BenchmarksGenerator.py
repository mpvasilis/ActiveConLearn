import os.path
import random
import re

import cpmpy as cp
from cpmpy import *
from cpmpy.expressions.utils import all_pairs
from cpmpy.transformations.normalize import toplevel_list
import json
import numpy as np
import concurrent.futures

from TestSetGenerator import _construct_greaterThanSudoku


def save_solution_to_json(grid, file_name, format_template):

    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {"formatTemplate": format_template, "solutions": []}

    flat_list = grid.value().tolist()

    num_rows, num_cols = grid.shape
    array = []
    for row in grid:
        current_row = []
        for cell in row:
            if hasattr(cell, 'value'):
                cell_value = cell.value()
                current_row.append(cell_value)
            else:
                current_row.append(cell)
        array.append(current_row)

    solution = {"array": array}

    data["solutions"].append(solution)

    with open(file_name, 'w') as f:
        json.dump(data, f, indent=4)

def save_constraints_to_txt(constraints, filename, grid_shape):
    def convert_2d_to_1d(index, shape):
        return index[0] * shape[1] + index[1]

    def extract_indices(var_name):
        match = re.search(r'\[(\d+),(\d+)\]', var_name)
        if match:
            return int(match.group(1)), int(match.group(2))
        raise ValueError(f"Invalid variable name format: {var_name}")


    with open(os.path.join("binary_cons",filename), 'w') as f:
        for constraint in constraints:
            if constraint.name == "!=":
                con_type = 0
                var1_name = constraint.args[0].name
                var2_name = constraint.args[1].name
                try:
                    var1_idx = extract_indices(var1_name)
                    var2_idx = extract_indices(var2_name)
                    var1 = convert_2d_to_1d(var1_idx, grid_shape)
                    var2 = convert_2d_to_1d(var2_idx, grid_shape)
                    f.write(f"{con_type} {var1} {var2}\n")
                except ValueError:
                    pass
def generate_solutions(model_func, json_filename, txt_filename, max_solutions=1000, min_hamming=5):
    grid, C_T, model, format_template = model_func()
    save_constraints_to_txt(C_T, txt_filename, grid.shape)
    print(f"Constraints saved for {json_filename}")

    solutions = []
    store = []
    flattened_grid = grid.flatten()

    s = SolverLookup.get('ortools', model)

    while len(solutions) < max_solutions:
        if not s.solve():
            break
        solution = flattened_grid.value().tolist()
        solutions.append(solution)
        store.append(solution)
        save_solution_to_json(grid, json_filename, format_template)
        hamming_constraints = []
        for sol in store:
            hamming_distance = sum(var != val for var, val in zip(flattened_grid, sol))
            hamming_constraints.append(hamming_distance >= min_hamming)
        # Combine all Hamming distance constraints
        model += hamming_constraints

        if len(solutions) % 100 == 0:
            print(f"Found {len(solutions)} solutions for {json_filename}")


def run_benchmarks_in_parallel():
    benchmarks = [
        # (_construct_4sudoku, '4sudoku_solution.json', '4sudoku_solution.txt'),
        # (_construct_9sudoku, '9sudoku_solution.json', '9sudoku_solution.txt'),
        (lambda: _construct_nurse_rostering(5, 3, 7), 'nurse_rostering_solution.json', 'nurse_rostering_solution.txt'),
        # (lambda: _construct_nurse_rostering_advanced(5, 3, 2, 7), 'nurse_rostering_advanced_solution.json', 'nurse_rostering_advanced_solution.txt'),
        # (lambda: _construct_examtt_simple(), 'examtt_simple_solution.json', 'examtt_simple_solution.txt'),
        # (lambda: _construct_examtt_advanced(), 'examtt_advanced_solution.json', 'examtt_advanced_solution.txt'),
        # (_construct_jsudoku, 'jsudoku_solution.json', 'jsudoku_solution.txt'),
        (_construct_murder_problem, 'murder_problem_solution.json', 'murder_problem_solution.txt'),
       #  (_construct_greaterThanSudoku, 'greaterThansudoku_solution.json', 'greaterThanSudoku_solution.txt'),
       # (lambda: construct_job_shop_scheduling_problem(n_jobs=3, machines=3, horizon=10, seed=42),
       #   'job_shop_scheduling_solution.json',
       #   'job_shop_scheduling_constraints.txt'),


        # (lambda: _construct_BIBD(7, 7, 3, 3, 1), 'BIBD_solution.json', 'BIBD_solution.txt'),
       # (lambda: _construct_golomb_ruler(6,), 'golomb_ruler_solution.json', 'golomb_ruler_solution.txt'),
       # (lambda: _construct_schurs_lemma(4,), 'schurs_lemma_solution.json', 'schurs_lemma_solution.txt')
       # (lambda:_construct_magic_square(3,), 'magic_square_solution.json', 'magic_square_solution.txt'),

    ]

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(generate_solutions, func, json_filename, txt_filename, 1000) for func, json_filename, txt_filename in benchmarks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Benchmark failed with exception: {e}")

# Helper functions to construct benchmarks

def construct_job_shop_scheduling_problem(n_jobs, machines, horizon, seed=0):
    """
    Constructs a Job Shop Scheduling problem.
    :param n_jobs: Number of jobs
    :param machines: Number of machines
    :param horizon: Time horizon
    :param seed: Random seed for reproducibility
    :return: grid (start and end times), constraints, model, and format template
    """
    random.seed(seed)
    np.random.seed(seed)
    max_time = horizon // n_jobs

    # Generate random durations for each job on each machine
    duration = [[random.randint(1, max_time) for _ in range(machines)] for _ in range(n_jobs)]

    # Assign machines to tasks (assuming each job has tasks that need to be processed on all machines)
    task_to_mach = [list(range(machines)) for _ in range(n_jobs)]
    for i in range(n_jobs):
        random.shuffle(task_to_mach[i])

    # Define precedence based on machine order
    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(n_jobs)]

    # Convert to numpy arrays for easier handling
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines_set = set(task_to_mach.flatten().tolist())

    model = cp.Model()

    # Decision variables
    start = cp.intvar(0, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(0, horizon, shape=task_to_mach.shape, name="end")

    # Define end times based on start times and durations
    model += (end == start + duration)

    # Precedence constraints within each job
    for job in range(n_jobs):
        for t in range(machines - 1):
            model += end[job, t] <= start[job, t + 1]

    # Non-overlapping constraints for tasks on the same machine
    for m in machines_set:
        # Get all tasks assigned to machine m
        tasks_on_mach = np.where(task_to_mach == m)
        tasks = list(zip(tasks_on_mach[0], tasks_on_mach[1]))
        for (j1, t1), (j2, t2) in all_pairs(tasks):
            # Either task1 finishes before task2 starts or vice versa
            model += (end[j1, t1] <= start[j2, t2]) | (end[j2, t2] <= start[j1, t1])

    # Optionally, add makespan variable and minimize it
    makespan = cp.intvar(0, horizon, name="makespan")
    model += makespan == cp.max(end)
    # To find any feasible solution, you might not need to minimize makespan
    # To enable minimization, uncomment the following line:
    # model.minimize(makespan)

    # Extract constraints and format template
    C_T = set(toplevel_list(model.constraints))
    format_template = {
        "array": [
            [{"high": horizon, "low": 0, "type": "dvar"} for _ in range(machines * 2)]  # start and end times
            for _ in range(n_jobs)
        ]
    }

    # Combine start and end into a grid for saving purposes
    # Note: Do NOT access start.value() or end.value() here
    grid = cp.cpm_array(np.hstack([start, end]))
    max_duration = max(duration)
    print(f"max_duration: {max_duration}")
    return grid, C_T, model, format_template

def _construct_magic_square(N):
    """
    Constructs a Magic Square problem.
    :param N: The size of the magic square (N x N)
    :return: square, constraints, model, and format template
    """
    # Create a square matrix of integer variables between 1 and N^2
    square = intvar(1, N * N, shape=(N, N), name="square")

    # Magic sum value
    sum_val = N * (N * N + 1) // 2  # Magic constant for rows, columns, and diagonals

    model = Model(
        # All values in the square must be distinct
        AllDifferent(square),

        # Each row must sum to the magic constant
        [sum(row) == sum_val for row in square],

        # Each column must sum to the magic constant
        [sum(col) == sum_val for col in square.T],

        # Main diagonal (top-left to bottom-right) must sum to the magic constant
        sum([square[i, i] for i in range(N)]) == sum_val,

        # Anti-diagonal (top-right to bottom-left) must sum to the magic constant
        sum([square[i, N - i - 1] for i in range(N)]) == sum_val
    )

    # Extract constraints and format template
    C_T = set(toplevel_list(model.constraints))
    format_template = {
        "array": [[{"high": N * N, "low": 1, "type": "dvar"} for _ in range(N)] for _ in range(N)]
    }

    return square, C_T, model, format_template
def _construct_schurs_lemma(N):
    """
    Constructs a Schur's Lemma problem.
    :param N: The number of balls (or size of the set)
    :return: balls, constraints, model, and format template
    """
    # Create a list of integer variables representing the partition (color) of each ball
    balls = intvar(1, 2, shape=N, name="balls")  # Two partitions/colors (can extend to more if needed)

    model = Model()

    # The 'not (x + y = z)' constraint: if x + y = z, then x, y, and z cannot all be the same color
    for x in range(1, N + 1):
        for y in range(1, N - x + 1):
            z = x + y
            model += ~((balls[x-1] == balls[y-1]) & (balls[x-1] == balls[z-1]) & (balls[y-1] == balls[z-1]))

    # Extract constraints and format template
    C_T = set(toplevel_list(model.constraints))
    format_template = {
        "array": [{"high": 2, "low": 1, "type": "dvar"} for _ in range(N)]
    }

    return balls, C_T, model, format_template
def _construct_golomb_ruler(num_marks):
    """
    Constructs a Golomb Ruler problem.
    :param num_marks: Number of marks on the ruler
    :return: marks, constraints, model, and format template
    """
    # Create a list of integer variables representing the marks on the ruler
    marks = intvar(0, num_marks ** 2, shape=num_marks, name="marks")  # Marks between 0 and num_marks^2

    model = Model()

    # Symmetry-breaking: First mark is at position 0, and marks are in increasing order
    model += marks[0] == 0
    model += [marks[i] < marks[i + 1] for i in range(num_marks - 1)]

    # The differences between every pair of marks must be distinct
    diffs = [marks[i] - marks[j] for i in range(num_marks) for j in range(i + 1, num_marks)]
    model += AllDifferent(diffs)

    # Extract constraints and format template
    C_T = set(toplevel_list(model.constraints))
    format_template = {
        "array": [{"high": num_marks ** 2, "low": 0, "type": "dvar"} for _ in range(num_marks)]
    }

    return marks, C_T, model, format_template
def _construct_BIBD(v, b, r, k, l):
    """
    Constructs a Balanced Incomplete Block Design (BIBD) problem.
    :param v: Number of treatments (points/varieties)
    :param b: Number of blocks
    :param r: Number of times each treatment is repeated
    :param k: Number of treatments per block
    :param l: Lambda, the number of times each pair of treatments appears together
    :return: grid, constraints, model, and format template
    """
    # v x b binary matrix representing whether treatment i is in block j
    matrix = intvar(0, 1, shape=(v, b), name="matrix")

    # Create the model
    model = Model(
        # Row constraints: Each treatment appears in exactly r blocks
        [sum(row) == r for row in matrix],

        # Column constraints: Each block contains exactly k treatments
        [sum(col) == k for col in matrix.T],

        # Pairwise constraint: Each pair of treatments appears together in exactly λ blocks
        [sum([matrix[i, block] * matrix[j, block] for block in range(b)]) == l
         for i in range(v) for j in range(i)]
    )

    # Extract constraints and format template
    C_T = set(toplevel_list(model.constraints))
    format_template = {
        "array": [[{"high": 1, "low": 0, "type": "dvar"} for _ in range(b)] for _ in range(v)]
    }

    return matrix, C_T, model, format_template
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
def _construct_vm_pm_allocation(n_vms, n_pms, seed=0):
    """
    Constructs a VM to PM Allocation problem.
    :param n_vms: Number of Virtual Machines
    :param n_pms: Number of Physical Machines
    :param seed: Random seed for reproducibility
    :return: allocation_matrix, constraints_set, model, format_template
    """
    import random
    import numpy as np
    import cpmpy as cp
    from cpmpy import intvar, Model, AllDifferent
    from cpmpy.expressions.utils import all_pairs
    from cpmpy.transformations.normalize import toplevel_list

    # Set seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)

    # Generate random CPU and Memory requirements for VMs
    vm_cpu = [random.randint(1, 10) for _ in range(n_vms)]
    vm_mem = [random.randint(1, 32) for _ in range(n_vms)]  # in GB

    # Generate random CPU and Memory capacities for PMs
    pm_cpu = [random.randint(int(sum(vm_cpu)/n_pms), int(sum(vm_cpu)/n_pms) + 10) for _ in range(n_pms)]
    pm_mem = [random.randint(int(sum(vm_mem)/n_pms), int(sum(vm_mem)/n_pms) + 16) for _ in range(n_pms)]  # in GB

    # Decision variables: x[i][j] = 1 if VM i is assigned to PM j, else 0
    allocation_matrix = cp.intvar(0, 1, shape=(n_vms, n_pms), name="x")

    model = cp.Model()

    # Constraint 1: Each VM is assigned to exactly one PM
    for i in range(n_vms):
        model += (sum(allocation_matrix[i, j] for j in range(n_pms)) == 1)

    # Constraint 2: Resource constraints for each PM
    for j in range(n_pms):
        # CPU Constraint
        model += (sum(allocation_matrix[i, j] * vm_cpu[i] for i in range(n_vms)) <= pm_cpu[j])
        # Memory Constraint
        model += (sum(allocation_matrix[i, j] * vm_mem[i] for i in range(n_vms)) <= pm_mem[j])

    # Extract constraints and format template
    C_T = set(toplevel_list(model.constraints))
    format_template = {
        "array": [[{"high": 1, "low": 0, "type": "bvar"} for _ in range(n_pms)] for _ in range(n_vms)],
        "vm_cpu": vm_cpu,
        "vm_memory": vm_mem,
        "pm_cpu": pm_cpu,
        "pm_memory": pm_mem
    }

    return allocation_matrix, C_T, model, format_template
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
    return grid, C_T, model, format_template


#generate_solutions((lambda: construct_job_shop_scheduling_problem(n_jobs=10, machines=3, horizon=40, seed=42)), "job.json", "job.txt", 1000)
run_benchmarks_in_parallel()
