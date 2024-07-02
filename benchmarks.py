import random
import cpmpy as cp
from cpmpy import *
from utils import *
from cpmpy.transformations.normalize import toplevel_list


# Benchmark construction
def construct_4sudoku():
    # Variables
    grid = intvar(1, 4, shape=(4, 4), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, 4, 2):
        for j in range(0, 4, 2):
            model += AllDifferent(grid[i:i + 2, j:j + 2]).decompose()  # python's indexing

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))


    return grid, C_T, model


def construct_9sudoku():
    # Variables
    grid = intvar(1, 9, shape=(9, 9), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # Constraints on blocks
    for i in range(0, 9, 3):
        for j in range(0, 9, 3):
            model += AllDifferent(grid[i:i + 3, j:j + 3]).decompose()  # python's indexing

    C = list( model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    print(len(C_T))

    return grid, C_T, model

def construct_nurse_rostering(num_nurses, shifts_per_day, num_days):

    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day), name="shifts")
    print(roster_matrix)

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day,:]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += (roster_matrix[day, shifts_per_day - 1] != roster_matrix[day + 1, 0])

    print(model)

    if model.solve():
        print(roster_matrix.value())
    else:
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    return roster_matrix, C_T, model

def construct_nurse_rostering_advanced(num_nurses, shifts_per_day, nurses_per_shift, num_days):

    # Define the variables
    roster_matrix = intvar(1, num_nurses, shape=(num_days, shifts_per_day, nurses_per_shift), name="shifts")

    # Define the constraints
    model = Model()

    # Constraint: Each shift in a day must be assigned to a different nurse
    for day in range(num_days):
        model += AllDifferent(roster_matrix[day,...]).decompose()

    # Constraint: The last shift of a day cannot have the same nurse as the first shift of the next day
    for day in range(num_days - 1):
        model += AllDifferent(roster_matrix[day, shifts_per_day - 1], roster_matrix[day + 1, 0]).decompose()

    if model.solve():
        print("solution exists")
    else:
        print("no solution")
        raise Exception("The problem has no solution")

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    C_T = set(toplevel_list(C))

    return roster_matrix, C_T, model


def construct_examtt_simple(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14):

    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()



    C = list(model.constraints)

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    C2 = [c for c in model.constraints if not isinstance(c, list)]

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(C))

    print("new model: ----------------------------------\n", C_T)
    print(C2)

    return courses, C_T, model


def construct_examtt_advanced(NSemesters=9, courses_per_semester=6, rooms=3, timeslots_per_day=3, days_for_exams=14,
                            NProfessors=30):
    total_courses = NSemesters * courses_per_semester
    slots_per_day = rooms * timeslots_per_day
    total_slots = slots_per_day * days_for_exams

    # Variables
    courses = intvar(1, total_slots, shape=(NSemesters, courses_per_semester), name="courses")
    all_courses = set(courses.flatten())

    model = Model()

    model += AllDifferent(all_courses).decompose()

    # Constraints on courses of same semester
    for row in courses:
        model += AllDifferent(row // slots_per_day).decompose()

    C = list(model.constraints)

    # Constraints of Professors - instance specific -------------------------------

    # first define the courses each professor is assigned to
    # this can be given, or random generated!!

    assert NProfessors <= total_courses
    courses_per_professor = total_courses // NProfessors
    remaining_courses = total_courses % NProfessors  # will assign 1 per professor to some professors

    # probabilities of additional constraints to be introduced
    pcon_close = 0.3  # probability of professor constraint to have his courses on close days
    # (e.g. because he lives in another city and has to come for the exams)

    #pcon_diff = 0.2  # probability of professor constraint to not have his exams in a certain day

    Prof_courses = list()
    for i in range(NProfessors):

        prof_courses = list()

        for j in range(courses_per_professor):  # assign the calculated number of courses to the professors
            prof_courses.append(all_courses.pop())  # it is a set, so it will pop a random one (there is no order)

        if i < remaining_courses:  # # assign the remaining courses to the professors
            prof_courses.append(all_courses.pop())  # it is a set, so it will pop a random one (there is no order)

        Prof_courses.append(prof_courses)

        if len(prof_courses) > 1:

            r = random.uniform(0, 1)

            if r < pcon_close:
                for c1, c2 in all_pairs(prof_courses):
                    model += abs(c1 - c2) // slots_per_day <= 2  # all her courses in 2 days

    print(model)

    if model.solve():
        print(courses.value())
    else:
        print("no solution")

    C2 = [c for c in model.constraints if not isinstance(c, list)]

    # it is needed to be able to make the "suboracle" in ask_query
    C_T = set(toplevel_list(C))


    print("new model: ----------------------------------\n", C_T)
    print(C2)

    return courses, C_T, model


def construct_jsudoku():
    # Variables
    grid = intvar(1, 9, shape=(9, 9), name="grid")

    model = Model()

    # Constraints on rows and columns
    for row in grid:
        model += AllDifferent(row).decompose()

    for col in grid.T:  # numpy's Transpose
        model += AllDifferent(col).decompose()

    # the 9 blocks of squares in the specific instance of jsudoku
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


    print(len(C_T))

    return grid, C_T, model

def construct_golomb8():
    # Variables
    grid = intvar(1, 35, shape=(1, 8), name="grid")

    model = Model()

    for i in range(8):
        for j in range(i + 1, 8):
            for x in range(j + 1, 7):
                for y in range(x + 1, 8):
                    if (y != i and x != j and x != i and y != j):
                        model += abs(grid[0, i] - grid[0, j]) != abs(grid[0, x] - grid[0, y])

    C_T = list(model.constraints)

    print(len(C_T))

    return grid, C_T, model


def construct_murder_problem():
    # Variables
    grid = intvar(1, 5, shape=(4, 5), name="grid")

    C_T = list()

    # Constraints on rows and columns
    model = Model([AllDifferent(row).decompose() for row in grid])

    # Additional constraints of the murder problem
    C_T += [grid[0, 1] == grid[1, 2]]
    C_T += [grid[0, 2] != grid[1, 4]]
    C_T += [grid[3, 2] != grid[1, 4]]
    C_T += [grid[0, 2] != grid[1, 0]]
    C_T += [grid[0, 2] != grid[3, 4]]
    C_T += [grid[3, 4] == grid[1, 3]]
    C_T += [grid[1, 1] == grid[2, 1]]
    C_T += [grid[2, 3] == grid[0, 3]]
    C_T += [grid[2, 0] == grid[3, 3]]
    C_T += [grid[0, 0] != grid[2, 4]]
    C_T += [grid[0, 0] != grid[1, 4]]
    C_T += [grid[0, 0] == grid[3, 0]]

    model += C_T

    for row in grid:
        C_T += list(AllDifferent(row).decompose())

    C_T = toplevel_list(C_T)

    return grid, C_T, model


def construct_job_shop_scheduling_problem(n_jobs, machines, horizon, seed=0):
    random.seed(seed)
    max_time = horizon // n_jobs

    duration = [[0] * machines for i in range(0, n_jobs)]
    for i in range(0, n_jobs):
        for j in range(0, machines):
            duration[i][j] = random.randint(1, max_time)

    task_to_mach = [list(range(0, machines)) for i in range(0, n_jobs)]

    for i in range(0, n_jobs):
        random.shuffle(task_to_mach[i])

    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(0, n_jobs)]

    # convert to numpy
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines = set(task_to_mach.flatten().tolist())

    model = cp.Model()

    # decision variables
    start = cp.intvar(1, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(1, horizon, shape=task_to_mach.shape, name="end")

    grid = cp.cpm_array(np.expand_dims(np.concatenate([start.flatten(), end.flatten()]), 0))

    # precedence constraints
    for chain in precedence:
        for (j1, t1), (j2, t2) in zip(chain[:-1], chain[1:]):
            model += end[j1, t1] <= start[j2, t2]

    # duration constraints
    model += (start + duration == end)

    # non_overlap constraints per machine
    for m in machines:
        tasks_on_mach = np.where(task_to_mach == m)
        for (j1, t1), (j2, t2) in all_pairs(zip(*tasks_on_mach)):
            m += (end[j1, t1] <= start[j2, t2]) | (end[j2, t2] <= start[j1, t1])

    C = list(model.constraints)

    # it is needed to be able to make the "suboracle" in ask_query ... Will fix later in a better way.
    temp = []
    for c in C:
        if isinstance(c, cp.expressions.core.Comparison):
            temp.append(c)
        elif isinstance(c, cp.expressions.variables.NDVarArray):
            _c = c.flatten()
            for __c in _c:
                temp.append(__c)
    # [temp.append(c) for c in C]
    C_T = set(temp)

    max_duration = max(duration)
    return grid, C_T, model, max_duration


def construct_job_shop_scheduling_with_precedence(n_jobs, machines, horizon, seed=0):
    random.seed(seed)
    max_time = horizon // n_jobs

    duration = [[0] * machines for i in range(n_jobs)]
    for i in range(n_jobs):
        for j in range(machines):
            duration[i][j] = random.randint(1, max_time)

    task_to_mach = [list(range(machines)) for i in range(n_jobs)]
    for i in range(n_jobs):
        random.shuffle(task_to_mach[i])

    precedence = [[(i, j) for j in task_to_mach[i]] for i in range(n_jobs)]

    # Convert to numpy
    task_to_mach = np.array(task_to_mach)
    duration = np.array(duration)
    precedence = np.array(precedence)

    machines_set = set(task_to_mach.flatten().tolist())

    model = cp.Model()

    # Decision variables
    start = cp.intvar(0, horizon, shape=task_to_mach.shape, name="start")
    end = cp.intvar(0, horizon, shape=task_to_mach.shape, name="end")
    demand = [1] * task_to_mach.size  # All tasks demand 1 unit of resource (machine)
    capacity = 1  # Each machine can handle 1 task at a time

    # Precedence constraints
    for chain in precedence:
        for (j1, t1), (j2, t2) in zip(chain[:-1], chain[1:]):
            model += end[j1, t1] <= start[j2, t2]

    # Duration constraints
    model += (start + duration == end)

    # Non-overlap constraints per machine
    for m in machines_set:
        tasks_on_mach = np.where(task_to_mach == m)
        model += Cumulative(start[tasks_on_mach], duration[tasks_on_mach], end[tasks_on_mach], demand[tasks_on_mach], capacity).decompose()

    C = list(model.constraints)
    C_T = set(toplevel_list(C))

    max_duration = max(duration.flatten())
    return start, end, C_T, model, max_duration

def construct_rcpsp(num_tasks, num_resources, max_duration, max_demand, horizon):
    random.seed(0)
    durations = [random.randint(1, max_duration) for _ in range(num_tasks)]
    demands = [[random.randint(1, max_demand) for _ in range(num_resources)] for _ in range(num_tasks)]
    total_resources = [random.randint(max_demand, max_demand * 2) for _ in range(num_resources)]

    precedence = [(random.randint(0, num_tasks-1), random.randint(0, num_tasks-1)) for _ in range(num_tasks // 2)]
    precedence = [(i, j) for i, j in precedence if i != j]

    model = cp.Model()

    # Decision variables
    start = cp.intvar(0, horizon, shape=num_tasks, name="start")
    end = cp.intvar(0, horizon, shape=num_tasks, name="end")

    # Duration constraints
    model += (start + durations == end)

    # Precedence constraints
    for i, j in precedence:
        model += end[i] <= start[j]

    # Resource constraints
    for r in range(num_resources):
        model += Cumulative(start, durations, end, [demands[i][r] for i in range(num_tasks)], total_resources[r]).decompose()

    C = list(model.constraints)
    C_T = set(toplevel_list(C))

    return start, end, C_T, model


def construct_balanced_assignment(num_tasks, num_workers):
    tasks = intvar(0, num_workers-1, shape=num_tasks, name="tasks")
    occurrences = [num_tasks // num_workers] * num_workers

    model = cp.Model()
    model += GlobalCardinalityCount(tasks, list(range(num_workers)), occurrences).decompose()

    C = list(model.constraints)
    C_T = set(toplevel_list(C))

    return tasks, C_T, model


def construct_tsp(num_cities, seed=0):
    random.seed(seed)
    np.random.seed(seed)

    # Generate random distances between cities
    distances = np.random.randint(1, 100, size=(num_cities, num_cities))
    np.fill_diagonal(distances, 0)  # Distance from a city to itself is 0

    # Variables
    path = intvar(0, num_cities-1, shape=num_cities, name="path")

    model = Model()

    # Ensure all cities are visited exactly once forming a valid circuit
    model += Circuit(path).decompose()

    # Define the cost variable
    cost = intvar(0, np.sum(distances), name="cost")

    # Constraint to calculate the total distance traveled
    model += (cost == sum(distances[path[i], path[(i+1) % num_cities]] for i in range(num_cities)))

    model.minimize(cost)

    if model.solve():
        print("Optimal path:", path.value())
        print("Minimum cost:", cost.value())
    else:
        print("No solution found")

    C = list(model.constraints)
    C_T = set(toplevel_list(C))

    return path, cost, C_T, model

# Example usage
num_cities = 5
path, cost, constraints, model = construct_tsp(num_cities)

print("Constraints:")
for constraint in constraints:
    print(constraint)
